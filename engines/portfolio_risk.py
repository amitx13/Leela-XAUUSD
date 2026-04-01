"""
engines/portfolio_risk.py — Portfolio Risk Brain (decision layer).

Calls PositionManager for raw data. Owns all cross-strategy exposure decisions.
PositionManager has zero awareness of these rules.

Controls (Phase 1):
  max_daily_var:     2% of account, all strategies combined
  max_session_lots:  0.15 lots total per session
  volatility_scaling: SUPER_TRENDING → 0.5× except S3 (S3 applies 0.5× in signal only)
  correlation_kill:  same direction simultaneously → second at 0.5×

VAR formula is intentionally conservative (sums ALL positions, no netting).
Opposing direction VAR is overstated on hedged days — acceptable Phase 1 behavior.
Correlation-adjusted VAR is Phase 3. See v1.1 notes.
"""
import config
from utils.logger import log_event, log_warning
from engines.position_manager import get_open_positions, get_total_open_lots


def check_portfolio_risk(candidate: dict, state: dict) -> tuple[bool, str]:
    """
    Called by dispatch_candidate() in main.py before place_order().
    Returns (permitted, reason).
    """
    from engines.regime_engine import get_safe_regime, RegimeState
    from utils.mt5_client import get_mt5

    mt5  = get_mt5()
    info = mt5.account_info()
    if info is None:
        log_warning("PORTFOLIO_RISK_ACCOUNT_INFO_FAILED")
        return True, "OK"   # can't check → don't block

    equity     = float(info.equity)
    direction  = candidate.get("direction", "LONG")
    lot_size   = candidate.get("lot_size", 0.01)

    # ── Daily VAR check ────────────────────────────────────────────────────────
    # F8 FIX: Use cached ATR from state instead of fetching on every call
    # The regime_job updates state["last_atr_h1_raw"] every 15 minutes
    atr_h1 = state.get("last_atr_h1_raw", 0.0)

    if atr_h1 <= 0:
        # Fallback: fetch if not cached (should rarely happen)
        from engines.data_engine import fetch_ohlcv
        import pandas_ta as ta
        import math
        df_h1 = fetch_ohlcv("H1", count=20)
        if df_h1 is not None and not df_h1.empty:
            df_h1["atr"] = ta.atr(df_h1["high"], df_h1["low"],
                                   df_h1["close"], length=14,
                                   mamode=config.ATR_MAMODE)
            val = df_h1["atr"].iloc[-1]
            if val is not None and not math.isnan(val):
                atr_h1 = float(val)

    if atr_h1 <= 0:
        log_warning("ATR_UNAVAILABLE_VAR_BLOCK")
        return False, "ATR_UNAVAILABLE"

    open_pos   = get_open_positions()
    contract   = config.CONTRACT_SPEC.get("contract_size", 100)
    current_var = sum(p["lot_size"] * atr_h1 * contract
                      for p in open_pos.values())
    new_var     = lot_size * atr_h1 * contract

    if equity > 0 and (current_var + new_var) / equity > config.MAX_DAILY_VAR_PCT:
        log_event("PORTFOLIO_VAR_BLOCKED",
                  current_var=round(current_var, 2),
                  new_var=round(new_var, 2),
                  equity=round(equity, 2))
        return False, "DAILY_VAR_LIMIT_REACHED"

    # ── Session lots cap ───────────────────────────────────────────────────────
    final_lots = candidate["lot_size"]

    if get_total_open_lots() + final_lots > config.MAX_SESSION_LOTS:
        return False, "SESSION_LOT_CAP_REACHED"

    # SIZE-4 FIX: Removed SUPER_TRENDING double-halving. The regime multiplier
    # already handles sizing (1.5× for SUPER). Halving here on top of that made
    # SUPER_TRENDING get LESS size than NORMAL_TRENDING — completely backward.
    # The strongest trend signal should get the largest size, not the smallest.

    # SIZE-5 FIX: Correlation kill — only apply when strategies are from the same family.
    # Same-direction ≠ correlated. S1 LONG and S7 LONG are independent strategies
    # on different timeframes. Only reduce when same family + same direction.
    TREND_FAMILY = {"S1_LONDON_BRK", "S1F_POST_TK", "S4_LONDON_PULL", "S5_NY_COMPRESS"}
    candidate_signal = candidate.get("signal_type", "")
    same_family_same_dir = any(
        p["direction"] == direction
        and sig in TREND_FAMILY
        and candidate_signal in TREND_FAMILY
        for sig, p in open_pos.items()
    )

    if same_family_same_dir:
        candidate["lot_size"] = max(
            config.CONTRACT_SPEC.get("volume_min", 0.01),
            round(candidate["lot_size"] * 0.65, 2)
        )
        log_event("CORRELATION_KILL_SAME_FAMILY",
                  lots=candidate["lot_size"],
                  same_direction=direction,
                  signal=candidate_signal)

    return True, "OK"

def run_correlation_check(state: dict) -> None:
    """
    Section 3.4 — P&L Correlation Matrix.
    Run after every PORTFOLIO_CORR_CHECK_EVERY_N new closed trades.
    If any strategy pair correlation > 0.65, log WARNING and reduce exposure.
    Only meaningful after 10+ trades per strategy.
    """
    import pandas as pd
    from db.connection import execute_query
    from utils.logger import log_event, log_warning
    import config

    rows = execute_query(
        """SELECT signal_type AS strategy_id,
                  DATE(exit_time AT TIME ZONE 'Asia/Kolkata') AS trade_date,
                  SUM(pnl_net_dollars) AS daily_pnl
           FROM system_state.trades
           WHERE exit_time IS NOT NULL
             AND signal_type IS NOT NULL
           GROUP BY signal_type, DATE(exit_time AT TIME ZONE 'Asia/Kolkata')
           ORDER BY trade_date""",
        {}
    )

    if not rows or len(rows) < 10:
        log_event("CORRELATION_CHECK_SKIPPED", reason="insufficient_data", rows=len(rows) if rows else 0)
        return

    df = pd.DataFrame(rows)
    df["daily_pnl"] = pd.to_numeric(df["daily_pnl"], errors="coerce").fillna(0)

    pivot = df.pivot_table(
        index="trade_date", columns="strategy_id", values="daily_pnl", aggfunc="sum"
    ).fillna(0)

    strategies = pivot.columns.tolist()
    if len(strategies) < 2:
        log_event("CORRELATION_CHECK_SKIPPED", reason="only_one_strategy")
        return

    corr_matrix = pivot.corr()
    high_corr_pairs = []

    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            corr_val = float(corr_matrix.loc[s1, s2])
            if abs(corr_val) > config.PORTFOLIO_CORR_THRESHOLD:
                high_corr_pairs.append((s1, s2, round(corr_val, 3)))
                log_warning("STRATEGY_CORRELATION_HIGH",
                            pair=f"{s1}/{s2}",
                            correlation=round(corr_val, 3),
                            threshold=config.PORTFOLIO_CORR_THRESHOLD,
                            action="combined_max_exposure_reduced_to_1.5x")

    if not high_corr_pairs:
        log_event("CORRELATION_CHECK_PASSED",
                  strategies=strategies,
                  note="No pairs exceed correlation threshold")
    else:
        # Throttle: set flag for portfolio_ks_job to enforce reduced exposure
        state["high_corr_pairs"] = high_corr_pairs

    log_event("CORRELATION_MATRIX_COMPUTED",
              strategies=strategies,
              pairs_checked=len(strategies) * (len(strategies) - 1) // 2,
              high_corr_count=len(high_corr_pairs))