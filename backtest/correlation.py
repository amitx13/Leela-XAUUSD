"""
backtest/correlation.py — Portfolio P&L correlation check for the backtest.

CRIT-3 FIX
──────────
The live system runs engines/portfolio_risk.run_correlation_check() after
every PORTFOLIO_CORR_CHECK_EVERY_N closed trades. It builds a per-strategy
daily-PnL pivot, computes Pearson correlations, and flags high-corr pairs
so that check_portfolio_risk() can apply a 0.65× lot-size reduction for
same-family same-direction concurrent entries.

The backtest now mirrors this exactly:
  run_portfolio_correlation_check(state, trade_log)
    - Called from BacktestEngine._on_position_closed() (engine.py)
      whenever a trade is added to self._trade_log.
    - Respects the PORTFOLIO_CORR_CHECK_EVERY_N throttle.
    - Updates state.corr_throttle_active and state.corr_throttle_pairs.
    - _evaluate_strategies() reads corr_throttle_active and, when True,
      reduces lot size by 0.65× for TREND_FAMILY same-direction orders.

Fidelity note:
  The live check queries a live Postgres DB; here we build the pivot from
  the in-memory TradeRecord list. Semantics are identical — daily PnL per
  strategy, Pearson correlation matrix across pairs.
"""
import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from backtest.models import SimulatedState, TradeRecord

logger = logging.getLogger("backtest.correlation")

# Strategies that belong to the trend family (mirrors live TREND_FAMILY set)
TREND_FAMILY: set[str] = {
    "S1_LONDON_BRK",
    "S1B_FAILED_BRK",
    "S1F_POST_TK",
    "S1E_PYRAMID",
    "S4_EMA_PULLBACK",
    "S5_NY_COMPRESS",
}


def run_portfolio_correlation_check(
    state: "SimulatedState",
    trade_log: list["TradeRecord"],
) -> None:
    """
    Compute the P&L correlation matrix from closed trades and update
    state.corr_throttle_active / state.corr_throttle_pairs.

    Mirrors live engines/portfolio_risk.run_correlation_check().

    Throttle:
      Only runs every PORTFOLIO_CORR_CHECK_EVERY_N closed trades
      (default 5 if not defined in config). Uses
      state._corr_check_trade_counter to track.

    Correlation threshold:
      config.PORTFOLIO_CORR_THRESHOLD (default 0.65 if not defined).
      Any pair with |Pearson r| > threshold is flagged.

    Minimum data:
      Requires at least 10 trade records per strategy before flagging.
      Below that, keeps current state unchanged (conservative).
    """
    try:
        import config as _config
        check_every   = getattr(_config, "PORTFOLIO_CORR_CHECK_EVERY_N", 5)
        corr_thresh   = getattr(_config, "PORTFOLIO_CORR_THRESHOLD",     0.65)
    except ImportError:
        check_every = 5
        corr_thresh = 0.65

    # Increment counter; only run on the Nth trade
    state._corr_check_trade_counter += 1
    if state._corr_check_trade_counter < check_every:
        return
    state._corr_check_trade_counter = 0

    if len(trade_log) < 10:
        logger.debug("Correlation check skipped: fewer than 10 closed trades.")
        return

    # ── Build daily-PnL pivot ────────────────────────────────────────────────
    records = [
        {
            "strategy":   t.strategy,
            "trade_date": t.exit_time.date(),
            "pnl":        t.pnl,
        }
        for t in trade_log
        if t.exit_time is not None
    ]
    if not records:
        return

    df = pd.DataFrame(records)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    pivot = (
        df.groupby(["trade_date", "strategy"])["pnl"]
        .sum()
        .unstack(fill_value=0.0)
    )

    strategies = pivot.columns.tolist()
    if len(strategies) < 2:
        logger.debug("Correlation check skipped: only one strategy in trade log.")
        return

    # Require at least 10 rows (trading days) per strategy for stable estimates
    min_rows = 10
    eligible = [s for s in strategies if pivot[s].abs().sum() > 0 and len(pivot[pivot[s] != 0]) >= min_rows]
    if len(eligible) < 2:
        logger.debug(
            f"Correlation check skipped: fewer than 2 strategies have "
            f">= {min_rows} active trading days."
        )
        return

    corr_matrix = pivot[eligible].corr()

    high_corr_pairs: list[tuple] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            s_a, s_b = eligible[i], eligible[j]
            val = float(corr_matrix.loc[s_a, s_b])
            if abs(val) > corr_thresh:
                high_corr_pairs.append((s_a, s_b, round(val, 3)))
                logger.warning(
                    f"[CRIT-3] High correlation: {s_a}/{s_b} r={val:.3f} "
                    f"(threshold={corr_thresh}) — lot throttle ACTIVE"
                )

    if high_corr_pairs:
        state.corr_throttle_active = True
        state.corr_throttle_pairs  = high_corr_pairs
        logger.info(
            f"Correlation throttle ACTIVATED: {len(high_corr_pairs)} pair(s) above {corr_thresh}. "
            f"TREND_FAMILY same-direction lots will be reduced to 0.65×."
        )
    else:
        # Clear the flag if no pairs currently exceed threshold
        if state.corr_throttle_active:
            logger.info("Correlation throttle CLEARED: no pairs above threshold.")
        state.corr_throttle_active = False
        state.corr_throttle_pairs  = []

    logger.debug(
        f"Correlation check complete: {len(strategies)} strategies, "
        f"{len(high_corr_pairs)} high-corr pairs."
    )


def apply_correlation_lot_reduction(
    order_strategy: str,
    order_direction: str,
    open_positions: list,   # list of SimPosition
    state: "SimulatedState",
    lots: float,
) -> float:
    """
    Apply the 0.65× lot reduction if corr_throttle_active is True AND
    the new order is from the TREND_FAMILY in the same direction as any
    currently open TREND_FAMILY position.

    Called by _evaluate_strategies() in engine.py just before appending
    a SimOrder to the pending orders list.

    Mirrors the SIZE-5 logic in live engines/portfolio_risk.check_portfolio_risk().

    Returns:
      Adjusted lot size (float). Minimum 0.01.
    """
    if not state.corr_throttle_active:
        return lots
    if order_strategy not in TREND_FAMILY:
        return lots

    same_family_same_dir = any(
        pos.strategy in TREND_FAMILY and pos.direction == order_direction
        for pos in open_positions
    )

    if same_family_same_dir:
        reduced = max(0.01, round(lots * 0.65, 2))
        logger.debug(
            f"[CRIT-3] Correlation lot reduction applied: {lots} → {reduced} "
            f"({order_strategy} {order_direction})"
        )
        return reduced

    return lots
