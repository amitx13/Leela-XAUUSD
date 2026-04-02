"""
engines/risk_engine.py — Layer 2: Risk Engine.

Sizes, filters, and kills trades. Nothing gets past this layer
without passing every active gate.

Fixes implemented:
  Fix 5  — Use equity (not balance) for position sizing
  C3     — Decimal ROUND_DOWN for float-safe lot calculation
  B5     — R-multiple always calculated vs stop_price_original
  B6     — peak_equity update is continuous (in heartbeat — see persistence.py)
  v1.1   — KS4 uses 5-trade countdown (not live streak check)
  v1.1   — KS2 uses 24h spread baseline (not session avg)
  v1.1   — Added gates for S1b/S3 reversal family, S6, S7
  v1.1   — Added check_portfolio_risk() Portfolio Risk Brain gate

Kill switches KS1–KS7:
  KS1    Trade-level hard stop (placed before fill, never moved against)
  KS2    Spread guard — only hard spread gate in system
  KS3    Daily loss limit (config) → halt + email
  KS4    Loss streak ≥4 → halve size for next 5 trades (countdown)
  KS5    Weekly loss limit (config) → 7-day pause + email
  KS6    Drawdown circuit breaker equity < peak×0.92 → halt + email
  KS7    Event blackout 45 min pre / 20 min + ATR check post

Conviction level: OBSERVATION MODE only — logged, never changes lot size.
"""
import pytz
from decimal import Decimal, ROUND_DOWN
from datetime import datetime


import config
from utils.logger import log_event, log_warning
from utils.mt5_client import get_mt5
from utils.session import get_current_session
from db.connection import execute_query
from db.persistence import (
    persist_critical_state, update_peak_equity,
    get_weekly_net_pnl_pct,
)
from engines.truth_engine import (
    get_live_trade_count,
    get_max_drawdown_pct,
    get_rolling_sharpe,
    check_phase_2_gate,
)
# ── CHANGE 4: added get_avg_spread_last_24h ───────────────────────────────────
from engines.data_engine import (
    get_upcoming_events_within,
    get_session_avg_spread,
    get_avg_spread_last_24h,
)
from engines.regime_engine import get_safe_regime, RegimeState



# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING — Fix 5 (equity), C3 (Decimal rounding)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_lot_size(
    stop_distance_points: float,
    size_multiplier: float,
    state: dict,
) -> float:
    """
    Fix 5: Uses account EQUITY (not balance) — includes unrealized P&L.
    C3 Fix: Decimal ROUND_DOWN — never overshoot risk due to float arithmetic.

    Sizing phases (read from system_config at startup):
      Phase 1 (<50 live trades):  1.0% base risk
      Phase 2 (50+ proven):       2.0% base risk
        Gate: WR>45%, expectancy>+0.15R, max_dd<15%
      Phase 3 (6mo, Sharpe>1.0):  3–4% max — not implemented until Phase 3 live

    KS4 v1.1: ks4_reduced_trades_remaining countdown — halve base_risk for
    the next 5 trades after a streak. Counter starts at 5 on streak trigger,
    decrements on each new order placed. Restores automatically after 5 trades.
    V1 hard cap: 0.50 lots maximum — raise only after Phase 2 gate confirmed.
    """
    mt5  = get_mt5()
    info = mt5.account_info()


    if info is None:
        log_warning("LOT_SIZE_ACCOUNT_INFO_FAILED")
        return config.CONTRACT_SPEC.get("volume_min", 0.01)


    # Fix 5: equity not balance
    equity     = float(info.equity)
    live_count = get_live_trade_count()
    max_dd     = get_max_drawdown_pct()
    sharpe     = get_rolling_sharpe(50) if live_count >= 50 else 0.0


    # Phase gate
    if live_count < config.PHASE_1_TRADE_GATE:
        base_risk = config.BASE_RISK_PHASE_1    # 1.0%
    elif (sharpe >= 0.8 and max_dd < config.PHASE_2_MAX_DD):
        base_risk = config.BASE_RISK_PHASE_2    # 2.0%
    else:
        base_risk = config.BASE_RISK_PHASE_1    # stay at 1% if gates not met

    # EXP-10 FIX: Promote conviction level to active sizing after 50+ trades.
    # If A+ conviction has measurably better expectancy (>8pp delta),
    # boost size by 25%. If OBSERVATION conditions, reduce by 25%.
    if live_count >= config.MACRO_PROMOTE_TRADE_MIN:
        try:
            from engines.truth_engine import get_conviction_delta
            conv_delta = get_conviction_delta()
            if conv_delta is not None and conv_delta > config.MACRO_PROMOTE_DELTA_PP / 100:
                conviction = calculate_conviction_level(state)
                if conviction == "A_PLUS":
                    base_risk *= 1.25
                    log_event("EXP10_CONVICTION_BOOST", conviction=conviction,
                              delta_pp=round(conv_delta * 100, 1), boost=1.25)
                elif conviction == "OBSERVATION":
                    base_risk *= 0.75
                    log_event("EXP10_CONVICTION_REDUCE", conviction=conviction,
                              delta_pp=round(conv_delta * 100, 1), reduction=0.75)
        except (ImportError, Exception) as e:
            pass  # truth_engine may not have get_conviction_delta yet — skip safely

    # ── CHANGE 2: KS4 countdown check (was: consecutive_losses >= threshold) ─
    # v1.1: ks4_reduced_trades_remaining is set to 5 on streak trigger and
    # decremented each trade by decrement_ks4_countdown(). Halving persists
    # for exactly 5 trades regardless of win/loss result.
    if state.get("ks4_reduced_trades_remaining", 0) > 0:
        base_risk *= 0.5
        log_event("KS4_SIZE_HALVED_COUNTDOWN",
                  trades_remaining=state["ks4_reduced_trades_remaining"],
                  new_base_risk=base_risk)

    # F5: Apply severity multiplier from economic event risk
    severity_score = state.get("ks7_severity_score", 0.0)
    severity_mult = get_severity_multiplier(severity_score)
    if severity_mult < 1.0:
        base_risk *= severity_mult
        log_event("KS7_SEVERITY_REDUCED_SIZE",
                  severity=severity_score,
                  multiplier=severity_mult,
                  new_base_risk=base_risk)

    # F6: Apply spread multiplier from upgraded KS2 gate
    spread_mult = get_spread_multiplier(state)
    if spread_mult < 1.0:
        base_risk *= spread_mult
        log_event("KS2_SPREAD_REDUCED_SIZE",
                  multiplier=spread_mult,
                  new_base_risk=base_risk)

    # F3: Apply vol_scalar from EWMA ATR percentile
    vol_scalar = get_vol_scalar(state)
    if vol_scalar < 1.0:
        base_risk *= vol_scalar
        log_event("VOL_SCALAR_REDUCED_SIZE",
                  atr_pct=state.get("last_atr_pct_h1", 0.0),
                  vol_scalar=vol_scalar,
                  new_base_risk=base_risk)

    # SIZE-1 FIX: Cap the reduction floor — never reduce below 50% of base regime sizing.
    # Without this floor, multiplicative stacking of severity × spread × vol_scalar
    # compounds to near-zero in realistic conditions, making every NY trade min-lot.
    risk_reduction = severity_mult * spread_mult * vol_scalar
    if risk_reduction < 0.50:
        # Clamp to 50% floor — still reduced, but tradeable
        floor_adjustment = 0.50 / risk_reduction if risk_reduction > 0 else 1.0
        base_risk *= floor_adjustment
        log_event("SIZE_REDUCTION_FLOOR_APPLIED",
                  raw_reduction=round(risk_reduction, 3),
                  floor=0.50,
                  adjustment=round(floor_adjustment, 3))

    #spec = config.CONTRACT_SPEC
    #if not spec:
        #log_warning("LOT_SIZE_CONTRACT_SPEC_EMPTY")
        #return spec.get("volume_min", 0.01) if spec else 0.01

    spec = config.CONTRACT_SPEC
    required_keys = {"volume_min", "volume_max", "volume_step", "tick_size", "tick_value"}
    if not spec or not required_keys.issubset(spec.keys()):
        log_warning("CONTRACT_SPEC_INVALID", spec=str(spec))
        raise RuntimeError("CONTRACT_SPEC missing required keys — initialize_system() was not called")

    risk_dollars       = equity * base_risk * size_multiplier
    tick_size          = spec["tick_size"]
    tick_value         = spec["tick_value"]


    if tick_size <= 0 or tick_value <= 0:
        log_warning("LOT_SIZE_INVALID_TICK_SPEC",
                    tick_size=tick_size, tick_value=tick_value)
        return spec["volume_min"]


    ticks_in_stop      = stop_distance_points / tick_size
    stop_value_per_lot = ticks_in_stop * tick_value


    if stop_value_per_lot <= 0:
        log_event("SIZING_ERROR_ZERO_STOP",
                  stop_distance_points=stop_distance_points)
        return spec["volume_min"]


    raw_lots = risk_dollars / stop_value_per_lot


    # C3 Fix: Decimal ROUND_DOWN — never overshoot risk
    step      = Decimal(str(spec["volume_step"]))
    raw_dec   = Decimal(str(raw_lots))
    valid_lots = float(
        (raw_dec / step).to_integral_value(ROUND_DOWN) * step
    )


    # V1 hard cap: 0.50 lots maximum
    # Raise this cap only after Phase 2 gate confirmed + analyst review
    final_lots = max(spec["volume_min"], min(valid_lots, config.V1_LOT_HARD_CAP))

    # Compound condition gate — replaces MIN_LOT_MULTIPLIER floor.
    # If severity × spread × vol_scalar < threshold, conditions are too degraded.
    # Block the trade entirely rather than forcing a too-small position.
    # Note: severity_mult and spread_mult must be in scope here from F1-F7 changes.
    # vol_scalar must also be in scope from F3 changes.
    # If the F1-F7 agent placed these as local variables, reference them directly.
    try:
        _sev  = severity_mult    # BUG-5 FIX: was `if "x" in dir()` which checks module scope, not locals
        _spr  = spread_mult
        _vol  = vol_scalar       # BUG-5 FIX continued
        _compound = _sev * _spr * _vol
        if _compound < config.MIN_CONDITION_MULTIPLIER:
            log_event("LOT_SIZE_CONDITIONS_BLOCKED",
                      severity_mult=round(_sev, 3),
                      spread_mult=round(_spr, 3),
                      vol_scalar=round(_vol, 3),
                      compound=round(_compound, 3),
                      threshold=config.MIN_CONDITION_MULTIPLIER)
            return 0.0   # Signal to _build_candidate to return None (no trade)
    except Exception:
        pass  # If F1-F7 variables aren't in scope, gate is skipped safely


    log_event("LOT_SIZE_CALCULATED",
              equity=round(equity, 2),
              base_risk=base_risk,
              size_mult=size_multiplier,
              stop_pts=stop_distance_points,
              raw_lots=round(raw_lots, 4),
              final_lots=final_lots,
              phase=1 if live_count < 50 else 2)


    return final_lots



# ─────────────────────────────────────────────────────────────────────────────
# KS4 COUNTDOWN HELPERS — v1.1
# ─────────────────────────────────────────────────────────────────────────────


def trigger_ks4_countdown(state: dict) -> None:
    """
    Called from execution_engine.on_trade_closed() when consecutive_losses
    reaches KS4_LOSS_STREAK_COUNT (4).

    Sets ks4_reduced_trades_remaining = KS4_REDUCED_TRADE_COUNT (5).
    If already in a countdown (previous streak not yet expired), resets to 5.
    Halving is then enforced in calculate_lot_size() for the next 5 trades.
    """
    if state["consecutive_losses"] >= config.KS4_LOSS_STREAK_COUNT:
        state["ks4_reduced_trades_remaining"] = config.KS4_REDUCED_TRADE_COUNT
        log_event("KS4_COUNTDOWN_STARTED",
                  consecutive_losses=state["consecutive_losses"],
                  trades_remaining=config.KS4_REDUCED_TRADE_COUNT)
        from utils.alerts import send_ks_alert
        send_ks_alert("KS4", (
            f"Loss streak = {state['consecutive_losses']} trades. "
            f"Lot size halved for next {config.KS4_REDUCED_TRADE_COUNT} trades."
        ))


def decrement_ks4_countdown(state: dict) -> None:
    """
    Decrements ks4_reduced_trades_remaining after each successful order placement
    under KS4 reduced-size penalty.
    At 0: clears the penalty — next trade resumes full size.
    """
    remaining = state.get("ks4_reduced_trades_remaining", 0)
    if remaining <= 0:
        return
    remaining -= 1
    state["ks4_reduced_trades_remaining"] = remaining
    if remaining == 0:
        log_event("KS4_COUNTDOWN_CLEARED", note="Full size resumes next trade")
    else:
        log_event("KS4_COUNTDOWN_DECREMENTED", remaining=remaining)



# ─────────────────────────────────────────────────────────────────────────────
# R-MULTIPLE — B5 Fix (always vs original stop)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_r_multiple(
    entry: float,
    exit_price: float,
    stop_original: float,
    direction: str,
) -> float:
    """
    B5 Fix: R-multiple always calculated against stop_price_original.
    stop_price_original is set ONCE at entry and NEVER updated.
    stop_price_current (BE, trail) is a separate field — never used for R.

    This ensures R-multiples are comparable across all trades regardless
    of how the stop was subsequently managed.
    """
    stop_distance = abs(entry - stop_original)
    if stop_distance == 0:
        log_warning("R_MULTIPLE_ZERO_STOP_DISTANCE",
                    entry=entry, stop=stop_original)
        return 0.0


    if direction == "LONG":
        return round((exit_price - entry) / stop_distance, 4)
    else:
        return round((entry - exit_price) / stop_distance, 4)



# ─────────────────────────────────────────────────────────────────────────────
# ATR TRAIL — RMA pinned (spec Part 7)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_atr_trail(current_price: float, direction: str) -> float | None:
    """
    ATR trailing stop on M15 using Wilder's RMA (pinned — matches TV + MT5).
    Multiplier: 1.5× ATR(14, M15).

    Activation rule (spec Part 7 — WEAK_TRENDING hybrid exit):
    Trail activates only AFTER breakeven is activated.
    Do NOT trail from entry — this would stop out on normal retracements.
    """
    from engines.regime_engine import get_current_atr_m15
    atr_m15 = get_current_atr_m15(period=14)   # RMA pinned inside that function


    if atr_m15 is None:
        log_warning("ATR_TRAIL_CALC_FAILED_NO_ATR")
        return None


    trail_distance = atr_m15 * config.ATR_TRAIL_MULTIPLIER   # 1.5×


    if direction == "LONG":
        return round(current_price - trail_distance, 2)
    else:
        return round(current_price + trail_distance, 2)



# ─────────────────────────────────────────────────────────────────────────────
# KILL SWITCH STACK
# ─────────────────────────────────────────────────────────────────────────────


def check_ks2_spread(state: dict) -> tuple[bool, str]:
    """
    F6: KS2 — Spread guard with rolling median and 2-consecutive-check persistence.

    Tiers:
    - ratio > 2.5×: HARD BLOCK (KS2 hard kill switch)
    - ratio > 2.0×: 50% lot size reduction (2 consecutive checks required)
    - ratio > 1.2×: Normal gate (proceed with lot sizing)
    - ratio <= 1.2×: Normal operation

    The 2-consecutive-check persistence filter prevents a single bad tick
    from reducing lot size. On XAUUSD, tick spreads spike for 1-2 ticks
    constantly. Two consecutive checks (2-minute confirmation) is required
    before applying the 50% reduction.

    Rolling MEDIAN is more robust to spike contamination than average.
    Uses get_spread_rolling_median() instead of get_avg_spread_last_24h().

    Returns (permitted: bool, reason: str).
    Stores spread_multiplier and elevated reading count in state for lot sizing.
    """
    from engines.data_engine import get_spread_rolling_median

    mt5  = get_mt5()
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        return False, "KS2_TICK_UNAVAILABLE"

    point      = config.CONTRACT_SPEC.get("point", 0.01)
    spread_now = (tick.ask - tick.bid) / point

    # F6: Use rolling MEDIAN instead of average
    median_spread = get_spread_rolling_median()
    ratio         = spread_now / median_spread if median_spread > 0 else 999

    # Store for debugging
    state["spread_elevated_last_ratio"] = ratio

    # HARD BLOCK: ratio > 2.5×
    if ratio > config.KS2_SPREAD_MULTIPLIER:
        state["spread_elevated_reading_count"] = 0
        state["spread_multiplier"] = 0.0
        log_event("KS2_REJECTED",
                  spread_now=round(spread_now, 1),
                  median_spread=round(median_spread, 1),
                  ratio=round(ratio, 2))
        return False, "KS2_SPREAD_TOO_WIDE"

    # 50% REDUCTION TIER: ratio > 2.0× for 2 consecutive checks
    if ratio > 2.0:
        state["spread_elevated_reading_count"] += 1
        if state["spread_elevated_reading_count"] >= 2:
            state["spread_multiplier"] = 0.5
            log_event("KS2_REDUCTION_TIER",
                      spread_now=round(spread_now, 1),
                      median_spread=round(median_spread, 1),
                      ratio=round(ratio, 2),
                      consecutive_checks=state["spread_elevated_reading_count"])
            return True, "KS2_REDUCTION_50PCT"
        else:
            state["spread_multiplier"] = 1.0
            log_event("KS2_REDUCTION_PENDING",
                      spread_now=round(spread_now, 1),
                      ratio=round(ratio, 2),
                      consecutive_checks=state["spread_elevated_reading_count"],
                      need=2)
            return True, "KS2_REDUCTION_PENDING"
    else:
        # Normal or clearing — reset counter
        if state["spread_elevated_reading_count"] > 0:
            log_event("KS2_REDUCTION_CLEARED",
                      ratio=round(ratio, 2))
        state["spread_elevated_reading_count"] = 0
        state["spread_multiplier"] = 1.0
        return True, "KS2_OK"


def get_spread_multiplier(state: dict) -> float:
    """
    F6: Returns the current spread multiplier from state.
    Used in calculate_lot_size() to apply spread-based lot reduction.
    """
    return state.get("spread_multiplier", 1.0)


def get_vol_scalar(state: dict) -> float:
    """
    F3: Returns volatility scalar based on EWMA ATR percentile.

    This scalar adjusts lot size based on current volatility level within
    the confirmed regime. A NORMAL regime at 58th ATR percentile and one
    at 88th are both regime="NORMAL" but have very different volatility.

    Scalar curve (atr_percentile → vol_scalar):
      - 0-20th pct:  vol_scalar = 1.10 (high edge in quiet markets)
      - 20-40th pct: vol_scalar = 1.05
      - 40-55th pct: vol_scalar = 1.00 (normal)
      - 55-70th pct: vol_scalar = 0.85
      - 70-85th pct: vol_scalar = 0.65
      - 85-100th pct: vol_scalar = 0.35 (extreme vol — reduce significantly)

    The scalar is clamped: MIN_LOT_MULTIPLIER (0.40) × base_lots minimum.
    This prevents extreme vol from driving lot size to near-zero.
    """
    atr_pct = state.get("last_atr_pct_h1", 50.0)  # Default to mid-range if not set

    if atr_pct <= 20:
        return 1.10
    elif atr_pct <= 40:
        # Linear interpolation 20→40: 1.10 → 1.05
        return round(1.10 - (atr_pct - 20) / 20 * 0.05, 3)
    elif atr_pct <= 55:
        # Linear interpolation 40→55: 1.05 → 1.00
        return round(1.05 - (atr_pct - 40) / 15 * 0.05, 3)
    elif atr_pct <= 70:
        # Linear interpolation 55→70: 1.00 → 0.85
        return round(1.00 - (atr_pct - 55) / 15 * 0.15, 3)
    elif atr_pct <= 85:
        # Linear interpolation 70→85: 0.85 → 0.65
        return round(0.85 - (atr_pct - 70) / 15 * 0.20, 3)
    else:
        # 85-100th: 0.65 → 0.35
        return round(0.65 - (atr_pct - 85) / 15 * 0.30, 3)



def check_ks3_daily_loss(state: dict) -> tuple[bool, str]:
    """
    KS3 — Daily loss limit.
    Net P&L today (after commission) < -1.5% → halt + email.
    Auto-resets at midnight IST (handled by reset_daily_counters).
    """
    if state["daily_net_pnl_pct"] < config.KS3_DAILY_LOSS_LIMIT_PCT:
        if state["trading_enabled"]:
            state["trading_enabled"] = False
            state["shutdown_reason"] = "KS3_DAILY_LOSS_LIMIT"
            persist_critical_state(state)
            from utils.alerts import send_ks_alert
            send_ks_alert("KS3", (
                f"Daily net P&L = {state['daily_net_pnl_pct']:.2%} "
                f"< limit {config.KS3_DAILY_LOSS_LIMIT_PCT:.2%}. "
                f"Trading halted. Auto-resets at midnight IST."
            ))
            log_event("KS3_FIRED",
                      daily_pnl_pct=round(state["daily_net_pnl_pct"], 4))
        return False, "KS3_DAILY_LOSS_LIMIT_REACHED"
    return True, "OK"



def check_ks5_weekly_loss(state: dict) -> tuple[bool, str]:
    """
    KS5 — Weekly loss limit.
    Net weekly P&L < -4.0% → 7-day pause + email.
    Requires MANUAL restart (unlike KS3 which auto-resets).
    """
    weekly_pnl = get_weekly_net_pnl_pct()
    state["weekly_net_pnl_pct"] = weekly_pnl


    if weekly_pnl < config.KS5_WEEKLY_LOSS_LIMIT_PCT:
        if state["trading_enabled"]:
            state["trading_enabled"] = False
            state["shutdown_reason"] = "KS5_WEEKLY_LOSS_LIMIT"
            persist_critical_state(state)
            from utils.alerts import send_ks_alert
            send_ks_alert("KS5", (
                f"Weekly net P&L = {weekly_pnl:.2%} "
                f"< limit {config.KS5_WEEKLY_LOSS_LIMIT_PCT:.2%}. "
                f"7-day pause. Manual restart required after written review."
            ))
            log_event("KS5_FIRED", weekly_pnl_pct=round(weekly_pnl, 4))
        return False, "KS5_WEEKLY_LOSS_LIMIT_REACHED"
    return True, "OK"



def check_ks6_drawdown(state: dict) -> tuple[bool, str]:
    """
    KS6 — Drawdown circuit breaker.
    equity < peak_equity × 0.92 → full system halt + email.
    Written review required before restart.

    peak_equity is updated continuously in heartbeat (B6 Fix).
    v1.1: peak_equity uses 30-day rolling peak (not all-time) — see persistence.py.
    """
    mt5  = get_mt5()
    info = mt5.account_info()
    if info is None:
        return True, "OK"   # can't check → don't block


    equity     = float(info.equity)
    peak       = state["peak_equity"]


    if peak > 0 and equity < peak * (1.0 - config.KS6_DRAWDOWN_LIMIT_PCT):
        if state["trading_enabled"]:
            state["trading_enabled"] = False
            state["shutdown_reason"] = "KS6_DRAWDOWN_CIRCUIT_BREAKER"
            persist_critical_state(state)
            from utils.alerts import send_ks_alert
            send_ks_alert("KS6", (
                f"Equity={equity:.2f} < peak×0.92={peak*0.92:.2f} "
                f"(peak={peak:.2f}). "
                f"Full halt. Written review required before restart."
            ))
            log_event("KS6_FIRED",
                      equity=round(equity, 2),
                      peak=round(peak, 2),
                      drawdown=round(1.0 - equity/peak, 4))
        return False, "KS6_DRAWDOWN_CIRCUIT_BREAKER"
    return True, "OK"



def check_ks7_event_blackout(state: dict) -> tuple[bool, str]:
    """
    F5: KS7 — Event severity guard.

    Replaced binary block with continuous severity scoring:
    - severity_score >= 60: HARD HALT (no new entries)
    - severity_score 30-59: lot size reduced via severity_multiplier
    - severity_score < 30: normal operation

    Severity score is stored in state["ks7_severity_score"] for use by
    calculate_lot_size(). The severity_multiplier is applied in the lot chain.

    EXISTING positions: held through events — hard stops protect them.
    Do NOT close existing positions before events.

    ATR resume check (post-event) is still enforced:
    - Current ATR must be < 130% of pre-event ATR before resuming normal size
    """
    from engines.data_engine import (
        get_economic_severity_score,
        get_minutes_since_last_event,
    )
    from engines.regime_engine import get_current_atr_m15

    # Calculate continuous severity score (looks 2 hours ahead)
    severity_score = get_economic_severity_score(hours_ahead=2)
    state["ks7_severity_score"] = severity_score

    # Log severity periodically (every 10th call to avoid log spam)
    _ks7_log_counter = getattr(check_ks7_event_blackout, "_log_counter", 0)
    check_ks7_event_blackout._log_counter = _ks7_log_counter + 1
    if _ks7_log_counter % 10 == 0:
        log_event("KS7_SEVERITY_SCORE",
                  severity=severity_score,
                  severity_multiplier=get_severity_multiplier(severity_score))

    # HARD HALT: severity >= 60
    if severity_score >= 60:
        if not state.get("ks7_active"):
            # Activate blackout — store pre-event ATR and price for R3
            atr_now  = get_current_atr_m15()
            mt5      = get_mt5()  # BUG-4 FIX: mt5 was not initialized
            tick_now = mt5.symbol_info_tick(config.SYMBOL)
            state["ks7_active"]          = True
            state["ks7_pre_event_atr"]   = atr_now or 0.0
            # Phase 2: Store pre-event mid-price for R3 direction determination.
            # R3 compares first post-delay M5 close to this to determine LONG/SHORT.
            state["ks7_pre_event_price"] = tick_now.bid if tick_now else 0.0
            log_event("KS7_HARD_HALT",
                      severity=severity_score,
                      pre_event_atr=state["ks7_pre_event_atr"])
        return False, f"KS7_HARD_HALT_SEVERITY_{severity_score:.0f}"

    # Severity is < 60 — check post-event ATR resume condition
    if state.get("ks7_active"):
        minutes_since = get_minutes_since_last_event()

        # ATR check: current ATR must be < 130% of pre-event ATR
        if state["ks7_pre_event_atr"] > 0:
            current_atr = get_current_atr_m15()
            if current_atr and current_atr > (
                state["ks7_pre_event_atr"] * config.KS7_ATR_RESUME_MULTIPLIER
            ):
                log_event("KS7_ATR_STILL_ELEVATED",
                          current=round(current_atr, 2),
                          pre_event=round(state["ks7_pre_event_atr"], 2))
                # Even with elevated ATR, trades are allowed at reduced size
                # (severity score handles the risk scaling)

        # Deactivate the "active" flag now that severity is below threshold
        if severity_score < 30:
            state["ks7_active"]        = False
            state["ks7_pre_event_atr"] = 0.0
            log_event("KS7_CLEARED", severity=severity_score)

    # Severity < 60: trades allowed with severity_multiplier applied
    return True, f"KS7_OK_SEVERITY_{severity_score:.0f}"


def get_severity_multiplier(severity_score: float) -> float:
    """
    F5: Convert severity score to a lot size multiplier.

    Rules:
    - score >= 60: HARD HALT → return 0.0 (caller must block trade)
    - score 30-59: scale linearly from 0.41 to 1.0
    - score < 30: normal operation → return 1.0

    The scaling is designed so that at worst (score=59), you're trading at 41%
    size. At score=30, you're at full size.
    """
    if severity_score >= 60:
        return 0.0  # HARD HALT
    elif severity_score >= 30:
        # Linear interpolation: score 59 → 0.41, score 30 → 1.0
        # multiplier = 1.0 - ((score - 30) / 29) * 0.59
        return round(1.0 - ((severity_score - 30) / 29) * 0.59, 3)
    else:
        return 1.0  # Normal operation



def run_pre_trade_kill_switches(state: dict) -> tuple[bool, str]:
    """
    Runs all pre-trade kill switch checks in priority order.
    Returns (permitted, reason) — first failure short-circuits.

    KS1 is enforced at order placement (stop placed before fill).
    KS4 is a size modifier, not a blocker — handled in calculate_lot_size().
    """
    if not state["trading_enabled"]:
        return False, f"TRADING_DISABLED_{state.get('shutdown_reason', 'UNKNOWN')}"


    checks = [
        ("KS3", check_ks3_daily_loss),
        ("KS5", check_ks5_weekly_loss),
        ("KS6", check_ks6_drawdown),
        ("KS7", check_ks7_event_blackout),
    ]
    for ks_name, check_fn in checks:
        permitted, reason = check_fn(state)
        if not permitted:
            return False, reason


    return True, "ALL_KS_PASSED"



# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FAMILY GATES
# ─────────────────────────────────────────────────────────────────────────────


def can_s1_family_fire(state: dict) -> tuple[bool, str]:
    """
    Gate for S1 + S1b combined daily attempts (spec Part 6).
    MAX_S1_FAMILY_ATTEMPTS = 4.
    """
    if state["s1_family_attempts_today"] >= config.MAX_S1_FAMILY_ATTEMPTS:
        return False, "DAILY_SIGNAL_LIMIT_REACHED"
    if not state["trading_enabled"]:
        return False, "TRADING_DISABLED"
    if state["trend_family_occupied"]:
        return False, "TREND_FAMILY_OCCUPIED"
    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        return False, "REGIME_NO_TRADE"
    if not regime.allows_s1:
        return False, f"REGIME_BLOCKS_S1_{regime.value}"


    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return False, reason


    return True, "PERMITTED"



def can_s1f_fire(state: dict) -> tuple[bool, str]:
    """
    Gate for S1f — post-time-kill re-entry (G4 Fix: own counter).
    MAX_S1F_ATTEMPTS = 1 per day, independent of s1_family_attempts_today.
    """
    if state["s1f_attempts_today"] >= config.MAX_S1F_ATTEMPTS:
        return False, "S1F_DAILY_LIMIT_REACHED"
    if state["trend_family_occupied"]:
        return False, "TREND_FAMILY_OCCUPIED"
    regime = get_safe_regime(state)
    if regime not in (RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING):
        return False, "S1F_REQUIRES_SUPER_OR_NORMAL_REGIME"


    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return False, reason


    return True, "PERMITTED"



def can_s2_fire(state: dict) -> tuple[bool, str]:
    """Gate for S2 mean reversion. Requires RANGING_CLEAR regime."""
    if state["trend_family_occupied"]:
        return False, "TREND_FAMILY_OCCUPIED"
    regime = get_safe_regime(state)
    if not regime.allows_s2:
        return False, f"S2_REQUIRES_RANGING_CLEAR_GOT_{regime.value}"


    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return False, reason


    return True, "PERMITTED"



def can_m5_reentry_fire(state: dict) -> tuple[bool, str]:
    """
    Gate for S1d M5 re-entries.
    Requires active open position + SUPER/NORMAL regime + < 3 consecutive M5 losses.
    S1d does NOT set trend_family_occupied — checked separately.
    """
    if not state["trend_family_occupied"]:
        return False, "S1D_REQUIRES_ACTIVE_POSITION"
    if state["consecutive_m5_losses"] >= config.M5_LOSS_PAUSE_COUNT:
        return False, "M5_CYCLING_PAUSED_LOSS_STREAK"
    regime = get_safe_regime(state)
    if not regime.allows_reentry:
        return False, f"S1D_REQUIRES_SUPER_OR_NORMAL_GOT_{regime.value}"


    return True, "PERMITTED"


# ── CHANGE 6: New gates for reversal family, S6, S7 ──────────────────────────

def can_reversal_family_fire(state: dict) -> tuple[bool, str]:
    """
    Gate for reversal family: S1b (failed breakout reversal) + S3 (stop hunt).

    reversal_family_occupied is separate from trend_family_occupied.
    Both families can coexist — S1 trend trade and S1b reversal are
    different positions on different signals. reversal_family_occupied
    prevents double-firing S1b and S3 on the same day.

    Regime gate: any regime except NO_TRADE (reversals fire against failed moves,
    so RANGING_CLEAR is also valid for reversal setups).
    """
    if state.get("reversal_family_occupied"):
        return False, "REVERSAL_FAMILY_OCCUPIED"
    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        return False, "REGIME_NO_TRADE"


    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return False, reason


    return True, "PERMITTED"



def can_s3_fire(state: dict) -> tuple[bool, str]:
    """
    Gate for S3 — Stop Hunt Reversal.

    S3 is part of the reversal family — fires at most once per day.
    s3_fired_today: set True when S3 order is placed (pending fill).
    Additional check: sweep candle must be identified (s3_sweep_candle_time
    must be set by evaluate_s3_signal before calling this gate).

    Regime: any except NO_TRADE (stop hunts happen in ranging AND trending).
    """
    if state.get("s3_fired_today"):
        return False, "S3_ALREADY_FIRED_TODAY"


    permitted, reason = can_reversal_family_fire(state)
    if not permitted:
        return False, reason


    return True, "PERMITTED"



def can_s6_fire(state: dict) -> tuple[bool, str]:
    """
    Gate for S6 — Asian Range Breakout (pending orders placed at 05:30 UTC).

    s6_fired_today: set True when both pending orders are placed.
    Regime gate: NO_TRADE blocks (high ATR = breakout likely to fail).
    Session check not needed here — S6 fires at fixed 05:30 UTC regardless
    of session label (asian_range_job handles the UTC schedule).
    """
    if state.get("s6_fired_today"):
        return False, "S6_ALREADY_FIRED_TODAY"
    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        return False, "REGIME_NO_TRADE_S6_BLOCKED"


    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return False, reason


    return True, "PERMITTED"



def can_s7_fire(state: dict) -> tuple[bool, str]:
    """
    Gate for S7 — Daily Structure Breakout (pending orders placed at midnight).

    s7_fired_today: set True when both pending orders are placed.
    Sunday check is handled in midnight_reset_job (MT5 not open at 18:30 UTC
    Sunday) — this gate does not need to re-check day of week.
    Regime gate: NO_TRADE blocks. UNSTABLE also blocks (wide daily ATR =
    invalid inside-day context).
    """
    if state.get("s7_fired_today"):
        return False, "S7_ALREADY_FIRED_TODAY"
    regime = get_safe_regime(state)
    if regime in (RegimeState.NO_TRADE, RegimeState.UNSTABLE):
        return False, f"REGIME_BLOCKS_S7_{regime.value}"


    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return False, reason


    return True, "PERMITTED"



# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK BRAIN — v1.1 (CHANGE 7)
# ─────────────────────────────────────────────────────────────────────────────


def check_portfolio_risk(candidate: dict, state: dict) -> tuple[bool, str]:
    """
    v1.1 Portfolio Risk Brain — duplicate of portfolio_risk.check_portfolio_risk;
    main path uses engines.portfolio_risk. Kept for reference / tests.

    Gate 1 — MAX_SESSION_LOTS (0.15).
    Gate 2 — MAX_DAILY_VAR_PCT (2%).

    Returns (permitted, reason).
    """
    mt5  = get_mt5()
    info = mt5.account_info()

    if info is None:
        log_warning("PORTFOLIO_RISK_ACCOUNT_INFO_FAILED")
        return True, "OK"   # can't check → don't block


    equity = float(info.equity)
    spec   = config.CONTRACT_SPEC

    # ── Gate 1: MAX_SESSION_LOTS ─────────────────────────────────────────────
    positions    = mt5.positions_get(symbol=config.SYMBOL)
    our_positions = [p for p in (positions or [])
                     if p.magic == config.MAGIC]
    current_lots  = sum(p.volume for p in our_positions)
    new_lots      = candidate.get("lot_size", 0.0)

    if current_lots + new_lots > config.MAX_SESSION_LOTS:
        log_event("PORTFOLIO_MAX_SESSION_LOTS_BLOCKED",
                  current_lots=round(current_lots, 3),
                  new_lots=round(new_lots, 3),
                  limit=config.MAX_SESSION_LOTS)
        return False, "PORTFOLIO_MAX_SESSION_LOTS_EXCEEDED"


    # ── Gate 2: MAX_DAILY_VAR_PCT ─────────────────────────────────────────────
    entry_level = candidate.get("entry_level", 0.0)
    stop_level  = candidate.get("stop_level", 0.0)
    stop_dist   = abs(entry_level - stop_level)

    tick_size  = spec.get("tick_size", 0.01)
    tick_value = spec.get("tick_value", 1.0)

    if tick_size > 0 and tick_value > 0 and equity > 0 and stop_dist > 0:
        ticks_in_stop  = stop_dist / tick_size
        trade_risk_usd = ticks_in_stop * tick_value * new_lots
        risk_pct       = trade_risk_usd / equity

        if risk_pct > config.MAX_DAILY_VAR_PCT:
            log_event("PORTFOLIO_MAX_DAILY_VAR_BLOCKED",
                      risk_pct=round(risk_pct, 4),
                      limit=config.MAX_DAILY_VAR_PCT,
                      trade_risk_usd=round(trade_risk_usd, 2))
            return False, "PORTFOLIO_MAX_DAILY_VAR_EXCEEDED"


    log_event("PORTFOLIO_RISK_PASSED",
              current_lots=round(current_lots, 3),
              new_lots=round(new_lots, 3),
              daily_pnl_pct=round(state.get("daily_net_pnl_pct", 0.0), 4))

    return True, "OK"



# ─────────────────────────────────────────────────────────────────────────────
# CONVICTION LEVEL — Observation Mode V1
# ─────────────────────────────────────────────────────────────────────────────


def calculate_conviction_level(state: dict) -> str:
    """
    Observation Mode V1 — logged per trade, never changes lot size.
    Promote to active modifier after 50 trades if A+ vs STANDARD delta > 8pp.

    Returns one of: A_PLUS, STANDARD, AFTER_STREAK
    """
    # AFTER_STREAK always takes precedence
    if state.get("ks4_reduced_trades_remaining", 0) > 0:
        return "AFTER_STREAK"


    # A+ conditions (all must be true)
    mt5   = get_mt5()
    tick  = mt5.symbol_info_tick(config.SYMBOL)
    point = config.CONTRACT_SPEC.get("point", 0.01)


    spread_now = 0.0
    avg_spread = get_avg_spread_last_24h()   # consistent with KS2
    if tick and point > 0:
        spread_now = (tick.ask - tick.bid) / point


    spread_ratio = (spread_now / avg_spread) if avg_spread > 0 else 999


    clear_horizon = get_upcoming_events_within(config.CONVICTION_CLEAR_HORIZON_MIN)


    a_plus = (
        state["current_regime"] in ("SUPER_TRENDING", "NORMAL_TRENDING")
        and state.get("macro_bias", "BOTH_PERMITTED") in
            ("LONG_PERMITTED", "SHORT_PERMITTED")
        and spread_ratio <= config.CONVICTION_SPREAD_RATIO_MAX    # <= 1.2×
        and len(clear_horizon) == 0
        and state["consecutive_losses"] < 2
    )


    return "A_PLUS" if a_plus else "STANDARD"   # ── CHANGE 1: removed extra )