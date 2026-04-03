"""
engines/signal_engine.py — Layer 3: Signal Engine.

Generates trade candidates only. Does NOT touch MT5 orders — execution
engine handles all order placement. Layer 3 never skips Layer 2.

Signals implemented:
  S1   — London Range Breakout (SUPER/NORMAL/WEAK regimes)
  S1b  — Failed Breakout Reversal (B3 Fix: candle auto-reset)
  S1c  — Stop Hunt Pre-Signal Detection (candle auto-reset)
  S1d  — M5 Pullback Re-entries (G3 Fix: body-close; B4 Fix: 5-min expiry)
  S1e  — Pyramid Into Confirmed Winners (one add per S1 only)
  S1f  — Post-Time-Kill Re-entry (G4 Fix: own counter, own cap)
  S2   — Mean Reversion (RANGING_CLEAR only, London-open restriction)
  S3   — Stop Hunt Reversal (sweep + reclaim pattern, reversal family)
  S6   — Asian Range Breakout (00:00–05:30 UTC, dual pending orders)
  S7   — Daily Structure Breakout (prev day high/low, midnight placement)

Position management:
  check_partial_exit_condition  — WEAK hybrid exit: 1.0R trigger
  check_be_activation_condition — WEAK hybrid exit: 0.75R + M15 swing (both required)
  check_momentum_cycle_exit     — SUPER/NORMAL: M5 EMA20 body break
  manage_open_position          — dispatcher called on every M5 candle close

"""
import uuid
import pytz
import pandas as pd
import pandas_ta as ta
from enum import Enum
from datetime import datetime, timedelta


import config
from utils.logger import log_event, log_warning
from utils.mt5_client import get_mt5
from utils.session import (
    get_current_session, get_london_local_time, get_ny_local_time
)
from engines.regime_engine import (
    get_safe_regime, RegimeState,
    get_adx_h4, get_current_atr_m15
)
from engines.risk_engine import (
    calculate_lot_size, calculate_conviction_level,
    can_s1_family_fire, can_s1f_fire, can_s2_fire,
    can_m5_reentry_fire, run_pre_trade_kill_switches,
    calculate_r_multiple, calculate_atr_trail,
    # ── CHANGE 3: added reversal family and Phase 1 strategy gates ───────────
    can_reversal_family_fire, can_s3_fire, can_s6_fire, can_s7_fire,
)
from engines.data_engine import (
    fetch_ohlcv, get_upcoming_events_within,
    # ── CHANGE 2: replaced get_session_avg_spread with 24h baseline ──────────
    get_avg_spread_last_24h,
    # ── CHANGE 2: Phase 1 strategy data functions ─────────────────────────────
    get_asian_range, get_prev_day_ohlc, get_daily_atr14,
)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL TYPE ENUM — G5 Fix (no raw strings in DB)
# ─────────────────────────────────────────────────────────────────────────────


class SignalType(str, Enum):
    """
    Signal type enumeration for different trading strategies.
    """
    S1_LONDON_BRK = "S1_LONDON_BRK"
    S1B_FAILED_BRK = "S1B_FAILED_BRK"
    S1C_STOP_HUNT = "S1C_STOP_HUNT"
    S1D_PYRAMID = "S1D_PYRAMID"
    S1E_PYRAMID = "S1E_PYRAMID"
    S1F_POST_TK = "S1F_POST_TK"
    S2_MEAN_REV = "S2_MEAN_REV"
    S3_STOP_HUNT_REV = "S3_STOP_HUNT_REV"
    S6_ASIAN_BRK = "S6_ASIAN_BRK"
    S7_DAILY_STRUCT = "S7_DAILY_STRUCT"
    # ── Phase 2 strategies ───────────────────────────────────────────────────
    R3_CAL_MOMENTUM = "R3_CAL_MOMENTUM"   # Post-event directional momentum
    S4_LONDON_PULL  = "S4_LONDON_PULL"    # London pullback to EMA20 M15
    S5_NY_COMPRESS  = "S5_NY_COMPRESS"    # NY session breakout of compressed London range
    S8_ATR_SPIKE = "S8_ATR_SPIKE"      # F7: ATR spike continuation trade
    MANUAL = "MANUAL"



# ─────────────────────────────────────────────────────────────────────────────
# TIME KILL HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def is_past_london_time_kill() -> bool:
    """
    True if current London local time is >= 16:30.
    G7 Fix: time kill must explicitly call cancel_all_pending_orders().
    Used by S1, S1b, S2, S1e, S1d exit.
    """
    lt = get_london_local_time()
    return (lt.hour > 16) or (lt.hour == 16 and lt.minute >= 30)



def is_past_ny_time_kill() -> bool:
    """
    True if current NY local time is >= 13:00.
    Used by S1f only. DST-safe via pytz.
    """
    nt = get_ny_local_time()
    return nt.hour >= 13



def is_after_london_open() -> bool:
    """
    C2 Fix: S1 evaluation only begins AFTER London 08:00 local.
    No pre-open triggers — prevents Asian-session false breakouts.
    """
    lt = get_london_local_time()
    return lt.hour >= 8



def is_within_s2_london_open_restriction() -> bool:
    """
    G7: S2 blocked within 30 min of London open (07:30–08:00 London local).
    Returns True if we are in the restricted window — S2 should NOT fire.
    """
    lt = get_london_local_time()
    total_min = lt.hour * 60 + lt.minute
    return 7 * 60 + 30 <= total_min < 8 * 60



def check_and_fire_london_time_kill(state: dict) -> bool:
    session = get_current_session()
    if session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return False
    if is_past_london_time_kill() and not state.get("london_tk_fired_today", False):
        from engines.execution_engine import cancel_all_pending_orders
        cancel_all_pending_orders()
        # Clear S1 pending ticket slots
        state["s1_pending_buy_ticket"]  = None
        state["s1_pending_sell_ticket"] = None
        # ── CHANGE 5: S7 orders live all day — cancel at London TK ──────────
        # S6 orders already expired at 08:00 UTC. S7 may still be pending.
        state["s7_pending_buy_ticket"]  = None
        state["s7_pending_sell_ticket"] = None
        state["london_tk_fired_today"]  = True
        log_event("LONDON_TIME_KILL_FIRED",
                  time=str(get_london_local_time().strftime("%H:%M")))
        return True
    return False



def check_and_fire_ny_time_kill(state: dict) -> bool:
    """
    S1f time kill: NY 13:00 local. Cancels all OUR pending orders.

    SESSION GUARD: Only fires when NY session is actually open
    (NY or LONDON_NY_OVERLAP). Prevents spurious fires at 06:48 IST
    when NY clock reads 21:18 — that is off-hours, not an active NY session.
    """
    session = get_current_session()
    if session not in ("NY", "LONDON_NY_OVERLAP"):
        return False
    if is_past_ny_time_kill() and not state.get("ny_tk_fired_today", False):
        from engines.execution_engine import cancel_all_pending_orders
        cancel_all_pending_orders()
        state["ny_tk_fired_today"] = True
        log_event("NY_TIME_KILL_FIRED", time=str(get_ny_local_time().strftime("%H:%M")))
        return True
    return False



# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def get_ema20_h1() -> float | None:
    """20 EMA on H1 bars. Used by S2 mean reversion signal."""
    df = fetch_ohlcv("H1", count=50)
    if df is None or df.empty:
        return None
    df["ema20"] = ta.ema(df["close"], length=20)
    val = df["ema20"].iloc[-1]
    return float(val) if not pd.isna(val) else None



def get_ema20_m5() -> float | None:
    """
    20 EMA on M5 bars. Used by S1d, S1e, S1f for body-close pullback signal.
    pandas_ta default for EMA is correct — no mamode needed (EMA != ATR).
    """
    df = fetch_ohlcv("M5", count=50)
    if df is None or df.empty:
        return None
    df["ema20"] = ta.ema(df["close"], length=20)
    val = df["ema20"].iloc[-1]
    return float(val) if not pd.isna(val) else None



def get_atr14_h1_rma() -> float | None:
    """ATR(14, H1, RMA). Used by S2 for signal threshold and stop distance."""
    df = fetch_ohlcv("H1", count=50)
    if df is None or df.empty:
        return None
    df["atr"] = ta.atr(df["high"], df["low"], df["close"],
                       length=14, mamode=config.ATR_MAMODE)  # RMA pinned
    val = df["atr"].iloc[-1]
    return float(val) if not pd.isna(val) else None



def get_last_m15_bar() -> dict | None:
    """Returns the last CLOSED M15 bar as a dict. Returns None on failure."""
    df = fetch_ohlcv("M15", count=3)
    if df is None or len(df) < 2:
        return None
    # iloc[-2] = last CLOSED bar (iloc[-1] is forming)
    row = df.iloc[-2]
    return {
        "open":  float(row["open"]),
        "high":  float(row["high"]),
        "low":   float(row["low"]),
        "close": float(row["close"]),
        "time":  row["time"],
    }



def get_last_m5_bar() -> dict | None:
    """Returns the last CLOSED M5 bar as a dict."""
    df = fetch_ohlcv("M5", count=3)
    if df is None or len(df) < 2:
        return None
    row = df.iloc[-2]
    return {
        "open":  float(row["open"]),
        "high":  float(row["high"]),
        "low":   float(row["low"]),
        "close": float(row["close"]),
        "time":  row["time"],
    }



def _body_close_above_ema20(bar: dict, ema20: float) -> bool:
    """
    G3 Fix: Full body close above EMA20 for LONG re-entry.
    BOTH candle open AND candle close must be above EMA20.
    Wick touches do NOT count. This single rule prevents wick-chasing losses.
    """
    return bar["open"] > ema20 and bar["close"] > ema20



def _body_close_below_ema20(bar: dict, ema20: float) -> bool:
    """G3 Fix: Full body close below EMA20 for SHORT re-entry."""
    return bar["open"] < ema20 and bar["close"] < ema20



# ─────────────────────────────────────────────────────────────────────────────
# TRADE CANDIDATE BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def _build_candidate(
    signal_type: SignalType,
    direction:   str,
    entry_level: float,
    stop_level:  float,
    lot_size:    float,
    state:       dict,
    range_data:  dict | None = None,
    extra:       dict | None = None,
) -> dict:
    """
    Builds a complete trade candidate dict.
    All Truth Engine Tier 1 fields populated here (logged per trade).
    Tier 2 fields (macro, conviction) populated here — logged, never gating V1.
    G5 Fix: signal_type stored as SignalType.value — never raw string.
    """
    mt5  = get_mt5()
    info = mt5.account_info()


    lt       = get_london_local_time()
    upcoming = get_upcoming_events_within(60)


    tick       = mt5.symbol_info_tick(config.SYMBOL)
    spread_now = 0.0
    # ── CHANGE 4: was get_session_avg_spread() — consistent with KS2 ─────────
    avg_spread = get_avg_spread_last_24h()
    if tick and config.CONTRACT_SPEC.get("point"):
        spread_now = (tick.ask - tick.bid) / config.CONTRACT_SPEC["point"]


    candidate = {
        # ── Identity (G5: enum value) ────────────────────────────────────
        "signal_type":      signal_type.value,
        "strategy_version": "V1",
        "campaign_id":      str(uuid.uuid4()),


        # ── Execution ────────────────────────────────────────────────────
        "direction":        direction,
        "entry_level":      round(entry_level, 3),
        "stop_level":       round(stop_level, 3),
        "lot_size":         lot_size,


        # ── Range context ────────────────────────────────────────────────
        "range_size":           range_data["range_size"]  if range_data else 0.0,
        "asian_range_high":     range_data["range_high"]  if range_data else 0.0,
        "asian_range_low":      range_data["range_low"]   if range_data else 0.0,
        "asian_range_size_pts": range_data["range_size"]  if range_data else 0.0,


        # ── Regime Tier 1 ────────────────────────────────────────────────
        "regime_at_entry":      state["current_regime"],
        "size_multiplier_used": state["size_multiplier"],
        "adx_h4_at_entry":      state.get("last_adx_h4", 0.0),
        "atr_h1_percentile":    state.get("last_atr_pct_h1", 0.0),
        "session":              get_current_session(),
        "regime_age_seconds":   _get_regime_age_sec(state),


        # ── Macro Tier 2 (observation) ───────────────────────────────────
        "macro_bias_at_entry":  state.get("macro_bias", "BOTH_PERMITTED"),
        "macro_boost_at_entry": bool(state.get("macro_boost", False)),
        "dxy_corr_at_entry":    float(state.get("dxy_corr_50", 0.0)),
        "macro_proxy_at_entry": state.get("macro_proxy_instrument", "TLT"),
        "tlt_3d_slope":         float(state.get("tlt_slope", 0.0)),


        # ── Conviction Tier 2 (observation) ─────────────────────────────
        "conviction_level":     calculate_conviction_level(state),
        "event_proximity_min":  len(upcoming),
        "spread_at_entry":      spread_now,
        "spread_vs_avg_ratio":  round(spread_now / avg_spread, 2) if avg_spread > 0 else 0.0,
        "slippage_points":      0.0,  # Will be updated on fill


        # ── Execution context ────────────────────────────────────────────
        "order_type_used":      "MARKET",  # Default, updated by execution_engine
        "risk_pct_used":        None,     # Populated by risk_engine
        "london_hour_at_entry": lt.hour,
        "s1_family_attempt_num": state.get("s1_family_attempts_today", 0) + 1,
        "equity_at_entry":      float(info.equity) if info else None,


        # ── Signal-specific flags ───────────────────────────────────────────
        "stop_hunt_detected":   state.get("stop_hunt_detected", False),
        "failed_breakout_trade": state.get("failed_breakout_flag", False),
        "post_time_kill_reentry": signal_type == SignalType.S1F_POST_TK,
    }


    if extra:
        candidate.update(extra)


    return candidate



def _get_regime_age_sec(state: dict) -> int:
    calc_at = state.get("regime_calculated_at")
    if calc_at is None:
        return 0
    return int((datetime.now(pytz.utc) - calc_at).total_seconds())



# ─────────────────────────────────────────────────────────────────────────────
# S1 — LONDON RANGE BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s1_signal(state: dict) -> dict | None:
    """
    S1 London Range Breakout.

    C2 Fix: Evaluation only begins AFTER London 08:00 local — no pre-open.
    Signal: M15 close > range_high + breakout_dist (12%) → LONG
            M15 close < range_low  - breakout_dist (12%) → SHORT
    Entry:  STOP ORDER at signal level.
    Chase:  Max range_size * 8% — execution engine checks at fill time.
    Stop:   LONG → range_low - range_size*0.10
            SHORT → range_high + range_size*0.10

    Returns candidate dict or None.
    """
    # C2 Fix: London open gate
    if not is_after_london_open():
        return None


    if is_past_london_time_kill():
        return None


    session = get_current_session()
    if session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return None


    # Range must be computed
    range_data = state.get("range_data")
    if not range_data:
        log_warning("S1_NO_RANGE_DATA")
        return None


    # Family + KS gate
    permitted, reason = can_s1_family_fire(state)
    if not permitted:
        return None


    bar = get_last_m15_bar()
    if bar is None:
        return None

    # EXP-2 FIX: Volume filter — reject low-volume breakouts.
    # Many false breakouts occur on low volume. Require at least 70% of
    # recent average volume for the breakout candle.
    if bar.get("tick_volume", 0) > 0:
        df_vol = fetch_ohlcv("M15", count=7)
        if df_vol is not None and len(df_vol) >= 6:
            avg_vol = float(df_vol["tick_volume"].iloc[-6:-1].mean())
            if avg_vol > 0 and bar["tick_volume"] < avg_vol * 0.7:
                log_event("S1_LOW_VOLUME_BREAKOUT_REJECTED",
                          bar_vol=bar["tick_volume"], avg=round(avg_vol, 1))
                return None

    rh  = range_data["range_high"]
    rl  = range_data["range_low"]
    rs  = range_data["range_size"]
    bd  = range_data["breakout_dist"]  # rs * 0.12


    # S1c stop hunt: reduce confirmation threshold if flag set
    threshold = bd
    if state.get("stop_hunt_detected"):
        hunt_dir  = state.get("stop_hunt_direction")
        bar_close = bar["close"]
        # Reduce threshold to rs * 0.08 in stop-hunt direction
        if (hunt_dir == "LONG"  and bar_close > rh) or \
           (hunt_dir == "SHORT" and bar_close < rl):
            threshold = range_data["hunt_threshold"]  # rs * 0.08
            log_event("S1_STOP_HUNT_THRESHOLD_APPLIED",
                      direction=hunt_dir, threshold=round(threshold, 2))


    direction = None
    if bar["close"] > rh + threshold:
        direction = "LONG"
    elif bar["close"] < rl - threshold:
        direction = "SHORT"
    else:
        return None


    # LOOP-1 FIX: ATR-based stop buffer instead of thin range-percentage.
    # 0.3× H1 ATR or minimum 5 points — prevents noise wicks from clipping stops.
    atr_h1 = state.get("last_atr_h1_raw", 0.0)
    stop_buffer = max(atr_h1 * 0.3, 5.0) if atr_h1 > 0 else rs * 0.15

    if direction == "LONG":
        entry = rh + threshold
        stop  = round(rl - stop_buffer, 3)
    else:
        entry = rl - threshold
        stop  = round(rh + stop_buffer, 3)


    stop_distance = abs(entry - stop)
    regime        = get_safe_regime(state)
    lot_size      = calculate_lot_size(stop_distance, state["size_multiplier"], state)

    # EXP-1 FIX: Add 2.5R take-profit target to S1.
    # Previously only R3 had TP. Adding TP captures winners cleanly
    # instead of relying solely on trailing stops which give back profits.
    tp = round(entry + stop_distance * 2.5, 3) if direction == "LONG" \
         else round(entry - stop_distance * 2.5, 3)

    candidate = _build_candidate(
        signal_type = SignalType.S1_LONDON_BRK,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        range_data  = range_data,
        extra       = {"s1_tp_level": tp},
    )


    log_event("S1_SIGNAL_GENERATED",
              direction=direction,
              entry=round(entry, 3),
              stop=round(stop, 3),
              lots=lot_size,
              regime=regime.value)
    return candidate



# ─────────────────────────────────────────────────────────────────────────────
# S1b — FAILED BREAKOUT REVERSAL (B3 Fix: candle auto-reset)
# ─────────────────────────────────────────────────────────────────────────────


def auto_reset_s1b_counter(state: dict) -> None:
    """
    B3 Fix: If failed_breakout_flag is set but no reversal signal fires
    within 6 M15 candles (~90 min), reset the flag and counter.
    Called on every M15 candle close when flag is active.
    """
    if not state.get("failed_breakout_flag"):
        return


    state["failed_breakout_flag_candles"] += 1


    if state["failed_breakout_flag_candles"] >= config.S1B_AUTO_RESET_CANDLES:
        state["failed_breakout_flag"]         = False
        state["failed_breakout_flag_candles"] = 0
        state["failed_breakout_direction"]    = None
        log_event("S1B_FLAG_AUTO_RESET",
                  reason="no_reversal_within_6_candles")



def check_s1b_trigger(state: dict) -> bool:
    """
    Checks if conditions are met to SET the failed_breakout_flag.
    Called after S1 close/loss — not an entry signal itself.

    Conditions:
      1. S1 fired in direction X
      2. S1 max_r reached -0.5R (partial loss, not full stop)
      3. M15 closes BACK INSIDE the Asian range
    Returns True if flag should be set.
    """
    if state.get("failed_breakout_flag"):
        return False  # Already set


    if state.get("last_s1_max_r", 0.0) > -0.5:
        return False  # Hasn't reached -0.5R


    range_data = state.get("range_data")
    if not range_data:
        return False


    bar = get_last_m15_bar()
    if bar is None:
        return False


    rh        = range_data["range_high"]
    rl        = range_data["range_low"]
    direction = state.get("last_s1_direction")


    # M15 must close back inside the range
    if bar["close"] < rh and bar["close"] > rl:
        state["failed_breakout_flag"]         = True
        state["failed_breakout_flag_candles"] = 0
        state["failed_breakout_direction"]    = direction
        log_event("S1B_FLAG_SET",
                  direction=direction,
                  last_max_r=state.get("last_s1_max_r", 0.0))
        return True
    return False



def evaluate_s1b_signal(state: dict) -> dict | None:
    """
    S1b: Failed Breakout Reversal.
    Fires when: failed_breakout_flag AND next M15 breaks OPPOSITE boundary.
    """
    if not state.get("failed_breakout_flag"):
        return None


    if is_past_london_time_kill():
        return None


    permitted, reason = can_s1_family_fire(state)
    if not permitted:
        return None


    range_data = state.get("range_data")
    if not range_data:
        return None


    bar = get_last_m15_bar()
    if bar is None:
        return None


    orig_dir  = state.get("failed_breakout_direction")
    rh        = range_data["range_high"]
    rl        = range_data["range_low"]
    rs        = range_data["range_size"]
    bd        = range_data["breakout_dist"]


    direction = None
    if orig_dir == "LONG"  and bar["close"] < rl - bd:
        direction = "SHORT"
    elif orig_dir == "SHORT" and bar["close"] > rh + bd:
        direction = "LONG"
    else:
        return None


    if direction == "LONG":
        entry = rh + bd
        stop  = round(rl - rs * 0.10, 3)
    else:
        entry = rl - bd
        stop  = round(rh + rs * 0.10, 3)


    stop_distance = abs(entry - stop)
    lot_size      = calculate_lot_size(stop_distance, state["size_multiplier"], state)


    candidate = _build_candidate(
        signal_type = SignalType.S1B_FAILED_BRK,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        range_data  = range_data,
        extra       = {"failed_breakout_trade": True},
    )

    # Reset flag AFTER candidate is built (not before, in case signal is rejected)
    # The actual flag reset happens in execution engine on_trade_opened
    # This ensures flags persist if signal is rejected by risk gates


    log_event("S1B_SIGNAL_GENERATED",
              direction=direction, orig_dir=orig_dir,
              entry=round(entry, 3))
    return candidate



# ─────────────────────────────────────────────────────────────────────────────
# S1c — STOP HUNT PRE-SIGNAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────


def detect_stop_hunt(state: dict) -> None:
    """
    S1c: Stop hunt detection. Sets state flags — does NOT generate a candidate.
    Pattern:
      - Price probes range extreme by >= hunt_threshold (8% of range)
      - Returns INSIDE range within 1-2 M15 candles (wick probe)
      - ADX still < 28 during probe (confirms ranging, not true breakout)

    Action: set stop_hunt_detected = True, stop_hunt_direction = OPPOSITE
    Candle reset: reset all flags if no signal fires within 3 M15 candles.
    """
    # Don't detect stop hunt if S1 has already fired today
    if state.get("s1_family_attempts_today", 0) >= config.MAX_S1_FAMILY_ATTEMPTS:
        return
    
    # Don't detect if trend family is already occupied
    if state.get("trend_family_occupied"):
        return
    
    range_data = state.get("range_data")
    if not range_data:
        return


    # Auto-reset candle counter
    if state.get("stop_hunt_detected"):
        state["stop_hunt_candles"] = state.get("stop_hunt_candles", 0) + 1
        if state["stop_hunt_candles"] >= config.S1C_AUTO_RESET_CANDLES:
            state["stop_hunt_detected"]  = False
            state["stop_hunt_direction"] = None
            state["stop_hunt_candles"]   = 0
            log_event("S1C_STOP_HUNT_AUTO_RESET",
                      reason="no_signal_within_3_candles")
        return  # Don't re-detect while flag is active


    bar = get_last_m15_bar()
    if bar is None:
        return


    rh = range_data["range_high"]
    rl = range_data["range_low"]
    ht = range_data["hunt_threshold"]  # rs * 0.08


    adx = get_adx_h4()
    if adx is None:
        return
    
    # Use regime-based ADX check: stop hunt only in ranging/weak trending
    # ADX >= 28 indicates real trend forming, not a stop hunt
    regime = get_safe_regime(state)
    if regime in (RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING):
        return  # Strong trend, not a stop hunt
    if adx >= 28:
        return  # ADX >= 28 = real trend forming, not a stop hunt


    probed_high = bar["high"] >= rh + ht and bar["close"] < rh
    probed_low  = bar["low"]  <= rl - ht and bar["close"] > rl


    if probed_high:
        state["stop_hunt_detected"]  = True
        state["stop_hunt_direction"] = "SHORT"  # Probe above = shorts hunted
        state["stop_hunt_candles"]   = 0
        log_event("STOP_HUNT_DETECTED",
                  direction="SHORT",
                  probe_high=round(bar["high"], 3),
                  range_high=round(rh, 3))


    elif probed_low:
        state["stop_hunt_detected"]  = True
        state["stop_hunt_direction"] = "LONG"   # Probe below = longs hunted
        state["stop_hunt_candles"]   = 0
        log_event("STOP_HUNT_DETECTED",
                  direction="LONG",
                  probe_low=round(bar["low"], 3),
                  range_low=round(rl, 3))



# ─────────────────────────────────────────────────────────────────────────────
# S1d — M5 RE-ENTRY CYCLING (B4, G3 Fix)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s1d_reentry(state: dict) -> dict | None:
    """
    S1d: M5 pullback re-entry into open S1 trend.

    G3 Fix: Body close required — BOTH open AND close above EMA20.
            Wick touches do NOT count (Rule 6 of the 19 Rules).

    B4 Fix: Limit order expiry = now + 5 minutes, ORDER_TIME_SPECIFIED.
            EMA20 moves every bar — a 3-bar-old limit is at a stale price.

    Max re-entries: SUPER=8, NORMAL=5.
    After 3 consecutive M5 losses: pause cycling. Main S1 unaffected.
    """
    # Verify open position is from S1 family (not S2, S6, S7)
    trend_strategy = state.get("trend_family_strategy")
    if trend_strategy not in (SignalType.S1_LONDON_BRK.value, SignalType.S1B_FAILED_BRK.value, 
                               SignalType.S1F_POST_TK.value, SignalType.S1E_PYRAMID.value):
        return None
    
    permitted, reason = can_m5_reentry_fire(state)
    if not permitted:
        return None


    if is_past_london_time_kill():
        return None


    session = get_current_session()
    if session not in ("LONDON", "LONDON_NY_OVERLAP"):
        return None


    ema20 = get_ema20_m5()
    if ema20 is None:
        return None


    bar = get_last_m5_bar()
    if bar is None:
        return None


    direction = state.get("last_s1_direction")
    if not direction:
        return None


    # G3 Fix: full body close — both open AND close
    # LOOP-3 FIX: ATR-based stop instead of fixed 10-12pt which is suicidal on XAUUSD M5.
    # Use 0.75× M15 ATR or config minimum, whichever is larger.
    from engines.data_engine import get_current_atr_m15
    atr_m15 = get_current_atr_m15(period=14)
    stop_distance = max(
        atr_m15 * 0.75 if atr_m15 else config.S1D_STOP_POINTS_MIN,
        config.S1D_STOP_POINTS_MIN
    )

    if direction == "LONG":
        if not _body_close_above_ema20(bar, ema20):
            return None
        entry = round(ema20, 3)
        stop  = round(ema20 - stop_distance, 3)
    else:
        if not _body_close_below_ema20(bar, ema20):
            return None
        entry = round(ema20, 3)
        stop  = round(ema20 + stop_distance, 3)


    stop_distance = abs(entry - stop)
    # S1d size = 0.5× normal lot size (spec Part 6)
    base_lots = calculate_lot_size(stop_distance, state["size_multiplier"], state)
    lot_size  = round(base_lots * 0.5, 2)
    lot_size  = max(config.CONTRACT_SPEC.get("volume_min", 0.01), lot_size)


    # B4 Fix: expiry timestamp (used by execution engine in type_time/expiration fields)
    expiry_utc = datetime.now(pytz.utc) + timedelta(minutes=config.M5_LIMIT_EXPIRY_MIN)


    candidate = _build_candidate(
        signal_type = SignalType.S1D_PYRAMID,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        extra       = {
            "order_expiry_utc": expiry_utc.isoformat(),  # B4 Fix
            "ema20_at_entry":   ema20,
            "tick_volume_m5":   0,
        }
    )


    log_event("S1D_REENTRY_SIGNAL",
              direction=direction,
              ema20=round(ema20, 3),
              lot_size=lot_size,
              m5_count=state.get("position_m5_count", 0),
              expiry=expiry_utc.strftime("%H:%M:%S"))
    return candidate



# ─────────────────────────────────────────────────────────────────────────────
# S1e — PYRAMID INTO CONFIRMED WINNERS
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s1e_pyramid(state: dict) -> dict | None:
    """
    S1e: ONE pyramid add per S1 trade — hard limit.
    Conditions: partial exit done, stop at BE or better,
    SUPER/NORMAL regime, pyramid_done = False.
    """
    if state.get("position_pyramid_done"):
        return None


    if not state.get("position_partial_done"):
        return None


    if not state.get("position_be_activated"):
        return None


    if is_past_london_time_kill():
        return None


    regime = get_safe_regime(state)
    if not regime.allows_reentry:
        return None


    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return None


    ema20 = get_ema20_m5()
    if ema20 is None:
        return None


    bar = get_last_m5_bar()
    if bar is None:
        return None


    direction = state.get("last_s1_direction")
    if not direction:
        return None


    # G3 Fix: full body close (same rule as S1d)
    if direction == "LONG":
        if not _body_close_above_ema20(bar, ema20):
            return None
        entry = round(ema20, 3)
        # Use actual current stop from position, or calculate from EMA20
        current_stop = state.get("stop_price_current")
        if current_stop and current_stop > 0:
            stop = current_stop
        else:
            stop = round(ema20 - config.S1D_STOP_POINTS_MIN, 3)
    else:
        if not _body_close_below_ema20(bar, ema20):
            return None
        entry = round(ema20, 3)
        current_stop = state.get("stop_price_current")
        if current_stop and current_stop > 0:
            stop = current_stop
        else:
            stop = round(ema20 + config.S1D_STOP_POINTS_MIN, 3)


    stop_distance = abs(entry - stop)
    # Guard against zero or very small stop distance
    if stop_distance < config.CONTRACT_SPEC.get("point", 0.01) * 5:
        log_warning("S1E_STOP_DISTANCE_TOO_SMALL", stop_distance=stop_distance)
        return None
    original_lots = state.get("original_lot_size", 0.01)
    lot_size      = max(
        config.CONTRACT_SPEC.get("volume_min", 0.01),
        round(original_lots * 0.5, 2)
    )


    expiry_utc = datetime.now(pytz.utc) + timedelta(minutes=config.M5_LIMIT_EXPIRY_MIN)


    candidate = _build_candidate(
        signal_type = SignalType.S1E_PYRAMID,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        extra       = {
            "order_expiry_utc": expiry_utc.isoformat(),
            "campaign_id":      state.get("open_campaign_id", str(uuid.uuid4())),
            "pyramid_add_done": True,
        }
    )


    log_event("S1E_PYRAMID_SIGNAL",
              direction=direction,
              entry=round(entry, 3),
              lot_size=lot_size)
    return candidate



# ─────────────────────────────────────────────────────────────────────────────
# S1f — POST-TIME-KILL RE-ENTRY (G4 Fix: own counter)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s1f_signal(state: dict) -> dict | None:
    """
    S1f: Post-time-kill NY re-entry.
    G4 Fix: uses s1f_attempts_today (max 1) — independent of s1_family_attempts_today.

    Conditions: London 16:30 TK fired, NY session active,
    SUPER/NORMAL regime, family vacant, s1f_attempts < 1.

    Stop: 15 points tighter than normal (late session).
    Target: 35 points minimum.
    Time kill: NY 13:00 local.
    """
    if not state.get("london_tk_fired_today", False):
        return None


    if is_past_ny_time_kill():
        return None


    session = get_current_session()
    if session not in ("NY", "LONDON_NY_OVERLAP"):
        return None


    permitted, reason = can_s1f_fire(state)
    if not permitted:
        return None


    ema20 = get_ema20_m5()
    if ema20 is None:
        return None


    bar = get_last_m5_bar()
    if bar is None:
        return None


    direction = state.get("last_s1_direction")
    if not direction:
        return None

    # CHANGE 3.5 / LOOP-9 FIX: H1 EMA20 direction validation.
    # H1 = macro safety gate (direction), M5 = entry timing. Both needed.
    # last_s1_direction persists even after S1 closes. If S1 was LONG that stopped out
    # and market reversed, S1f would still try LONG — trading against the trend.
    ema20_h1 = get_ema20_h1()
    if ema20_h1 is not None:
        mt5_s1f = get_mt5()
        tick_s1f = mt5_s1f.symbol_info_tick(config.SYMBOL)
        if tick_s1f is not None:
            current_mid = (tick_s1f.ask + tick_s1f.bid) / 2
            if direction == "LONG" and current_mid < ema20_h1:
                log_event("S1F_REJECTED_H1_REVERSAL",
                          direction=direction,
                          price=round(current_mid, 3),
                          ema20_h1=round(ema20_h1, 3))
                return None
            elif direction == "SHORT" and current_mid > ema20_h1:
                log_event("S1F_REJECTED_H1_REVERSAL",
                          direction=direction,
                          price=round(current_mid, 3),
                          ema20_h1=round(ema20_h1, 3))
                return None

    # G3 Fix: full body close
    if direction == "LONG":
        if not _body_close_above_ema20(bar, ema20):
            return None
        entry = round(ema20, 3)
        stop  = round(ema20 - config.S1F_STOP_POINTS, 3)
    else:
        if not _body_close_below_ema20(bar, ema20):
            return None
        entry = round(ema20, 3)
        stop  = round(ema20 + config.S1F_STOP_POINTS, 3)


    stop_distance = abs(entry - stop)
    lot_size      = calculate_lot_size(stop_distance, state["size_multiplier"], state)


    expiry_utc = datetime.now(pytz.utc) + timedelta(minutes=config.M5_LIMIT_EXPIRY_MIN)


    candidate = _build_candidate(
        signal_type = SignalType.S1F_POST_TK,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        extra       = {
            "order_expiry_utc":       expiry_utc.isoformat(),
            "post_time_kill_reentry": True,
            "target_min_pts":         config.S1F_TARGET_POINTS_MIN,
        }
    )


    log_event("S1F_SIGNAL_GENERATED",
              direction=direction,
              entry=round(entry, 3),
              stop=round(stop, 3),
              s1f_count=state["s1f_attempts_today"])
    return candidate



# ─────────────────────────────────────────────────────────────────────────────
# S2 — MEAN REVERSION (RANGING_CLEAR only)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s2_signal(state: dict) -> dict | None:
    """
    S2 Mean Reversion. RANGING_CLEAR regime only.
    Exits immediately if regime transitions to ANY trending state.

    Signal: Second CONSECUTIVE H1 close > 2.5× ATR(14,H1,RMA) away from 20 EMA.
    ATR percentile: between 30th–70th (not too quiet, not chaotic).
    G7: NOT within 30 min of London open (07:30–08:00 London local).
    Entry: LIMIT ORDER at 20 EMA — wait for return, never chase.
    Stop: 1.5× ATR beyond extreme point.
    """
    if is_past_london_time_kill():
        return None


    # G7 preserved: not within 30 min of London open
    if is_within_s2_london_open_restriction():
        log_event("S2_BLOCKED_LONDON_OPEN_RESTRICTION")
        return None


    permitted, reason = can_s2_fire(state)
    if not permitted:
        return None


    # ATR percentile gate: 30th–70th only
    atr_pct = state.get("last_atr_pct_h1", 0.0)
    if not (config.ATR_PCT_QUIET_REF <= atr_pct <= config.ATR_PCT_UNSTABLE_THRESHOLD):
        return None


    ema20 = get_ema20_h1()
    atr   = get_atr14_h1_rma()
    if ema20 is None or atr is None or atr == 0:
        return None


    df = fetch_ohlcv("H1", count=5)
    if df is None or len(df) < 3:
        return None


    # LOOP-2 FIX: Reduced threshold from 2.5× to 1.5× ATR with RSI confirmation.
    # 2.5× ATR required price 30-62pts from EMA20 for TWO consecutive hours — near-impossible.
    # 1.5× ATR + RSI filter provides quality mean-reversion signals that actually fire.
    c2 = float(df["close"].iloc[-2])  # last closed bar

    # Validate bar gap
    t1 = df["time"].iloc[-3]
    t2 = df["time"].iloc[-2]
    if hasattr(t1, "timestamp") and hasattr(t2, "timestamp"):
        time_diff_hours = (t2 - t1).total_seconds() / 3600
        if time_diff_hours > 3:
            log_warning("S2_NON_CONSECUTIVE_BARS", gap_hours=round(time_diff_hours, 1))
            return None

    threshold    = 1.5 * atr

    # RSI confirmation — require overbought/oversold for quality
    rsi_series = ta.rsi(df["close"], length=14)
    rsi_val = float(rsi_series.iloc[-2]) if rsi_series is not None and not pd.isna(rsi_series.iloc[-2]) else 50.0

    short_signal = (c2 > ema20 + threshold) and rsi_val > 70
    long_signal  = (c2 < ema20 - threshold) and rsi_val < 30


    if not short_signal and not long_signal:
        return None


    direction = "SHORT" if short_signal else "LONG"


    if direction == "SHORT":
        entry = round(ema20, 3)
        stop  = round(c2 + 1.5 * atr, 3)
    else:
        entry = round(ema20, 3)
        stop  = round(c2 - 1.5 * atr, 3)


    stop_distance = abs(entry - stop)
    lot_size      = calculate_lot_size(stop_distance, state["size_multiplier"], state)


    candidate = _build_candidate(
        signal_type = SignalType.S2_MEAN_REV,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        extra       = {
            "ema20_h1":        round(ema20, 3),
            "atr_h1":          round(atr, 4),
            "target_1_2r_min": True,
        }
    )


    log_event("S2_SIGNAL_GENERATED",
              direction=direction,
              ema20=round(ema20, 3),
              threshold=round(threshold, 3),
              atr_pct=round(atr_pct, 1))
    return candidate



# ─────────────────────────────────────────────────────────────────────────────
# S3 — STOP HUNT REVERSAL (CHANGE 6 — new)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_s3_signal(state: dict) -> dict | None:
    """
    S3 Stop Hunt Reversal.
    LONG: bearish sweep below range_low by 0.3×ATR14,
          then M15 close reclaims above range_low within 3 bars.
          Entry = BUY STOP 2pts above reclaim candle HIGH.
          Stop  = sweep_low - 0.5×ATR.
    SHORT: exact mirror above range_high.
    Regime: RANGING_CLEAR, NORMAL_TRENDING, or SUPER_TRENDING (latter at 0.5× lots).
    Max 1 S3 per session. Reversal family blocks S1b same day and vice versa.
    """
    # ── Gates ────────────────────────────────────────────────────────────────
    # CHANGE 3.3: Allow UNSTABLE — S3 is a stop-hunt reversal and UNSTABLE
    # (85-95th ATR percentile) is prime stop-hunting territory.
    # Size is already reduced to 0.4× by regime multiplier.
    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        return None

    if state.get("s3_fired_today", False):
        return None

    if state.get("reversal_family_occupied", False):
        return None

    range_data = state.get("range_data")
    if not range_data:
        return None

    # London / London-NY overlap only (08:00–16:30 UTC)
    utc_now = datetime.now(pytz.utc)
    if not (8 <= utc_now.hour < 16 or (utc_now.hour == 16 and utc_now.minute <= 30)):
        return None

    # LOOP-5 FIX: Use dynamic range from last 12 M15 bars (3 hours) instead of
    # stale pre-London range. By 12:00-14:00 UTC when S3 typically fires,
    # the pre-London range computed at 07:55 UTC is outdated.
    df_dynamic = fetch_ohlcv("M15", count=12)
    if df_dynamic is not None and len(df_dynamic) >= 8:
        # Use dynamic range (exclude current forming bar)
        range_high = float(df_dynamic["high"].iloc[:-1].max())
        range_low  = float(df_dynamic["low"].iloc[:-1].min())
        log_event("S3_USING_DYNAMIC_RANGE",
                  dynamic_high=round(range_high, 3),
                  dynamic_low=round(range_low, 3),
                  static_high=round(range_data["range_high"], 3),
                  static_low=round(range_data["range_low"], 3))
    else:
        # Fallback to pre-London range if dynamic data unavailable
        range_high = range_data["range_high"]
        range_low  = range_data["range_low"]

    # ── ATR14 H1 — FIX: compute from fetched df_h1, _calculate_atr14 doesn't exist ──
    df_h1 = fetch_ohlcv("H1", count=25)
    if df_h1 is None or len(df_h1) < 16:
        return None

    df_h1["atr"] = ta.atr(df_h1["high"], df_h1["low"], df_h1["close"],
                           length=14, mamode=config.ATR_MAMODE)
    atr_val = df_h1["atr"].iloc[-1]
    atr = float(atr_val) if not pd.isna(atr_val) else 0.0
    if atr <= 0:
        return None

    df = fetch_ohlcv("M15", count=6)
    if df is None or len(df) < 5:
        return None

    point           = config.CONTRACT_SPEC.get("point", 0.01)
    sweep_threshold = atr * config.S3_SWEEP_THRESHOLD_ATR

    # Last 3 CLOSED bars = iloc[-4], [-3], [-2]  (iloc[-1] is forming)
    closed = df.iloc[-4:-1]
    bar    = df.iloc[-2]   # most recent closed bar

    # ── FIX: use _build_candidate — not a raw dict —————————————————————————
    # This ensures conviction_level, event_proximity_min, spread_vs_avg_ratio
    # are all computed properly, same as every other signal in this file.
    def _make_candidate(direction, entry, stop):
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        lots = calculate_lot_size(risk, state.get("size_multiplier", 1.0), state)
        if regime == RegimeState.SUPER_TRENDING:
            lots = max(
                config.CONTRACT_SPEC.get("volume_min", 0.01),
                round(lots * 0.5, 2),
            )
        return _build_candidate(
            signal_type = SignalType.S3_STOP_HUNT_REV,   # FIX: was "S3_STOP_HUNT" raw string
            direction   = direction,
            entry_level = entry,
            stop_level  = stop,
            lot_size    = lots,
            state       = state,
            range_data  = range_data,
            extra       = {
                "stop_hunt_detected": True,
                "failed_breakout_trade":   False,
                "post_time_kill_reentry":  False,
            }
        )

    # ── LONG: bearish sweep below range_low ──────────────────────────────────
    for i in range(len(closed)):
        b = closed.iloc[i]
        if float(b["low"]) < range_low - sweep_threshold:
            sweep_low_price = float(b["low"])
            sweep_time      = b["time"]
            if float(bar["close"]) > range_low:
                bars_elapsed = int((bar["time"] - sweep_time).total_seconds() // 900)
                if bars_elapsed <= config.S3_WINDOW_CANDLES:
                    entry     = float(bar["high"]) + config.S3_RECLAIM_OFFSET_PTS * point
                    stop      = sweep_low_price - config.S3_STOP_ATR_MULT * atr
                    candidate = _make_candidate("LONG", entry, stop)
                    if candidate:
                        log_event("S3_SIGNAL_DETECTED", direction="LONG",
                                  sweep_low=round(sweep_low_price, 3),
                                  reclaim_close=round(float(bar["close"]), 3),
                                  entry=round(entry, 3), stop=round(stop, 3))
                        return candidate

    # ── SHORT: bullish sweep above range_high ─────────────────────────────────
    for i in range(len(closed)):
        b = closed.iloc[i]
        if float(b["high"]) > range_high + sweep_threshold:
            sweep_high_price = float(b["high"])
            sweep_time       = b["time"]
            if float(bar["close"]) < range_high:
                bars_elapsed = int((bar["time"] - sweep_time).total_seconds() // 900)
                if bars_elapsed <= config.S3_WINDOW_CANDLES:
                    entry     = float(bar["low"]) - config.S3_RECLAIM_OFFSET_PTS * point
                    stop      = sweep_high_price + config.S3_STOP_ATR_MULT * atr
                    candidate = _make_candidate("SHORT", entry, stop)
                    if candidate:
                        log_event("S3_SIGNAL_DETECTED", direction="SHORT",
                                  sweep_high=round(sweep_high_price, 3),
                                  reclaim_close=round(float(bar["close"]), 3),
                                  entry=round(entry, 3), stop=round(stop, 3))
                        return candidate

    return None

# ─────────────────────────────────────────────────────────────────────────────
# S6 — ASIAN RANGE BREAKOUT (CHANGE 7 — new, dual pending)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s6_signal(state: dict) -> dict | None:
    """
    S6: Asian Range Breakout.
    Called from asian_range_job() at 05:30 UTC daily — NOT from m15_dispatch_job.

    Asian session range: 00:00–05:30 UTC (computed by get_asian_range()).
    Min range: 8 pts — skip if tighter (config.S6_MIN_RANGE_PTS).

    Returns a dict with buy_candidate + sell_candidate (dual pending orders).
    Both BUY STOP + SELL STOP are placed simultaneously.
    On fill: execution engine cancels the opposite leg automatically.

    Entry:
      BUY STOP  at range_high + 5% of range
      SELL STOP at range_low  - 5% of range
    Stop:
      LONG  stop = range_low  - 0.5 × ATR14_H1
      SHORT stop = range_high + 0.5 × ATR14_H1
    Expiry: 08:00 UTC today (London open — stale breakout levels after this)

    Returns None if any gate fails or data unavailable.
    Returns {"buy_candidate": ..., "sell_candidate": ..., "range_data": ...}
    """
    permitted, reason = can_s6_fire(state)
    if not permitted:
        log_event("S6_GATE_BLOCKED", reason=reason)
        return None


    # Get Asian range for today
    range_data = get_asian_range()
    if range_data is None:
        return None


    # ATR14 H1 for stop distance (0.5 × ATR — wider than fixed pts)
    atr_h1 = get_atr14_h1_rma()
    if atr_h1 is None:
        # Try to get ATR from state (cached by regime engine)
        atr_h1 = state.get("last_atr_h1_raw")
        if atr_h1 is None or atr_h1 <= 0:
            log_warning("S6_NO_ATR_H1_USING_FALLBACK")
            # Use range-based fallback: 50% of range size as ATR estimate
            atr_h1 = range_data["range_size"] * 0.5
            if atr_h1 <= 0:
                atr_h1 = 15.0  # absolute fallback


    rh = range_data["range_high"]
    rl = range_data["range_low"]
    bd = range_data["breakout_dist"]   # 5% of range

    stop_buffer = round(atr_h1 * 0.5, 2)   # 0.5 × ATR14_H1


    # ── BUY STOP candidate ────────────────────────────────────────────────────
    buy_entry     = round(rh + bd, 3)
    buy_stop      = round(rl - stop_buffer, 3)
    buy_stop_dist = abs(buy_entry - buy_stop)
    buy_lots      = calculate_lot_size(buy_stop_dist, state["size_multiplier"], state)


    # ── SELL STOP candidate ───────────────────────────────────────────────────
    sell_entry     = round(rl - bd, 3)
    sell_stop      = round(rh + stop_buffer, 3)
    sell_stop_dist = abs(sell_entry - sell_stop)
    sell_lots      = calculate_lot_size(sell_stop_dist, state["size_multiplier"], state)


    # Expiry: 08:00 UTC today
    now_utc    = datetime.now(pytz.utc)
    expiry_utc = now_utc.replace(hour=8, minute=0, second=0, microsecond=0)

    # Safety: if it's already past 08:00 UTC (job ran late), skip
    if expiry_utc <= now_utc:
        log_event("S6_PAST_EXPIRY_SKIPPED",
                  now_utc=str(now_utc.strftime("%H:%M UTC")))
        return None


    expiry_iso = expiry_utc.isoformat()

    buy_candidate = _build_candidate(
        signal_type = SignalType.S6_ASIAN_BRK,   # BUG-3 FIX: was S6_ASIAN_RNG (doesn't exist)
        direction   = "LONG",
        entry_level = buy_entry,
        stop_level  = buy_stop,
        lot_size    = buy_lots,
        state       = state,
        range_data  = range_data,
        extra       = {
            "order_expiry_utc": expiry_iso,
            "s6_leg":           "BUY",
            "atr_h1_at_signal": round(atr_h1, 4),
            "stop_buffer_pts":  stop_buffer,
        }
    )

    sell_candidate = _build_candidate(
        signal_type = SignalType.S6_ASIAN_BRK,   # BUG-3 FIX: was S6_ASIAN_RNG (doesn't exist)
        direction   = "SHORT",
        entry_level = sell_entry,
        stop_level  = sell_stop,
        lot_size    = sell_lots,
        state       = state,
        range_data  = range_data,
        extra       = {
            "order_expiry_utc": expiry_iso,
            "s6_leg":           "SELL",
            "atr_h1_at_signal": round(atr_h1, 4),
            "stop_buffer_pts":  stop_buffer,
        }
    )


    # EXP-9 FIX: ADX trend filter — in strong trends, only place the trending direction.
    # Prevents counter-trend STOP orders from filling on quick spikes then reversing.
    df_adx = fetch_ohlcv("H1", count=20)
    if df_adx is not None and len(df_adx) >= 16:
        adx_df = ta.adx(df_adx["high"], df_adx["low"], df_adx["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            adx_val = float(adx_df["ADX_14"].iloc[-1]) if not pd.isna(adx_df["ADX_14"].iloc[-1]) else 0.0
            plus_di = float(adx_df["DMP_14"].iloc[-1]) if not pd.isna(adx_df["DMP_14"].iloc[-1]) else 0.0
            minus_di = float(adx_df["DMN_14"].iloc[-1]) if not pd.isna(adx_df["DMN_14"].iloc[-1]) else 0.0

            if adx_val > 25:
                if plus_di > minus_di * 1.3:
                    # Strong uptrend — only place BUY STOP
                    sell_candidate = None
                    log_event("S6_TREND_FILTER_SELL_REMOVED", adx=round(adx_val, 1),
                              plus_di=round(plus_di, 1), minus_di=round(minus_di, 1))
                elif minus_di > plus_di * 1.3:
                    # Strong downtrend — only place SELL STOP
                    buy_candidate = None
                    log_event("S6_TREND_FILTER_BUY_REMOVED", adx=round(adx_val, 1),
                              plus_di=round(plus_di, 1), minus_di=round(minus_di, 1))

    # If both candidates were filtered out, return None
    if buy_candidate is None and sell_candidate is None:
        log_event("S6_BOTH_LEGS_FILTERED_BY_TREND")
        return None

    log_event("S6_SIGNAL_GENERATED",
              range_high=rh,
              range_low=rl,
              range_size=range_data["range_size"],
              buy_entry=buy_entry,
              sell_entry=sell_entry,
              stop_buffer=stop_buffer,
              expiry="08:00 UTC")

    return {
        "buy_candidate":  buy_candidate,
        "sell_candidate": sell_candidate,
        "range_data":     range_data,
    }



# ─────────────────────────────────────────────────────────────────────────────
# S7 — DAILY STRUCTURE BREAKOUT (CHANGE 8 — new, dual pending)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s7_signal(state: dict) -> dict | None:
    """
    S7: Daily Structure Breakout.
    Called from midnight_reset_job() — NOT from m15_dispatch_job.

    Prev day high/low from get_prev_day_ohlc(). Daily ATR14 from get_daily_atr14().

    Inside day filter: if prev_day_range < 0.75 × daily_ATR14, skip.
    An inside day has a compressed range — breakout of a compressed day is
    statistically less reliable and stops are too wide relative to the move.

    Returns dict with buy_candidate + sell_candidate (dual pending orders).
    Both BUY STOP + SELL STOP placed at midnight. On fill: opposite leg cancelled.

    Entry:
      BUY STOP  at prev_day_high + S7_ENTRY_OFFSET_PTS (5 pts)
      SELL STOP at prev_day_low  - S7_ENTRY_OFFSET_PTS (5 pts)
    Stop:
      LONG  stop = prev_day_low  - S7_STOP_OFFSET_PTS (10 pts)
      SHORT stop = prev_day_high + S7_STOP_OFFSET_PTS (10 pts)
    Size: 0.5× base lot (S7_LOT_MULTIPLIER) — wider stops need smaller size.

    Returns None if gate fails, inside day, or data unavailable.
    """
    permitted, reason = can_s7_fire(state)
    if not permitted:
        log_event("S7_GATE_BLOCKED", reason=reason)
        return None


    prev_day = get_prev_day_ohlc()
    if prev_day is None:
        log_warning("S7_NO_PREV_DAY_DATA")
        return None
    
    # Validate prev_day data freshness — must be from actual previous trading day
    # If today is Monday, prev_day should be Friday
    # If today is Sunday, prev_day should be Friday (skip Saturday)
    now_utc = datetime.now(pytz.utc)
    today_weekday = now_utc.weekday()  # 0=Mon, 6=Sun
    
    # Check if prev_day has a 'time' field to validate freshness
    if "time" in prev_day:
        prev_day_time = prev_day["time"]
        if hasattr(prev_day_time, "date"):
            prev_date = prev_day_time.date() if hasattr(prev_day_time, "date") else prev_day_time
            today_date = now_utc.date()
            days_diff = (today_date - prev_date).days
            
            # If more than 3 days old (weekend + holiday), data may be stale
            if days_diff > 3:
                log_warning("S7_PREV_DAY_DATA_STALE", days_old=days_diff)
                # Don't return None — still use it but log warning


    daily_atr14 = get_daily_atr14()
    if daily_atr14 is None:
        log_warning("S7_NO_DAILY_ATR14")
        return None


    ph         = prev_day["high"]
    pl         = prev_day["low"]
    prev_range = ph - pl


    # Inside day filter: S7_INSIDE_DAY_FILTER = 0.75
    if prev_range < daily_atr14 * config.S7_MIN_RANGE_ATR_RATIO:
        log_event("S7_INSIDE_DAY_FILTERED",
                  prev_range=round(prev_range, 2),
                  daily_atr14=round(daily_atr14, 2),
                  threshold=round(daily_atr14 * config.S7_MIN_RANGE_ATR_RATIO, 2))
        return None


    # Canonical state keys (validate_state_keys) — execution also sets on place
    state["s7_prev_day_high"] = ph
    state["s7_prev_day_low"]  = pl


    # EXP-6 FIX: Use ATR-based stop instead of prev_day opposite extreme.
    # With daily ranges of 40-80pts, the old stop was 50-90pts wide — poor R:R.
    # 50% of daily ATR gives a reasonable stop that auto-scales with volatility.
    stop_dist = daily_atr14 * 0.5

    # ── BUY STOP candidate ────────────────────────────────────────────────────
    buy_entry     = round(ph + config.S7_ENTRY_OFFSET_PTS, 3)   # 5 pts above prev high
    buy_stop      = round(buy_entry - stop_dist, 3)              # EXP-6: ATR-based stop
    buy_stop_dist = abs(buy_entry - buy_stop)
    base_buy  = calculate_lot_size(buy_stop_dist, state["size_multiplier"] * config.S7_SIZE_MULTIPLIER, state)
    buy_lots  = max(
        config.CONTRACT_SPEC.get("volume_min", 0.01),
        round(base_buy, 2)
    )


    # ── SELL STOP candidate ───────────────────────────────────────────────────
    sell_entry     = round(pl - config.S7_ENTRY_OFFSET_PTS, 3)  # 5 pts below prev low
    sell_stop      = round(sell_entry + stop_dist, 3)            # EXP-6: ATR-based stop
    sell_stop_dist = abs(sell_entry - sell_stop)
    base_sell  = calculate_lot_size(sell_stop_dist, state["size_multiplier"] * config.S7_SIZE_MULTIPLIER, state)
    sell_lots  = max(
        config.CONTRACT_SPEC.get("volume_min", 0.01),
        round(base_sell, 2)
    )


    buy_candidate = _build_candidate(
        signal_type = SignalType.S7_DAILY_STRUCT,
        direction   = "LONG",
        entry_level = buy_entry,
        stop_level  = buy_stop,
        lot_size    = buy_lots,
        state       = state,
        extra       = {
            "s7_leg":        "BUY",
            "prev_day_high": ph,
            "prev_day_low":  pl,
            "prev_range":    round(prev_range, 2),
            "daily_atr14":   round(daily_atr14, 2),
        }
    )

    sell_candidate = _build_candidate(
        signal_type = SignalType.S7_DAILY_STRUCT,
        direction   = "SHORT",
        entry_level = sell_entry,
        stop_level  = sell_stop,
        lot_size    = sell_lots,
        state       = state,
        extra       = {
            "s7_leg":        "SELL",
            "prev_day_high": ph,
            "prev_day_low":  pl,
            "prev_range":    round(prev_range, 2),
            "daily_atr14":   round(daily_atr14, 2),
        }
    )


    # EXP-9 FIX: ADX trend filter for S7 — same logic as S6.
    # In strong trends, only place the trending direction to avoid counter-trend fills.
    df_adx_s7 = fetch_ohlcv("H1", count=20)
    if df_adx_s7 is not None and len(df_adx_s7) >= 16:
        adx_df_s7 = ta.adx(df_adx_s7["high"], df_adx_s7["low"], df_adx_s7["close"], length=14)
        if adx_df_s7 is not None and not adx_df_s7.empty:
            adx_val_s7 = float(adx_df_s7["ADX_14"].iloc[-1]) if not pd.isna(adx_df_s7["ADX_14"].iloc[-1]) else 0.0
            plus_di_s7 = float(adx_df_s7["DMP_14"].iloc[-1]) if not pd.isna(adx_df_s7["DMP_14"].iloc[-1]) else 0.0
            minus_di_s7 = float(adx_df_s7["DMN_14"].iloc[-1]) if not pd.isna(adx_df_s7["DMN_14"].iloc[-1]) else 0.0

            if adx_val_s7 > 25:
                if plus_di_s7 > minus_di_s7 * 1.3:
                    sell_candidate = None
                    log_event("S7_TREND_FILTER_SELL_REMOVED", adx=round(adx_val_s7, 1))
                elif minus_di_s7 > plus_di_s7 * 1.3:
                    buy_candidate = None
                    log_event("S7_TREND_FILTER_BUY_REMOVED", adx=round(adx_val_s7, 1))

    if buy_candidate is None and sell_candidate is None:
        log_event("S7_BOTH_LEGS_FILTERED_BY_TREND")
        return None

    log_event("S7_SIGNAL_GENERATED",
              prev_high=ph,
              prev_low=pl,
              prev_range=round(prev_range, 2),
              daily_atr14=round(daily_atr14, 2),
              buy_entry=buy_entry,
              sell_entry=sell_entry,
              buy_lots=buy_lots,
              sell_lots=sell_lots)

    return {
        "buy_candidate":  buy_candidate,
        "sell_candidate": sell_candidate,
        "prev_day_high":  ph,
        "prev_day_low":   pl,
    }



# ─────────────────────────────────────────────────────────────────────────────
# F7: S8 ATR SPIKE TRADE
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_s8_signal(state: dict) -> dict | None:
    """
    F7: ATR Spike Trade — Flash spike continuation strategy.

    Market structure: Flash spikes on XAUUSD frequently precede continuation moves,
    especially during London institutional sweeps. This strategy catches the
    confirmation candle that follows a spike.

    Logic:
    1. DETECT spike: M15 candle range > 1.5× ATR(14,H1) on same timeframe
    2. ARM: Store spike high/low, direction, arm_time
    3. CONFIRM: Next M15 candle closes past spike midpoint
    4. TRADE: STOP order in direction of continuation

    Constraints:
    - Does NOT fire when trend_family_occupied (S1 pending/active)
    - Does NOT fire when s8_fired_today = True
    - Max 1 S8 per day
    - 0.5× base lot size (tight stops)

    Runs in: m5_mgmt_job() — M5-level monitoring
    """
    from engines.data_engine import get_current_atr_m15, fetch_ohlcv

    # Already fired today — skip
    if state.get("s8_fired_today"):
        return None

    # S8 independent lane — blocked when already open, not by trend_family
    if state.get("s8_open_ticket"):
        return None

    # Regime gate — block NO_TRADE only. UNSTABLE allowed because
    # 0.4× regime multiplier × 0.5× S8 lot = 0.2× effective risk.
    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        return None

    # Don't fire if S1 pending orders are active (prevents S8 market fill +
    # S1 pending fill creating two simultaneous positions)
    if state.get("s1_pending_buy_ticket") or state.get("s1_pending_sell_ticket"):
        return None

    # Get current ATR for spike threshold
    current_atr = get_current_atr_m15(period=14)
    if current_atr is None:
        return None

    spike_threshold = current_atr * 1.5  # Spike = 1.5× ATR

    # Get recent M15 candles
    df = fetch_ohlcv("M15", count=5)
    if df is None or len(df) < 3:
        return None

    # Analyze most recent closed candle
    last_candle = df.iloc[-2]  # Previous closed candle (not current forming)
    prev_candle = df.iloc[-3]

    last_range = last_candle["high"] - last_candle["low"]

    # Check if we're currently armed and waiting for confirmation
    if state.get("s8_armed"):
        return _check_s8_confirmation(state, df, spike_threshold)

    # Check if last candle was a spike (potential arm condition)
    if last_range > spike_threshold:
        # Determine direction based on spike candle's position relative to previous candle
        # If spike candle's high is above prev candle's high → bullish spike (long)
        # If spike candle's low is below prev candle's low → bearish spike (short)
        if last_candle["high"] > prev_candle["high"] and last_candle["low"] >= prev_candle["low"]:
            direction = "long"
        elif last_candle["low"] < prev_candle["low"] and last_candle["high"] <= prev_candle["high"]:
            direction = "short"
        else:
            # Mixed spike — use close vs open as tiebreaker
            direction = "long" if last_candle["close"] > last_candle["open"] else "short"

        # Arm the trade — wait for confirmation
        state["s8_armed"] = True
        state["s8_arm_time"] = datetime.now(pytz.utc)
        state["s8_spike_high"] = float(last_candle["high"])
        state["s8_spike_low"] = float(last_candle["low"])
        state["s8_direction"] = direction
        state["s8_spike_atr"] = float(current_atr)
        state["s8_arm_candle_time"] = last_candle["time"]
        # Track the spike candle index so we can find the NEXT candle for confirmation
        state["s8_spike_candle_idx"] = len(df) - 2  # index of last_candle in current df

        log_event("S8_ARMED",
                  direction=direction,
                  spike_high=round(float(last_candle["high"]), 3),
                  spike_low=round(float(last_candle["low"]), 3),
                  spike_range=round(last_range, 2),
                  atr=round(current_atr, 2))

        return None  # Wait for confirmation

    return None


def _check_s8_confirmation(state: dict, df: pd.DataFrame, spike_threshold: float) -> dict | None:
    """
    F7: Check if S8 confirmation candle has formed.

    Confirmation rules:
    - Direction = LONG: candle closes above spike midpoint
    - Direction = SHORT: candle closes below spike midpoint
    - Must be within 3 candles of arm_time (45 min window)
    - Must be a NEW candle (not the spike candle itself)
    """
    arm_time = state.get("s8_arm_time")
    arm_candle_time = state.get("s8_arm_candle_time")
    spike_candle_idx = state.get("s8_spike_candle_idx")

    # Check window expiry using candle count (3 M15 candles = 45 min)
    if arm_candle_time:
        # Count candles since arm by checking how many closed bars are after arm candle
        candles_since_arm = 0
        for i in range(len(df) - 2, -1, -1):  # iterate from most recent closed bar backwards
            bar_time = df.iloc[i]["time"]
            if bar_time > arm_candle_time:
                candles_since_arm += 1
            else:
                break
        
        if candles_since_arm > 3:
            # Window expired — disarm
            log_event("S8_WINDOW_EXPIRED", candles_since=candles_since_arm)
            state["s8_armed"] = False
            state["s8_arm_time"] = None
            state["s8_arm_candle_time"] = None
            state["s8_spike_candle_idx"] = None
            return None
        
        # Must have at least 1 new candle since arm
        if candles_since_arm < 1:
            return None  # Still on spike candle, wait for next
    elif arm_time:
        # Fallback to wall clock if arm_candle_time not set
        now = datetime.now(pytz.utc)
        minutes_since_arm = (now - arm_time).total_seconds() / 60
        if minutes_since_arm > 45:
            log_event("S8_WINDOW_EXPIRED", minutes_since=round(minutes_since_arm, 1))
            state["s8_armed"] = False
            state["s8_arm_time"] = None
            state["s8_spike_candle_idx"] = None
            return None

    direction = state.get("s8_direction")
    spike_high = state.get("s8_spike_high", 0)
    spike_low = state.get("s8_spike_low", 0)
    spike_midpoint = (spike_high + spike_low) / 2

    # Get confirmation candle — must be the NEXT candle after the spike
    # Use spike_candle_idx to find the correct confirmation candle
    if spike_candle_idx is not None:
        # The confirmation candle should be at spike_candle_idx + 1 in the original df
        # But since df may have changed, find it by time
        spike_time = state.get("s8_arm_candle_time")
        if spike_time:
            # Find the first candle after the spike
            confirm_candle = None
            for i in range(len(df) - 2, -1, -1):
                if df.iloc[i]["time"] > spike_time:
                    confirm_candle = df.iloc[i]
                    break
            
            if confirm_candle is None:
                return None  # No confirmation candle yet
            
            confirm_close = float(confirm_candle["close"])
        else:
            # Fallback: use the candle at spike_candle_idx + 1 if available
            confirm_idx = spike_candle_idx + 1
            if confirm_idx >= len(df) - 1:  # -1 because we skip the forming candle
                return None  # Not enough candles yet
            confirm_candle = df.iloc[confirm_idx]
            confirm_close = float(confirm_candle["close"])
    else:
        # Fallback: use most recent closed candle (legacy behavior)
        confirm_candle = df.iloc[-2]
        confirm_close = float(confirm_candle["close"])
        
        # Verify this is not the spike candle
        if arm_candle_time and confirm_candle["time"] <= arm_candle_time:
            return None  # Still on spike candle, wait for next

    # Check confirmation
    confirmed = False
    if direction == "long" and confirm_close > spike_midpoint:
        confirmed = True
    elif direction == "short" and confirm_close < spike_midpoint:
        confirmed = True

    if not confirmed:
        return None

    # Build S8 candidate
    from engines.risk_engine import calculate_lot_size

    # LOOP-6 FIX: Enter at market on confirmation, not at spike midpoint.
    # If price confirmed by closing past midpoint, it's already BEYOND the midpoint.
    # A limit order at midpoint may never fill if momentum continues.
    mt5_s8 = get_mt5()
    tick_s8 = mt5_s8.symbol_info_tick(config.SYMBOL)
    atr_stop_buffer = float(state.get("s8_spike_atr", 0)) * 0.5
    size_mult = 0.5  # 0.5× base lot (tight stops)

    if direction == "long":
        entry = tick_s8.ask if tick_s8 else spike_midpoint
        stop = spike_low - atr_stop_buffer
    else:
        entry = tick_s8.bid if tick_s8 else spike_midpoint
        stop = spike_high + atr_stop_buffer

    # CRITICAL FIX (Change 3.2): Use actual distance from entry to stop, NOT just
    # the ATR buffer. Entry is market price at confirmation, which can be far from
    # the spike extreme. Using atr_stop_buffer alone would calculate lots for 5pt
    # risk when actual risk is 25pts → 5× oversize.
    # Floor: never calculate on <5-point stop (prevents lot explosion).
    actual_stop_dist = max(
        abs(entry - stop),
        5.0 * config.CONTRACT_SPEC.get("point", 0.01)
    )

    lots = calculate_lot_size(actual_stop_dist, state["size_multiplier"] * size_mult, state)

    candidate = _build_candidate(
        signal_type=SignalType.S8_ATR_SPIKE,
        direction=direction.upper(),
        entry_level=round(entry, 3),
        stop_level=round(stop, 3),
        lot_size=lots,
        state=state,
        extra={
            "s8_direction": direction,
            "spike_high": spike_high,
            "spike_low": spike_low,
            "spike_midpoint": spike_midpoint,
            "confirm_close": confirm_close,
        }
    )

    # Disarm after generating candidate
    state["s8_armed"] = False
    state["s8_arm_time"] = None
    state["s8_arm_candle_time"] = None
    state["s8_fired_today"] = True

    log_event("S8_SIGNAL_GENERATED",
              direction=direction,
              entry=round(entry, 3),
              stop=round(stop, 3),
              lots=lots,
              spike_midpoint=round(spike_midpoint, 3))

    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# S8 POSITION MANAGEMENT (Change 3.8)
# ─────────────────────────────────────────────────────────────────────────────

def _s8_modify_stop(ticket: int, new_stop: float, reason: str, state: dict) -> bool:
    """
    Inline stop modification for S8 — bypasses shared modify_stop()
    which reads S1-specific state keys (last_s1_direction, stop_price_current).
    Using modify_stop() for S8 would read wrong direction and wrong stop.
    S8 MUST use its own inline modifier.
    """
    mt5_mod = get_mt5()
    pos = next(
        (p for p in (mt5_mod.positions_get(symbol=config.SYMBOL) or [])
         if p.ticket == ticket and p.magic == config.MAGIC),
        None
    )
    if not pos:
        return False
    request = {
        "action":   mt5_mod.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   config.SYMBOL,
        "sl":       new_stop,
        "tp":       pos.tp,
        "magic":    config.MAGIC,
    }
    result = mt5_mod.order_send(request)
    if result and result.retcode == mt5_mod.TRADE_RETCODE_DONE:
        log_event(reason, ticket=ticket, new_stop=round(new_stop, 3))
        return True
    log_event(f"{reason}_FAILED", ticket=ticket,
              retcode=result.retcode if result else "NO_RESULT")
    return False


def manage_s8_position(state: dict) -> None:
    """
    S8 position management — runs every M5 bar.
    Independent from trend family management.
    Simplified: BE at 1.5R + ATR trail only.
    No partial exit (0.5x lot too small to split).
    No momentum cycle exit (wrong pattern for spikes).
    """
    ticket = state.get("s8_open_ticket")
    if not ticket:
        return

    mt5_s8 = get_mt5()
    positions = mt5_s8.positions_get(symbol=config.SYMBOL) or []
    positions = [p for p in positions if p.ticket == ticket and p.magic == config.MAGIC]

    # ── Position closed by SL/TP on broker side? ──────────────────────────────
    if not positions:
        log_event("S8_POSITION_CLOSED_BROKER", ticket=ticket)
        deals = mt5_s8.history_deals_get(position=ticket) or []
        exit_price = state.get("s8_entry_price", 0.0)
        for d in deals:
            if d.entry == mt5_s8.DEAL_ENTRY_OUT:
                exit_price = d.price
                break
        # Deferred import to avoid circular dependency
        from engines.execution_engine import on_trade_closed
        on_trade_closed(ticket, exit_price, "S8_BROKER_CLOSE", state)
        return

    pos       = positions[0]
    entry     = state["s8_entry_price"]
    stop_orig = state["s8_stop_price_original"]
    direction = state["s8_trade_direction"]

    # ── BE Activation at 1.5R ─────────────────────────────────────────────────
    if not state["s8_be_activated"]:
        r_now = calculate_r_multiple(entry, pos.price_current, stop_orig, direction)
        if r_now >= config.BE_ACTIVATION_R:    # 1.5
            success = _s8_modify_stop(ticket, entry, "S8_BE_ACTIVATED", state)
            if success:
                state["s8_be_activated"]        = True
                state["s8_stop_price_current"]  = entry
                log_event("S8_BE_ACTIVATED",
                          r_now=round(r_now, 2), ticket=ticket)
        return    # Don't trail until BE is activated

    # ── ATR Trail (only after BE) ─────────────────────────────────────────────
    new_trail = calculate_atr_trail(pos.price_current, direction)
    if new_trail is None:
        return
    current_stop = state["s8_stop_price_current"]

    if direction == "LONG" and new_trail > current_stop:
        success = _s8_modify_stop(ticket, new_trail, "S8_ATR_TRAIL", state)
        if success:
            state["s8_stop_price_current"] = new_trail
    elif direction == "SHORT" and new_trail < current_stop:
        success = _s8_modify_stop(ticket, new_trail, "S8_ATR_TRAIL", state)
        if success:
            state["s8_stop_price_current"] = new_trail


def _on_s8_closed(state: dict) -> None:
    """Clean up all S8 position state. Does NOT reset s8_fired_today."""
    state["s8_open_ticket"]         = None
    state["s8_entry_price"]         = 0.0
    state["s8_stop_price_original"] = 0.0
    state["s8_stop_price_current"]  = 0.0
    state["s8_trade_direction"]     = None
    state["s8_be_activated"]        = False
    state["s8_open_time_utc"]       = None


def check_partial_exit_condition(state: dict) -> bool:
    """
    WEAK_TRENDING hybrid exit: partial close 50% at 1.0R.
    Returns True when condition is met and partial not yet done.
    Execution engine handles the actual close.
    """
    if state.get("position_partial_done"):
        return False


    open_ticket = state.get("open_position")
    if not open_ticket:
        return False


    mt5 = get_mt5()
    pos = _get_our_position(mt5)
    if pos is None:
        return False


    entry         = state.get("entry_price", 0.0)
    stop_original = state.get("stop_price_original", 0.0)
    direction     = state.get("last_s1_direction", "LONG")


    r_now = calculate_r_multiple(entry, pos.price_current, stop_original, direction)
    return r_now >= config.PARTIAL_EXIT_R  # EXP-3 FIX: was hardcoded 1.0, now uses config (2.0)



def check_be_activation_condition(state: dict) -> bool:
    """
    WEAK_TRENDING breakeven activation.
    Spec: BOTH conditions required — NOT just distance alone.
      1. Profit R >= 0.75
      2. Recent M15 swing formed beyond entry (structural confirmation)

    This two-condition rule prevents premature BE moves on shallow pullbacks.
    """
    if state.get("position_be_activated"):
        return False


    mt5 = get_mt5()
    pos = _get_our_position(mt5)
    if pos is None:
        return False


    entry         = state.get("entry_price", 0.0)
    stop_original = state.get("stop_price_original", 0.0)
    direction     = state.get("last_s1_direction", "LONG")


    r_now = calculate_r_multiple(entry, pos.price_current, stop_original, direction)
    if r_now < config.BE_ACTIVATION_R:  # EXP-4 FIX: was hardcoded 0.75, now uses config (1.5)
        return False


    # Condition 2: structural swing beyond entry on M15
    df = fetch_ohlcv("M15", count=6)
    if df is None or df.empty:
        return False


    if direction == "LONG":
        recent_high = float(df["high"].iloc[-5:-1].max())
        return recent_high > entry
    else:
        recent_low = float(df["low"].iloc[-5:-1].min())
        return recent_low < entry



def check_momentum_cycle_exit(state: dict) -> bool:
    """
    SUPER/NORMAL: exit when M5 EMA20 body BREAKS against position.
    Returns True = execution engine should close position and allow S1d re-entry.
    """
    mt5 = get_mt5()
    pos = _get_our_position(mt5)
    if pos is None:
        return False


    ema20 = get_ema20_m5()
    if ema20 is None:
        return False


    bar = get_last_m5_bar()
    if bar is None:
        return False


    direction = state.get("last_s1_direction", "LONG")


    if direction == "LONG":
        return _body_close_below_ema20(bar, ema20)
    else:
        return _body_close_above_ema20(bar, ema20)



def check_s2_regime_exit(state: dict) -> bool:
    """
    S2 exit rule: if regime transitions to ANY trending state → exit at market.
    Called on every regime update when S2 position is open.
    """
    if state.get("trend_family_strategy") != SignalType.S2_MEAN_REV.value:
        return False
    regime = get_safe_regime(state)
    return regime.is_trending



def _get_our_position(mt5):
    """Returns our open XAUUSD position filtered by magic number (C5 Fix)."""
    positions = mt5.positions_get(symbol=config.SYMBOL)
    if not positions:
        return None
    ours = [p for p in positions if p.magic == config.MAGIC]
    return ours[0] if ours else None



def manage_open_position(state: dict) -> dict:
    """
    Dispatcher: called on every M5 candle close when position is open.
    Returns action dict consumed by execution engine.

    Actions:
      "NONE"           — no action needed
      "PARTIAL_EXIT"   — close 50% at market (WEAK hybrid)
      "ACTIVATE_BE"    — move stop to entry (WEAK hybrid, both conditions met)
      "CYCLE_EXIT"     — close full position for momentum cycle re-entry (SUPER/NORMAL)
      "S2_FORCE_EXIT"  — S2 regime-change exit
      "S1D_REENTRY"    — candidate dict for S1d re-entry after cycle exit
    """
    # S2 does not use trend_family_occupied — still needs regime-based exit.
    if (state.get("open_position")
            and state.get("trend_family_strategy") == SignalType.S2_MEAN_REV.value):
        if check_s2_regime_exit(state):
            log_event("S2_REGIME_CHANGE_EXIT")
            return {"action": "S2_FORCE_EXIT"}
        return {"action": "NONE"}

    if not state.get("trend_family_occupied"):
        return {"action": "NONE"}

    regime   = get_safe_regime(state)
    strategy = state.get("trend_family_strategy", "")

    # LOOP-4 FIX: S4/S5 are trend-continuation — use same momentum trailing as S1.
    # Also apply BE activation after sufficient profit.
    if strategy in ("S4_LONDON_PULL", "S5_NY_COMPRESS"):
        if regime in (RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING):
            if check_momentum_cycle_exit(state):
                log_event("MOMENTUM_CYCLE_EXIT", strategy=strategy)
                return {"action": "CYCLE_EXIT"}
        # BE activation for S4/S5 in any trending regime
        if not state.get("position_be_activated"):
            if check_be_activation_condition(state):
                log_event("BE_ACTIVATION_TRIGGERED", strategy=strategy)
                return {"action": "ACTIVATE_BE"}
        # Partial exit for S4/S5 in WEAK regime
        if regime == RegimeState.WEAK_TRENDING:
            if check_partial_exit_condition(state):
                log_event("PARTIAL_EXIT_TRIGGERED", strategy=strategy, r_threshold=config.PARTIAL_EXIT_R)
                return {"action": "PARTIAL_EXIT"}
        return {"action": "NONE"}

    # LOOP-4 FIX: S6/S7 — apply BE activation + ATR trail (no momentum cycling)
    if strategy in ("S6_ASIAN_BRK", "S7_DAILY_STRUCT"):
        if not state.get("position_be_activated"):
            if check_be_activation_condition(state):
                log_event("BE_ACTIVATION_TRIGGERED", strategy=strategy)
                return {"action": "ACTIVATE_BE"}
        if not state.get("position_partial_done"):
            if check_partial_exit_condition(state):
                log_event("PARTIAL_EXIT_TRIGGERED", strategy=strategy, r_threshold=config.PARTIAL_EXIT_R)
                return {"action": "PARTIAL_EXIT"}
        return {"action": "NONE"}

    # S1 family: original logic
    if regime in (RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING):
        if check_momentum_cycle_exit(state):
            log_event("MOMENTUM_CYCLE_EXIT")
            return {"action": "CYCLE_EXIT"}


    elif regime == RegimeState.WEAK_TRENDING:
        if check_partial_exit_condition(state):
            log_event("PARTIAL_EXIT_TRIGGERED", r_threshold=config.PARTIAL_EXIT_R)
            return {"action": "PARTIAL_EXIT"}
        if check_be_activation_condition(state):
            log_event("BE_ACTIVATION_TRIGGERED")
            return {"action": "ACTIVATE_BE"}


    return {"action": "NONE"}
