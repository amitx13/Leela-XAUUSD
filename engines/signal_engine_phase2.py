"""
engines/signal_engine_phase2.py — Phase 2 Strategies: R3, S4, S5

This is a standalone module. Import and wire into main.py directly.
See main_phase2_wiring.py for exact wiring instructions.

REQUIRED ADDITIONS to engines/signal_engine.py's SignalType enum
(add these 3 lines inside the class body):
    R3_CAL_MOMENTUM = "R3_CAL_MOMENTUM"
    S4_LONDON_PULL  = "S4_LONDON_PULL"
    S5_NY_COMPRESS  = "S5_NY_COMPRESS"

REQUIRED ADDITION to engines/regime_engine.py:
    Add get_adx_h4_slope() — see regime_engine_adx_slope.py

REQUIRED PATCH to engines/execution_engine.py:
    on_trade_opened() needs R3 branch — see execution_engine_r3_patch.py
    place_order() needs TP support — see execution_engine_r3_patch.py
"""

import pytz
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta

import config
from utils.logger import log_event, log_warning
from utils.mt5_client import get_mt5
from utils.session import get_current_session
from db.connection import execute_query, execute_write

# ── Imports from existing signal engine ──────────────────────────────────────
from engines.signal_engine import (
    _build_candidate,           # Private but used cross-module — this is intentional
    get_last_m5_bar,
    get_last_m15_bar,
    get_atr14_h1_rma,
    is_past_london_time_kill,
)
from engines.signal_engine import SignalType   # R3/S4/S5 must be added to this enum

from engines.regime_engine import (
    get_safe_regime,
    RegimeState,
    get_adx_h4_slope,           # Add this function to regime_engine.py (see patch file)
)
from engines.risk_engine import (
    calculate_lot_size,
    run_pre_trade_kill_switches,
    can_s1_family_fire,
)
from engines.data_engine import fetch_ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_ema20_m15_value() -> float | None:
    """EMA(20) on M15 bars. Used by S4 touch detection."""
    df = fetch_ohlcv("M15", count=30)
    if df is None or df.empty:
        return None
    df["ema20"] = ta.ema(df["close"], length=20)
    val = df["ema20"].iloc[-1]
    return float(val) if not pd.isna(val) else None


def _get_recent_high_event_db(min_delay_min: int, max_window_min: int) -> dict | None:
    """
    Returns most recent HIGH impact event that occurred between
    min_delay_min and max_window_min ago.
    Works even when released_flag = NULL (hardcoded fallback events).
    """
    rows = execute_query(
        """SELECT event_name, scheduled_utc
           FROM market_data.economic_events
           WHERE impact_level = 'HIGH'
             AND scheduled_utc <= NOW() AT TIME ZONE 'UTC'
                                 - INTERVAL '1 minute' * :min_delay
             AND scheduled_utc >= NOW() AT TIME ZONE 'UTC'
                                 - INTERVAL '1 minute' * :max_window
           ORDER BY scheduled_utc DESC
           LIMIT 1""",
        {"min_delay": min_delay_min, "max_window": max_window_min}
    )
    return rows[0] if rows else None


# ─────────────────────────────────────────────────────────────────────────────
# FAMILY GATES
# ─────────────────────────────────────────────────────────────────────────────

def _can_r3_fire(state: dict) -> tuple[bool, str]:
    """
    R3 — independent family. No trend_family_occupied check.
    Can coexist with an active S1, S2, S4, or S5 position.
    One R3 per day maximum.
    """
    if state.get("r3_fired_today"):
        return False, "R3_ALREADY_FIRED_TODAY"
    if not state.get("trading_enabled", True):
        return False, "TRADING_DISABLED"
    # R3 does NOT check trend_family_occupied — it's fully independent.
    # It DOES go through KS3/KS5/KS6/KS7 checks.
    permitted, reason = run_pre_trade_kill_switches(state)
    if not permitted:
        return False, reason
    return True, "PERMITTED"


def _can_s4_fire(state: dict) -> tuple[bool, str]:
    """
    S4 — trend family. Requires first EMA20 touch of London session.
    Reuses can_s1_family_fire for the trend family occupancy check.
    """
    if state.get("s4_fired_today"):
        return False, "S4_ALREADY_FIRED_TODAY"
    if not state.get("s4_ema_touched"):
        return False, "S4_EMA_NOT_TOUCHED_YET"
    # Trend family gate (same as S1)
    permitted, reason = can_s1_family_fire(state)
    if not permitted:
        return False, reason
    return True, "PERMITTED"


def _can_s5_fire(state: dict) -> tuple[bool, str]:
    """
    S5 — trend family. Requires London compression confirmed at noon.
    """
    if state.get("s5_fired_today"):
        return False, "S5_ALREADY_FIRED_TODAY"
    if not state.get("s5_compression_confirmed"):
        return False, "S5_NO_COMPRESSION_THIS_SESSION"
    # Trend family gate (same as S1)
    permitted, reason = can_s1_family_fire(state)
    if not permitted:
        return False, reason
    return True, "PERMITTED"


# ─────────────────────────────────────────────────────────────────────────────
# R3 — CALENDAR MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def arm_r3_if_ready(state: dict) -> None:
    """
    Called on every M5 close from m5_mgmt_job.

    Arms R3 when:
      1. A HIGH event occurred 5-35 min ago (delay cleared, window still open)
      2. r3_fired_today = False and r3_armed = False
      3. S8 not armed (S8 takes severity < 35 spikes; R3 takes severity >= 35)
      4. ks7_pre_event_price is stored (KS7 activated pre-event — always true
         for scheduled events since KS7 fires 45 min before)
      5. Current M5 close has moved enough from pre-event price to confirm direction

    Direction is LOCKED on first arm. If gates block entry on that bar,
    R3 retries on subsequent M5s within the 35-min window.
    """
    if state.get("r3_fired_today") or state.get("r3_armed"):
        return

    # Mutual exclusion: S8 has this spike if severity < R3_SEVERITY_THRESHOLD
    if state.get("s8_armed"):
        return

    event = _get_recent_high_event_db(
        min_delay_min=config.R3_MIN_DELAY_MIN,
        max_window_min=config.R3_ARMED_WINDOW_MIN,
    )
    if event is None:
        return

    pre_event_price = state.get("ks7_pre_event_price", 0.0)
    if pre_event_price <= 0.0:
        # KS7 didn't activate before this event (unscheduled? very short-notice?)
        # R3 cannot determine direction without a pre-event baseline.
        log_event("R3_ARM_SKIPPED_NO_PRE_EVENT_PRICE",
                  event=event.get("event_name", "UNKNOWN"))
        return

    bar = get_last_m5_bar()
    if bar is None:
        return

    current_close = bar["close"]
    price_diff = current_close - pre_event_price

    # Direction confirmation: require minimum move of R3_DIRECTION_MIN_MOVE_RATIO × H1 ATR
    # to avoid arming on noise when price barely moved after the event.
    atr_h1 = get_atr14_h1_rma()
    min_move = (atr_h1 * config.R3_DIRECTION_MIN_MOVE_RATIO) if atr_h1 else 2.0

    if abs(price_diff) < min_move:
        log_event("R3_DIRECTION_AMBIGUOUS",
                  diff=round(price_diff, 3),
                  min_required=round(min_move, 3),
                  event=event.get("event_name", "UNKNOWN"))
        return

    direction = "LONG" if price_diff > 0 else "SHORT"

    state["r3_armed"]               = True
    state["r3_arm_time"]            = datetime.now(pytz.utc)
    state["r3_direction"]           = direction
    state["r3_event_scheduled_utc"] = event["scheduled_utc"]

    log_event("R3_ARMED",
              event=event.get("event_name", "UNKNOWN"),
              direction=direction,
              pre_event_price=round(pre_event_price, 3),
              current_close=round(current_close, 3),
              price_diff=round(price_diff, 3))


def evaluate_r3_signal(state: dict) -> dict | None:
    """
    R3 entry evaluation. Called on M5 close when r3_armed = True.

    Entry:  Market order in r3_direction (first M5 close after delay confirms direction)
    Stop:   0.5 × H1 ATR from entry (no structural anchor post-event)
    TP:     0.75 × H1 ATR from entry → 1.5:1 RR
    Hold:   Max 30 min — checked separately in check_r3_hard_exit()
    Window: Expires R3_ARMED_WINDOW_MIN after the event
    """
    if not state.get("r3_armed"):
        return None

    # Check arming window hasn't expired
    arm_time = state.get("r3_arm_time")
    if arm_time is None:
        state["r3_armed"] = False
        return None

    now_utc = datetime.now(pytz.utc)
    elapsed_min = (now_utc - arm_time).total_seconds() / 60
    if elapsed_min > config.R3_ARMED_WINDOW_MIN:
        state["r3_armed"]     = False
        state["r3_arm_time"]  = None
        state["r3_direction"] = None
        log_event("R3_WINDOW_EXPIRED", elapsed_min=round(elapsed_min, 1))
        return None

    permitted, reason = _can_r3_fire(state)
    if not permitted:
        return None

    direction = state.get("r3_direction")
    if not direction:
        return None

    atr_h1 = get_atr14_h1_rma()
    if atr_h1 is None:
        log_warning("R3_NO_ATR_H1")
        return None

    # LOOP-7 FIX: Require post-event move to exceed minimum ATR threshold.
    # On minor data releases where price barely moves, R3 should not enter.
    pre_event_price = state.get("ks7_pre_event_price", 0.0)
    mt5  = get_mt5()
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        return None

    if pre_event_price > 0 and atr_h1 > 0:
        current_mid = (tick.ask + tick.bid) / 2
        move_from_pre = abs(current_mid - pre_event_price)
        min_move_threshold = atr_h1 * 0.3  # Require at least 0.3× H1 ATR move
        if move_from_pre < min_move_threshold:
            log_event("R3_MOVE_TOO_SMALL",
                      move=round(move_from_pre, 2),
                      threshold=round(min_move_threshold, 2),
                      pre_event_price=round(pre_event_price, 3))
            return None

    # Market entry at current bid/ask
    entry = round(tick.ask if direction == "LONG" else tick.bid, 3)

    stop_dist = atr_h1 * config.R3_STOP_ATR_MULT    # 0.5 × H1 ATR
    tp_dist   = atr_h1 * config.R3_TP_ATR_MULT      # 0.75 × H1 ATR

    if direction == "LONG":
        stop = round(entry - stop_dist, 3)
        tp   = round(entry + tp_dist,   3)
    else:
        stop = round(entry + stop_dist, 3)
        tp   = round(entry - tp_dist,   3)

    lot_size = calculate_lot_size(abs(entry - stop), state["size_multiplier"], state)
    if lot_size <= 0.0:
        log_event("R3_BLOCKED_CONDITIONS_TOO_POOR", reason="lot_size_zero")
        return None

    candidate = _build_candidate(
        signal_type = SignalType.R3_CAL_MOMENTUM,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        extra       = {
            "r3_tp_level":           tp,     # Used by execution_engine to set MT5 TP
            "r3_event_time":         str(state.get("r3_event_scheduled_utc", "")),
            "r3_pre_event_price":    state.get("ks7_pre_event_price", 0.0),
            "r3_armed_window_min":   round(elapsed_min, 1),
            # R3 does NOT set post_time_kill_reentry or failed_breakout_trade
            "post_time_kill_reentry": False,
            "failed_breakout_trade":  False,
        }
    )

    if candidate is None:
        return None

    # Disarm — do not re-enter R3 this session regardless of result
    state["r3_armed"]     = False

    log_event("R3_SIGNAL_GENERATED",
              direction=direction,
              entry=round(entry, 3),
              stop=round(stop, 3),
              tp=round(tp, 3),
              lots=lot_size,
              stop_dist=round(stop_dist, 2),
              elapsed_min=round(elapsed_min, 1))

    return candidate


def check_r3_closed_by_broker(state: dict) -> None:
    """
    Called every M5 close. Detects if R3's MT5 position was closed
    by the broker (SL hit, TP hit). Updates R3 state accordingly.

    R3 uses its own ticket (r3_open_ticket), separate from open_position,
    because R3 can coexist with a trend-family trade.
    """
    r3_ticket = state.get("r3_open_ticket")
    if not r3_ticket:
        return

    mt5 = get_mt5()
    open_tickets = {
        p.ticket
        for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
        if p.magic == config.MAGIC
    }

    if r3_ticket in open_tickets:
        return  # Still open — no action

    # Position is gone from MT5 — find exit price from deal history
    deals      = mt5.history_deals_get(position=r3_ticket) or []
    exit_price = None
    for d in deals:
        if d.entry == mt5.DEAL_ENTRY_OUT:
            exit_price = d.price
            break

    if exit_price is None:
        tick       = mt5.symbol_info_tick(config.SYMBOL)
        exit_price = tick.bid if tick else 0.0
        log_warning("R3_EXIT_PRICE_UNKNOWN_USING_BID", ticket=r3_ticket)

    _finalize_r3_close(r3_ticket, exit_price, "BROKER_CLOSE_SL_OR_TP", state)


def check_r3_hard_exit(state: dict) -> bool:
    """
    Returns True if R3's 30-minute hold limit has elapsed.
    Called every M5 close from m5_mgmt_job.
    """
    if not state.get("r3_open_ticket"):
        return False

    r3_open_time = state.get("r3_open_time")
    if r3_open_time is None:
        return False

    elapsed_min = (datetime.now(pytz.utc) - r3_open_time).total_seconds() / 60
    if elapsed_min >= config.R3_MAX_HOLD_MIN:
        log_event("R3_HARD_EXIT_TRIGGERED",
                  elapsed_min=round(elapsed_min, 1),
                  limit=config.R3_MAX_HOLD_MIN)
        return True
    return False


def execute_r3_hard_exit(state: dict) -> None:
    """
    Force-closes R3 position at market. Called when 30-min limit elapses.
    """
    r3_ticket = state.get("r3_open_ticket")
    if not r3_ticket:
        return

    mt5 = get_mt5()
    pos = next(
        (p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
         if p.ticket == r3_ticket and p.magic == config.MAGIC),
        None
    )

    if pos is None:
        # Already closed (SL/TP hit between this check and the broker close check)
        state["r3_open_ticket"] = None
        return

    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    result = mt5.order_send({
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    config.SYMBOL,
        "volume":    pos.volume,
        "type":      close_type,
        "position":  r3_ticket,
        "deviation": config.ORDER_DEVIATION_POINTS,
        "magic":     config.MAGIC,
        "comment":   "R3_30MIN_HARD_EXIT",
    })

    retcode = result.retcode if result else "NONE"
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        _finalize_r3_close(r3_ticket, result.price, "R3_30MIN_EXPIRY", state)
    else:
        log_warning("R3_HARD_EXIT_ORDER_FAILED",
                    ticket=r3_ticket, retcode=retcode)


def _finalize_r3_close(ticket: int, exit_price: float,
                        reason: str, state: dict) -> None:
    """Internal: calls on_trade_closed and clears R3 state."""
    from engines.execution_engine import on_trade_closed
    on_trade_closed(ticket, exit_price, reason, state)

    state["r3_open_ticket"] = None
    state["r3_open_time"]   = None
    state["r3_entry_price"] = 0.0
    state["r3_stop_price"]  = 0.0
    state["r3_tp_price"]    = 0.0

    log_event("R3_STATE_CLEARED", ticket=ticket, reason=reason)


# ─────────────────────────────────────────────────────────────────────────────
# S4 — LONDON PULLBACK
# ─────────────────────────────────────────────────────────────────────────────

def check_s4_ema_touch(state: dict) -> None:
    """
    Called on every M15 close during London session (07:00-12:00 UTC).

    Detects FIRST touch of EMA20 M15 during this session.
    Once touched, sets s4_ema_touched = True and records the M15 bar
    levels needed for stop calculation.

    Touch definition:
      LONG setup  (price > EMA20): M15 LOW enters within touch_zone of EMA20
      SHORT setup (price < EMA20): M15 HIGH enters within touch_zone of EMA20
      touch_zone = S4_TOUCH_ATR_FACTOR × H1 ATR (default 0.10)

    Uses the OPEN of the candle to confirm price was above/below EMA20
    at bar start — prevents touching from the wrong side counting.
    """
    if state.get("s4_ema_touched") or state.get("s4_fired_today"):
        return

    now_utc = datetime.now(pytz.utc)
    if not (config.S4_SESSION_START_HOUR_UTC <= now_utc.hour < config.S4_SESSION_END_HOUR_UTC):
        return

    bar = get_last_m15_bar()
    if bar is None:
        return

    ema20 = _get_ema20_m15_value()
    if ema20 is None:
        return

    atr_h1 = get_atr14_h1_rma()
    if atr_h1 is None:
        return

    touch_zone = atr_h1 * config.S4_TOUCH_ATR_FACTOR

    # LONG setup: price was above EMA20 at bar open → pulled back down to touch
    # Bar open must be above EMA20 (confirms the pullback is from above, not a cross)
    if bar["open"] > ema20 and bar["low"] <= ema20 + touch_zone:
        state["s4_ema_touched"]   = True
        state["s4_touch_bar_low"] = bar["low"]
        log_event("S4_EMA_TOUCH_LONG",
                  bar_open=round(bar["open"], 3),
                  bar_low=round(bar["low"], 3),
                  ema20=round(ema20, 3),
                  touch_zone=round(touch_zone, 2))

    # SHORT setup: price was below EMA20 at bar open → pulled back up to touch
    elif bar["open"] < ema20 and bar["high"] >= ema20 - touch_zone:
        state["s4_ema_touched"]    = True
        state["s4_touch_bar_high"] = bar["high"]
        log_event("S4_EMA_TOUCH_SHORT",
                  bar_open=round(bar["open"], 3),
                  bar_high=round(bar["high"], 3),
                  ema20=round(ema20, 3),
                  touch_zone=round(touch_zone, 2))


def evaluate_s4_signal(state: dict) -> dict | None:
    """
    S4 London Pullback — fires on the M15 bar AFTER the EMA20 touch is confirmed.

    Additional entry filter: ADX H4 > 20 AND increasing (last 2 closed H4 bars).
    Without an accelerating trend, London pullbacks frequently fail.

    Entry: LIMIT at EMA20 M15 (fill on return to EMA)
    Stop:  touch_bar_low  - 0.3 × H1 ATR  (LONG)
           touch_bar_high + 0.3 × H1 ATR  (SHORT)
    TP:    1.5 × stop distance from entry
    PM:    Same as S1 — BE at 1R, then ATR trail (handled by manage_open_position)
    Hard exit: 16:00 UTC (S4 thesis is London continuation, not NY)
    """
    if not state.get("s4_ema_touched") or state.get("s4_fired_today"):
        return None

    now_utc = datetime.now(pytz.utc)
    if not (config.S4_SESSION_START_HOUR_UTC <= now_utc.hour < config.S4_SESSION_END_HOUR_UTC):
        return None

    permitted, reason = _can_s4_fire(state)
    if not permitted:
        return None

    # ADX gate: trend must be present AND accelerating
    adx_current, adx_increasing = get_adx_h4_slope()
    if adx_current is None:
        return None
    if adx_current < config.S4_ADX_MIN_THRESHOLD:
        log_event("S4_ADX_TOO_LOW", adx=round(adx_current, 2),
                  threshold=config.S4_ADX_MIN_THRESHOLD)
        return None
    if not adx_increasing:
        log_event("S4_ADX_NOT_INCREASING", adx=round(adx_current, 2))
        return None

    ema20 = _get_ema20_m15_value()
    if ema20 is None:
        return None

    atr_h1 = get_atr14_h1_rma()
    if atr_h1 is None:
        return None

    stop_buffer = atr_h1 * config.S4_STOP_ATR_BUFFER

    # Determine direction from which bar made the touch
    touch_low  = state.get("s4_touch_bar_low",  0.0)
    touch_high = state.get("s4_touch_bar_high", 0.0)

    if touch_low > 0.0:
        direction = "LONG"
        entry     = round(ema20, 3)
        stop      = round(touch_low - stop_buffer, 3)
    elif touch_high > 0.0:
        direction = "SHORT"
        entry     = round(ema20, 3)
        stop      = round(touch_high + stop_buffer, 3)
    else:
        return None

    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return None

    tp = round(
        entry + stop_dist * config.S4_TP_RR_RATIO if direction == "LONG"
        else entry - stop_dist * config.S4_TP_RR_RATIO,
        3
    )

    lot_size = calculate_lot_size(stop_dist, state["size_multiplier"], state)
    if lot_size <= 0.0:
        log_event("S4_BLOCKED_CONDITIONS_TOO_POOR", reason="lot_size_zero")
        return None

    # B4 pattern: LIMIT order at EMA20, 15-min expiry (1 M15 candle)
    expiry_utc = datetime.now(pytz.utc) + timedelta(minutes=15)

    candidate = _build_candidate(
        signal_type = SignalType.S4_LONDON_PULL,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        extra       = {
            "order_expiry_utc":      expiry_utc.isoformat(),
            "s4_tp_level":           tp,
            "s4_adx_at_entry":       round(adx_current, 2),
            "s4_adx_increasing":     adx_increasing,
            "s4_ema20_m15":          round(ema20, 3),
            "s4_hard_exit_utc":      f"{config.S4_HARD_EXIT_HOUR_UTC:02d}:00 UTC",
            "post_time_kill_reentry": False,
            "failed_breakout_trade":  False,
        }
    )

    if candidate is None:
        return None

    log_event("S4_SIGNAL_GENERATED",
              direction=direction,
              entry=round(entry, 3),
              stop=round(stop, 3),
              tp=round(tp, 3),
              lots=lot_size,
              adx=round(adx_current, 2))

    return candidate


def check_s4_hard_exit(state: dict) -> bool:
    """
    S4 hard exit: 16:00 UTC. Returns True when it's time to force-close.
    ONLY fires when trend_family_strategy == 'S4_LONDON_PULL'.
    Caller in m15_dispatch_job must execute the close.
    """
    if state.get("trend_family_strategy") != SignalType.S4_LONDON_PULL.value:
        return False
    if not state.get("open_position"):
        return False

    now_utc = datetime.now(pytz.utc)
    if now_utc.hour >= config.S4_HARD_EXIT_HOUR_UTC:
        log_event("S4_HARD_EXIT_16UTC")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# S5 — NY COMPRESSION BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────

def update_london_session_tracking(state: dict) -> None:
    """
    Called on every M15 close during London session (07:00-12:00 UTC).

    Maintains running high/low of the London session for S5 compression check.
    Initialized on first bar of session (london_session_tracking_active = False).
    The tracking flag is cleared after the noon compression check runs.
    Reset at midnight by reset_daily_counters().
    """
    now_utc = datetime.now(pytz.utc)
    if not (config.S4_SESSION_START_HOUR_UTC <= now_utc.hour < config.S4_SESSION_END_HOUR_UTC):
        return

    bar = get_last_m15_bar()
    if bar is None:
        return

    if not state.get("london_session_tracking_active"):
        # First bar of today's London session — initialize
        state["london_session_high"]             = bar["high"]
        state["london_session_low"]              = bar["low"]
        state["london_session_tracking_active"]  = True
        log_event("LONDON_TRACKING_STARTED",
                  initial_high=round(bar["high"], 3),
                  initial_low=round(bar["low"], 3))
    else:
        # Update running extremes
        if bar["high"] > state["london_session_high"]:
            state["london_session_high"] = bar["high"]
        if bar["low"] < state["london_session_low"]:
            state["london_session_low"] = bar["low"]


def check_s5_compression_at_noon(state: dict) -> None:
    """
    Called ONCE per day at 12:00 UTC from m15_dispatch_job.

    Computes whether the London session range was compressed relative to
    the daily ATR14. Sets s5_compression_confirmed.

    london_session_tracking_active is set to False after this check —
    that flag serves double duty as "tracking done today."
    """
    if not state.get("london_session_tracking_active"):
        return  # Tracking never started today or already checked

    london_high  = state.get("london_session_high", 0.0)
    london_low   = state.get("london_session_low",  0.0)
    london_range = london_high - london_low

    d1_atr = state.get("d1_atr_14", 0.0)
    if d1_atr <= 0:
        log_warning("S5_NOON_CHECK_NO_D1_ATR",
                    note="d1_atr_14 should have been set at midnight")
        state["london_session_tracking_active"] = False
        return

    ratio = london_range / d1_atr if d1_atr > 0 else 1.0
    compression_confirmed = ratio < config.S5_COMPRESSION_RATIO

    state["s5_compression_confirmed"]       = compression_confirmed
    state["london_session_tracking_active"] = False  # Tracking complete

    log_event("S5_COMPRESSION_CHECKED",
              london_range=round(london_range, 2),
              d1_atr=round(d1_atr, 2),
              ratio=round(ratio, 3),
              confirmed=compression_confirmed,
              threshold=config.S5_COMPRESSION_RATIO)


def evaluate_s5_signal(state: dict) -> dict | None:
    """
    S5 NY Compression Breakout.

    Entry: M15 CLOSE outside London session range (not touch — close required).
    Window: 12:00-15:00 UTC. Entry after 15:00 UTC is not taken.

    Stop:  opposite London extreme + 0.3 × H1 ATR buffer
           (wide stop by design — lot sizing auto-scales down with wide stops)
    TP:    1.0 × London range distance from entry → R/R varies by entry location
    PM:    Same as S1/S4 — BE at 1R, ATR trail (manage_open_position handles this)
    Hard exit: 22:00 UTC
    """
    if not state.get("s5_compression_confirmed") or state.get("s5_fired_today"):
        return None

    now_utc = datetime.now(pytz.utc)
    if not (config.S5_ENTRY_START_HOUR_UTC <= now_utc.hour < config.S5_ENTRY_END_HOUR_UTC):
        return None

    permitted, reason = _can_s5_fire(state)
    if not permitted:
        return None

    bar = get_last_m15_bar()
    if bar is None:
        return None

    london_high  = state.get("london_session_high", 0.0)
    london_low   = state.get("london_session_low",  0.0)
    london_range = london_high - london_low

    if london_range <= 0 or london_low <= 0:
        return None

    atr_h1 = get_atr14_h1_rma()
    if atr_h1 is None:
        return None

    stop_buffer = atr_h1 * config.S5_STOP_ATR_BUFFER

    # Breakout detection: M15 CLOSE must clear the London range boundary
    direction = None
    if bar["close"] > london_high:
        direction = "LONG"
    elif bar["close"] < london_low:
        direction = "SHORT"
    else:
        return None  # No breakout on this bar

    # LOOP-8 FIX: Use STOP order beyond breakout level instead of M15 close.
    # By the time this signal is detected, the M15 close is historical.
    # Using a STOP order 2pts beyond the London boundary ensures we only enter
    # if price continues in the breakout direction.
    mt5_s5 = get_mt5()
    tick_s5 = mt5_s5.symbol_info_tick(config.SYMBOL)
    spread_price_s5 = (tick_s5.ask - tick_s5.bid) if tick_s5 else 0.0

    if direction == "LONG":
        entry = round(london_high + spread_price_s5 + 2.0, 3)  # BUY STOP 2pts above London high
        stop  = round(london_low - stop_buffer, 3)
        tp    = round(entry + london_range * config.S5_TP_RANGE_MULT, 3)
    else:
        entry = round(london_low - 2.0, 3)  # SELL STOP 2pts below London low
        stop  = round(london_high + stop_buffer, 3)
        tp    = round(entry - london_range * config.S5_TP_RANGE_MULT, 3)

    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return None

    lot_size = calculate_lot_size(stop_dist, state["size_multiplier"], state)
    if lot_size <= 0.0:
        log_event("S5_BLOCKED_CONDITIONS_TOO_POOR", reason="lot_size_zero")
        return None

    candidate = _build_candidate(
        signal_type = SignalType.S5_NY_COMPRESS,
        direction   = direction,
        entry_level = entry,
        stop_level  = stop,
        lot_size    = lot_size,
        state       = state,
        extra       = {
            "s5_tp_level":           tp,
            "s5_london_range":       round(london_range, 2),
            "s5_london_high":        round(london_high, 3),
            "s5_london_low":         round(london_low, 3),
            "s5_hard_exit_utc":      f"{config.S5_HARD_EXIT_HOUR_UTC:02d}:00 UTC",
            "post_time_kill_reentry": False,
            "failed_breakout_trade":  False,
        }
    )

    if candidate is None:
        return None

    log_event("S5_SIGNAL_GENERATED",
              direction=direction,
              entry=round(entry, 3),
              stop=round(stop, 3),
              tp=round(tp, 3),
              lots=lot_size,
              london_range=round(london_range, 2))

    return candidate


def check_s5_hard_exit(state: dict) -> bool:
    """
    S5 hard exit: 22:00 UTC. Returns True when it's time to force-close.
    ONLY fires when trend_family_strategy == 'S5_NY_COMPRESS'.
    Caller in m15_dispatch_job must execute the close.
    """
    if state.get("trend_family_strategy") != SignalType.S5_NY_COMPRESS.value:
        return False
    if not state.get("open_position"):
        return False

    now_utc = datetime.now(pytz.utc)
    if now_utc.hour >= config.S5_HARD_EXIT_HOUR_UTC:
        log_event("S5_HARD_EXIT_22UTC")
        return True
    return False
