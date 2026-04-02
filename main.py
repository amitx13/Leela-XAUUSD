"""
main.py — Entry point and APScheduler wiring for XAUUSD Algo.

All jobs: coalesce=True, max_instances=1 — spec law (Part 16).
All cron triggers use explicit timezone — DST-safe (v1.1).
Scheduler itself runs on UTC — never local time (v1.1 fix).

Scheduled jobs:
  1.  pre_london_range_job — 07:55 Europe/London (cron)
  2.  regime_job           — every 15 min
  3.  spread_logger_job    — every 5 min
  4.  tlt_macro_job        — 09:00 IST daily
  5.  calendar_job         — 00:01 IST daily (G1 Fix)
  6.  midnight_reset_job   — 00:00 IST daily (KS3 + S7 orders)
  7.  m15_dispatch_job     — every 1 min (M15 candle detection)
  8.  m5_mgmt_job          — every 1 min (M5 candle + fill detection)
  9.  portfolio_ks_job     — every 1 min (KS3/KS5/KS6 + correlation)
  10. dxy_update_job       — every 1 hour
  11. asian_range_job      — 05:30 UTC daily (S6 pending orders)

CLI:
  --live       Live trading (requires explicit flag)
  --paper      Paper mode — signals logged, no orders placed
  --checklist  Pre-session checklist only, then exit
  --weekly     Weekly Truth Engine review, then exit
"""

import sys
import time
import argparse
import pytz
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config
from state import build_initial_state, validate_state_keys
from utils.logger import log_event, log_warning, log_critical
from utils.alerts import send_ks_alert

from db.connection import test_db_connection, execute_query
from db.schema import get_config_value
from db.persistence import persist_critical_state, weekly_review

from engines.data_engine import (
    calculate_pre_london_range,
    log_spread,
    calculate_macro_bias,
    fetch_economic_calendar,
    fetch_ohlcv,
    get_daily_atr14
)
from engines.regime_engine import (
    regime_job as regime_engine_job,
    get_safe_regime,
    RegimeState,
    bootstrap_regime_from_history,
)
from engines.risk_engine import (
    check_ks3_daily_loss,
    check_ks5_weekly_loss,
    check_ks6_drawdown,
    decrement_ks4_countdown,
)
from engines.signal_engine import (
    evaluate_s1_signal,
    evaluate_s1b_signal,
    evaluate_s1d_reentry,
    evaluate_s1e_pyramid,
    evaluate_s1f_signal,
    evaluate_s2_signal,
    evaluate_s3_signal,
    evaluate_s6_signal,
    evaluate_s7_signal,
    detect_stop_hunt,
    auto_reset_s1b_counter,
    check_and_fire_london_time_kill,
    check_and_fire_ny_time_kill,
    manage_open_position,
)
from engines.execution_engine import (
    place_order,
    place_s1_pending_orders,
    place_s6_pending_orders,
    place_s7_pending_orders,
    on_trade_opened_from_pending_fill,
    cancel_all_pending_orders,
    emergency_shutdown,
    initialize_system,
    pre_session_checklist,
    on_trade_closed,
    modify_stop,
)
from state import reset_daily_counters   # BUG-1 FIX: use state.py version (resets Phase 2 + S8 daily flags)
from engines.portfolio_risk import check_portfolio_risk, run_correlation_check
from engines.starvation_tracker import StarvationTracker
from engines.truth_engine import daily_edge_check

from engines.signal_engine_phase2 import (
    # R3 — Calendar Momentum
    arm_r3_if_ready,
    evaluate_r3_signal,
    check_r3_closed_by_broker,
    check_r3_hard_exit,
    execute_r3_hard_exit,

    # S4 — London Pullback
    check_s4_ema_touch,
    evaluate_s4_signal,
    check_s4_hard_exit,

    # S5 — NY Compression Breakout
    update_london_session_tracking,
    check_s5_compression_at_noon,
    evaluate_s5_signal,
    check_s5_hard_exit,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

STATE              = build_initial_state()
PAPER_MODE         = False
STARVATION_TRACKER = StarvationTracker()

_last_m15_time: datetime | None = None
_last_m5_time:  datetime | None = None
_prev_session:  str | None      = None   # session boundary detection for starvation checks

# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    from utils.session import is_trading_hours
    return is_trading_hours()


def _safe_execute(job_name: str, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        log_warning(f"JOB_EXCEPTION_{job_name}", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# JOB 1 — PRE-LONDON RANGE (07:55 London local, G2 Fix)
# ─────────────────────────────────────────────────────────────────────────────

def pre_london_range_job() -> None:
    """
    Computes Asian/pre-London range from M15 candles 04:00–07:55 London.
    Immediately places spread-aware BUY STOP + SELL STOP at range boundaries.
    """
    log_event("JOB_START", job="pre_london_range")

    range_data = calculate_pre_london_range()
    if range_data:
        STATE["range_data"] = range_data
        log_event("PRE_LONDON_RANGE_SET",
                  range_high=round(range_data["range_high"], 3),
                  range_low=round(range_data["range_low"], 3),
                  range_size=round(range_data["range_size"], 2),
                  breakout_dist=round(range_data["breakout_dist"], 2))

        regime = get_safe_regime(STATE)
        if regime not in (RegimeState.NO_TRADE, RegimeState.UNSTABLE):
            if not PAPER_MODE:
                _safe_execute("s1_pending", place_s1_pending_orders, STATE)
            else:
                log_event("PAPER_MODE_S1_PENDING_SKIPPED",
                          buy_level=round(range_data["range_high"] + range_data["breakout_dist"], 2),
                          sell_level=round(range_data["range_low"] - range_data["breakout_dist"], 2))
        else:
            log_event("S1_PENDING_SKIPPED_REGIME_AT_RANGE_TIME", regime=regime.value)
    else:
        STATE["range_data"] = None
        log_warning("PRE_LONDON_RANGE_FAILED", note="S1 blocked until range computed")


# ─────────────────────────────────────────────────────────────────────────────
# JOB 2 — REGIME RECALCULATION (every 15 min)
# ─────────────────────────────────────────────────────────────────────────────

def regime_job() -> None:
    log_event("JOB_START", job="regime")
    validate_state_keys(STATE)
    regime_engine_job(STATE)
    log_event("REGIME_JOB_DONE",
              regime=STATE["current_regime"],
              size_mult=round(STATE["size_multiplier"], 3))


# ─────────────────────────────────────────────────────────────────────────────
# JOB 3 — SPREAD LOGGER (every 5 min)
# ─────────────────────────────────────────────────────────────────────────────

def spread_logger_job() -> None:
    if not _is_market_hours():
        return
    log_event("JOB_START", job="spread_logger")
    log_spread(STATE)


# ─────────────────────────────────────────────────────────────────────────────
# JOB 4 — TLT MACRO PULL (09:00 IST daily)
# ─────────────────────────────────────────────────────────────────────────────

def tlt_macro_job() -> None:
    log_event("JOB_START", job="tlt_macro")
    result = calculate_macro_bias()
    STATE["macro_bias"]             = result["macro_bias_label"]
    STATE["tlt_slope"]              = result["proxy_3d_slope"]
    STATE["macro_proxy_instrument"] = result["macro_proxy_instrument"]
    log_event("MACRO_BIAS_UPDATED",
              bias=STATE["macro_bias"],
              slope=round(STATE["tlt_slope"], 5),
              proxy=STATE["macro_proxy_instrument"])


# ─────────────────────────────────────────────────────────────────────────────
# JOB 5 — ECONOMIC CALENDAR (00:01 IST daily, G1 Fix)
# ─────────────────────────────────────────────────────────────────────────────

def calendar_job() -> None:
    log_event("JOB_START", job="calendar")
    count = fetch_economic_calendar()
    log_event("CALENDAR_REFRESHED", events_stored=count)


# ─────────────────────────────────────────────────────────────────────────────
# JOB 6 — MIDNIGHT RESET (00:00 IST daily)
# ─────────────────────────────────────────────────────────────────────────────

def midnight_reset_job() -> None:
    """
    C4 Fix: resets daily counters only.
    Rolling state (consecutive_losses, peak_equity) is NEVER reset by date.
    KS3 auto-resets here.
    v1.1: resets 5 new daily flags + places S7 pending orders (Sunday skip).
    """
    log_event("JOB_START", job="midnight_reset")

    # KS3 auto-reset
    if not STATE.get("trading_enabled") and STATE.get("shutdown_reason", "").startswith("KS3"):
        STATE["trading_enabled"] = True
        STATE["shutdown_reason"] = None
        log_event("KS3_AUTO_RESET_MIDNIGHT")

    reset_daily_counters(STATE)

    # Spread tracking reset for new session
    STATE["session_spread_initialized"] = False
    STATE["spread_fallback_active"]     = True

    # ── v1.1: daily flag resets ──────────────────────────────────────────────
    STATE["s1d_ema_touched_today"]    = False   # S1d first-touch guard
    STATE["s1d_fired_today"]          = False   # S1d fired guard
    STATE["s3_sweep_candle_time"]     = None    # S3 window anchor
    STATE["s3_sweep_low"]             = 0.0
    STATE["s3_sweep_high"]            = 0.0
    STATE["s3_fired_today"]           = False
    STATE["reversal_family_occupied"] = False   # S1b/S3 reversal family
    STATE["s1b_pending_ticket"]       = None    # safety clear on new day
    STATE["s6_fired_today"]           = False
    STATE["s7_fired_today"]           = False
    
    try:
        d1_atr = get_daily_atr14()
        if d1_atr is not None:
            STATE["d1_atr_14"] = d1_atr
            log_event("D1_ATR14_COMPUTED", value=round(d1_atr, 2))
        else:
            log_warning("D1_ATR14_UNAVAILABLE", note="S5 will skip compression today")
    except Exception as _e:
        log_warning("D1_ATR14_COMPUTE_ERROR", error=str(_e))

    # ── S7: Daily Structure pending orders (v1.1: Sunday skip) ──────────────
    if datetime.now(pytz.utc).weekday() == 6:   # 6 = Sunday, MT5 not open
        log_event("S7_SKIPPED_SUNDAY_PREMARKET")
    else:
        s7_result = evaluate_s7_signal(STATE)
        if s7_result:
            if not PAPER_MODE:
                _safe_execute("s7_place_orders", place_s7_pending_orders, STATE, s7_result)
            else:
                log_event("PAPER_MODE_S7_SKIPPED",
                          buy_entry=s7_result["buy_candidate"]["entry_level"],
                          sell_entry=s7_result["sell_candidate"]["entry_level"])

    # ── Phase 1A: Starvation daily summary + reset ──────────────────────────
    _safe_execute("starvation_daily", STARVATION_TRACKER.daily_summary)
    STARVATION_TRACKER.reset()

    # ── Phase 1A: Edge decay daily check ─────────────────────────────────────
    _safe_execute("edge_decay_daily", daily_edge_check)

    log_event("MIDNIGHT_RESET_COMPLETE",
              date=datetime.now(pytz.timezone("Asia/Kolkata")).date().isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# JOB 11 — S6 ASIAN RANGE (05:30 UTC daily)
# ─────────────────────────────────────────────────────────────────────────────

def asian_range_job() -> None:
    """
    S6: Computes 00:00–05:30 UTC Asian range and places BUY STOP + SELL STOP.
    Expiry: 08:00 UTC (London open auto-cancel).
    Min range 8 pts — skip logged inside evaluate_s6_signal().
    NO_TRADE regime check inside evaluate_s6_signal().
    Sunday skip — MT5 not open.
    """
    log_event("JOB_START", job="asian_range_s6")

    if datetime.now(pytz.utc).weekday() == 6:   # Sunday
        log_event("S6_SKIPPED_SUNDAY")
        return

    s6_result = evaluate_s6_signal(STATE)
    if s6_result is None:
        return

    if not PAPER_MODE:
        _safe_execute("s6_place_orders", place_s6_pending_orders, STATE, s6_result)
    else:
        log_event("PAPER_MODE_S6_SKIPPED",
                  buy_entry=s6_result["buy_candidate"]["entry_level"],
                  sell_entry=s6_result["sell_candidate"]["entry_level"])


# ─────────────────────────────────────────────────────────────────────────────
# JOB 7 — M15 SIGNAL DISPATCH (poll every 1 min, fire on new M15 close)
# ─────────────────────────────────────────────────────────────────────────────

def m15_dispatch_job() -> None:
    """
    Polls every 1 min. Executes signal logic only on new M15 candle close.

    Dispatch order:
      1.  Time kills — London 16:30, NY 13:00 (G7 Fix)
      2.  S1c stop hunt detection
      3.  S1b auto-reset candle counter (B3 Fix)
      4.  S1 London Breakout — pending order architecture
      5.  S1b Failed Breakout Reversal
      5a. S3 Stop Hunt Reversal
      6.  S2 Mean Reversion
      7.  S1f Post-Time-Kill re-entry
      ★8.  S4 London Pullback — EMA20 touch detection + signal evaluation
      ★9.  S5 London session tracking + noon compression check + signal evaluation
      ★10. Hard exits: S4 at 16:00 UTC, S5 at 22:00 UTC
    """
    global _last_m15_time, _prev_session

    if not STATE.get("trading_enabled"):
        return

    df = fetch_ohlcv("M15", count=3)
    if df is None or len(df) < 2:
        return

    bar_time = df["time"].iloc[-2]
    if bar_time == _last_m15_time:
        return
    _last_m15_time = bar_time

    log_event("M15_CANDLE_CLOSE", time=str(bar_time))
    validate_state_keys(STATE)

    # ── Phase 1A: Session boundary starvation check ──────────────────────────
    from utils.session import get_current_session as _get_sess
    _cur_session = _get_sess()
    if _prev_session is not None and _cur_session != _prev_session:
        _safe_execute("starvation_check", STARVATION_TRACKER.check_starvation, _prev_session)
    _prev_session = _cur_session

    # ── 1. Time kills ────────────────────────────────────────────────────────
    check_and_fire_london_time_kill(STATE)
    check_and_fire_ny_time_kill(STATE)

    if not STATE.get("trading_enabled"):
        return

    regime = get_safe_regime(STATE)

    # ── ★ S4/S5 Hard exits (checked every M15 close, before new signals) ─────
    # These must run regardless of trading_enabled for trend kills
    if check_s4_hard_exit(STATE):
        _safe_execute("s4_hard_exit", _execute_generic_market_close,
                      STATE.get("open_position"), "S4_HARD_EXIT_16UTC")

    if check_s5_hard_exit(STATE):
        _safe_execute("s5_hard_exit", _execute_generic_market_close,
                      STATE.get("open_position"), "S5_HARD_EXIT_22UTC")

    # ── 2. Stop hunt detection ────────────────────────────────────────────────
    if regime not in (RegimeState.NO_TRADE, RegimeState.UNSTABLE):
        _safe_execute("s1c_detect", detect_stop_hunt, STATE)

    # ── 3. S1b candle counter reset ───────────────────────────────────────────
    _safe_execute("s1b_reset", auto_reset_s1b_counter, STATE)

    # ── ★ S4: EMA20 touch detection (London session only, 07:00-12:00 UTC) ────
    _safe_execute("s4_touch_check", check_s4_ema_touch, STATE)

    # ── ★ S5: London session range tracking (07:00-12:00 UTC) ─────────────────
    _safe_execute("s5_tracking", update_london_session_tracking, STATE)

    # ── ★ S5: Noon compression check — triggers once at 12:00 UTC ─────────────
    import pytz as _pytz
    _now_utc = datetime.now(_pytz.utc)
    if _now_utc.hour == 12 and not STATE.get("s5_compression_confirmed"):
        _safe_execute("s5_noon_check", check_s5_compression_at_noon, STATE)

    # ── 4. S1: London Breakout — pending order architecture ──────────────────
    if not STATE.get("trend_family_occupied") and not STATE.get("open_position"):
        pending_buy  = STATE.get("s1_pending_buy_ticket")
        pending_sell = STATE.get("s1_pending_sell_ticket")

        if pending_buy or pending_sell:
            log_event("S1_PENDING_ACTIVE",
                      buy_ticket=pending_buy, sell_ticket=pending_sell)
        else:
            range_data = STATE.get("range_data")
            if range_data and regime not in (RegimeState.NO_TRADE, RegimeState.UNSTABLE):
                if not PAPER_MODE:
                    _safe_execute("s1_pending_late", place_s1_pending_orders, STATE)
                    log_event("S1_PENDING_PLACED_LATE", regime=regime.value)
                else:
                    STARVATION_TRACKER.record_evaluation()
                    candidate = evaluate_s1_signal(STATE)
                    if candidate:
                        STARVATION_TRACKER.record_signal()
                        _dispatch_candidate(candidate)
                        return

    # ── 5. S1b: Failed Breakout Reversal ─────────────────────────────────────
    if (STATE.get("failed_breakout_flag")
            and not STATE.get("trend_family_occupied")
            and not STATE.get("reversal_family_occupied")):
        STARVATION_TRACKER.record_evaluation()
        candidate = evaluate_s1b_signal(STATE)
        if candidate:
            STARVATION_TRACKER.record_signal()
            _dispatch_candidate(candidate)
            return

    # ── 5a. S3: Stop Hunt Reversal ────────────────────────────────────────────
    if (regime not in (RegimeState.NO_TRADE, RegimeState.UNSTABLE)
            and not STATE.get("trend_family_occupied")
            and not STATE.get("reversal_family_occupied")
            and not STATE.get("s3_fired_today")):
        STARVATION_TRACKER.record_evaluation()
        candidate = evaluate_s3_signal(STATE)
        if candidate:
            STARVATION_TRACKER.record_signal()
            STATE["s3_fired_today"]           = True
            STATE["reversal_family_occupied"] = True
            _dispatch_candidate(candidate)
            return

    # ── 6. S2: Mean Reversion ─────────────────────────────────────────────────
    if regime == RegimeState.RANGING_CLEAR and not STATE.get("trend_family_occupied"):
        STARVATION_TRACKER.record_evaluation()
        candidate = evaluate_s2_signal(STATE)
        if candidate:
            STARVATION_TRACKER.record_signal()
            _dispatch_candidate(candidate)
            return

    # ── 7. S1f: Post-Time-Kill re-entry ──────────────────────────────────────
    if STATE.get("london_tk_fired_today") and not STATE.get("trend_family_occupied"):
        STARVATION_TRACKER.record_evaluation()
        candidate = evaluate_s1f_signal(STATE)
        if candidate:
            STARVATION_TRACKER.record_signal()
            _dispatch_candidate(candidate)
            return

    # ── ★ 8. S4: London Pullback ──────────────────────────────────────────────
    # Fires only when EMA20 was touched this London session and trend family is free.
    # s4_fired_today gate is inside evaluate_s4_signal().
    if (STATE.get("s4_ema_touched")
            and not STATE.get("trend_family_occupied")
            and not STATE.get("s4_fired_today")):
        STARVATION_TRACKER.record_evaluation()
        candidate = evaluate_s4_signal(STATE)
        if candidate:
            STARVATION_TRACKER.record_signal()
            STATE["s4_fired_today"] = True
            _dispatch_candidate(candidate)
            return

    # ── ★ 9. S5: NY Compression Breakout ──────────────────────────────────────
    # Fires only when London session was compressed and trend family is free.
    # s5_fired_today gate is inside evaluate_s5_signal().
    if (STATE.get("s5_compression_confirmed")
            and not STATE.get("trend_family_occupied")
            and not STATE.get("s5_fired_today")):
        STARVATION_TRACKER.record_evaluation()
        candidate = evaluate_s5_signal(STATE)
        if candidate:
            STARVATION_TRACKER.record_signal()
            STATE["s5_fired_today"] = True
            _dispatch_candidate(candidate)
            return


# ─────────────────────────────────────────────────────────────────────────────
# JOB 8 — M5 POSITION MANAGEMENT (poll every 1 min, fire on new M5 close)
# ─────────────────────────────────────────────────────────────────────────────

def m5_mgmt_job() -> None:
    """
    Polls every 1 min. Fires on new M5 candle close.

    Order:
      1. Detect broker-closed positions (main trend family)
      2. Detect S1 pending fills
      3. Detect S6 pending fills
      4. Detect S7 pending fills
      ★5. R3 arm detection (checks for recent HIGH events)
      ★6. R3 broker-close detection (SL/TP hit on R3 independent ticket)
      ★7. R3 hard exit (30-min hold limit)
      ★8. R3 signal evaluation (when armed + conditions pass)
      9. Main position management (S1/S2/S4/S5 trend family)
      10. Addon signals (S1d re-entry, S1e pyramid)
    """
    global _last_m5_time

    if not STATE.get("trading_enabled"):
        return

    df = fetch_ohlcv("M5", count=3)
    if df is None or len(df) < 2:
        return

    bar_time = df["time"].iloc[-2]
    if bar_time == _last_m5_time:
        return
    _last_m5_time = bar_time

    # ── 1-4. Existing fill detection (unchanged) ─────────────────────────────
    _check_for_closed_positions()
    _check_for_s1_pending_fill()
    _check_for_s6_pending_fill()
    _check_for_s7_pending_fill()

    # ── ★ 5. R3: Attempt to arm on this M5 close ─────────────────────────────
    if not STATE.get("r3_fired_today"):
        _safe_execute("r3_arm", arm_r3_if_ready, STATE)

    # ── ★ 6. R3: Check if broker closed the R3 position (SL or TP hit) ────────
    if STATE.get("r3_open_ticket"):
        _safe_execute("r3_close_check", check_r3_closed_by_broker, STATE)

    # ── ★ 7. R3: Hard exit check (30-min hold limit) ──────────────────────────
    if STATE.get("r3_open_ticket"):
        if check_r3_hard_exit(STATE):
            _safe_execute("r3_hard_exit", execute_r3_hard_exit, STATE)

    # ── ★ 8. R3: Signal evaluation (fires market order if all gates pass) ──────
    if STATE.get("r3_armed") and not STATE.get("r3_open_ticket"):
        candidate = evaluate_r3_signal(STATE)
        if candidate:
            if not PAPER_MODE:
                _dispatch_candidate(candidate)
            else:
                log_event("PAPER_MODE_R3_SKIPPED",
                          direction=candidate.get("direction"),
                          entry=candidate.get("entry_level"))

    # ── 9. Main trend family position management (unchanged) ─────────────────
    if not STATE.get("open_position"):
        if STATE.get("trend_family_occupied"):
            _evaluate_addon_signals()
        return

    validate_state_keys(STATE)
    action_dict = manage_open_position(STATE)
    action      = action_dict.get("action", "NONE")

    if action == "NONE":
        if STATE.get("position_be_activated"):
            _safe_execute("atr_trail", _update_atr_trail)
        return

    if action == "PARTIAL_EXIT":
        _safe_execute("partial_exit", _execute_partial_exit)
    elif action == "ACTIVATE_BE":
        _safe_execute("be_activation", _execute_be_activation)
    elif action == "CYCLE_EXIT":
        _safe_execute("cycle_exit", _execute_cycle_exit)
        _safe_execute("s1d_reentry", _evaluate_addon_signals)
    elif action == "S2_FORCE_EXIT":
        _safe_execute("s2_exit", _execute_s2_force_exit)

# ─────────────────────────────────────────────────────────────────────────────
# JOB 9 — PORTFOLIO KILL SWITCH MONITOR (every 1 min)
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_ks_job() -> None:
    if not STATE.get("trading_enabled"):
        return

    # ── KS3 daily loss ────────────────────────────────────────────────────────
    ks3_ok, ks3_reason = check_ks3_daily_loss(STATE)
    if not ks3_ok:
        cancel_all_pending_orders()
        log_event("KS3_FIRED_CANCEL_SENT", reason=ks3_reason)
        return

    # ── KS5 weekly loss ───────────────────────────────────────────────────────
    ks5_ok, ks5_reason = check_ks5_weekly_loss(STATE)
    if not ks5_ok:
        cancel_all_pending_orders()
        log_event("KS5_FIRED_CANCEL_SENT", reason=ks5_reason)
        return

    # ── KS6 drawdown from 30-day peak ────────────────────────────────────────
    ks6_ok, ks6_reason = check_ks6_drawdown(STATE)
    if not ks6_ok:
        emergency_shutdown(f"KS6_DRAWDOWN_{ks6_reason}", STATE)
        log_critical("KS6_FIRED", reason=ks6_reason)
        return

    # ── Section 3.4: P&L Correlation check (every N closed trades) ───────────
    try:
        rows = execute_query(
            "SELECT COUNT(*) as cnt FROM system_state.trades WHERE exit_time IS NOT NULL",
            {}
        )
        if rows:
            n = int(rows[0]["cnt"] or 0)
            if n > 0 and n % config.PORTFOLIO_CORR_CHECK_EVERY_N == 0:
                _safe_execute("corr_check", run_correlation_check, STATE)
    except Exception as e:
        log_warning("PORTFOLIO_CORR_COUNT_FAILED", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# JOB 10 — DXY UPDATE (every 1 hour)
# ─────────────────────────────────────────────────────────────────────────────

def dxy_update_job() -> None:
    from engines.data_engine import calculate_dxy_correlation, calculate_dxy_ewma_variance
    log_event("JOB_START", job="dxy_update")

    # Calculate DXY correlation
    corr = calculate_dxy_correlation(lookback=config.DXY_CORR_LOOKBACK)
    STATE["dxy_corr_50"] = corr
    log_event("DXY_CORRELATION_UPDATED", corr=round(corr, 4))

    # F4: Calculate DXY EWMA variance for stability check
    variance = calculate_dxy_ewma_variance(lookback=20)
    STATE["dxy_ewma_variance"] = variance


# ─────────────────────────────────────────────────────────────────────────────
# FILL DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _check_for_closed_positions() -> None:
    """Detects broker-closed positions (stop hit, TP, manual close)."""
    ticket = STATE.get("open_position")
    if not ticket:
        return

    from utils.mt5_client import get_mt5
    mt5 = get_mt5()
    open_tickets = {p.ticket for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
                    if p.magic == config.MAGIC}

    if ticket not in open_tickets:
        deals = mt5.history_deals_get(position=ticket)
        exit_price = None
        if deals:
            close_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
            if close_deals:
                exit_price = close_deals[-1].price
        if exit_price is None:
            log_warning("CLOSED_DEAL_PRICE_UNKNOWN", ticket=ticket)
            exit_price = mt5.symbol_info_tick(config.SYMBOL).bid

        on_trade_closed(ticket, exit_price, "BROKER_CLOSE_STOP_OR_TP", STATE)
        log_event("POSITION_CLOSED_BY_BROKER",
                  ticket=ticket, exit_price=round(exit_price, 3))


def _check_for_s1_pending_fill() -> None:
    """
    Detects S1 BUY/SELL STOP fills and S1b pending fill (v1.1).
    On S1 fill: writes DB row, cancels opposite leg, clears both slots.
    On S1b fill: writes DB row, sets reversal_family_occupied = True.
    C5: magic filter enforced throughout.
    """
    from utils.mt5_client import get_mt5
    mt5 = get_mt5()

    buy_ticket  = STATE.get("s1_pending_buy_ticket")
    sell_ticket = STATE.get("s1_pending_sell_ticket")
    s1b_ticket  = STATE.get("s1b_pending_ticket")

    if not buy_ticket and not sell_ticket and not s1b_ticket:
        return

    open_positions = {
        p.ticket: p
        for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
        if p.magic == config.MAGIC
    }
    pending_orders = {
        o.ticket
        for o in (mt5.orders_get(symbol=config.SYMBOL) or [])
        if o.magic == config.MAGIC
    }

    # ── S1 BUY/SELL STOP fills ───────────────────────────────────────────────
    for ticket, direction, opposite_key, own_key in [
        (buy_ticket,  "LONG",  "s1_pending_sell_ticket", "s1_pending_buy_ticket"),
        (sell_ticket, "SHORT", "s1_pending_buy_ticket",  "s1_pending_sell_ticket"),
    ]:
        if not ticket:
            continue
        if ticket in pending_orders:
            continue    # still resting at broker
        if ticket in open_positions:
            pos = open_positions[ticket]
            log_event("S1_PENDING_FILLED",
                      ticket=ticket, direction=direction,
                      price=round(pos.price_open, 3),
                      lots=pos.volume, sl=round(pos.sl, 3))
            STARVATION_TRACKER.record_fill()
            on_trade_opened_from_pending_fill(
                ticket=ticket, pos_price=pos.price_open,
                pos_volume=pos.volume, pos_sl=pos.sl,
                direction=direction, state=STATE,
                signal_type="S1_LONDON_BRK",
            )
            opp_ticket = STATE.get(opposite_key)
            if opp_ticket and opp_ticket in pending_orders:
                result  = mt5.order_delete(opp_ticket)
                retcode = result.retcode if result else "NONE"
                log_event("S1_PENDING_OPPOSITE_CANCELLED",
                          ticket=opp_ticket, retcode=retcode)
            STATE[own_key]      = None
            STATE[opposite_key] = None
            return
        # Gone from pending, not in positions → expired or time-killed
        log_event("S1_PENDING_EXPIRED_OR_CANCELLED",
                  ticket=ticket, direction=direction)
        STATE[own_key] = None

    # ── v1.1: S1b pending STOP fill ──────────────────────────────────────────
    if s1b_ticket:
        if s1b_ticket in pending_orders:
            pass    # still resting — nothing to do
        elif s1b_ticket in open_positions:
            pos       = open_positions[s1b_ticket]
            direction = "LONG" if pos.type == 0 else "SHORT"
            log_event("S1B_PENDING_FILLED",
                      ticket=s1b_ticket, direction=direction,
                      price=round(pos.price_open, 3),
                      lots=pos.volume, sl=round(pos.sl, 3))
            STARVATION_TRACKER.record_fill()
            on_trade_opened_from_pending_fill(
                ticket=s1b_ticket, pos_price=pos.price_open,
                pos_volume=pos.volume, pos_sl=pos.sl,
                direction=direction, state=STATE,
                signal_type="S1B_FAILED_BRK",
            )
            STATE["reversal_family_occupied"] = True   # blocks S3 rest of day
            STATE["s1b_pending_ticket"]       = None
        else:
            log_event("S1B_PENDING_EXPIRED_OR_CANCELLED", ticket=s1b_ticket)
            STATE["s1b_pending_ticket"] = None


def _check_for_s6_pending_fill() -> None:
    """v1.1: Detects S6 BUY or SELL STOP fill. Cancels opposite leg on fill."""
    from utils.mt5_client import get_mt5
    mt5 = get_mt5()

    for side, key, opp_key in (
        ("BUY",  "s6_pending_buy_ticket",  "s6_pending_sell_ticket"),
        ("SELL", "s6_pending_sell_ticket", "s6_pending_buy_ticket"),
    ):
        ticket = STATE.get(key)
        if not ticket:
            continue

        pos = next(
            (p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
             if p.ticket == ticket and p.magic == config.MAGIC),
            None
        )
        if pos:
            direction = "LONG" if pos.type == 0 else "SHORT"
            log_event("S6_PENDING_FILLED",
                      ticket=ticket, direction=direction,
                      price=round(pos.price_open, 3), lots=pos.volume)
            STARVATION_TRACKER.record_fill()
            on_trade_opened_from_pending_fill(
                ticket=ticket, pos_price=pos.price_open,
                pos_volume=pos.volume, pos_sl=pos.sl,
                direction=direction, state=STATE,
                signal_type="S6_ASIAN_BRK",   # BUG-3 FIX: was S6_ASIAN_RNG
            )
            STATE[key] = None
            opp_ticket = STATE.get(opp_key)
            if opp_ticket:
                mt5.order_delete(opp_ticket)
                STATE[opp_key] = None
                log_event("S6_OPPOSITE_LEG_CANCELLED",
                          cancelled_ticket=opp_ticket, filled_side=side)
            STATE["s6_fired_today"] = True
            return

        # Check if expired (not in positions, not in pending)
        pending_orders = {o.ticket for o in (mt5.orders_get(symbol=config.SYMBOL) or [])
                         if o.magic == config.MAGIC}
        if ticket not in pending_orders:
            log_event("S6_PENDING_EXPIRED_OR_CANCELLED", ticket=ticket, side=side)
            STATE[key] = None


def _check_for_s7_pending_fill() -> None:
    """v1.1: Detects S7 BUY or SELL STOP fill. Cancels opposite leg on fill."""
    from utils.mt5_client import get_mt5
    mt5 = get_mt5()

    for side, key, opp_key in (
        ("BUY",  "s7_pending_buy_ticket",  "s7_pending_sell_ticket"),
        ("SELL", "s7_pending_sell_ticket", "s7_pending_buy_ticket"),
    ):
        ticket = STATE.get(key)
        if not ticket:
            continue

        pos = next(
            (p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
             if p.ticket == ticket and p.magic == config.MAGIC),
            None
        )
        if pos:
            direction = "LONG" if pos.type == 0 else "SHORT"
            log_event("S7_PENDING_FILLED",
                      ticket=ticket, direction=direction,
                      price=round(pos.price_open, 3), lots=pos.volume)
            STARVATION_TRACKER.record_fill()
            on_trade_opened_from_pending_fill(
                ticket=ticket, pos_price=pos.price_open,
                pos_volume=pos.volume, pos_sl=pos.sl,
                direction=direction, state=STATE,
                signal_type="S7_DAILY_STRUCT",
            )
            STATE[key] = None
            opp_ticket = STATE.get(opp_key)
            if opp_ticket:
                mt5.order_delete(opp_ticket)
                STATE[opp_key] = None
                log_event("S7_OPPOSITE_LEG_CANCELLED",
                          cancelled_ticket=opp_ticket, filled_side=side)
            STATE["s7_fired_today"] = True
            return

        pending_orders = {o.ticket for o in (mt5.orders_get(symbol=config.SYMBOL) or [])
                         if o.magic == config.MAGIC}
        if ticket not in pending_orders:
            log_event("S7_PENDING_EXPIRED_OR_CANCELLED", ticket=ticket, side=side)
            STATE[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MANAGEMENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_addon_signals() -> None:
    candidate = evaluate_s1d_reentry(STATE)
    if candidate:
        _dispatch_candidate(candidate)
        return
    candidate = evaluate_s1e_pyramid(STATE)
    if candidate:
        _dispatch_candidate(candidate)


def _execute_partial_exit() -> None:
    ticket = STATE.get("open_position")
    if not ticket:
        return

    from utils.mt5_client import get_mt5
    mt5 = get_mt5()
    pos = next((p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
                if p.ticket == ticket and p.magic == config.MAGIC), None)
    if not pos:
        return

    half_lots  = max(config.CONTRACT_SPEC.get("volume_min", 0.01), round(pos.volume / 2, 2))
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY

    result = mt5.order_send({
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    config.SYMBOL,
        "volume":    half_lots,
        "type":      close_type,
        "position":  ticket,
        "deviation": 10,
        "magic":     config.MAGIC,
        "comment":   "PARTIAL_1R",
    })

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        STATE["position_partial_done"] = True
        from db.connection import execute_write
        execute_write(
            "UPDATE system_state.trades SET partial_exit_done = TRUE "
            "WHERE mt5_ticket = :ticket AND exit_time IS NULL",
            {"ticket": ticket}
        )
        persist_critical_state(STATE)
        log_event("PARTIAL_EXIT_DONE",
                  ticket=ticket, lots=half_lots, price=round(result.price, 3))
    else:
        log_warning("PARTIAL_EXIT_FAILED", ticket=ticket,
                    retcode=result.retcode if result else "NONE")


def _execute_be_activation() -> None:
    ticket   = STATE.get("open_position")
    entry_px = STATE.get("entry_price", 0.0)
    if not ticket or not entry_px:
        return

    success = modify_stop(ticket, entry_px, "BE_ACTIVATED", STATE)
    if success:
        STATE["position_be_activated"] = True
        from db.connection import execute_write
        execute_write(
            "UPDATE system_state.trades SET be_activated = TRUE "
            "WHERE mt5_ticket = :t AND exit_time IS NULL",
            {"t": ticket}
        )
        persist_critical_state(STATE)


def _execute_cycle_exit() -> None:
    ticket = STATE.get("open_position")
    if not ticket:
        return

    from utils.mt5_client import get_mt5
    mt5 = get_mt5()
    pos = next((p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
                if p.ticket == ticket and p.magic == config.MAGIC), None)
    if not pos:
        return

    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    result = mt5.order_send({
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    config.SYMBOL,
        "volume":    pos.volume,
        "type":      close_type,
        "position":  ticket,
        "deviation": 10,
        "magic":     config.MAGIC,
        "comment":   "CYCLE_EXIT_EMA",
    })

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        on_trade_closed(ticket, result.price, "MOMENTUM_CYCLE_EMA_BREAK", STATE)
        log_event("CYCLE_EXIT_DONE", ticket=ticket, price=round(result.price, 3))
    else:
        log_warning("CYCLE_EXIT_FAILED", ticket=ticket,
                    retcode=result.retcode if result else "NONE")


def _execute_s2_force_exit() -> None:
    ticket = STATE.get("open_position")
    if not ticket:
        return

    from utils.mt5_client import get_mt5
    mt5 = get_mt5()
    pos = next((p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
                if p.ticket == ticket and p.magic == config.MAGIC), None)
    if not pos:
        return

    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    result = mt5.order_send({
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    config.SYMBOL,
        "volume":    pos.volume,
        "type":      close_type,
        "position":  ticket,
        "deviation": 10,
        "magic":     config.MAGIC,
        "comment":   "S2_REGIME_EXIT",
    })

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        on_trade_closed(ticket, result.price, "S2_REGIME_CHANGE_EXIT", STATE)
        log_event("S2_FORCE_EXIT_DONE", ticket=ticket, price=round(result.price, 3))
    else:
        log_warning("S2_FORCE_EXIT_FAILED", ticket=ticket,
                    retcode=result.retcode if result else "NONE")


def _update_atr_trail() -> None:
    ticket    = STATE.get("open_position")
    direction = STATE.get("last_s1_direction")
    if not ticket or not direction:
        return

    from engines.risk_engine import calculate_atr_trail
    from utils.mt5_client import get_mt5
    mt5 = get_mt5()
    pos = next((p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
                if p.ticket == ticket and p.magic == config.MAGIC), None)
    if not pos:
        return

    new_trail    = calculate_atr_trail(pos.price_current, direction)
    current_stop = STATE.get("stop_price_current", 0.0)
    if new_trail is None:
        return

    if direction == "LONG"  and new_trail > current_stop:
        modify_stop(ticket, new_trail, "ATR_TRAIL_UPDATED", STATE)
    elif direction == "SHORT" and new_trail < current_stop:
        modify_stop(ticket, new_trail, "ATR_TRAIL_UPDATED", STATE)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DISPATCH — final gate before execution engine
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_candidate(candidate: dict) -> None:
    """
    PAPER MODE: log only, no order.
    LIVE MODE (3 gates in sequence):
      1. Phase gate  — blocks Phase 2 strategies until PHASE config >= 2
      2. Portfolio Risk Brain — VAR, correlation, session cap
      3. place_order() — execution engine handles KS2, spread, C5, C6
    v1.1: decrements KS4 countdown on every successful placement.
    """
    validate_state_keys(STATE)

    if PAPER_MODE:
        log_event("PAPER_MODE_CANDIDATE",
                  signal=candidate["signal_type"],
                  direction=candidate["direction"],
                  entry=candidate["entry_level"],
                  stop=candidate["stop_level"],
                  lots=candidate["lot_size"])
        return

    # ── 1. Phase gate ─────────────────────────────────────────────────────────
    PHASE2_STRATEGIES = {"S4_NY_CONT", "S5_NY_CONT", "R3_CAL_MOM"}
    current_phase = int(get_config_value("PHASE") or 0)
    if candidate["signal_type"] in PHASE2_STRATEGIES and current_phase < 2:
        STARVATION_TRACKER.record_block(candidate["signal_type"], "compound_gate", "phase<2")
        log_event("PHASE_GATE_BLOCKED", signal=candidate["signal_type"])
        return

    # ── 2. Portfolio Risk Brain ───────────────────────────────────────────────
    permitted, reason = check_portfolio_risk(candidate, STATE)
    if not permitted:
        STARVATION_TRACKER.record_block(candidate["signal_type"], "portfolio", reason)
        log_event("PORTFOLIO_RISK_BLOCKED",
                  signal=candidate["signal_type"], reason=reason)
        return

    log_event("DISPATCHING_CANDIDATE",
              signal=candidate["signal_type"],
              direction=candidate["direction"],
              entry=candidate["entry_level"],
              lots=candidate["lot_size"])

    # ── 3. Execute ────────────────────────────────────────────────────────────
    ticket = place_order(candidate, STATE)

    if ticket:
        STARVATION_TRACKER.record_order()

    # v1.1: KS4 countdown — decrement on every successful placement
    if ticket and STATE.get("ks4_reduced_trades_remaining", 0) > 0:
        decrement_ks4_countdown(STATE)


def _execute_generic_market_close(ticket: int | None, reason: str) -> None:
    """
    Force-closes a position at market. Used for S4/S5 hard session exits.
    Delegates to on_trade_closed for DB update and state cleanup.
    """
    if not ticket:
        return

    from utils.mt5_client import get_mt5
    mt5 = get_mt5()
    pos = next((p for p in (mt5.positions_get(symbol=config.SYMBOL) or [])
                if p.ticket == ticket and p.magic == config.MAGIC), None)
    if not pos:
        return

    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    result = mt5.order_send({
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    config.SYMBOL,
        "volume":    pos.volume,
        "type":      close_type,
        "position":  ticket,
        "deviation": config.ORDER_DEVIATION_POINTS,
        "magic":     config.MAGIC,
        "comment":   reason[:31],
    })

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        from engines.execution_engine import on_trade_closed
        on_trade_closed(ticket, result.price, reason, STATE)
        log_event("GENERIC_MARKET_CLOSE_DONE",
                  ticket=ticket, reason=reason, price=round(result.price, 3))
    else:
        retcode = result.retcode if result else "NONE"
        log_warning("GENERIC_MARKET_CLOSE_FAILED",
                    ticket=ticket, reason=reason, retcode=retcode)
        
# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler() -> BackgroundScheduler:
    """
    All jobs: coalesce=True, max_instances=1.
    v1.1: scheduler timezone = UTC (not IST) — DST-proof.
    Cron triggers that need local time pass explicit tz= to CronTrigger.
    """
    tz_utc    = pytz.utc
    tz_ist    = pytz.timezone("Asia/Kolkata")
    tz_london = pytz.timezone("Europe/London")

    scheduler = BackgroundScheduler(timezone=tz_utc)   # v1.1: UTC base

    # Job 1: Pre-London range — 07:55 London local
    scheduler.add_job(
        func=lambda: _safe_execute("pre_london_range", pre_london_range_job),
        trigger=CronTrigger(hour=7, minute=55, timezone=tz_london),
        id="pre_london_range",
        name="Pre-London Range G2 Fix",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 2: Regime recalculation — every 15 min (fires immediately on start)
    scheduler.add_job(
        func=lambda: _safe_execute("regime", regime_job),
        trigger=IntervalTrigger(minutes=config.REGIME_JOB_INTERVAL_MIN),
        id="regime",
        name="Regime Recalculation",
        coalesce=True, max_instances=1, replace_existing=True,
        next_run_time=datetime.now(pytz.utc),
    )

    # Job 3: Spread logger — every 5 min (fires immediately on start)
    scheduler.add_job(
        func=lambda: _safe_execute("spread", spread_logger_job),
        trigger=IntervalTrigger(minutes=5),
        id="spread_logger",
        name="Spread Logger C1 Fix",
        coalesce=True, max_instances=1, replace_existing=True,
        next_run_time=datetime.now(pytz.utc),
    )

    # Job 4: TLT macro pull — 09:00 IST daily
    scheduler.add_job(
        func=lambda: _safe_execute("tlt", tlt_macro_job),
        trigger=CronTrigger(hour=9, minute=0, timezone=tz_ist),
        id="tlt_macro",
        name="TLT Macro Bias Daily",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 5: Economic calendar — 00:01 IST daily (G1 Fix: daily not weekly)
    scheduler.add_job(
        func=lambda: _safe_execute("calendar", calendar_job),
        trigger=CronTrigger(hour=0, minute=1, timezone=tz_ist),
        id="calendar",
        name="Economic Calendar G1 Fix Daily",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 6: Midnight reset — 00:00 IST (KS3 + v1.1 daily flags + S7 orders)
    scheduler.add_job(
        func=lambda: _safe_execute("midnight_reset", midnight_reset_job),
        trigger=CronTrigger(hour=0, minute=0, timezone=tz_ist),
        id="midnight_reset",
        name="Midnight IST Reset KS3+Counters+S7",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 7: M15 signal dispatch — every 1 min (M15 candle close detection)
    scheduler.add_job(
        func=lambda: _safe_execute("m15_dispatch", m15_dispatch_job),
        trigger=IntervalTrigger(seconds=60),
        id="m15_dispatch",
        name="M15 Signal Dispatch",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 8: M5 position management — every 1 min (M5 candle + fill detection)
    scheduler.add_job(
        func=lambda: _safe_execute("m5_mgmt", m5_mgmt_job),
        trigger=IntervalTrigger(seconds=60),
        id="m5_mgmt",
        name="M5 Position Management",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 9: Portfolio KS monitor — every 1 min (KS3/KS5/KS6 + correlation)
    scheduler.add_job(
        func=lambda: _safe_execute("portfolio_ks", portfolio_ks_job),
        trigger=IntervalTrigger(seconds=60),
        id="portfolio_ks",
        name="Portfolio Kill Switch Monitor",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 10: DXY correlation — every 1 hour
    scheduler.add_job(
        func=lambda: _safe_execute("dxy_update", dxy_update_job),
        trigger=IntervalTrigger(hours=1),
        id="dxy_update",
        name="DXY Correlation Hourly Update",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    # Job 11: S6 Asian Range — 05:30 UTC daily (v1.1)
    scheduler.add_job(
        func=lambda: _safe_execute("s6_asian_range", asian_range_job),
        trigger=CronTrigger(hour=5, minute=30, timezone=tz_utc),
        id="s6_asian_range",
        name="Asian Range S6 Orders",
        coalesce=True, max_instances=1, replace_existing=True,
    )

    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="XAUUSD Algo — Leela Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --checklist  Run pre-session checklist only\n"
            "  python main.py --paper      Paper trading (log signals, no orders)\n"
            "  python main.py --live       Live trading (requires explicit flag)\n"
            "  python main.py --weekly     Run Truth Engine weekly review\n"
        )
    )
    parser.add_argument("--live",      action="store_true", help="Enable live order placement")
    parser.add_argument("--paper",     action="store_true", help="Paper mode: log signals, no orders")
    parser.add_argument("--checklist", action="store_true", help="Run pre-session checklist and exit")
    parser.add_argument("--weekly",    action="store_true", help="Run weekly Truth Engine review and exit")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global PAPER_MODE, STATE

    args = parse_args()

    # ── Single-shot modes ─────────────────────────────────────────────────────
    if args.checklist:
        pre_session_checklist()
        sys.exit(0)

    if args.weekly:
        weekly_review()
        sys.exit(0)

    # ── Safety: require explicit --live or --paper ────────────────────────────
    if not args.live and not args.paper:
        print("ERROR: Must specify --live or --paper.")
        print("Rule 1: Never skip from backtest to live without paper trading first.")
        sys.exit(1)

    PAPER_MODE = args.paper
    mode_label = "PAPER" if PAPER_MODE else "LIVE"

    print(f"\n{'='*60}")
    print(f"  XAUUSD Algo — {mode_label} MODE")
    print(f"  {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"{'='*60}\n")

    # ── DB + system init ──────────────────────────────────────────────────────
    try:
        if not test_db_connection():
            print("❌ PostgreSQL unreachable. Run: docker-compose up -d")
            sys.exit(1)

        initialize_system(STATE)

        # v1.1: bootstrap regime from H4/H1 history — avoids 30-min cold wait
        bootstrap_confirmed = bootstrap_regime_from_history(STATE)
        if bootstrap_confirmed:
            log_event("STARTUP_REGIME_BOOTSTRAPPED",
                      regime=STATE.get("current_regime"))
        else:
            log_event("STARTUP_BOOTSTRAP_PARTIAL",
                      note="Needs 1-2 live regime readings, max 30 min wait")

    except ConnectionError as e:
        log_critical("STARTUP_MT5_FAILED", error=str(e))
        print(f"STARTUP FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        log_critical("STARTUP_EXCEPTION", error=str(e))
        print(f"STARTUP EXCEPTION: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Restart-safe range + S1 pending injection ─────────────────────────────
    # If we restart mid-London session and range already happened at 07:55,
    # recalculate it and re-place S1 pending orders so the session isn't lost.
    try:
        from utils.session import get_current_session as _sess
        from engines.data_engine import calculate_pre_london_range as _clr

        if _sess() in ("LONDON", "LONDON_NY_OVERLAP") and not STATE.get("range_data"):
            _r = _clr()
            if _r:
                STATE["range_data"] = _r
                log_event("STARTUP_RANGE_INJECTED",
                          high=round(_r["range_high"], 2),
                          low=round(_r["range_low"], 2),
                          size=round(_r["range_size"], 2))
                _regime         = get_safe_regime(STATE)
                _no_pending_yet = (not STATE.get("s1_pending_buy_ticket")
                                   and not STATE.get("s1_pending_sell_ticket"))
                _no_open_pos    = not STATE.get("open_position")
                if (not PAPER_MODE
                        and _regime not in (RegimeState.NO_TRADE, RegimeState.UNSTABLE)
                        and _no_pending_yet
                        and _no_open_pos):
                    _safe_execute("s1_pending_restart", place_s1_pending_orders, STATE)
                    log_event("STARTUP_PENDING_ORDERS_PLACED", regime=_regime.value)
    except Exception as _re:
        log_warning("STARTUP_RANGE_INJECTION_ERROR", error=str(_re))

    # ── Pre-session checklist (warn, do not block) ────────────────────────────
    checklist_ok = pre_session_checklist()
    if not checklist_ok:
        log_warning("CHECKLIST_FAILED_CONTINUING", mode=mode_label,
                    note="Manual override — operator responsibility")

    # ── Build + start scheduler ───────────────────────────────────────────────
    scheduler = build_scheduler()
    scheduler.start()

    log_event("SCHEDULER_STARTED",
              mode=mode_label,
              jobs=[j.id for j in scheduler.get_jobs()])
    print(f"Scheduler running — {len(scheduler.get_jobs())} jobs active.")
    print("Press Ctrl+C to stop.")

    # ── Main loop — heartbeat every 5 min ────────────────────────────────────
    try:
        while True:
            time.sleep(300)
            if STATE.get("trading_enabled"):
                log_event("SYSTEM_ALIVE",
                          regime=STATE["current_regime"],
                          position=STATE.get("open_position"),
                          r3_open=STATE.get("r3_open_ticket"),
                          daily_pnl_pct=round(STATE.get("daily_net_pnl_pct", 0) * 100, 3),
                          weekly_pnl_pct=round(STATE.get("weekly_net_pnl_pct", 0) * 100, 3),
                          peak_equity=round(STATE.get("peak_equity", 0), 2),
                          spread_mult=round(STATE.get("spread_multiplier", 1.0), 2))

    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down gracefully...")
        log_event("SHUTDOWN_REQUESTED")
        scheduler.shutdown(wait=False)
        cancel_all_pending_orders()
        persist_critical_state(STATE)
        log_event("SHUTDOWN_COMPLETE")
        print("All pending orders cancelled. State persisted. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
