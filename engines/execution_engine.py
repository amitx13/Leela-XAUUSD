"""
engines/execution_engine.py

Handles: order placement, fill detection, trade open/close, stop modification,
emergency shutdown, system initialization, pre-session checklist.

v1.1 additions:
  - place_s6_pending_orders() stub
  - place_s7_pending_orders() stub
  - reset_daily_counters() — re-exported for main.py
  - initialize_system() — MT5 + DB startup
  - pre_session_checklist() — pre-session sanity gates
  - Change 56: P&L correlation check every 10 closed trades
  - KS4 countdown hook in on_trade_closed()
  - Spread-aware BUY STOP on all pending placement
"""

import uuid
import time
import threading
from datetime import datetime
import pytz

import config
from utils.logger import log_event, log_warning, log_critical
from utils.mt5_client import get_mt5, reset_mt5_connection
from utils.alerts import send_ks_alert
from db.connection import execute_query, execute_write, test_db_connection
from db.connection import get_config_value
from db.persistence import (
    persist_critical_state,
    restore_critical_state,
    update_peak_equity,
)
from engines.data_engine import (
    get_upcoming_events_within,
    get_avg_spread_last_24h,
    get_s1_preplacement_spread_baseline,
)
from engines.risk_engine import (
    calculate_r_multiple,
    trigger_ks4_countdown,
    decrement_ks4_countdown,
    check_ks7_event_blackout,
)
from engines.position_manager import (
    can_open,
    clear_all_positions,
    on_fill as pm_on_fill,
    on_close as pm_on_close,
)


# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL RETCODES — C6
# ─────────────────────────────────────────────────────────────────────────────

def _get_critical_retcodes(mt5) -> set:
    """
    C6 Fix: Retcodes that indicate systemic failure → emergency_shutdown.
    REMOVED: mt5.TRADE_RETCODE_SERVER_DISCON — does NOT exist in Python MT5 lib.
    10004 (TRADE_RETCODE_CONNECTION) covers disconnection.
    """
    return {
        mt5.TRADE_RETCODE_CONNECTION,
        mt5.TRADE_RETCODE_TRADE_DISABLED,
        mt5.TRADE_RETCODE_MARKET_CLOSED,
        mt5.TRADE_RETCODE_TIMEOUT,
        mt5.TRADE_RETCODE_FROZEN,
        mt5.TRADE_RETCODE_TOO_MANY_REQUESTS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _reconcile_position_manager_from_db() -> None:
    """Repopulate Position Manager after restart so VAR / session caps see open risk."""
    clear_all_positions()
    rows = execute_query(
        """SELECT mt5_ticket, signal_type, direction, lot_size
           FROM system_state.trades
           WHERE exit_time IS NULL AND mt5_ticket IS NOT NULL""",
        {},
    )
    if not rows:
        return
    for r in rows:
        try:
            pm_on_fill(
                str(r["signal_type"]),
                int(r["mt5_ticket"]),
                str(r["direction"]),
                float(r["lot_size"] or 0),
            )
        except (TypeError, ValueError) as e:
            log_warning("PM_RECONCILE_ROW_SKIPPED", error=str(e), row=r)


def initialize_system(state: dict) -> None:
    """
    Startup sequence:
      1. Test DB connectivity — raises ConnectionError if unreachable
      2. Connect MT5, populate CONTRACT_SPEC from symbol metadata
      3. Restore critical state from DB (regime, equity, counters)
      4. Sync current equity + update 30-day peak

    Called once in main() before bootstrap and scheduler.
    Raises ConnectionError on any critical failure.
    """
    # 1. DB gate
    if not test_db_connection():
        raise ConnectionError("PostgreSQL unreachable — run: docker-compose up -d")

    # 2. MT5 connection + symbol metadata
    mt5 = get_mt5()
    if not mt5.initialize():
        err = mt5.last_error()
        raise ConnectionError(f"MT5 initialization failed: {err}")

    symbol_info = mt5.symbol_info(config.SYMBOL)
    if symbol_info is None:
        raise ConnectionError(
            f"Symbol {config.SYMBOL} not found in MT5 — "
            "check broker feed and symbol name"
        )

    config.CONTRACT_SPEC.update({
        "point":         symbol_info.point,
        "contract_size": symbol_info.trade_contract_size,
        "volume_min":    symbol_info.volume_min,
        "volume_max":    symbol_info.volume_max,
        "volume_step":   symbol_info.volume_step,
        "digits":        symbol_info.digits,
        "tick_size":     symbol_info.trade_tick_size,
        "tick_value":    symbol_info.trade_tick_value,
        "currency_profit": symbol_info.currency_profit,
    })

    log_event("CONTRACT_SPEC_LOADED",
              symbol=config.SYMBOL,
              point=symbol_info.point,
              contract_size=symbol_info.trade_contract_size,
              volume_min=symbol_info.volume_min,
              volume_max=symbol_info.volume_max)

    # 3. Restore persisted state (regime, peak_equity, daily counters)
    restore_critical_state(state)
    _reconcile_position_manager_from_db()

    # 4. Live equity sync
    info = mt5.account_info()
    if info:
        equity = float(info.equity)
        state["current_equity"] = equity
        update_peak_equity(state)
        log_event("EQUITY_SYNCED_AT_STARTUP",
                  balance=round(info.balance, 2),
                  equity=round(equity, 2),
                  server=info.server)
    else:
        log_warning("ACCOUNT_INFO_UNAVAILABLE_AT_STARTUP",
                    note="Equity not synced — check MT5 connection")

    log_event("SYSTEM_INITIALIZED",
              symbol=config.SYMBOL,
              regime=state.get("current_regime", "NO_TRADE"),
              trading_enabled=state.get("trading_enabled", True))


# ─────────────────────────────────────────────────────────────────────────────
# PRE-SESSION CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────

def pre_session_checklist() -> bool:
    """
    Pre-session health check. Calls mt5.initialize() before account_info().
    Returns True only if ALL critical checks pass.
    Called from main() --checklist path.
    """
    from engines.data_engine import get_upcoming_events_within

    all_pass = True
    print("=" * 50)
    print("  Pre-Session Checklist")
    print("=" * 50)

    # ── 1. MT5 — initialize first, THEN account_info ─────────────────────────
    try:
        mt5 = get_mt5()
        if not mt5.initialize():
            err = mt5.last_error()
            _print_check(False, f"MT5 connected  [{err}]")
            log_critical("CHECKLIST_MT5_FAIL",
                         error=str(err),
                         note="mt5.initialize() failed — is MT5 terminal running inside Bottles?")
            all_pass = False
        else:
            info = mt5.account_info()
            if info is None:
                _print_check(False, "MT5 account_info() returned None")
                log_critical("CHECKLIST_MT5_FAIL",
                             note="initialize() OK but account_info() None — check MT5 login")
                all_pass = False
            else:
                _print_check(True, (
                    f"MT5 connected  |  balance={round(info.balance,2)}"
                    f"  equity={round(info.equity,2)}"
                    f"  server={info.server}"
                ))
                log_event("CHECKLIST_MT5_OK",
                          balance=round(info.balance, 2),
                          equity=round(info.equity, 2),
                          server=info.server)
    except Exception as e:
        _print_check(False, f"MT5 exception: {e}")
        log_critical("CHECKLIST_MT5_EXCEPTION", error=str(e))
        all_pass = False

    # ── 2. PostgreSQL ─────────────────────────────────────────────────────────
    db_ok = test_db_connection()
    _print_check(db_ok, "PostgreSQL reachable")
    if db_ok:
        log_event("CHECKLIST_DB_OK")
    else:
        log_critical("CHECKLIST_DB_FAIL", note="Run: docker-compose up -d")
        all_pass = False

    # ── 3. Power (Linux sysfs — desktop always passes) ───────────────────────
    power_ok = _check_power_status()
    _print_check(power_ok, "Power / AC connected")

    # ── 4. No imminent high-impact events (non-critical — KS7 guards live) ───
    try:
        upcoming = get_upcoming_events_within(minutes=config.KS7_PRE_EVENT_MINUTES)
        no_events = len(upcoming) == 0
        _print_check(no_events, (
            "No imminent high-impact events"
            if no_events else
            f"WARNING: {len(upcoming)} event(s) within {config.KS7_PRE_EVENT_MINUTES} min"
            f" — KS7 will auto-block"
        ))
        log_event("CHECKLIST_NO_IMMINENT_EVENTS" if no_events else "CHECKLIST_EVENT_WARNING",
                  count=len(upcoming))
    except Exception as e:
        _print_check(True, f"Calendar check skipped ({e})")

    print("=" * 50)
    if all_pass:
        print("  ✅  All clear — safe to start session")
        log_event("CHECKLIST_PASSED")
    else:
        print("  ❌  CHECKLIST FAILED — DO NOT START")
        log_warning("CHECKLIST_FAILED")

    print("=" * 50)
    return all_pass


def _print_check(ok: bool, label: str) -> None:
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}")


def _check_power_status() -> bool:
    """Returns True if AC power connected (Linux sysfs). Desktop always True."""
    try:
        with open("/sys/class/power_supply/AC/online") as f:
            return f.read().strip() == "1"
    except FileNotFoundError:
        return True

# ─────────────────────────────────────────────────────────────────────────────
# DAILY COUNTER RESET — re-exported for main.py import
# ─────────────────────────────────────────────────────────────────────────────

def reset_daily_counters(state: dict) -> None:
    """
    Resets intra-day counters ONLY.
    Rolling state (consecutive_losses, peak_equity) is NEVER reset here.
    Called by midnight_reset_job() in main.py.
    v1.1 flags (s3_fired_today, s6_fired_today, s7_fired_today) reset in
    midnight_reset_job() directly alongside this call.
    """
    state["daily_net_pnl_pct"]          = 0.0
    state["daily_commission_paid"]      = 0.0
    state["s1_family_attempts_today"]   = 0
    state["s1f_attempts_today"]         = 0
    state["consecutive_m5_losses"]      = 0
    state["session_spread_initialized"] = False
    state["spread_fallback_active"]     = True
    state["spread_readings_count"]      = 0
    state["london_tk_fired_today"]      = False
    state["ny_tk_fired_today"]          = False
    log_event("DAILY_COUNTERS_RESET")


# ─────────────────────────────────────────────────────────────────────────────
# CANCEL ALL OUR PENDING ORDERS — C5
# ─────────────────────────────────────────────────────────────────────────────

def cancel_all_pending_orders() -> int:
    """
    C5 Fix: Cancels ALL our pending orders on XAUUSD.
    Magic filter: config.MAGIC. NEVER touches orders from other EAs.
    Called by: time kills, regime→NO_TRADE, emergency_shutdown.
    Returns count of cancelled orders.
    """
    mt5     = get_mt5()
    pending = mt5.orders_get(symbol=config.SYMBOL) or []
    count   = 0

    for order in pending:
        if order.magic != config.MAGIC:
            continue    # C5: hard filter — never touch other EAs

        result  = mt5.order_delete(order.ticket)
        retcode = result.retcode if result else "NO_RESULT"

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log_event("PENDING_ORDER_CANCELLED",
                      ticket=order.ticket, type=order.type)
            count += 1
        else:
            log_warning("PENDING_ORDER_CANCEL_FAILED",
                        ticket=order.ticket, retcode=retcode)

    if count > 0:
        log_event("ALL_PENDING_CANCELLED", count=count)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# S1 PRE-LONDON PENDING ORDERS — spread-aware v1.1
# ─────────────────────────────────────────────────────────────────────────────

def place_s1_pending_orders(state: dict) -> None:
    """
    Places BUY STOP + SELL STOP at range boundaries at 07:55 London.
    Orders rest at MT5 broker — fill happens at EXACT price.
    Eliminates SIGNAL_MISSED_CHASE for S1.

    Expiry: London 16:30 (same as time kill). Magic: config.MAGIC.
    Tickets stored in state["s1_pending_buy_ticket/sell_ticket"].
    Cleared by: fill detection, time kill, regime→NO_TRADE.

    v1.1: Spread-aware BUY STOP (Master Plan §3.5):
      BUY STOP  = breakout_level + current_spread_pts
      SELL STOP = breakout_level (fills on bid, no adjustment)
      Pre-placement gate: 1.2× get_s1_preplacement_spread_baseline() (session window, 5d).
    """
    from engines.regime_engine import get_safe_regime, RegimeState
    from engines.risk_engine import calculate_lot_size

    ks7_ok, ks7_reason = check_ks7_event_blackout(state)
    if not ks7_ok:
        log_event("KS7_BLOCKED_NEW_PLACEMENT", reason=ks7_reason, path="place_s1_pending")
        return

    mt5        = get_mt5()
    range_data = state.get("range_data")
    if not range_data:
        log_warning("S1_PENDING_NO_RANGE_DATA")
        return

    regime = get_safe_regime(state)
    if regime in (RegimeState.NO_TRADE, RegimeState.UNSTABLE):
        log_event("S1_PENDING_SKIPPED_REGIME", regime=regime.value)
        return

    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        log_warning("S1_PENDING_NO_TICK")
        return

    point         = config.CONTRACT_SPEC.get("point", 0.01)
    spread_price  = tick.ask - tick.bid
    spread_pts    = spread_price / point if point else 0.0
    avg_spread    = get_s1_preplacement_spread_baseline()

    # Pre-placement gate: 1.2× session-window baseline (07:45–08:05 UTC, last 5 Mon–Fri days)
    if avg_spread > 0 and spread_pts > avg_spread * config.PREPLACEMENT_SPREAD_MULTIPLIER:
        log_event("S1_PENDING_PREPLACEMENT_SPREAD_BLOCKED",
                  spread_pts=round(spread_pts, 1),
                  avg=round(avg_spread, 1),
                  ratio=round(spread_pts / avg_spread, 2))
        return

    rh = range_data["range_high"]
    rl = range_data["range_low"]
    rs = range_data["range_size"]
    bd = range_data["breakout_dist"]

    # Spread-aware BUY STOP: add bid/ask distance in price units (not point count)
    buy_entry  = round(rh + bd + spread_price, 2)
    sell_entry = round(rl - bd, 2)

    buy_sl  = round(rl - rs * 0.10, 2)
    sell_sl = round(rh + rs * 0.10, 2)

    buy_lots  = calculate_lot_size(abs(buy_entry  - buy_sl),  state["size_multiplier"], state)
    sell_lots = calculate_lot_size(abs(sell_entry - sell_sl), state["size_multiplier"], state)

    # Expiry = London 16:30 today (DST-safe via Europe/London tz)
    london_tz  = pytz.timezone("Europe/London")
    now_london = datetime.now(london_tz)
    expiry_dt  = now_london.replace(hour=16, minute=30, second=0, microsecond=0)
    expiry_ts  = int(expiry_dt.astimezone(pytz.utc).timestamp())

    orders = [
        (mt5.ORDER_TYPE_BUY_STOP,  buy_entry,  buy_sl,  buy_lots,  "LONG",  "s1_pending_buy_ticket"),
        (mt5.ORDER_TYPE_SELL_STOP, sell_entry, sell_sl, sell_lots, "SHORT", "s1_pending_sell_ticket"),
    ]

    for order_type, price, sl, lots, direction, ticket_key in orders:
        request = {
            "action":     mt5.TRADE_ACTION_PENDING,
            "symbol":     config.SYMBOL,
            "volume":     float(lots),
            "type":       order_type,
            "price":      price,
            "sl":         sl,
            "type_time":  mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiry_ts,
            "magic":      config.MAGIC,
            "comment":    f"S1_{direction}_STOP"[:31],
        }
        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            state[ticket_key] = result.order
            log_event("S1_PENDING_PLACED",
                      direction=direction, ticket=result.order,
                      price=price, sl=sl, lots=lots,
                      expiry=expiry_dt.strftime("%H:%M"))
        else:
            retcode = result.retcode if result else "NONE"
            comment = getattr(result, "comment", "") if result else ""
            log_warning("S1_PENDING_FAILED",
                        direction=direction, retcode=retcode, comment=comment)


# ─────────────────────────────────────────────────────────────────────────────
# S6 ASIAN BREAKOUT PENDING ORDERS — Phase 1 Step 7 stub
# ─────────────────────────────────────────────────────────────────────────────

def place_s6_pending_orders(state: dict, s6result: dict) -> None:
    """
    S6 Asian Breakout — places BUY STOP / SELL STOP at 05:30 UTC.
    Expiry: 08:00 UTC (London open auto-cancel).
    Entries, stops, and lot sizes are pre-computed by evaluate_s6_signal().
    KS2 spread re-checked at placement time — spread can widen between
    signal generation and this call.
    """
    ks7_ok, ks7_reason = check_ks7_event_blackout(state)
    if not ks7_ok:
        log_event("KS7_BLOCKED_NEW_PLACEMENT", reason=ks7_reason, path="place_s6_pending")
        return

    mt5  = get_mt5()
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        log_warning("S6_PENDING_NO_TICK")
        return

    point         = config.CONTRACT_SPEC.get("point", 0.01)
    spread_price  = tick.ask - tick.bid
    spread_pts    = spread_price / point if point else 0.0
    avg_spread    = get_avg_spread_last_24h()
    if avg_spread > 0 and spread_pts > avg_spread * config.KS2_SPREAD_MULTIPLIER:
        log_event("S6_PENDING_KS2_BLOCKED",
                  spread_pts=round(spread_pts, 1),
                  avg=round(avg_spread, 1))
        return

    buy_c  = s6result.get("buy_candidate")
    sell_c = s6result.get("sell_candidate")
    if not buy_c or not sell_c:
        log_warning("S6_PENDING_NO_CANDIDATES")
        return

    # Use expiry already embedded in candidate by evaluate_s6_signal
    expiry_iso = buy_c.get("order_expiry_utc")
    expiry_ts  = int(datetime.fromisoformat(expiry_iso).timestamp()) if expiry_iso else \
                 int(datetime.now(pytz.utc).replace(hour=8, minute=0, second=0, microsecond=0).timestamp())

    orders = [
        (mt5.ORDER_TYPE_BUY_STOP,  buy_c["entry_level"],  buy_c["stop_level"],  buy_c["lot_size"],  "LONG",  "s6_pending_buy_ticket"),
        (mt5.ORDER_TYPE_SELL_STOP, sell_c["entry_level"], sell_c["stop_level"], sell_c["lot_size"], "SHORT", "s6_pending_sell_ticket"),
    ]

    for order_type, price, sl, lots, direction, ticket_key in orders:
        place_price = round(price + spread_price, 3) if order_type == mt5.ORDER_TYPE_BUY_STOP else price
        request = {
            "action":     mt5.TRADE_ACTION_PENDING,
            "symbol":     config.SYMBOL,
            "volume":     float(lots),
            "type":       order_type,
            "price":      place_price,
            "sl":         sl,
            "type_time":  mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiry_ts,
            "magic":      config.MAGIC,
            "comment":    f"S6{direction[:1]}STOP"[:31],
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            state[ticket_key] = result.order
            log_event("S6_PENDING_PLACED",
                      direction=direction, ticket=result.order,
                      price=place_price, sl=sl, lots=lots,
                      expiry=datetime.fromtimestamp(expiry_ts, tz=pytz.utc).strftime("%H:%M UTC"))
        else:
            retcode = result.retcode if result else "NO_RESULT"
            comment = getattr(result, "comment", "") if result else ""
            log_warning("S6_PENDING_FAILED",
                        direction=direction, retcode=retcode,
                        price=price, comment=comment)

    # ✅ FIXED: range_data is nested, not flat
    state["s6_range_high"] = s6result.get("range_data", {}).get("range_high", 0.0)
    state["s6_range_low"]  = s6result.get("range_data", {}).get("range_low",  0.0)


def place_s7_pending_orders(state: dict, s7result: dict) -> None:
    """
    S7 Daily Structure Breakout — places BUY STOP / SELL STOP at midnight IST.
    Sunday skip handled by caller (midnight_reset_job).
    No KS2 check: placed off-hours, fills during active London/NY sessions.
    """
    ks7_ok, ks7_reason = check_ks7_event_blackout(state)
    if not ks7_ok:
        log_event("KS7_BLOCKED_NEW_PLACEMENT", reason=ks7_reason, path="place_s7_pending")
        return

    mt5  = get_mt5()
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        log_warning("S7_PENDING_NO_TICK")
        return

    buy_c  = s7result.get("buy_candidate")
    sell_c = s7result.get("sell_candidate")
    if not buy_c or not sell_c:
        log_warning("S7_PENDING_NO_CANDIDATES")
        return

    # Expiry: 21:00 IST today → UTC
    ist_now   = datetime.now(pytz.timezone("Asia/Kolkata"))
    expiry_dt = ist_now.replace(hour=21, minute=0, second=0, microsecond=0)
    expiry_ts = int(expiry_dt.astimezone(pytz.utc).timestamp())

    spread_price = tick.ask - tick.bid

    orders = [
        (mt5.ORDER_TYPE_BUY_STOP,  buy_c["entry_level"],  buy_c["stop_level"],  buy_c["lot_size"],  "LONG",  "s7_pending_buy_ticket"),
        (mt5.ORDER_TYPE_SELL_STOP, sell_c["entry_level"], sell_c["stop_level"], sell_c["lot_size"], "SHORT", "s7_pending_sell_ticket"),
    ]

    for order_type, price, sl, lots, direction, ticket_key in orders:
        place_price = round(price + spread_price, 3) if order_type == mt5.ORDER_TYPE_BUY_STOP else price
        request = {
            "action":     mt5.TRADE_ACTION_PENDING,
            "symbol":     config.SYMBOL,
            "volume":     float(lots),
            "type":       order_type,
            "price":      place_price,
            "sl":         sl,
            "type_time":  mt5.ORDER_TIME_SPECIFIED,
            "expiration": expiry_ts,
            "magic":      config.MAGIC,
            "comment":    f"S7{direction[:1]}STOP"[:31],
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            state[ticket_key] = result.order
            log_event("S7_PENDING_PLACED",
                      direction=direction, ticket=result.order,
                      price=place_price, sl=sl, lots=lots,
                      expiry=expiry_dt.strftime("%H:%M IST"))
        else:
            retcode = result.retcode if result else "NO_RESULT"
            comment = getattr(result, "comment", "") if result else ""
            log_warning("S7_PENDING_FAILED",
                        direction=direction, retcode=retcode,
                        price=place_price, comment=comment)

    # prev_day_high/low are top-level in s7result ✅
    state["s7_prev_day_high"] = s7result.get("prev_day_high", 0.0)
    state["s7_prev_day_low"]  = s7result.get("prev_day_low",  0.0)
    
# ─────────────────────────────────────────────────────────────────────────────
# PENDING FILL HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def on_trade_opened_from_pending_fill(
    ticket:       int,
    pos_price:    float,
    pos_volume:   float,
    pos_sl:       float,
    direction:    str,
    state:        dict,
    signal_type:  str = "S1_LONDON_BRK",
) -> None:
    """
    Called when a pending STOP order fills (detected by m5_mgmt fill monitor).
    Reconstructs candidate dict from current state, delegates to on_trade_opened().
    Slippage = 0 — STOP orders fill at exact requested price.
    signal_type: must match SignalType.value (S1_LONDON_BRK, S1B_FAILED_BRK, S6_ASIAN_BRK, S7_DAILY_STRUCT).
    """
    commission = config.COMMISSION_PER_LOT_PER_SIDE * pos_volume
    range_data = state.get("range_data") or {}
    mt5        = get_mt5()
    info       = mt5.account_info()
    from utils.session import get_current_session

    sess = get_current_session()

    candidate = {
        "signal_type":           signal_type,
        "strategy_version":      "V1",
        "campaign_id":           str(uuid.uuid4()),
        "direction":             direction,
        "entry_level":           round(pos_price, 3),
        "stop_level":            round(pos_sl, 3),
        "lot_size":              pos_volume,
        "range_size":            range_data.get("range_size", 0.0),
        "asian_range_high":      range_data.get("range_high", 0.0),
        "asian_range_low":       range_data.get("range_low", 0.0),
        "asian_range_size_pts":  range_data.get("range_size", 0.0),
        "regime_at_entry":       state["current_regime"],
        "size_multiplier_used":  state["size_multiplier"],
        "adx_h4_at_entry":       state.get("last_adx_h4", 0.0),
        "atr_h1_percentile":     state.get("last_atr_pct_h1", 0.0),
        "session":               sess,
        "regime_age_seconds":    0,
        "macro_bias_at_entry":   state.get("macro_bias", "BOTH_PERMITTED"),
        "macro_boost_at_entry":  bool(state.get("macro_boost", False)),
        "dxy_corr_at_entry":     float(state.get("dxy_corr_50", 0.0)),
        "macro_proxy_at_entry":  state.get("macro_proxy_instrument", "TLT"),
        "tlt_3d_slope":          float(state.get("tlt_slope", 0.0)),
        "conviction_level":      state.get("conviction_level", "STANDARD"),
        "stop_hunt_detected":    bool(state.get("stop_hunt_detected", False)),
        "spread_at_entry":       0.0,
        "spread_vs_avg_ratio":   0.0,
        "equity_at_entry":       float(info.equity) if info else 0.0,
        "london_hour_at_entry":  0,
        "s1_family_attempt_num": state["s1_family_attempts_today"] + 1,
        "event_proximity_min":   999,
        "failed_breakout_trade": False,
        "post_time_kill_reentry": False,
    }

    on_trade_opened(ticket, pos_price, 0.0, commission, candidate, state)


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC PLACE_ORDER — spread-aware BUY STOPs v1.1
# ─────────────────────────────────────────────────────────────────────────────

def place_order(candidate: dict, state: dict) -> int | None:
    """
    Places a single order from a signal engine candidate dict.

    Chase logic:
      LONG:  price < entry → BUY STOP (spread-adjusted); ≤ entry+max_chase → BUY market
      SHORT: price > entry → SELL STOP; ≥ entry-max_chase → SELL market
      Outside chase window → SIGNAL_MISSED_CHASE

    KS2: ONLY hard spread gate in system (2.5× 24h baseline).
    C6:  Critical retcodes → emergency_shutdown immediately.
    B5:  stop_price_original written once here — passed to on_trade_opened().
    G8:  persist_critical_state called inside on_trade_opened().

    v1.1: BUY STOP price = entry + current_spread_pts (§3.5).
          SELL STOP left unadjusted (fills on bid).
    """
    mt5  = get_mt5()
    tick = mt5.symbol_info_tick(config.SYMBOL)
    if tick is None:
        log_warning("PLACE_ORDER_NO_TICK")
        return None

    ks7_ok, ks7_reason = check_ks7_event_blackout(state)
    if not ks7_ok:
        log_event("KS7_BLOCKED_NEW_PLACEMENT", reason=ks7_reason, path="place_order")
        return None

    ask        = tick.ask
    bid        = tick.bid
    direction  = candidate["direction"]
    entry      = candidate["entry_level"]
    range_size = candidate.get("range_size", 0.0)
    max_chase  = range_size * config.HUNT_THRESHOLD_PCT

    point         = config.CONTRACT_SPEC.get("point", 0.01)
    spread_price  = ask - bid
    spread_pts    = spread_price / point if point else 0.0
    avg_spread    = get_avg_spread_last_24h()

    permitted_pm, pm_reason = can_open(
        candidate["signal_type"], direction, float(candidate["lot_size"]), state
    )
    if not permitted_pm:
        log_event("POSITION_MANAGER_BLOCKED",
                  signal=candidate["signal_type"], reason=pm_reason)
        return None

    # KS2 hard gate (2.5× 24h baseline — Master Plan v1.1 §3.1)
    if avg_spread > 0 and spread_pts > avg_spread * config.KS2_SPREAD_MULTIPLIER:
        log_event("KS2_REJECTED_AT_PLACEMENT",
                  spread_pts=round(spread_pts, 1),
                  avg=round(avg_spread, 1),
                  ratio=round(spread_pts / avg_spread, 2))
        return None

    # ── Chase / order type + spread-aware pending ─────────────────────────────
    if direction == "LONG":
        diff = ask - entry
        if diff < 0:
            order_type = mt5.ORDER_TYPE_BUY_STOP
            price      = round(entry + spread_price, 3)
        elif diff <= max_chase:
            order_type = mt5.ORDER_TYPE_BUY
            price      = ask
        else:
            log_event("SIGNAL_MISSED_CHASE",
                      direction=direction, ask=ask, entry=entry,
                      diff=round(diff, 3), max_chase=round(max_chase, 3))
            return None
    else:
        diff = entry - bid
        if diff < 0:
            order_type = mt5.ORDER_TYPE_SELL_STOP
            price      = entry    # SELL STOP hits on bid — no spread adjustment
        elif diff <= max_chase:
            order_type = mt5.ORDER_TYPE_SELL
            price      = bid
        else:
            log_event("SIGNAL_MISSED_CHASE",
                      direction=direction, bid=bid, entry=entry,
                      diff=round(diff, 3), max_chase=round(max_chase, 3))
            return None

    is_pending = order_type in (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP)
    comment    = f"{candidate['signal_type']}{datetime.now().strftime('%H%M')}"

    request = {
        "action":    mt5.TRADE_ACTION_PENDING if is_pending else mt5.TRADE_ACTION_DEAL,
        "symbol":    config.SYMBOL,
        "volume":    float(candidate["lot_size"]),
        "type":      order_type,
        "price":     price,
        "sl":        float(candidate["stop_level"]),
        "deviation": 10,
        "magic":     config.MAGIC,
        "comment":   comment[:31],
        "type_time": mt5.ORDER_TIME_DAY,
    }

    # EXP-1 FIX: Set TP level for ALL strategies that provide one (not just R3).
    # Check multiple TP key names used by different strategies.
    _tp_level = (
        candidate.get("r3_tp_level")
        or candidate.get("s1_tp_level")
        or candidate.get("s4_tp_level")
        or candidate.get("s5_tp_level")
        or candidate.get("s6_tp_level")
        or candidate.get("s7_tp_level")
        or candidate.get("tp_level")
    )
    if _tp_level and _tp_level > 0:
        request["tp"] = float(_tp_level)

    # B4 Fix: explicit expiry for S1d/S1e/S1f/S6/S7
    if "order_expiry_utc" in candidate:
        try:
            expiry_dt = datetime.fromisoformat(candidate["order_expiry_utc"])
            request["type_time"]  = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(expiry_dt.timestamp())
        except Exception as e:
            log_warning("ORDER_EXPIRY_PARSE_FAILED", error=str(e))

    result = mt5.order_send(request)

    if result is None:
        log_warning("ORDER_SEND_RETURNED_NONE", signal=candidate["signal_type"])
        return None

    # C6: Critical retcode → emergency shutdown
    critical = _get_critical_retcodes(mt5)
    if result.retcode in critical:
        log_critical("CRITICAL_RETCODE_EMERGENCY_SHUTDOWN",
                     retcode=result.retcode,
                     signal=candidate["signal_type"])
        emergency_shutdown(f"CRITICAL_RETCODE_{result.retcode}", state)
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log_warning("ORDER_REJECTED",
                    retcode=result.retcode,
                    signal=candidate["signal_type"],
                    comment=getattr(result, "comment", ""))
        return None

    ticket       = result.order
    actual_price = result.price if hasattr(result, "price") else price
    slippage_pts = abs(actual_price - price) / point
    commission   = config.COMMISSION_PER_LOT_PER_SIDE * candidate["lot_size"]

    log_event("ORDER_PLACED",
              ticket=ticket,
              signal=candidate["signal_type"],
              direction=direction,
              price=round(actual_price, 3),
              lots=candidate["lot_size"],
              slippage_pts=round(slippage_pts, 2),
              commission_entry=round(commission, 4))

    on_trade_opened(ticket, actual_price, slippage_pts, commission, candidate, state)
    return ticket


# ─────────────────────────────────────────────────────────────────────────────
# TRADE OPEN
# ─────────────────────────────────────────────────────────────────────────────

def on_trade_opened(
    ticket:           int,
    actual_price:     float,
    slippage_pts:     float,
    commission_entry: float,
    candidate:        dict,
    state:            dict,
) -> None:
    """
    Writes full Truth Engine row on OPEN (Fix 3).
    B5 Fix: stop_price_original = candidate stop — NEVER updated after entry.
    G8 Fix: persist_critical_state() called here.
    C5:     trend_family_occupied set ONLY for S1-family signals.
    """
    trade_id = str(uuid.uuid4())

    execute_write(
        """
        INSERT INTO system_state.trades (
            trade_id, signal_type, strategy_version, mt5_ticket, campaign_id,
            direction, entry_time, entry_price,
            stop_price_original, stop_price_current, lot_size,
            commission_entry,
            regime_at_entry, size_multiplier_used, adx_h4_at_entry,
            atr_h1_percentile, session, regime_age_seconds,
            macro_bias, tlt_3d_slope, dxy_corr_at_entry, macro_boost_at_entry,
            conviction_level,
            asian_range_high, asian_range_low, asian_range_size_pts,
            stop_hunt_detected,
            spread_at_entry, spread_vs_avg_ratio, slippage_points,
            order_type_used, event_proximity_min, risk_pct_used,
            london_hour_at_entry, s1_family_attempt_num, equity_at_entry,
            partial_exit_done, be_activated, pyramid_add_done, m5_reentry_count,
            failed_breakout_trade, post_time_kill_reentry
        ) VALUES (
            :trade_id, :signal_type, :strategy_version, :mt5_ticket, :campaign_id,
            :direction, :entry_time, :entry_price,
            :stop_original, :stop_current, :lot_size,
            :commission_entry,
            :regime_at_entry, :size_mult, :adx_h4, :atr_pct, :session, :regime_age,
            :macro_bias, :tlt_slope, :dxy_corr, :macro_boost,
            :conviction,
            :range_high, :range_low, :range_size,
            :stop_hunt,
            :spread, :spread_ratio, :slippage,
            :order_type, :event_prox, :risk_pct,
            :london_hour, :s1_attempt_num, :equity_at_entry,
            FALSE, FALSE, FALSE, 0,
            :failed_brk, :post_tk
        )
        """,
        {
            "trade_id":        trade_id,
            "signal_type":     candidate["signal_type"],
            "strategy_version": candidate.get("strategy_version", "V1"),
            "mt5_ticket":      ticket,
            "campaign_id":     candidate.get("campaign_id", str(uuid.uuid4())),
            "direction":       candidate["direction"],
            "entry_time":      datetime.now(pytz.utc),
            "entry_price":     actual_price,
            "stop_original":   candidate["stop_level"],
            "stop_current":    candidate["stop_level"],
            "lot_size":        candidate["lot_size"],
            "commission_entry": commission_entry,
            "regime_at_entry": candidate.get("regime_at_entry", state["current_regime"]),
            "size_mult":       candidate.get("size_multiplier_used", state["size_multiplier"]),
            "adx_h4":          candidate.get("adx_h4_at_entry"),
            "atr_pct":         candidate.get("atr_h1_percentile"),
            "session":         candidate.get("session", "UNKNOWN"),
            "regime_age":      candidate.get("regime_age_seconds", 0),
            "macro_bias":      candidate.get("macro_bias_at_entry"),
            "tlt_slope":       candidate.get("tlt_3d_slope"),
            "dxy_corr":        candidate.get("dxy_corr_at_entry"),
            "macro_boost":     candidate.get("macro_boost_at_entry", False),
            "conviction":      candidate.get("conviction_level", "STANDARD"),
            "range_high":      candidate.get("asian_range_high"),
            "range_low":       candidate.get("asian_range_low"),
            "range_size":      candidate.get("asian_range_size_pts"),
            "stop_hunt":       candidate.get("stop_hunt_detected", False),
            "spread":          candidate.get("spread_at_entry"),
            "spread_ratio":    candidate.get("spread_vs_avg_ratio"),
            "slippage":        round(slippage_pts, 2),
            "order_type":      "STOP" if "STOP" in str(candidate.get("signal_type", "")) else "LIMIT",
            "event_prox":      candidate.get("event_proximity_min"),
            "risk_pct":        None,
            "london_hour":     candidate.get("london_hour_at_entry"),
            "s1_attempt_num":  candidate.get("s1_family_attempt_num"),
            "equity_at_entry": candidate.get("equity_at_entry"),
            "failed_brk":      candidate.get("failed_breakout_trade", False),
            "post_tk":         candidate.get("post_time_kill_reentry", False),
        }
    )

    signal = candidate["signal_type"]
    
    # ── Phase 2: R3 Independent Family Branch ────────────────────────────────
    # R3 does not occupy the trend family and tracks its own position ticket.
    if signal == "R3_CAL_MOMENTUM":
        state["r3_open_ticket"] = ticket
        state["r3_open_time"]   = datetime.now(pytz.utc)
        state["r3_entry_price"] = actual_price
        state["r3_stop_price"]  = candidate["stop_level"]
        state["r3_tp_price"]    = candidate.get("r3_tp_level", 0.0)
        state["r3_fired_today"] = True
        # R3 does NOT set: open_position, trend_family_occupied, trend_family_strategy
        # But it DOES update the PM and persist state:
        from engines.position_manager import pm_on_fill
        from db.persistence import persist_critical_state
        pm_on_fill(signal, ticket, candidate["direction"], float(candidate["lot_size"]))
        persist_critical_state(state)
        log_event("R3_TRADE_OPENED",
                  trade_id=trade_id, ticket=ticket,
                  direction=candidate["direction"],
                  price=round(actual_price, 3),
                  lots=candidate["lot_size"])
        return   # ← EARLY RETURN: skip the main family state updates below

    # ── Standard path (all non-R3 signals) ─────────────────────────────────--
    # Everything below is EXISTING code, unchanged.
    state["open_position"]       = ticket
    state["entry_price"]         = actual_price
    state["stop_price_original"] = candidate["stop_level"]
    state["stop_price_current"]  = candidate["stop_level"]
    state["original_lot_size"]   = candidate["lot_size"]
    state["open_trade_id"]       = trade_id
    state["open_campaign_id"]    = candidate.get("campaign_id")
    state["last_s1_direction"]   = candidate["direction"]
    state["last_s1_max_r"]       = 0.0

    # C5: S1, S1b, S1f, S1e occupy trend_family. S1d does NOT.
    # BUG-2 FIX: Added S4_LONDON_PULL, S5_NY_COMPRESS to trend family.
    # Without this, S4/S5 don't block other strategies and their hard exits never fire.
    if signal in ("S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK", "S1E_PYRAMID",
                  "S4_LONDON_PULL", "S5_NY_COMPRESS"):
        state["trend_family_occupied"] = True
        state["trend_family_strategy"] = signal

    # v1.1: S1b + S3 share reversal_family
    if signal in ("S1B_FAILED_BRK", "S3_STOP_HUNT_REV"):
        state["reversal_family_occupied"] = True

    # Attempt counters
    if signal in ("S1_LONDON_BRK", "S1B_FAILED_BRK"):
        state["s1_family_attempts_today"] += 1
    if signal == "S1F_POST_TK":
        state["s1f_attempts_today"] += 1
    if signal == "S1D_PYRAMID":
        state["position_m5_count"] += 1
    if signal == "S1E_PYRAMID":
        state["position_pyramid_done"] = True

    if signal in ("S2_MEAN_REV", "S6_ASIAN_BRK", "S7_DAILY_STRUCT"):
        state["trend_family_strategy"] = signal

    pm_on_fill(signal, ticket, candidate["direction"], float(candidate["lot_size"]))

    # G8 Fix: persist on EVERY trade open
    persist_critical_state(state)

    log_event("TRADE_OPENED",
              trade_id=trade_id, ticket=ticket,
              signal=signal, direction=candidate["direction"],
              price=round(actual_price, 3), lots=candidate["lot_size"])


# ─────────────────────────────────────────────────────────────────────────────
# TRADE CLOSE — KS4 countdown + Change 56 correlation check
# ─────────────────────────────────────────────────────────────────────────────

def on_trade_closed(
    ticket:      int,
    exit_price:  float,
    exit_reason: str,
    state:       dict,
) -> None:
    """
    Updates Truth Engine row on CLOSE (Fix 3).
    B5: R-multiple always vs stop_price_original (never stop_price_current).
    v1.1: trigger_ks4_countdown() when loss streak hits KS4_LOSS_STREAK_COUNT.
    Change 56: run P&L correlation check every 10 closed trades.
    """
    rows = execute_query(
        """SELECT trade_id, entry_price, stop_price_original, lot_size,
                  direction, commission_entry, signal_type, entry_time
           FROM system_state.trades
           WHERE mt5_ticket = :ticket AND exit_time IS NULL
           LIMIT 1""",
        {"ticket": ticket}
    )

    if not rows:
        log_warning("ON_TRADE_CLOSED_NO_OPEN_ROW", ticket=ticket)
        return

    row        = rows[0]
    signal     = row["signal_type"]
    entry      = float(row["entry_price"])
    stop_orig  = float(row["stop_price_original"])
    lot_size   = float(row["lot_size"])
    direction  = row["direction"]
    comm_entry = float(row["commission_entry"])

    pt          = config.CONTRACT_SPEC.get("point", 0.01)
    tick_size   = config.CONTRACT_SPEC.get("tick_size", pt)
    tick_value  = config.CONTRACT_SPEC.get("tick_value", 1.0)
    contract_sz = config.CONTRACT_SPEC.get("contract_size", 100)
    # BUG-9 FIX: Use tick_value/tick_size formula instead of double-counting point.
    # pnl_points is in broker points (price_diff / point).
    # Correct formula: pnl_gross = (price_diff / tick_size) * tick_value * lot_size
    price_diff  = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    pnl_points  = price_diff / pt
    pnl_gross   = (price_diff / tick_size) * tick_value * lot_size

    comm_exit  = config.COMMISSION_PER_LOT_PER_SIDE * lot_size
    total_comm = comm_entry + comm_exit
    pnl_net    = pnl_gross - total_comm

    r_multiple = calculate_r_multiple(entry, exit_price, stop_orig, direction)
    outcome    = "WIN" if pnl_net > 0 else "LOSS" if pnl_net < 0 else "BREAKEVEN"


    execute_write(
        """UPDATE system_state.trades SET
                exit_time         = :exit_time,
                exit_price        = :exit_price,
                pnl_gross_dollars = :pnl_gross,
                commission_exit   = :comm_exit,
                total_commission  = :total_comm,
                pnl_net_dollars   = :pnl_net,
                pnl_points        = :pnl_points,
                r_multiple        = :r_mult,
                outcome           = :outcome,
                exit_reason       = :exit_reason,
                time_in_trade_min = EXTRACT(EPOCH FROM (:exit_time - entry_time))/60
       WHERE mt5_ticket = :ticket AND exit_time IS NULL""",
        {
            "exit_time":   datetime.now(pytz.utc),
            "exit_price":  exit_price,
            "pnl_gross":   round(pnl_gross, 2),
            "comm_exit":   round(comm_exit, 4),
            "total_comm":  round(total_comm, 4),
            "pnl_net":     round(pnl_net, 2),
            "pnl_points":  round(pnl_points, 2),
            "r_mult":      round(r_multiple, 4),
            "outcome":     outcome,
            "exit_reason": exit_reason[:30],
            "ticket":      ticket,
        }
    )

    pm_on_close(str(signal), ticket)

    # ── Streak tracking ───────────────────────────────────────────────────────
    if outcome == "WIN":
        state["consecutive_losses"] = 0
        if signal == "S1D_PYRAMID":
            state["consecutive_m5_losses"] = 0
    else:
        state["consecutive_losses"] += 1
        if signal == "S1D_PYRAMID":
            state["consecutive_m5_losses"] += 1

        # v1.1 KS4: start 5-trade reduced-size countdown on streak threshold
        if state["consecutive_losses"] == config.KS4_LOSS_STREAK_COUNT:
            trigger_ks4_countdown(state)

    # ── State cleanup ─────────────────────────────────────────────────────────
    # ── Phase 2: R3 state cleanup ─────────────────────────────────────────────
    if signal == "R3_CAL_MOMENTUM":
        # R3 does not own open_position — only clear R3-specific state
        state["r3_open_ticket"] = None
        state["r3_open_time"]   = None
        state["r3_entry_price"] = 0.0
        state["r3_stop_price"]  = 0.0
        state["r3_tp_price"]    = 0.0
        # Fall through to equity/P&L update below (shared with all signals)
    else:
        # ── Existing non-R3 state cleanup (UNCHANGED) ─────────────────────────
        if signal not in ("S1D_PYRAMID", "S1E_PYRAMID"):
            state["open_position"]          = None
            state["trend_family_occupied"]  = False
            state["trend_family_strategy"]  = None
            state["position_partial_done"]  = False
            state["position_be_activated"]  = False
            state["position_pyramid_done"]  = False
            state["position_m5_count"]      = 0

    # ── Equity + P&L update ───────────────────────────────────────────────────
    mt5  = get_mt5()
    info = mt5.account_info()
    if info:
        equity = float(info.equity)
        if equity > 0:
            state["daily_net_pnl_pct"] += pnl_net / equity
        state["daily_commission_paid"] += total_comm
        update_peak_equity(state, equity)

    persist_critical_state(state)

    log_event("TRADE_CLOSED",
              ticket=ticket, signal=signal,
              outcome=outcome, r_multiple=round(r_multiple, 4),
              pnl_net=round(pnl_net, 2),
              exit_reason=exit_reason)

    # ── Change 56: P&L correlation check every 10 trades (v1.1 spec §3.4) ───
    try:
        trade_count = int(get_config_value("CLOSED_TRADE_COUNT") or 0) + 1
        execute_write(
            """INSERT INTO system_state.system_config (key, value)
               VALUES ('CLOSED_TRADE_COUNT', :v)
               ON CONFLICT (key) DO UPDATE SET value = :v""",
            {"v": str(trade_count)}
        )
        if trade_count % 10 == 0:
            from engines.portfolio_risk import run_correlation_check
            run_correlation_check(state)
            log_event("CORRELATION_CHECK_RUN", trade_count=trade_count)
    except Exception as e:
        log_warning("CORRELATION_CHECK_FAILED", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# STOP MODIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def modify_stop(ticket: int, new_stop: float,
                reason: str, state: dict) -> bool:
    """
    B5 Fix: Modifies stop_price_current ONLY.
    stop_price_original is NEVER touched after entry — used for R-multiple calc.
    Used for: breakeven activation, ATR trail moves.
    Returns True on success.
    """
    mt5 = get_mt5()
    pos = _get_our_position_by_ticket(mt5, ticket)
    if pos is None:
        log_warning("MODIFY_STOP_POSITION_NOT_FOUND", ticket=ticket)
        return False

    direction    = state.get("last_s1_direction", "LONG")
    entry        = state.get("entry_price", 0.0)
    current_stop = state.get("stop_price_current", 0.0)

    # KS1: never move stop against trade direction
    if direction == "LONG" and new_stop <= current_stop:
        log_warning("MODIFY_STOP_REJECTED_AGAINST_TRADE",
                    direction=direction, current=current_stop, new=new_stop)
        return False
    if direction == "SHORT" and new_stop >= current_stop:
        log_warning("MODIFY_STOP_REJECTED_AGAINST_TRADE",
                    direction=direction, current=current_stop, new=new_stop)
        return False

    result = mt5.order_send({
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   config.SYMBOL,
        "sl":       new_stop,
        "tp":       pos.tp,
        "position": ticket,
        "magic":    config.MAGIC,
    })

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        state["stop_price_current"] = new_stop
        execute_write(
            """UPDATE system_state.trades
               SET stop_price_current = :new_stop
               WHERE mt5_ticket = :ticket AND exit_time IS NULL""",
            {"new_stop": new_stop, "ticket": ticket}
        )
        log_event(reason,
                  ticket=ticket,
                  new_stop=round(new_stop, 3),
                  entry=round(entry, 3))
        return True
    else:
        retcode = result.retcode if result else "NO_RESULT"
        log_warning("MODIFY_STOP_FAILED", ticket=ticket, retcode=retcode)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# EMERGENCY SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

def emergency_shutdown(reason: str, state: dict) -> None:
    """
    C5 Fix: Magic filter enforced on EVERY position and order.
    Never closes positions or cancels orders with wrong magic number.
    Sends SMTP alert. Calls mt5.shutdown() + reset_mt5_connection().
    """
    log_critical("EMERGENCY_SHUTDOWN_INITIATED", reason=reason)

    cancel_all_pending_orders()

    mt5       = get_mt5()
    positions = mt5.positions_get(symbol=config.SYMBOL) or []

    for pos in positions:
        if pos.magic != config.MAGIC:
            continue    # C5: NEVER touch other EAs

        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        result     = mt5.order_send({
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    config.SYMBOL,
            "volume":    pos.volume,
            "type":      close_type,
            "position":  pos.ticket,
            "deviation": config.EMERGENCY_DEVIATION_POINTS,
            "magic":     config.MAGIC,
            "comment":   f"EMRG_{reason[:15]}",
        })
        retcode = result.retcode if result else "NONE"
        log_event("EMERGENCY_POSITION_CLOSED",
                  ticket=pos.ticket, volume=pos.volume, retcode=retcode)

    state["trading_enabled"]       = False
    state["shutdown_reason"]       = reason
    state["trend_family_occupied"] = False
    state["open_position"]         = None
    persist_critical_state(state)

    send_ks_alert("EMERGENCY_SHUTDOWN", (
        f"Reason: {reason} | "
        f"Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')} | "
        f"Action: Review Truth Engine. Manual restart required."
    ))

    log_critical("EMERGENCY_SHUTDOWN_COMPLETE", reason=reason)
    mt5.shutdown()
    reset_mt5_connection()


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_our_position_by_ticket(mt5, ticket: int):
    """Returns position by ticket, enforcing magic filter (C5 Fix)."""
    positions = mt5.positions_get(symbol=config.SYMBOL) or []
    for pos in positions:
        if pos.ticket == ticket and pos.magic == config.MAGIC:
            return pos
    return None