"""
engines/s8_position_management.py

Step 8 (Changes 3.1, 3.2, 3.10) + Step 9 (Change 3.8) implementation.

This module patches the S8 ATR Spike strategy in signal_engine.py with:

  STEP 8 — evaluate_s8_signal gate + lot size fixes:
    Change 3.1  — Gate: replace trend_family_occupied with s8_open_ticket check
                  + NO_TRADE-only regime block (UNSTABLE allowed at 0.4 mult)
    Change 3.2  — Lot size: use actual_stop_dist = max(abs(entry-stop), 5*point)
                  instead of just atr_stop_buffer (prevented 5x lot explosion)
    Change 3.10 — Race condition guard: verify s8_armed=False is set BEFORE
                  return in check_s8_confirmation (already correct, documented)

  STEP 9 — S8 independent position management:
    Change 3.8  — s8_modify_stop() inline helper (bypasses shared modify_stop)
                  manage_s8_position() — BE at 1.5R, ATR trail after BE
                  on_s8_closed() — cleans S8 state, keeps s8_fired_today

IMPORTANT: signal_engine.py imports and calls these functions.
The patched evaluate_s8_signal and check_s8_confirmation replace the originals.
The new management functions are called from m5_mgmt_job in main.py.
"""
import pytz
from datetime import datetime

import config
from utils.logger import log_event
from utils.mt5_client import get_mt5
from engines.regime_engine import get_safe_regime, RegimeState, get_current_atr_m15
from engines.risk_engine import calculate_lot_size, calculate_r_multiple


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — CHANGE 3.1: evaluate_s8_signal gate replacement
# ─────────────────────────────────────────────────────────────────────────────

def _apply_s8_gate_patch(state: dict) -> tuple[bool, str]:
    """
    Change 3.1 — Replacement gate logic for evaluate_s8_signal.

    OLD gate (broken):
        if state.get("trend_family_occupied"):  # WRONG — blocks S8 when S1 is open
            return None

    NEW gate:
        if state.get("s8_open_ticket"):         # Block if S8 already has open position
            return False, "S8_ALREADY_OPEN"

        regime gate: NO_TRADE blocks, UNSTABLE allowed
        (UNSTABLE = 0.4 size_multiplier * 0.5 S8 lot = 0.2 effective risk — acceptable)

    Also keeps the existing s1_pending_buy/sell_ticket check:
        if state.get("s1_pending_buy_ticket") or state.get("s1_pending_sell_ticket"):
            return False, "S1_PENDING_ORDERS_ON_SERVER"
        This prevents S8 market fill + S1 pending fill = double position.

    Returns (allowed: bool, reason: str)
    """
    # Already have an open S8 position
    if state.get("s8_open_ticket"):
        return False, "S8_ALREADY_OPEN"

    # Already fired today
    if state.get("s8_fired_today"):
        return False, "S8_FIRED_TODAY"

    # S1 pending orders on server — prevent double fill
    if state.get("s1_pending_buy_ticket") or state.get("s1_pending_sell_ticket"):
        return False, "S8_BLOCKED_S1_PENDING_ON_SERVER"

    # Regime gate: NO_TRADE only blocks. UNSTABLE allowed.
    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        return False, "S8_BLOCKED_NO_TRADE_REGIME"

    return True, "PERMITTED"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — CHANGE 3.2: lot size fix for check_s8_confirmation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_s8_lot_size(entry: float, stop: float, state: dict) -> float:
    """
    Change 3.2 — CRITICAL lot size fix.

    OLD (broken):
        stop_dist = float(state.get("s8_spike_atr", 0)) * 0.5
        lots = calculate_lot_size(stop_dist, ...)
        # BUG: stop_dist = ATR*0.5 ≈ 5pts, but actual entry-to-stop = 25pts
        # => 5x lot oversize

    NEW (correct):
        actual_stop_dist = max(abs(entry - stop), 5.0 * point)
        lots = calculate_lot_size(actual_stop_dist, ...)
        # Uses real entry-to-stop distance, floor of 5pts prevents lot explosion
        # on near-zero distances

    Also applies size_multiplier from state (regime-based).
    S8 spec: always 0.5x base lot (spike trade, not full conviction).
    So: base_lots = calculate_lot_size(actual_stop_dist, size_mult, state)
        s8_lots   = round(base_lots * 0.5, 2) — capped at volume_min
    """
    point = config.CONTRACT_SPEC.get("point", 0.01)
    actual_stop_dist = max(abs(entry - stop), 5.0 * point)

    # Apply regime size multiplier from state
    size_mult = state.get("size_multiplier", 1.0)

    base_lots = calculate_lot_size(actual_stop_dist, size_mult, state)

    # S8 is a spike continuation — 0.5x base lot
    s8_lots = round(base_lots * 0.5, 2)
    s8_lots = max(config.CONTRACT_SPEC.get("volume_min", 0.01), s8_lots)

    return s8_lots


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — CHANGE 3.10: race condition guard (verification + documentation)
# ─────────────────────────────────────────────────────────────────────────────

def verify_s8_disarm_order(state_before_return: dict) -> bool:
    """
    Change 3.10 — Race condition guard verification.

    In check_s8_confirmation(), the disarm MUST happen BEFORE return:

        CORRECT order (already in the file — this function confirms it):
            state["s8_armed"]          = False   # <-- disarm FIRST
            state["s8_arm_time"]       = None
            state["s8_arm_candle_time"]= None
            state["s8_fired_today"]    = True    # <-- then mark fired
            ...build candidate...
            return candidate                     # <-- then return

        WRONG order (would cause race condition):
            ...build candidate...
            return candidate                     # premature return
            state["s8_armed"] = False            # never reached!

    If s8_armed were still True when the candidate is returned, a second M5
    evaluation could enter check_s8_confirmation again before place_order()
    completes and build a duplicate candidate.

    Setting s8_fired_today=True in BOTH check_s8_confirmation AND
    on_trade_opened is fine — double-setting True is harmless (defense-in-depth).

    This function returns True if the state shows the disarm already happened
    (s8_armed is False after confirmation fires).
    """
    return not state_before_return.get("s8_armed", False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — CHANGE 3.8: S8 inline stop modifier (bypasses shared modify_stop)
# ─────────────────────────────────────────────────────────────────────────────

def s8_modify_stop(ticket: int, new_stop: float, reason: str, state: dict) -> bool:
    """
    Change 3.8 — Inline stop modification for S8.

    WHY this exists (do NOT use the shared modify_stop function for S8):
    The shared modify_stop() reads state["last_s1_direction"] and
    state["stop_price_current"] internally. These are S1's keys, NOT S8's.
    Using the shared helper for S8 would:
      - Read wrong direction (S1's, possibly opposite to S8's)
      - Read wrong stop reference (S1's current stop, not S8's)
      - KS1 checks inside modify_stop would see S1 state, give wrong result

    This function reads only S8's own state keys and sends a direct MT5
    TRADE_ACTION_SLTP request for the S8 ticket.
    """
    mt5_mod = get_mt5()
    positions = mt5_mod.positions_get(ticket=ticket)
    if not positions:
        return False

    pos = positions[0]
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
    log_event(f"{reason}_FAILED",
              ticket=ticket,
              retcode=result.retcode if result else "NO_RESULT")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — CHANGE 3.8: S8 position management
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_atr_trail(current_price: float, direction: str) -> float | None:
    """
    Calculate ATR-based trailing stop level.
    Uses 1.0x M15 ATR(14) from current price as trail distance.
    Returns new stop level, or None if ATR unavailable.
    """
    atr_m15 = get_current_atr_m15(period=14)
    if not atr_m15 or atr_m15 <= 0:
        return None

    trail_dist = atr_m15 * 1.0  # 1x M15 ATR trailing distance

    if direction == "LONG":
        return round(current_price - trail_dist, 3)
    else:
        return round(current_price + trail_dist, 3)


def manage_s8_position(state: dict) -> None:
    """
    Change 3.8 — S8 position management. Called every M5 bar close.

    Independent from trend family management — S8 has its own position lane.
    Uses only S8-specific state keys (s8_open_ticket, s8_entry_price, etc.)
    and the s8_modify_stop() inline helper.

    Logic:
      1. Check if position still exists (may have closed on SL/TP broker-side)
      2. BE activation at 1.5R — move stop to entry
      3. ATR trail after BE activated only

    NO partial exit: S8 is always 0.5x lot — too small to split.
    NO momentum cycle exit: that pattern is for S1 trend trades, not spikes.
    """
    ticket = state.get("s8_open_ticket")
    if not ticket:
        return

    mt5_s8 = get_mt5()
    positions = mt5_s8.positions_get(ticket=ticket)

    # Position closed by SL/TP on broker side?
    if not positions:
        log_event("S8_POSITION_CLOSED_BROKER", ticket=ticket)
        # Reconstruct exit price from history
        deals = mt5_s8.history_deals_get(position=ticket)
        if deals and len(deals) > 0:
            exit_price = deals[-1].price
        else:
            exit_price = state.get("s8_entry_price", 0.0)
        on_s8_closed(state)
        log_event("S8_POSITION_CLEANED_UP",
                  ticket=ticket,
                  exit_price=round(exit_price, 3))
        return

    pos       = positions[0]
    entry     = state["s8_entry_price"]
    stop_orig = state["s8_stop_price_original"]
    direction = state["s8_trade_direction"]

    if not entry or not stop_orig or not direction:
        log_event("S8_MGMT_MISSING_STATE", ticket=ticket)
        return

    # ── BE Activation at 1.5R ────────────────────────────────────────────────
    if not state["s8_be_activated"]:
        r_now = calculate_r_multiple(
            entry_price   = entry,
            current_price = pos.price_current,
            stop_original = stop_orig,
            direction     = direction,
        )
        if r_now >= config.BE_ACTIVATION_R:  # 1.5R
            success = s8_modify_stop(
                ticket   = ticket,
                new_stop = entry,
                reason   = "S8_BE_ACTIVATED",
                state    = state,
            )
            if success:
                state["s8_be_activated"]      = True
                state["s8_stop_price_current"] = entry
                log_event("S8_BE_ACTIVATED",
                          r_now=round(r_now, 2),
                          ticket=ticket)
        return  # Don't trail until BE is activated

    # ── ATR Trail (only after BE activated) ─────────────────────────────────
    new_trail = _calculate_atr_trail(pos.price_current, direction)
    if new_trail is None:
        return

    current_stop = state["s8_stop_price_current"]

    if direction == "LONG" and new_trail > current_stop:
        success = s8_modify_stop(
            ticket   = ticket,
            new_stop = new_trail,
            reason   = "S8_ATR_TRAIL",
            state    = state,
        )
        if success:
            state["s8_stop_price_current"] = new_trail

    elif direction == "SHORT" and new_trail < current_stop:
        success = s8_modify_stop(
            ticket   = ticket,
            new_stop = new_trail,
            reason   = "S8_ATR_TRAIL",
            state    = state,
        )
        if success:
            state["s8_stop_price_current"] = new_trail


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — CHANGE 3.8: S8 state cleanup on close
# ─────────────────────────────────────────────────────────────────────────────

def on_s8_closed(state: dict) -> None:
    """
    Change 3.8 — Clean up all S8 position state on trade close.

    Called by:
      - manage_s8_position() when broker closes on SL/TP
      - on_trade_closed() in execution_engine.py when signal == S8_ATR_SPIKE
      - execute_s8_friday_close() in main.py on Friday 20:30 UTC shutdown

    IMPORTANT: Does NOT reset s8_fired_today.
    s8_fired_today is reset in reset_daily_counters() (midnight only).
    This prevents a second S8 from firing the same day after the first closes.
    """
    state["s8_open_ticket"]        = None
    state["s8_entry_price"]        = 0.0
    state["s8_stop_price_original"] = 0.0
    state["s8_stop_price_current"]  = 0.0
    state["s8_trade_direction"]     = None
    state["s8_be_activated"]        = False
    state["s8_open_time_utc"]       = None
    # s8_fired_today intentionally NOT reset here
    log_event("S8_STATE_CLEANED_UP")
