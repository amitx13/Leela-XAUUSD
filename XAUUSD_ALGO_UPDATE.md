# LEELA-XAUUSD COMPLETE ACTION PLAN v4.0 (FINAL)
### Single Source of Truth — All Changes, All Files, All Steps
### Date: Based on full 6-batch codebase audit + agent verification

---

## WHAT WAS ALREADY DONE (SKIP THESE)

| Item | Status | Proof |
|------|--------|-------|
| `MAX_S1_FAMILY_ATTEMPTS = 4` | ✅ Already at target | Agent verified in config.py |
| `M5_LOSS_PAUSE_COUNT = 5` | ✅ Already at target | Agent verified in config.py |
| KS4 duplicate consolidation | ✅ Already resolved | Alias pattern `KS4_REDUCED_TRADE_COUNT = KS4_REDUCED_TRADES` |

---

## FILE 1 — `config.py`

```
CHANGE 1.1 — Kill Switch Thresholds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND:    KS5_WEEKLY_LOSS_LIMIT_PCT  = -0.100
REPLACE: KS5_WEEKLY_LOSS_LIMIT_PCT  = -0.120

FIND:    KS6_DRAWDOWN_LIMIT_PCT     = 0.12
REPLACE: KS6_DRAWDOWN_LIMIT_PCT     = 0.20

NO OTHER config.py CHANGES NEEDED.
```

---

## FILE 2 — `state.py`

```
CHANGE 2.1 — S8 Independent Position State Keys
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In build_initial_state(), ADD these keys near the existing s8_fired_today 
/ s8_armed block:

    "s8_open_ticket":         None,
    "s8_entry_price":         0.0,
    "s8_stop_price_original": 0.0,
    "s8_stop_price_current":  0.0,
    "s8_trade_direction":     None,
    "s8_be_activated":        False,
    "s8_open_time_utc":       None,

VERIFY: The following keys should already exist from the original S8 
code. If any are missing, add them:

    "s8_fired_today":         False,
    "s8_armed":               False,
    "s8_arm_time":            None,
    "s8_arm_candle_time":     None,
    "s8_spike_high":          0.0,
    "s8_spike_low":           0.0,
    "s8_direction":           None,
    "s8_spike_atr":           0.0,
    "s8_spike_candle_idx":    None,
```

---

## FILE 3 — `signal_engine.py`

```
CHANGE 3.1 — evaluate_s8_signal() — Gate Update
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND (near top of function, after docstring):
    if state.get("s8_fired_today"):
        return None
    if state.get("trend_family_occupied"):
        return None

REPLACE WITH:
    if state.get("s8_fired_today"):
        return None
    if state.get("s8_open_ticket"):
        return None

    # Regime gate — block NO_TRADE only. UNSTABLE allowed because
    # 0.4× regime multiplier × 0.5× S8 lot = 0.2× effective risk.
    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        return None

NOTE: Keep the existing s1_pending_buy/sell_ticket check — it prevents
S8 from firing while S1 pending orders are on the server, avoiding
the scenario where S8 fills at market and then S1 pending also fills.


CHANGE 3.2 — _check_s8_confirmation() — Lot Size Fix (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND (inside _check_s8_confirmation, near bottom where candidate 
is built):
    stop_dist = float(state.get("s8_spike_atr", 0)) * 0.5
    ...
    entry = tick_s8.ask if direction == "long" else tick_s8.bid
    stop  = state["s8_spike_low"] - stop_dist if direction == "long" \
            else state["s8_spike_high"] + stop_dist

    lots = calculate_lot_size(stop_dist, state["size_multiplier"] * size_mult, state)

REPLACE WITH:
    atr_stop_buffer = float(state.get("s8_spike_atr", 0)) * 0.5

    entry = tick_s8.ask if direction == "long" else tick_s8.bid
    stop  = state["s8_spike_low"] - atr_stop_buffer if direction == "long" \
            else state["s8_spike_high"] + atr_stop_buffer

    # CRITICAL FIX: Use actual distance from entry to stop, NOT just
    # the ATR buffer. Entry is market price at confirmation, which can
    # be far from the spike extreme. Using atr_stop_buffer alone would
    # calculate lots for 5pt risk when actual risk is 25pts → 5× oversize.
    # Floor: never calculate on <5-point stop (prevents lot explosion).
    actual_stop_dist = max(
        abs(entry - stop),
        5.0 * config.CONTRACT_SPEC.get("point", 0.01)
    )

    lots = calculate_lot_size(actual_stop_dist, state["size_multiplier"] * size_mult, state)

ALSO: If _build_candidate uses stop_dist for range_size or chase 
window, pass actual_stop_dist there too.


CHANGE 3.3 — S3 Regime Gate — Allow UNSTABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND (in evaluate_s3_signal, near top):
    if regime in (RegimeState.NO_TRADE, RegimeState.UNSTABLE):
        return None

REPLACE WITH:
    if regime == RegimeState.NO_TRADE:
        return None

REASON: S3 is a stop-hunt reversal. UNSTABLE (85-95% ATR) is prime
stop-hunting territory. Size already reduced to 0.4× by regime 
multiplier.


CHANGE 3.4 — S6 Regime Gate — Block UNSTABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND (in can_s6_fire or evaluate_s6_signal):
    if regime == RegimeState.NO_TRADE:
        return False, "REGIME_NO_TRADE"

REPLACE WITH:
    if regime in (RegimeState.NO_TRADE, RegimeState.UNSTABLE):
        return False, f"REGIME_BLOCKS_S6_{regime.value}"

REASON: S6 Asian breakout relies on clean range formation. In 
UNSTABLE, Asian ranges are noisy — breakout levels unreliable.


CHANGE 3.5 — S1F H1 EMA20 Direction Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND: evaluate_s1f_signal() (or wherever S1F evaluation happens).

ADD after regime check, BEFORE M5 EMA20 entry logic:

    # H1 EMA20 direction validation — reject if trend reversed
    ema20_h1 = get_ema20_h1()
    direction = state.get("last_s1_direction")
    mt5_s1f = get_mt5()
    tick_s1f = mt5_s1f.symbol_info_tick(config.SYMBOL)
    current_mid = (tick_s1f.ask + tick_s1f.bid) / 2

    if ema20_h1 is not None and direction is not None:
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

NOTE: get_ema20_h1() already exists in data_engine.py (used by S2).
The existing M5 EMA20 body-close check stays below this unchanged.
H1 = macro safety gate, M5 = entry timing. Both needed.


CHANGE 3.6 — S6 ADX/DI Trend Filter in place_s6_pending_orders()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREREQUISITE: DI+/DI- values must be cached in STATE by the regime 
engine (see Change 9.1 below). Read from state, do NOT recompute.

ADD before the order placement loop:

    place_buy = True
    place_sell = True

    adx = get_adx_h4()
    di_plus = state.get("last_di_plus_h4")
    di_minus = state.get("last_di_minus_h4")

    if (adx is not None and adx > 25
            and di_plus is not None and di_minus is not None):
        if di_minus > 0:
            di_ratio = di_plus / di_minus
        else:
            di_ratio = 999.0

        if di_ratio > 1.3:
            place_sell = False
            log_event("S6_SELL_FILTERED_STRONG_UPTREND",
                      adx=round(adx, 1), di_ratio=round(di_ratio, 2))
        elif di_ratio < (1.0 / 1.3):    # ~0.769
            place_buy = False
            log_event("S6_BUY_FILTERED_STRONG_DOWNTREND",
                      adx=round(adx, 1), di_ratio=round(di_ratio, 2))

    if place_buy:
        # ... existing buy stop placement code ...
    if place_sell:
        # ... existing sell stop placement code ...


CHANGE 3.7 — S7 ADX/DI Trend Filter in place_s7_pending_orders()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IDENTICAL logic to Change 3.6. Copy the ADX/DI filter block.
Only change log event names to S7_SELL_FILTERED / S7_BUY_FILTERED.


CHANGE 3.8 — S8 Position Management Function + Inline Stop Modifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL NOTE: The shared modify_stop() function reads 
state["last_s1_direction"] and state["stop_price_current"] internally.
These are S1's keys, NOT S8's. Using modify_stop() for S8 would read 
wrong direction, wrong stop, and KS1 would falsely reject valid moves.
S8 MUST use its own inline stop modifier.

ADD this helper function:

    def _s8_modify_stop(ticket, new_stop, reason, state):
        """
        Inline stop modification for S8 — bypasses shared modify_stop()
        which reads S1-specific state keys.
        """
        mt5_mod = get_mt5()
        pos = mt5_mod.positions_get(ticket=ticket)
        if not pos:
            return False
        request = {
            "action":   mt5_mod.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   config.SYMBOL,
            "sl":       new_stop,
            "tp":       pos[0].tp,
            "magic":    config.MAGIC,
        }
        result = mt5_mod.order_send(request)
        if result and result.retcode == mt5_mod.TRADE_RETCODE_DONE:
            log_event(reason, ticket=ticket, new_stop=round(new_stop, 3))
            return True
        log_event(f"{reason}_FAILED", ticket=ticket,
                  retcode=result.retcode if result else "NO_RESULT")
        return False

ADD the main management function:

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
        positions = mt5_s8.positions_get(ticket=ticket)

        # ── Position closed by SL/TP on broker side? ──
        if not positions:
            log_event("S8_POSITION_CLOSED_BROKER", ticket=ticket)
            deals = mt5_s8.history_deals_get(position=ticket)
            if deals and len(deals) > 0:
                exit_price = deals[-1].price
            else:
                exit_price = state.get("s8_entry_price", 0.0)
            on_trade_closed(ticket, exit_price, "S8_BROKER_CLOSE", state)
            return

        pos = positions[0]
        entry     = state["s8_entry_price"]
        stop_orig = state["s8_stop_price_original"]
        direction = state["s8_trade_direction"]

        # ── BE Activation at 1.5R ──
        if not state["s8_be_activated"]:
            r_now = calculate_r_multiple(entry, pos.price_current,
                                         stop_orig, direction)
            if r_now >= config.BE_ACTIVATION_R:    # 1.5
                success = _s8_modify_stop(ticket, entry,
                                          "S8_BE_ACTIVATED", state)
                if success:
                    state["s8_be_activated"] = True
                    state["s8_stop_price_current"] = entry
                    log_event("S8_BE_ACTIVATED",
                              r_now=round(r_now, 2), ticket=ticket)
            return    # Don't trail until BE is activated

        # ── ATR Trail (only after BE) ──
        new_trail = calculate_atr_trail(pos.price_current, direction)
        if new_trail is None:
            return
        current_stop = state["s8_stop_price_current"]

        if direction == "LONG" and new_trail > current_stop:
            success = _s8_modify_stop(ticket, new_trail,
                                      "S8_ATR_TRAIL", state)
            if success:
                state["s8_stop_price_current"] = new_trail
        elif direction == "SHORT" and new_trail < current_stop:
            success = _s8_modify_stop(ticket, new_trail,
                                      "S8_ATR_TRAIL", state)
            if success:
                state["s8_stop_price_current"] = new_trail

ADD the cleanup function:

    def _on_s8_closed(state: dict) -> None:
        """Clean up all S8 position state. Does NOT reset s8_fired_today."""
        state["s8_open_ticket"]         = None
        state["s8_entry_price"]         = 0.0
        state["s8_stop_price_original"] = 0.0
        state["s8_stop_price_current"]  = 0.0
        state["s8_trade_direction"]     = None
        state["s8_be_activated"]        = False
        state["s8_open_time_utc"]       = None


CHANGE 3.9 — S4/S5 Gate Separation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This change may be in signal_engine.py or signal_engine_phase2.py
depending on where _can_s4_fire and _can_s5_fire live.

ADD new shared function (near can_s1_family_fire):

    def _can_trend_family_fire(state: dict) -> tuple[bool, str]:
        """
        Shared gate for S4/S5 — checks trend family occupancy +
        regime + kill switches, but NOT s1_family_attempts_today.
        S4 and S5 should not be limited by S1/S1B attempt count.
        """
        if not state["trading_enabled"]:
            return False, "TRADING_DISABLED"
        if state["trend_family_occupied"]:
            return False, "TREND_FAMILY_OCCUPIED"
        regime = get_safe_regime(state)
        if regime == RegimeState.NO_TRADE:
            return False, "REGIME_NO_TRADE"
        if not regime.allows_s1:
            return False, f"REGIME_BLOCKS_TREND_{regime.value}"
        permitted, reason = run_pre_trade_kill_switches(state)
        if not permitted:
            return False, reason
        return True, "PERMITTED"

MODIFY _can_s4_fire():

    def _can_s4_fire(state: dict) -> tuple[bool, str]:
        if state.get("s4_fired_today"):
            return False, "S4_ALREADY_FIRED_TODAY"
        if not state.get("s4_ema_touched"):
            return False, "S4_EMA_NOT_TOUCHED_YET"
        return _can_trend_family_fire(state)    # ← WAS can_s1_family_fire

MODIFY _can_s5_fire():

    def _can_s5_fire(state: dict) -> tuple[bool, str]:
        if state.get("s5_fired_today"):
            return False, "S5_ALREADY_FIRED_TODAY"
        if not state.get("s5_compression_confirmed"):
            return False, "S5_NO_COMPRESSION_THIS_SESSION"
        return _can_trend_family_fire(state)    # ← WAS can_s1_family_fire

ALSO: Search entire codebase for places where S4 or S5 firing 
increments s1_family_attempts_today. If found, REMOVE those increments.
Search for all occurrences of: s1_family_attempts_today += 1

can_s1_family_fire() STAYS UNCHANGED — S1 and S1B continue using it.


CHANGE 3.10 — _check_s8_confirmation() — Race Condition Guard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERIFY: Inside _check_s8_confirmation(), confirm that 
state["s8_armed"] = False is set BEFORE the candidate dict is 
returned. If the candidate is returned while s8_armed is still True,
a second M5 evaluation could enter _check_s8_confirmation again and 
build a duplicate candidate before place_order() completes.

The existing code (from Batch 2 answers) shows:
    # Disarm
    state["s8_armed"]           = False
    state["s8_arm_time"]        = None
    state["s8_arm_candle_time"] = None
    state["s8_fired_today"]     = True
    ...
    return candidate

This order is CORRECT — disarm happens before return. VERIFY this 
is the actual order in the live file. If s8_fired_today = True is 
set here AND in on_trade_opened(), that's fine — double-setting True 
is harmless and provides defense-in-depth.
```

---

## FILE 4 — `signal_engine_phase2.py`

```
CHANGE 4.1 — R3 Regime Gate — Block NO_TRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND: evaluate_r3_signal() (or _can_r3_fire)

ADD at top of function, after existing checks:

    regime = get_safe_regime(state)
    if regime == RegimeState.NO_TRADE:
        log_event("R3_BLOCKED_NO_TRADE_REGIME")
        return None

If _can_r3_fire is the gate function (returning tuple), use:
    if regime == RegimeState.NO_TRADE:
        return False, "REGIME_NO_TRADE"

REASON: In >95th percentile ATR, post-event moves are 
indistinguishable from noise. All other regimes fine — events 
create their own micro-context.
```

---

## FILE 5 — `execution_engine.py`

```
CHANGE 5.1 — on_trade_opened() — S8 Independent Handling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

place_order() auto-calls on_trade_opened() (confirmed V11).

FIND the block that sets trend_family_occupied:
    if signal in ("S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK",
                   "S1E_PYRAMID", "S4_LONDON_PULL", "S5_NY_COMPRESS"):
        state["trend_family_occupied"] = True
        state["trend_family_strategy"] = signal

ADD THIS BLOCK *BEFORE* that existing block:

    # ── S8: Independent position lane (R3 pattern) ──
    # S8 does NOT occupy trend_family. Coexists with S1/S4/S5.
    if signal == "S8_ATR_SPIKE":
        state["s8_open_ticket"]         = ticket
        state["s8_entry_price"]         = actual_price
        state["s8_stop_price_original"] = candidate.get("stop_level", 0.0)
        state["s8_stop_price_current"]  = candidate.get("stop_level", 0.0)
        state["s8_trade_direction"]     = candidate.get("direction", None)
        state["s8_be_activated"]        = False
        state["s8_open_time_utc"]       = datetime.now(pytz.utc).isoformat()
        state["s8_fired_today"]         = True
        log_event("S8_POSITION_OPENED",
                  ticket=ticket,
                  entry=actual_price,
                  stop=candidate.get("stop_level"),
                  direction=candidate.get("direction"))
        return    # ← EARLY RETURN: skip trend_family_occupied setter

DO NOT add S8 to the trend_family_occupied setter list.


CHANGE 5.2 — on_trade_closed() — S8 Guard (CRITICAL BUG PREVENTION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without this fix, closing S8 while S1 is open would accidentally 
clear trend_family_occupied → allow duplicate S1 fills.

FIND (inside on_trade_closed()):
    if signal == "R3_CAL_MOMENTUM":
        state["r3_open_ticket"] = None
        state["r3_open_time"]   = None
        state["r3_entry_price"] = 0.0
        state["r3_stop_price"]  = 0.0
        state["r3_tp_price"]    = 0.0
    else:
        if signal not in ("S1D_PYRAMID", "S1E_PYRAMID"):
            state["open_position"]          = None
            state["trend_family_occupied"]  = False
            ...

REPLACE WITH:
    if signal == "R3_CAL_MOMENTUM":
        state["r3_open_ticket"] = None
        state["r3_open_time"]   = None
        state["r3_entry_price"] = 0.0
        state["r3_stop_price"]  = 0.0
        state["r3_tp_price"]    = 0.0
    elif signal == "S8_ATR_SPIKE":
        # S8 independent lane — clean up S8 state only.
        # Do NOT touch trend_family_occupied (S1 may still be open).
        state["s8_open_ticket"]         = None
        state["s8_entry_price"]         = 0.0
        state["s8_stop_price_original"] = 0.0
        state["s8_stop_price_current"]  = 0.0
        state["s8_trade_direction"]     = None
        state["s8_be_activated"]        = False
        state["s8_open_time_utc"]       = None
    else:
        if signal not in ("S1D_PYRAMID", "S1E_PYRAMID"):
            state["open_position"]          = None
            state["trend_family_occupied"]  = False
            state["trend_family_strategy"]  = None
            ...

IMPORTANT: The P&L calculation, equity update, consecutive loss 
tracking, KS4 evaluation, trade DB insert, and all other logic that 
runs AFTER this branching block must still execute for S8. Only the 
STATE CLEANUP branching needs the guard. Make sure S8 falls through 
to the shared P&L/logging section below the branch.


CHANGE 5.3 — on_trade_opened() — S6/S7 Cross-Cancellation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Confirmed state key names: s6_pending_buy_ticket, s6_pending_sell_ticket,
s7_pending_buy_ticket, s7_pending_sell_ticket.

FIND: After the trend_family_occupied setter block (after S8 early 
return, after existing S1/S1B/S1F/S4/S5 setter).

ADD:

    # ── S6/S7 Cross-Cancellation ──
    # When S6 or S7 fills, cancel all other S6/S7 pending orders.
    # Prevents position collision (two fills → open_position overwrite).
    if signal in ("S6_ASIAN_BRK", "S7_DAILY_STRUCT"):
        mt5_cancel = get_mt5()

        for key in ("s6_pending_buy_ticket", "s6_pending_sell_ticket",
                     "s7_pending_buy_ticket", "s7_pending_sell_ticket"):
            pending_ticket = state.get(key)
            if pending_ticket and pending_ticket != ticket:
                try:
                    mt5_cancel.order_delete(pending_ticket)
                    log_event("S6_S7_CROSS_CANCEL",
                              cancelled_ticket=pending_ticket,
                              filled_strategy=signal,
                              cancelled_key=key)
                except Exception:
                    pass
                state[key] = None


CHANGE 5.4 — emergency_shutdown() — S8 + R3 Cleanup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND: emergency_shutdown() function. It currently resets 
trend_family_occupied, open_position, trading_enabled but has 
no awareness of S8 or R3 independent lanes.

ADD to emergency_shutdown(), near the existing state cleanup:

    # Clean independent position lanes
    state["s8_open_ticket"]    = None
    state["s8_entry_price"]    = 0.0
    state["s8_be_activated"]   = False
    state["s8_open_time_utc"]  = None
    state["s8_trade_direction"] = None
    state["s8_stop_price_original"] = 0.0
    state["s8_stop_price_current"]  = 0.0
    state["r3_open_ticket"]    = None    # Fix existing R3 gap too
    state["r3_open_time"]      = None

REASON: After emergency shutdown, stale S8/R3 state keys would 
cause the bot to think positions are open when they aren't on 
restart.
```

---

## FILE 6 — `main.py`

```
CHANGE 6.1 — Import S8 Functions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADD to imports at top of file:

    from engines.signal_engine import evaluate_s8_signal, manage_s8_position

If manage_s8_position ends up in a different file, adjust import.


CHANGE 6.2 — S8 Blocks in m5_mgmt_job()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADD two S8 blocks inside m5_mgmt_job(). Place them AFTER the R3 
management block and BEFORE or AFTER the S4/S5 evaluation block.

    # ════════════════════════════════════════════
    # S8: Independent Position Management
    # ════════════════════════════════════════════
    if STATE.get("s8_open_ticket"):
        _safe_execute("s8_mgmt", manage_s8_position, STATE)

    # ════════════════════════════════════════════
    # S8: Signal Evaluation (no open S8 + not fired today)
    # ════════════════════════════════════════════
    elif not STATE.get("s8_fired_today"):
        candidate = evaluate_s8_signal(STATE)
        if candidate:
            permitted, reason = run_pre_trade_kill_switches(STATE)
            if not permitted:
                log_event("S8_BLOCKED_BY_KS", reason=reason)
            else:
                if not PAPER_MODE:
                    ticket = place_order(candidate, STATE)
                    # on_trade_opened is called automatically inside
                    # place_order (confirmed V11)
                    if ticket:
                        log_event("S8_ORDER_PLACED", ticket=ticket)
                else:
                    log_event("PAPER_MODE_S8_SIGNAL",
                              direction=candidate.get("direction"))

NOTE: elif ensures we either manage OR evaluate, never both on 
same M5 bar. Clean and prevents race conditions.


CHANGE 6.3 — S8 Midnight Reset (CRITICAL — WITHOUT THIS S8 FIRES ONCE EVER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND: midnight_reset_job() or reset_daily_counters() — wherever 
s6_fired_today, s7_fired_today, r3_fired_today are reset.

ADD alongside the existing daily resets:

    STATE["s8_fired_today"]      = False
    STATE["s8_armed"]            = False
    STATE["s8_arm_time"]         = None
    STATE["s8_arm_candle_time"]  = None
    STATE["s8_spike_candle_idx"] = None

NOTE: Do NOT reset s8_open_ticket here — an S8 position could be 
held overnight (not common but possible). Only reset the daily 
signal flags and spike detection state.

If these resets happen in state.py reset_daily_counters() instead 
of main.py midnight_reset_job(), add them there following the same 
pattern as the other strategy resets.


CHANGE 6.4 — Friday Close Job
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADD scheduler job registration (near other scheduler.add_job calls):

    scheduler.add_job(
        friday_close_job,
        'cron',
        day_of_week='fri',
        hour=20,
        minute=30,
        timezone='UTC',
        id='friday_close'
    )

ADD the function:

    def friday_close_job():
        """
        Close all positions and cancel all pending orders before weekend.
        Friday 20:30 UTC (Saturday 02:00 IST).
        XAUUSD weekend gaps can be 50-200pts on geopolitical events.
        Trailing stops don't update while bot is offline.
        """
        log_event("FRIDAY_CLOSE_INITIATED")

        # 1. Close trend family position
        if STATE.get("open_position"):
            _safe_execute("friday_close_trend",
                         _execute_generic_market_close,
                         STATE["open_position"],
                         "FRIDAY_WEEKEND_CLOSE")

        # 2. Close R3 position
        if STATE.get("r3_open_ticket"):
            _safe_execute("friday_close_r3",
                         execute_r3_hard_exit, STATE)

        # 3. Close S8 position
        if STATE.get("s8_open_ticket"):
            _safe_execute("friday_close_s8",
                         _execute_s8_friday_close, STATE)

        # 4. Cancel all pending orders
        _safe_execute("friday_cancel_pending",
                     cancel_all_pending_orders)

        log_event("FRIDAY_CLOSE_COMPLETE")

    def _execute_s8_friday_close(state):
        """Close S8 position at market for Friday shutdown."""
        ticket = state.get("s8_open_ticket")
        if not ticket:
            return
        mt5_close = get_mt5()
        positions = mt5_close.positions_get(ticket=ticket)
        if not positions:
            _on_s8_closed(state)
            return
        pos = positions[0]
        close_type = (mt5_close.ORDER_TYPE_SELL if pos.type == 0
                      else mt5_close.ORDER_TYPE_BUY)
        result = mt5_close.order_send({
            "action":   mt5_close.TRADE_ACTION_DEAL,
            "symbol":   config.SYMBOL,
            "volume":   pos.volume,
            "type":     close_type,
            "position": ticket,
            "magic":    config.MAGIC,
            "comment":  "S8_FRIDAY_CLOSE",
        })
        if result and result.retcode == mt5_close.TRADE_RETCODE_DONE:
            on_trade_closed(ticket, result.price,
                           "S8_FRIDAY_CLOSE", state)

NOTE: If _execute_generic_market_close() can accept any ticket as 
parameter (not just STATE["open_position"]), you can simplify the 
S8 close to reuse it. Check the function signature.
```

---

## FILE 7 — `risk_engine.py`

```
CHANGE 7.1 — KS6 Docstring Fix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND (in check_ks6_drawdown docstring or comments):
    Any reference to "×0.92" or "8%" or "peak × 0.88"

REPLACE WITH:
    # KS6: equity < peak × (1.0 - KS6_DRAWDOWN_LIMIT_PCT)
    # Current threshold: 20% drawdown from 30-day rolling peak
    # equity < peak × 0.80 → emergency halt + email
```

---

## FILE 8 — `db/persistence.py` + Database Schema

```
CHANGE 8.1 — SQL Schema Migration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RUN THIS SQL (safe to re-run with IF NOT EXISTS):

    ALTER TABLE system_state.system_state_persistent
        ADD COLUMN IF NOT EXISTS s8_fired_today         BOOLEAN       DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS s8_open_ticket          BIGINT        DEFAULT NULL,
        ADD COLUMN IF NOT EXISTS s8_entry_price          DECIMAL(10,3) DEFAULT 0.0,
        ADD COLUMN IF NOT EXISTS s8_stop_price_original  DECIMAL(10,3) DEFAULT 0.0,
        ADD COLUMN IF NOT EXISTS s8_stop_price_current   DECIMAL(10,3) DEFAULT 0.0,
        ADD COLUMN IF NOT EXISTS s8_trade_direction      VARCHAR(10)   DEFAULT NULL,
        ADD COLUMN IF NOT EXISTS s8_be_activated         BOOLEAN       DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS s8_open_time_utc        TIMESTAMPTZ   DEFAULT NULL;


CHANGE 8.2 — persist_critical_state()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADD these columns to:
1. The column list in the INSERT statement
2. The VALUES placeholder list
3. The ON CONFLICT DO UPDATE SET block
4. The params dict

    params["s8_fired_today"]         = state.get("s8_fired_today", False)
    params["s8_open_ticket"]         = state.get("s8_open_ticket")
    params["s8_entry_price"]         = state.get("s8_entry_price", 0.0)
    params["s8_stop_price_original"] = state.get("s8_stop_price_original", 0.0)
    params["s8_stop_price_current"]  = state.get("s8_stop_price_current", 0.0)
    params["s8_trade_direction"]     = state.get("s8_trade_direction")
    params["s8_be_activated"]        = state.get("s8_be_activated", False)
    params["s8_open_time_utc"]       = state.get("s8_open_time_utc")

Follow EXACT same pattern used by existing columns (r3_fired_today,
peak_equity, consecutive_losses, etc.).


CHANGE 8.3 — restore_critical_state()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADD to the column→key mapping section:

    state["s8_fired_today"]         = row.get("s8_fired_today", False)
    state["s8_open_ticket"]         = row.get("s8_open_ticket")
    state["s8_entry_price"]         = float(row.get("s8_entry_price", 0) or 0)
    state["s8_stop_price_original"] = float(row.get("s8_stop_price_original", 0) or 0)
    state["s8_stop_price_current"]  = float(row.get("s8_stop_price_current", 0) or 0)
    state["s8_trade_direction"]     = row.get("s8_trade_direction")
    state["s8_be_activated"]        = row.get("s8_be_activated", False)
    state["s8_open_time_utc"]       = row.get("s8_open_time_utc")

Follow same null-safety pattern: float(x or 0) for numeric fields.


CHANGE 8.4 — schema.py DDL Update
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIND: The CREATE TABLE statement for system_state_persistent in 
schema.py (used by create_all_schemas() for fresh deployments).

ADD the same 8 S8 columns with their types and defaults so new 
deployments have the schema from the start.
```

---

## FILE 9 — `regime_engine.py` (or wherever regime is computed)

```
CHANGE 9.1 — Cache DI+/DI- Values in STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREREQUISITE for Changes 3.6 and 3.7 (S6/S7 ADX filter).

The regime engine already computes ADX which requires DI+ and DI-.
These values are computed internally but not exposed. We need to 
cache them in STATE so S6/S7 placement can read them.

FIND: The section of regime_engine.py (or data_engine.py) where ADX 
is calculated for H4. It will have DI+ and DI- computed as 
intermediate values.

ADD after the ADX calculation, before returning:

    state["last_di_plus_h4"]  = float(di_plus)     # Use actual variable names
    state["last_di_minus_h4"] = float(di_minus)     # from the ADX calculation

ALSO ADD to state.py build_initial_state():

    "last_di_plus_h4":   None,
    "last_di_minus_h4":  None,

The exact variable names for DI+ and DI- will depend on the 
implementation. Common patterns:
    plus_di, minus_di
    di_plus, di_minus
    adx_plus, adx_minus

The agent should find the correct names in the ADX calculation code 
and cache them.
```

---

## IMPLEMENTATION ORDER

```
STEP  1: config.py — KS thresholds          (Change 1.1)        [5 min]
         → COMMIT: "config: KS5=-12%, KS6=20%"

STEP  2: state.py — S8 keys                 (Change 2.1)        [5 min]
         → COMMIT: "state: add S8 independent position keys"

STEP  3: SQL migration                      (Change 8.1)        [5 min]
         → Run ALTER TABLE
         → COMMIT: "db: S8 columns in persistent state"

STEP  4: db/persistence.py                  (Changes 8.2-8.4)   [15 min]
         → Wire S8 into persist + restore + schema DDL
         → COMMIT: "persistence: S8 warm start support"

STEP  5: execution_engine.py on_trade_closed (Change 5.2)       [10 min]
         → CRITICAL: S8 guard before S8 goes live
         → COMMIT: "execution: S8 close guard — prevent trend family collision"

STEP  6: execution_engine.py on_trade_opened (Change 5.1)       [10 min]
         → S8 independent lane with early return
         → COMMIT: "execution: S8 independent lane in on_trade_opened"

STEP  7: execution_engine.py emergency      (Change 5.4)        [5 min]
         → S8 + R3 cleanup in emergency_shutdown
         → COMMIT: "execution: S8/R3 cleanup in emergency_shutdown"

STEP  8: signal_engine.py S8 gate + lot fix (Changes 3.1, 3.2)  [15 min]
         → Replace trend_family gate with s8_open_ticket
         → Add regime gate
         → Fix lot size with actual_stop_dist + floor
         → Verify s8_armed disarm order (Change 3.10)
         → COMMIT: "signal: S8 gate update + critical lot size fix"

STEP  9: signal_engine.py manage_s8         (Change 3.8)        [20 min]
         → _s8_modify_stop inline helper
         → manage_s8_position function
         → _on_s8_closed cleanup function
         → COMMIT: "signal: S8 position management (BE + trail)"

STEP 10: main.py S8 wiring + midnight reset (Changes 6.1-6.3)   [15 min]
         → Import, m5_mgmt_job blocks, midnight reset
         → COMMIT: "main: wire S8 + midnight reset — strategy live"

         ═══ S8 IS NOW FULLY OPERATIONAL ═══

STEP 11: signal_engine.py regime gates      (Changes 3.3, 3.4)  [5 min]
         → S3 allow UNSTABLE, S6 block UNSTABLE
         → COMMIT: "signal: fix S3/S6 regime gates"

STEP 12: signal_engine_phase2.py R3 gate    (Change 4.1)        [5 min]
         → R3 block NO_TRADE
         → COMMIT: "signal: R3 NO_TRADE regime block"

STEP 13: signal_engine.py S1F H1 check      (Change 3.5)        [10 min]
         → H1 EMA20 direction validation
         → COMMIT: "signal: S1F H1 reversal safety check"

STEP 14: regime_engine.py DI cache          (Change 9.1)        [15 min]
         → Cache DI+/DI- in STATE during regime calc
         → COMMIT: "regime: expose DI+/DI- for S6/S7 filter"

STEP 15: signal_engine.py S6/S7 ADX filter  (Changes 3.6, 3.7)  [20 min]
         → Add trend filter to both placement functions
         → COMMIT: "signal: S6/S7 skip counter-trend in strong trends"

STEP 16: execution_engine.py S6/S7 cancel   (Change 5.3)        [10 min]
         → Cross-cancel pending orders on fill
         → COMMIT: "execution: S6/S7 cross-cancel on fill"

STEP 17: signal_engine*.py S4/S5 gate       (Change 3.9)        [15 min]
         → Create _can_trend_family_fire
         → Update _can_s4_fire and _can_s5_fire
         → Verify no s1_family_attempts_today increment for S4/S5
         → COMMIT: "signal: separate S4/S5 from S1 attempt counter"

STEP 18: main.py Friday close               (Change 6.4)        [15 min]
         → New scheduler job + S8 close function
         → COMMIT: "main: Friday 20:30 UTC weekend close"

STEP 19: risk_engine.py docstring           (Change 7.1)        [2 min]
         → COMMIT: "docs: fix KS6 docstring"

TOTAL: ~3.5 hours for 19 steps, 27 discrete changes across 9 files.
```

---

## POST-IMPLEMENTATION VERIFICATION CHECKLIST

```
S8 VERIFICATION:
☐ Bot starts → S8 state keys initialize correctly
☐ evaluate_s8_signal() called in m5_mgmt_job (add temp log)
☐ S8 spike detection arms correctly (check logs)
☐ S8 confirmation generates candidate with correct lot size
  └─ Verify actual_stop_dist used, not atr_stop_buffer
  └─ Verify 5-point floor prevents lot explosion
☐ on_trade_opened routes S8 to independent lane (not trend family)
  └─ Verify trend_family_occupied stays False after S8 fill
☐ S8 gets BE activation at 1.5R via _s8_modify_stop
  └─ Verify it does NOT use shared modify_stop()
☐ S8 gets ATR trailing after BE via _s8_modify_stop
☐ S8 close does NOT clear trend_family_occupied
  └─ Test: open S1 + S8, close S8, verify S1 still tracked
☐ S8 state survives bot restart (warm start from DB)
☐ S8 midnight reset clears s8_fired_today + armed state
  └─ Verify next day S8 can fire again
☐ emergency_shutdown clears S8 + R3 state
☐ S8 closes on Friday 20:30 UTC

REGIME GATE VERIFICATION:
☐ S3 fires in UNSTABLE regime
☐ S6 blocks in UNSTABLE regime
☐ R3 blocks in NO_TRADE regime
☐ S8 blocks in NO_TRADE regime
☐ S8 fires in UNSTABLE regime (with reduced size)

FILTER VERIFICATION:
☐ S1F rejects when price wrong side of H1 EMA20
☐ S6 skips counter-trend leg when ADX>25 + DI ratio>1.3
☐ S7 skips counter-trend leg when ADX>25 + DI ratio>1.3
☐ DI+/DI- cached in STATE by regime engine
☐ S6/S7 cross-cancel fires when one fills

GATE VERIFICATION:
☐ S4 NOT blocked by s1_family_attempts_today
☐ S5 NOT blocked by s1_family_attempts_today
☐ S4/S5 still blocked by trend_family_occupied
☐ S1 IS still blocked after 4 attempts
☐ S4/S5 do NOT increment s1_family_attempts_today

KILL SWITCH VERIFICATION:
☐ KS5 triggers at -12% weekly
☐ KS6 triggers at 20% drawdown from peak
☐ Both still require manual restart

FRIDAY CLOSE VERIFICATION:
☐ friday_close_job runs at 20:30 UTC Friday
☐ Closes trend family position
☐ Closes R3 position
☐ Closes S8 position
☐ Cancels all pending orders
```

---

## COMPLETE CHANGE COUNT

| File | Changes | Est. Lines |
|------|---------|-----------|
| config.py | 2 threshold changes | ~4 |
| state.py | 9 new keys | ~20 |
| signal_engine.py | 10 changes (S8 gate, lot fix, race verify, S3/S6 regime, S1F, S6/S7 ADX, S8 mgmt+helper, S4/S5 gate) | ~250 |
| signal_engine_phase2.py | 1 change (R3 regime) | ~5 |
| execution_engine.py | 4 changes (S8 open, S8 close guard, S6/S7 cancel, emergency) | ~90 |
| main.py | 4 changes (S8 import+wiring, midnight reset, Friday close) | ~100 |
| db/persistence.py | 3 changes (persist, restore, schema DDL) | ~40 |
| risk_engine.py | 1 docstring fix | ~3 |
| regime_engine.py | 1 DI cache change | ~10 |
| **SQL migration** | 1 ALTER TABLE | ~10 |
| **TOTAL** | **27 changes across 9 files + 1 SQL** | **~532 lines** |