"""
Position Management

Handles position limits, state management, and exit logic.
Mirrors live system position management exactly.

IMPORTANT: check_position_limits() is a NEW-ORDER guard only.
Never call it on an already-open position during management.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from ..models import SimPosition, SimulatedState


def check_position_limits(
    state: SimulatedState,
    new_order_direction: str,
    new_order_lots: float
) -> Tuple[bool, str]:
    """
    Check if a NEW order can be opened given current position state.

    This is a pre-entry guard. Call it before placing a new order.
    Do NOT call it on positions that are already open and being managed
    (SL/TP/trailing) — that will spuriously block management once any
    trend-family slot is occupied.
    """
    if not state.trading_enabled:
        return False, "TRADING_DISABLED"

    # Trend-family slot already occupied — block another trend-family entry
    if state.trend_family_occupied and new_order_direction in ["LONG", "SHORT"]:
        return False, "TREND_FAMILY_OCCUPIED"

    max_lots    = 1.0
    current_lots = get_current_position_lots(state)
    if current_lots + new_order_lots > max_lots:
        return False, f"MAX_LOTS_EXCEEDED: {current_lots + new_order_lots:.2f} > {max_lots}"

    if state.current_session == "OFF_HOURS":
        return False, "OFF_HOURS"

    return True, "OK"


def get_current_position_lots(state: SimulatedState) -> float:
    """Get total lots currently open across all lanes."""
    total_lots = 0.0
    if state.open_position and state.original_lot_size > 0:
        total_lots += state.original_lot_size
    if state.s8_open_ticket and state.s8_entry_price > 0:
        total_lots += 0.01
    if state.r3_open_ticket and state.r3_entry_price > 0:
        total_lots += 0.01
    return total_lots


def update_position_state(
    state: SimulatedState,
    position: SimPosition,
    action: str
) -> Dict[str, Any]:
    """
    Return a dict of state field updates for a position open/close/modify.
    Caller applies via setattr.
    """
    updates: Dict[str, Any] = {}

    if action == "OPEN":
        if position.strategy == "S8_ATR_SPIKE":
            updates.update({
                "s8_open_ticket":         position.strategy,
                "s8_entry_price":         position.entry_price,
                "s8_stop_price_original": position.stop_price_original,
                "s8_stop_price_current":  position.current_sl,
                "s8_trade_direction":     position.direction,
                "s8_be_activated":        position.be_activated,
                "s8_open_time_utc":       position.entry_time.isoformat(),
            })
        elif position.strategy == "R3_CAL_MOMENTUM":
            updates.update({
                "r3_open_ticket": position.strategy,
                "r3_open_time":   position.entry_time,
                "r3_entry_price": position.entry_price,
                "r3_stop_price":  position.current_sl,
                "r3_tp_price":    position.tp,
            })
        else:
            updates.update({
                "trend_family_occupied":  True,
                "trend_family_strategy": position.strategy,
                "trend_trade_direction": position.direction,
                "open_position":         position.strategy,
                "entry_price":           position.entry_price,
                "stop_price_original":   position.stop_price_original,
                "stop_price_current":    position.current_sl,
                "original_lot_size":     position.lots,
                "position_partial_done": position.partial_done,
                "position_be_activated": position.be_activated,
            })

    elif action == "CLOSE":
        if position.strategy == "S8_ATR_SPIKE":
            updates.update({
                "s8_open_ticket":         None,
                "s8_entry_price":         0.0,
                "s8_stop_price_original": 0.0,
                "s8_stop_price_current":  0.0,
                "s8_trade_direction":     None,
                "s8_be_activated":        False,
                "s8_open_time_utc":       None,
            })
        elif position.strategy == "R3_CAL_MOMENTUM":
            updates.update({
                "r3_open_ticket": None,
                "r3_open_time":   None,
                "r3_entry_price": 0.0,
                "r3_stop_price":  0.0,
                "r3_tp_price":    0.0,
            })
        else:
            updates.update({
                "trend_family_occupied":  False,
                "trend_family_strategy": None,
                "trend_trade_direction": None,
                "open_position":         None,
                "entry_price":           0.0,
                "stop_price_original":   0.0,
                "stop_price_current":    0.0,
                "original_lot_size":     0.0,
                "position_partial_done": False,
                "position_be_activated": False,
            })

    elif action == "MODIFY":
        if position.strategy == "S8_ATR_SPIKE":
            updates["s8_stop_price_current"] = position.current_sl
            updates["s8_be_activated"]       = position.be_activated
        elif position.strategy == "R3_CAL_MOMENTUM":
            updates["r3_stop_price"] = position.current_sl
        else:
            updates["stop_price_current"]    = position.current_sl
            updates["position_be_activated"] = position.be_activated
            updates["position_partial_done"] = position.partial_done

    return updates


def manage_position_exits(
    state: SimulatedState,
    positions: List[SimPosition],
    current_price: float,
    current_time: datetime,
    indicators: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Enumerate exit actions needed for a list of positions."""
    exit_actions = []
    for position in positions:
        for check in (
            check_breakeven_activation,
            lambda p, cp: check_trailing_stop_adjustment(p, cp, indicators),
            check_partial_exit_conditions,
            check_hard_exit_conditions,
        ):
            action = check(position, current_price) if check != check_hard_exit_conditions else check(position, current_time)
            if action:
                exit_actions.append(action)
    return exit_actions


def check_breakeven_activation(
    position: SimPosition,
    current_price: float
) -> Optional[Dict[str, Any]]:
    if position.be_activated:
        return None
    if position.current_r(current_price) >= 0.75:
        return {
            "action": "BE_ACTIVATION",
            "position_id": position.strategy,
            "new_stop": position.entry_price,
            "current_r": position.current_r(current_price),
        }
    return None


def check_trailing_stop_adjustment(
    position: SimPosition,
    current_price: float,
    indicators: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not position.be_activated:
        return None
    atr = indicators.get("atr_m15") or indicators.get("atr_h1")
    if not atr or atr <= 0:
        return None
    trail_distance = atr * 0.5
    if position.direction == "LONG":
        new_stop = current_price - trail_distance
        if new_stop > position.current_sl:
            return {"action": "TRAILING_STOP_ADJUST", "position_id": position.strategy, "new_stop": new_stop, "old_stop": position.current_sl}
    else:
        new_stop = current_price + trail_distance
        if new_stop < position.current_sl:
            return {"action": "TRAILING_STOP_ADJUST", "position_id": position.strategy, "new_stop": new_stop, "old_stop": position.current_sl}
    return None


def check_partial_exit_conditions(
    position: SimPosition,
    current_price: float
) -> Optional[Dict[str, Any]]:
    if position.partial_done:
        return None
    if position.current_r(current_price) >= 1.0:
        return {"action": "PARTIAL_EXIT", "position_id": position.strategy, "exit_ratio": 0.5, "current_r": position.current_r(current_price)}
    return None


def check_hard_exit_conditions(
    position: SimPosition,
    current_time: datetime
) -> Optional[Dict[str, Any]]:
    if position.strategy == "R3_CAL_MOMENTUM":
        elapsed = (current_time - position.entry_time).total_seconds() / 60
        if elapsed >= 30:
            return {"action": "HARD_EXIT", "position_id": position.strategy, "reason": "R3_TIME_EXIT", "elapsed_minutes": elapsed}
    return None


def reconcile_positions(
    state: SimulatedState,
    mt5_positions: List[Dict[str, Any]],
    simulated_positions: List[SimPosition]
) -> Dict[str, Any]:
    return {
        "ghost_positions": [],
        "orphan_positions": [],
        "state_updates": {},
        "reconciliation_status": "HEALTHY"
    }
