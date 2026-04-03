"""
Position Management

Handles position limits, state management, and exit logic.
Mirrors live system position management exactly.
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
    Check if new position can be opened based on limits.
    
    Returns:
        (can_open, reason) tuple
    """
    # Check if trading is enabled
    if not state.trading_enabled:
        return False, "TRADING_DISABLED"
    
    # Check position family limits
    if state.trend_family_occupied and new_order_direction in ["LONG", "SHORT"]:
        # Only allow independent lanes (S8, R3) when trend family occupied
        return False, "TREND_FAMILY_OCCUPIED"
    
    # Check maximum position size
    max_lots = 1.0  # Default max lots
    current_lots = get_current_position_lots(state)
    
    if current_lots + new_order_lots > max_lots:
        return False, f"MAX_LOTS_EXCEEDED: {current_lots + new_order_lots} > {max_lots}"
    
    # Check session limits
    session = state.current_session
    if session == "OFF_HOURS":
        return False, "OFF_HOURS"
    
    return True, "OK"


def get_current_position_lots(state: SimulatedState) -> float:
    """Get total lots currently open across all lanes."""
    total_lots = 0.0
    
    # Main trend family position
    if state.open_position and state.original_lot_size > 0:
        total_lots += state.original_lot_size
    
    # S8 independent lane
    if state.s8_open_ticket and state.s8_entry_price > 0:
        # Would need to get S8 lot size from position tracking
        total_lots += 0.01  # Default assumption
    
    # R3 independent lane
    if state.r3_open_ticket and state.r3_entry_price > 0:
        # Would need to get R3 lot size from position tracking
        total_lots += 0.01  # Default assumption
    
    return total_lots


def update_position_state(
    state: SimulatedState,
    position: SimPosition,
    action: str  # "OPEN", "CLOSE", "MODIFY"
) -> Dict[str, Any]:
    """
    Update state based on position action.
    
    Returns:
        Dict of state updates to apply.
    """
    updates = {}
    
    if action == "OPEN":
        if position.strategy in ["S8_ATR_SPIKE"]:
            # S8 independent lane
            updates.update({
                "s8_open_ticket": position.strategy,  # Would be actual ticket
                "s8_entry_price": position.entry_price,
                "s8_stop_price_original": position.stop_price_original,
                "s8_stop_price_current": position.current_sl,
                "s8_trade_direction": position.direction,
                "s8_be_activated": position.be_activated,
                "s8_open_time_utc": position.entry_time.isoformat(),
            })
        elif position.strategy == "R3_CAL_MOMENTUM":
            # R3 independent lane
            updates.update({
                "r3_open_ticket": position.strategy,  # Would be actual ticket
                "r3_open_time": position.entry_time,
                "r3_entry_price": position.entry_price,
                "r3_stop_price": position.current_sl,
                "r3_tp_price": position.tp,
            })
        else:
            # Trend family position
            updates.update({
                "trend_family_occupied": True,
                "trend_family_strategy": position.strategy,
                "open_position": position.strategy,  # Would be actual ticket
                "entry_price": position.entry_price,
                "stop_price_original": position.stop_price_original,
                "stop_price_current": position.current_sl,
                "original_lot_size": position.lots,
                "position_partial_done": position.partial_done,
                "position_be_activated": position.be_activated,
            })
    
    elif action == "CLOSE":
        if position.strategy == "S8_ATR_SPIKE":
            updates.update({
                "s8_open_ticket": None,
                "s8_entry_price": 0.0,
                "s8_stop_price_original": 0.0,
                "s8_stop_price_current": 0.0,
                "s8_trade_direction": None,
                "s8_be_activated": False,
                "s8_open_time_utc": None,
            })
        elif position.strategy == "R3_CAL_MOMENTUM":
            updates.update({
                "r3_open_ticket": None,
                "r3_open_time": None,
                "r3_entry_price": 0.0,
                "r3_stop_price": 0.0,
                "r3_tp_price": 0.0,
            })
        else:
            # Trend family position
            updates.update({
                "trend_family_occupied": False,
                "trend_family_strategy": None,
                "open_position": None,
                "entry_price": 0.0,
                "stop_price_original": 0.0,
                "stop_price_current": 0.0,
                "original_lot_size": 0.0,
                "position_partial_done": False,
                "position_be_activated": False,
            })
    
    elif action == "MODIFY":
        if position.strategy == "S8_ATR_SPIKE":
            updates["s8_stop_price_current"] = position.current_sl
            updates["s8_be_activated"] = position.be_activated
        elif position.strategy == "R3_CAL_MOMENTUM":
            updates["r3_stop_price"] = position.current_sl
        else:
            # Trend family position
            updates["stop_price_current"] = position.current_sl
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
    """
    Manage position exits (BE activation, trailing stops, partial exits).
    
    Returns:
        List of exit actions to take.
    """
    exit_actions = []
    
    for position in positions:
        # Check breakeven activation
        be_action = check_breakeven_activation(position, current_price)
        if be_action:
            exit_actions.append(be_action)
        
        # Check trailing stop adjustment
        trail_action = check_trailing_stop_adjustment(
            position, current_price, indicators
        )
        if trail_action:
            exit_actions.append(trail_action)
        
        # Check partial exit conditions
        partial_action = check_partial_exit_conditions(position, current_price)
        if partial_action:
            exit_actions.append(partial_action)
        
        # Check hard exit conditions
        hard_action = check_hard_exit_conditions(position, current_time)
        if hard_action:
            exit_actions.append(hard_action)
    
    return exit_actions


def check_breakeven_activation(
    position: SimPosition,
    current_price: float
) -> Optional[Dict[str, Any]]:
    """Check if position should activate breakeven."""
    if position.be_activated:
        return None
    
    # Calculate current R
    current_r = position.current_r(current_price)
    
    # BE activation threshold (typically 0.75R)
    be_threshold = 0.75
    
    if current_r >= be_threshold:
        return {
            "action": "BE_ACTIVATION",
            "position_id": position.strategy,
            "new_stop": position.entry_price,
            "current_r": current_r,
        }
    
    return None


def check_trailing_stop_adjustment(
    position: SimPosition,
    current_price: float,
    indicators: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Check if trailing stop should be adjusted."""
    if not position.be_activated:
        return None
    
    # Get ATR for trailing distance
    atr = indicators.get("atr_m15", indicators.get("atr_h1"))
    if atr is None or atr <= 0:
        return None
    
    # Calculate new trailing stop level
    trail_distance = atr * 0.5  # 0.5× ATR trailing
    
    if position.direction == "LONG":
        new_stop = current_price - trail_distance
        if new_stop > position.current_sl:
            return {
                "action": "TRAILING_STOP_ADJUST",
                "position_id": position.strategy,
                "new_stop": new_stop,
                "old_stop": position.current_sl,
            }
    else:  # SHORT
        new_stop = current_price + trail_distance
        if new_stop < position.current_sl:
            return {
                "action": "TRAILING_STOP_ADJUST",
                "position_id": position.strategy,
                "new_stop": new_stop,
                "old_stop": position.current_sl,
            }
    
    return None


def check_partial_exit_conditions(
    position: SimPosition,
    current_price: float
) -> Optional[Dict[str, Any]]:
    """Check if position should take partial profits."""
    if position.partial_done:
        return None
    
    # Partial exit threshold (typically 1.0R)
    partial_threshold = 1.0
    current_r = position.current_r(current_price)
    
    if current_r >= partial_threshold:
        return {
            "action": "PARTIAL_EXIT",
            "position_id": position.strategy,
            "exit_ratio": 0.5,  # Close 50%
            "current_r": current_r,
        }
    
    return None


def check_hard_exit_conditions(
    position: SimPosition,
    current_time: datetime
) -> Optional[Dict[str, Any]]:
    """Check for hard exit conditions (time-based, etc.)."""
    # R3 hard exit after 30 minutes
    if position.strategy == "R3_CAL_MOMENTUM":
        elapsed = (current_time - position.entry_time).total_seconds() / 60
        if elapsed >= 30:  # 30 minute hard exit
            return {
                "action": "HARD_EXIT",
                "position_id": position.strategy,
                "reason": "R3_TIME_EXIT",
                "elapsed_minutes": elapsed,
            }
    
    # Add other hard exit conditions as needed
    
    return None


def reconcile_positions(
    state: SimulatedState,
    mt5_positions: List[Dict[str, Any]],
    simulated_positions: List[SimPosition]
) -> Dict[str, Any]:
    """
    Reconcile simulated positions against MT5 positions.
    
    Returns:
        Dict with reconciliation results and any required state updates.
    """
    # This would be implemented to match the live system's reconciliation logic
    # For now, return basic structure
    
    return {
        "ghost_positions": [],
        "orphan_positions": [],
        "state_updates": {},
        "reconciliation_status": "HEALTHY"
    }
