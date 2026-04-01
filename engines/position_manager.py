"""
engines/position_manager.py — Position Manager (data layer only).

Tracks ALL open positions across ALL strategies.
Answers queries. Makes ZERO portfolio decisions.
Portfolio Risk Brain calls get_open_positions() and makes all exposure decisions.

v1.1: Position Manager also enforces reversal_family_occupied in can_open().
"""
import config
from utils.logger import log_event, log_warning


_open_positions: dict[str, dict] = {}
# key = strategy_id, value = {ticket, direction, lot_size}


def clear_all_positions() -> None:
    """Call before DB reconcile on startup so stale in-memory map cannot linger."""
    _open_positions.clear()


def can_open(strategy_id: str, direction: str, lot_size: float,
             state: dict) -> tuple[bool, str]:
    """
    Returns (permitted, reason).
    Checks:
      1. trend_family_occupied (only 1 S1-family at a time)
      2. reversal_family_occupied (S1b + S3 share one slot)
      3. total open lots <= MAX_SESSION_LOTS
      4. strategy already has open position
    """
    # Only primary trend entries compete for trend_family_occupied.
    # S1d/S1e are add-ons while an S1-family core trade is already open.
    PRIMARY_TREND   = {"S1_LONDON_BRK", "S1B_FAILED_BRK", "S1F_POST_TK"}
    REVERSAL_FAMILY = {"S1B_FAILED_BRK", "S3_STOP_HUNT_REV"}

    if strategy_id in PRIMARY_TREND and state.get("trend_family_occupied"):
        return False, "TREND_FAMILY_OCCUPIED"

    if strategy_id in REVERSAL_FAMILY and state.get("reversal_family_occupied"):
        return False, "REVERSAL_FAMILY_OCCUPIED"

    if strategy_id in _open_positions:
        return False, f"STRATEGY_{strategy_id}_ALREADY_OPEN"

    total_lots = get_total_open_lots()
    if total_lots + lot_size > config.MAX_SESSION_LOTS:
        return False, f"SESSION_LOT_CAP_REACHED_{total_lots:.2f}"

    return True, "PERMITTED"


def on_fill(strategy_id: str, ticket: int, direction: str, lot_size: float) -> None:
    _open_positions[strategy_id] = {
        "ticket":    ticket,
        "direction": direction,
        "lot_size":  lot_size,
    }
    log_event("PM_POSITION_REGISTERED",
              strategy=strategy_id, ticket=ticket,
              direction=direction, lots=lot_size)


def on_close(strategy_id: str, ticket: int) -> None:
    if strategy_id in _open_positions:
        del _open_positions[strategy_id]
        log_event("PM_POSITION_DEREGISTERED", strategy=strategy_id, ticket=ticket)
    else:
        log_warning("PM_CLOSE_UNKNOWN_STRATEGY", strategy=strategy_id)


def get_open_positions() -> dict:
    return dict(_open_positions)


def get_total_open_lots() -> float:
    return sum(p["lot_size"] for p in _open_positions.values())


def get_direction_exposure(direction: str) -> float:
    return sum(p["lot_size"] for p in _open_positions.values()
               if p["direction"] == direction)