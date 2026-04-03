"""
Backtest Risk Management Package

Implements all kill switches and risk management logic from the live system.
"""

from .kill_switches import (
    check_ks3_daily_loss,
    check_ks5_weekly_loss,
    check_ks6_drawdown,
    check_ks7_event_blackout,
    run_all_kill_switches
)

from .position_manager import (
    check_position_limits,
    update_position_state,
    manage_position_exits
)

__all__ = [
    "check_ks3_daily_loss",
    "check_ks5_weekly_loss", 
    "check_ks6_drawdown",
    "check_ks7_event_blackout",
    "run_all_kill_switches",
    "check_position_limits",
    "update_position_state",
    "manage_position_exits",
]
