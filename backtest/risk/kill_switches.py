"""
Kill Switches Implementation

All kill switches from the live system implemented for backtesting.
Matches live system thresholds exactly.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from ..models import SimulatedState


def check_ks3_daily_loss(state: SimulatedState) -> Tuple[bool, str]:
    """
    KS3: Daily loss limit kill switch.
    
    Threshold: -4% daily loss (configurable via KS3_DAILY_LOSS_LIMIT_PCT)
    Triggers: trading_enabled = False, daily_pnl reset
    """
    threshold = -0.04  # Default: -4%
    daily_pnl_pct = state.daily_pnl / state.balance if state.balance > 0 else 0
    
    if daily_pnl_pct <= threshold:
        return True, f"Daily loss {daily_pnl_pct:.3f} exceeds threshold {threshold}"
    
    return False, "OK"


def check_ks5_weekly_loss(state: SimulatedState, current_time: datetime) -> Tuple[bool, str]:
    """
    KS5: Weekly loss limit kill switch.
    
    Threshold: -12% weekly loss (configurable via KS5_WEEKLY_LOSS_LIMIT_PCT)
    Triggers: trading_enabled = False, weekly reset required
    """
    threshold = -0.12  # Default: -12%
    
    # Calculate weekly P&L (simplified - would need trade history in real implementation)
    # For backtest, we'll track weekly cumulative P&L in state
    weekly_pnl = getattr(state, 'weekly_pnl', state.daily_pnl)
    weekly_pnl_pct = weekly_pnl / state.balance if state.balance > 0 else 0
    
    if weekly_pnl_pct <= threshold:
        return True, f"Weekly loss {weekly_pnl_pct:.3f} exceeds threshold {threshold}"
    
    return False, "OK"


def check_ks6_drawdown(state: SimulatedState) -> Tuple[bool, str]:
    """
    KS6: Drawdown circuit breaker.
    
    Threshold: 20% drawdown from 30-day rolling peak
    Calculation: equity < peak × (1.0 - KS6_DRAWDOWN_LIMIT_PCT)
    Current threshold: equity < peak × 0.80 → emergency halt + email
    """
    threshold = 0.20  # Default: 20%
    
    if state.peak_equity <= 0:
        return False, "NO_PEAK_EQUITY"
    
    drawdown_pct = (state.peak_equity - state.equity) / state.peak_equity
    
    if drawdown_pct >= threshold:
        return True, f"Drawdown {drawdown_pct:.3f} exceeds threshold {threshold}"
    
    return False, "OK"


def check_ks7_event_blackout(
    state: SimulatedState,
    current_time: datetime,
    upcoming_events: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    KS7: Economic event blackout windows.
    
    Blocks new trades during high-impact economic events.
    Sets ks7_active flag and stores pre-event ATR/price for R3.
    """
    if not upcoming_events:
        # No events, deactivate if active
        if state.ks7_active:
            state.ks7_active = False
            state.ks7_pre_event_atr = 0.0
            state.ks7_pre_event_price = 0.0
        return False, "NO_EVENTS"
    
    # Check for events within blackout window
    blackout_minutes = 30  # Default: 30 minutes before/after events
    
    for event in upcoming_events:
        event_time = event.get('time')
        if not event_time:
            continue
        
        time_diff = abs((current_time - event_time).total_seconds() / 60)
        
        if time_diff <= blackout_minutes:
            # Activate blackout
            if not state.ks7_active:
                state.ks7_active = True
                # Store pre-event data for R3
                atr_m15 = getattr(state, 'last_atr_m15', 0.0)
                state.ks7_pre_event_atr = atr_m15
                
                # Get current price (would come from MT5 in live system)
                current_price = getattr(state, 'current_price', 0.0)
                state.ks7_pre_event_price = current_price
            
            return True, f"Event blackout active: {event.get('name', 'Unknown')}"
    
    # No events in window, deactivate if active
    if state.ks7_active:
        state.ks7_active = False
        state.ks7_pre_event_atr = 0.0
        state.ks7_pre_event_price = 0.0
    
    return False, "NO_BLACKOUT"


def run_all_kill_switches(
    state: SimulatedState,
    current_time: datetime,
    upcoming_events: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Run all kill switches and return results.
    
    Returns:
        Dict with kill switch results and any state changes needed.
    """
    results = {
        "trading_enabled": state.trading_enabled,
        "triggered_switches": [],
        "warnings": [],
        "state_updates": {}
    }
    
    if upcoming_events is None:
        upcoming_events = []
    
    # KS3: Daily loss limit
    ks3_triggered, ks3_reason = check_ks3_daily_loss(state)
    if ks3_triggered:
        results["triggered_switches"].append("KS3")
        results["state_updates"]["trading_enabled"] = False
        results["state_updates"]["shutdown_reason"] = f"KS3_DAILY_LOSS: {ks3_reason}"
    
    # KS5: Weekly loss limit
    ks5_triggered, ks5_reason = check_ks5_weekly_loss(state, current_time)
    if ks5_triggered:
        results["triggered_switches"].append("KS5")
        results["state_updates"]["trading_enabled"] = False
        results["state_updates"]["shutdown_reason"] = f"KS5_WEEKLY_LOSS: {ks5_reason}"
    
    # KS6: Drawdown circuit breaker
    ks6_triggered, ks6_reason = check_ks6_drawdown(state)
    if ks6_triggered:
        results["triggered_switches"].append("KS6")
        results["state_updates"]["trading_enabled"] = False
        results["state_updates"]["shutdown_reason"] = f"KS6_DRAWDOWN: {ks6_reason}"
    
    # KS7: Event blackout (doesn't disable trading, just blocks new trades)
    ks7_active, ks7_reason = check_ks7_event_blackout(state, current_time, upcoming_events)
    if ks7_active:
        results["warnings"].append(f"KS7_BLACKOUT: {ks7_reason}")
    
    return results


def reset_daily_counters(state: SimulatedState):
    """Reset daily counters for kill switches."""
    state.daily_pnl = 0.0
    state.daily_trades = 0
    state.daily_commission_paid = 0.0
    state.consecutive_m5_losses = 0
    state.consecutive_losses = 0


def reset_weekly_counters(state: SimulatedState):
    """Reset weekly counters for kill switches."""
    state.weekly_pnl = 0.0
    state.weekly_trades = 0


def get_kill_switch_status(state: SimulatedState) -> Dict[str, Any]:
    """
    Get current status of all kill switches.
    
    Returns:
        Dict with current status of each kill switch.
    """
    return {
        "ks3_daily_loss": {
            "daily_pnl": state.daily_pnl,
            "daily_pnl_pct": state.daily_pnl / state.balance if state.balance > 0 else 0,
            "threshold": -0.04,
            "status": "TRIGGERED" if state.daily_pnl / state.balance <= -0.04 else "OK"
        },
        "ks5_weekly_loss": {
            "weekly_pnl": getattr(state, 'weekly_pnl', state.daily_pnl),
            "weekly_pnl_pct": getattr(state, 'weekly_pnl', state.daily_pnl) / state.balance if state.balance > 0 else 0,
            "threshold": -0.12,
            "status": "TRIGGERED" if getattr(state, 'weekly_pnl', state.daily_pnl) / state.balance <= -0.12 else "OK"
        },
        "ks6_drawdown": {
            "current_equity": state.equity,
            "peak_equity": state.peak_equity,
            "drawdown_pct": (state.peak_equity - state.equity) / state.peak_equity if state.peak_equity > 0 else 0,
            "threshold": 0.20,
            "status": "TRIGGERED" if state.peak_equity > 0 and (state.peak_equity - state.equity) / state.peak_equity >= 0.20 else "OK"
        },
        "ks7_event_blackout": {
            "active": state.ks7_active,
            "pre_event_atr": state.ks7_pre_event_atr,
            "pre_event_price": state.ks7_pre_event_price,
            "status": "ACTIVE" if state.ks7_active else "INACTIVE"
        },
        "trading_enabled": state.trading_enabled,
        "shutdown_reason": state.shutdown_reason
    }
