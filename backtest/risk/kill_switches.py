"""
backtest/risk_updated/kill_switches.py — Updated Kill Switches for Backtesting

Updated to match current live system thresholds and logic:
- KS3: Daily loss > -7% (was -4%)
- KS4: 4 consecutive losses (was 6) with 3-trade countdown (was 5)
- KS5: Weekly loss > -15% (was -12%)
- KS6: Drawdown > 20% from peak (unchanged but implementation updated)
- KS7: Economic event blackout 45min pre/20min post (implementation updated)
- KS2: Spread > 2.5× 24h median (implementation updated)
- KS1: Stop modification protection (implementation updated)

All thresholds match config.py values exactly.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal, ROUND_DOWN

# Import updated components
from backtest.strategies import STRATEGY_CONFIGS

# ─────────────────────────────────────────────────────────────────────────────
# KILL SWITCH THRESHOLDS - Updated to match config.py
# ─────────────────────────────────────────────────────────────────────────────

KS3_DAILY_LOSS_LIMIT_PCT = -0.070    # -7% (was -4%)
KS4_LOSS_STREAK_COUNT = 4               # 4 consecutive losses (was 6)
KS4_REDUCED_TRADES = 3                  # 3 trades with 50% size (was 5)
KS5_WEEKLY_LOSS_LIMIT_PCT = -0.150   # -15% (was -12%)
KS6_DRAWDOWN_LIMIT_PCT = 0.20         # 20% drawdown
KS7_PRE_MINUTES = 45                    # 45 minutes before event
KS7_POST_MINUTES = 20                   # 20 minutes after event
KS2_SPREAD_MULTIPLIER = 2.5             # 2.5× median spread

# ─────────────────────────────────────────────────────────────────────────────
# KILL SWITCH IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def check_ks1_stop_modification(
    order_type: str, 
    new_stop: float, 
    original_stop: float,
    direction: str
) -> Tuple[bool, str]:
    """
    KS1: Stop modification protection.
    
    Blocks any stop modification that moves stop against trade direction.
    """
    if direction == "LONG" and new_stop < original_stop:
        return True, f"KS1: Cannot move LONG stop down from {original_stop} to {new_stop}"
    elif direction == "SHORT" and new_stop > original_stop:
        return True, f"KS1: Cannot move SHORT stop up from {original_stop} to {new_stop}"
    
    return False, "OK"

def check_ks2_spread_guard(
    current_spread: float, 
    median_spread_24h: float
) -> Tuple[bool, str]:
    """
    KS2: Spread guard.
    
    Blocks orders when spread > 2.5× 24h median.
    """
    if median_spread_24h <= 0:
        return False, "NO_MEDIAN_SPREAD"
    
    spread_ratio = current_spread / median_spread_24h
    if spread_ratio > KS2_SPREAD_MULTIPLIER:
        return True, f"KS2: Spread {current_spread:.1f} is {spread_ratio:.1f}× median ({median_spread_24h:.1f})"
    
    return False, "OK"

def check_ks3_daily_loss(
    daily_pnl: float, 
    starting_balance: float
) -> Tuple[bool, str]:
    """
    KS3: Daily loss limit kill switch.
    
    Threshold: -7% daily loss
    Action: trading_enabled = False for remainder of day
    """
    if starting_balance <= 0:
        return False, "INVALID_BALANCE"
    
    daily_pnl_pct = daily_pnl / starting_balance
    
    if daily_pnl_pct <= KS3_DAILY_LOSS_LIMIT_PCT:
        return True, f"KS3: Daily loss {daily_pnl_pct:.2%} exceeds threshold {KS3_DAILY_LOSS_LIMIT_PCT:.2%}"
    
    return False, "OK"

def check_ks4_loss_streak(
    consecutive_losses: int,
    ks4_reduced_trades_remaining: int = 0
) -> Tuple[bool, str, int]:
    """
    KS4: Loss streak kill switch with countdown.
    
    Triggers: 4 consecutive losses
    Action: Reduce size by 50% for next 3 trades
    """
    if consecutive_losses >= KS4_LOSS_STREAK_COUNT and ks4_reduced_trades_remaining == 0:
        return True, f"KS4: {consecutive_losses} consecutive losses - reducing size for {KS4_REDUCED_TRADES} trades", KS4_REDUCED_TRADES
    
    # Decrement countdown if active
    if ks4_reduced_trades_remaining > 0:
        new_countdown = ks4_reduced_trades_remaining - 1
        return False, f"KS4: Size reduction active - {new_countdown} trades remaining", new_countdown
    
    return False, "OK", ks4_reduced_trades_remaining

def check_ks5_weekly_loss(
    weekly_pnl: float, 
    starting_balance: float
) -> Tuple[bool, str]:
    """
    KS5: Weekly loss limit kill switch.
    
    Threshold: -15% weekly loss
    Action: trading_enabled = False for remainder of week
    """
    if starting_balance <= 0:
        return False, "INVALID_BALANCE"
    
    weekly_pnl_pct = weekly_pnl / starting_balance
    
    if weekly_pnl_pct <= KS5_WEEKLY_LOSS_LIMIT_PCT:
        return True, f"KS5: Weekly loss {weekly_pnl_pct:.2%} exceeds threshold {KS5_WEEKLY_LOSS_LIMIT_PCT:.2%}"
    
    return False, "OK"

def check_ks6_drawdown(
    current_equity: float, 
    peak_equity: float,
    state: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    KS6: Drawdown circuit breaker.
    
    In live mode: Threshold: 20% drawdown from 30-day rolling peak
    Action: Emergency shutdown (permanent)
    
    In backtest mode with auto-reset: 
    Action: Emergency close + 24-hour cooldown + auto-resume
    """
    threshold = 0.20  # Default: 20%
    
    if peak_equity <= 0:
        return False, "NO_PEAK_EQUITY"
    
    drawdown_pct = (peak_equity - current_equity) / peak_equity
    
    if drawdown_pct >= threshold:
        # Check if we should use auto-reset logic
        if state and _should_use_ks6_auto_reset():
            return True, f"KS6_AUTO_RESET: Drawdown {drawdown_pct:.3f} exceeds threshold {threshold:.3f}%"
        else:
            return True, f"KS6: Drawdown {drawdown_pct:.3f} exceeds threshold {threshold:.3f}%"
    
    return False, "OK"

def _should_use_ks6_auto_reset() -> bool:
    """Check if KS6 auto-reset should be used."""
    try:
        import config
        return (getattr(config, 'BACKTEST_MODE', False) and 
                getattr(config, 'BACKTEST_KS6_AUTO_RESET', False))
    except ImportError:
        return False

def check_ks7_event_blackout(
    current_time: datetime,
    upcoming_events: List[Dict[str, Any]]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    KS7: Economic event blackout windows.
    
    Blocks new trades during high-impact events:
      - 45 minutes before event
      - 20 minutes after event
    Stores pre-event ATR/price for R3 volatility filter.
    """
    if not upcoming_events:
        return False, "NO_EVENTS", {}
    
    blackout_start = None
    blackout_end = None
    pre_event_atr = None
    pre_event_price = None
    
    for event in upcoming_events:
        event_time = event.get('time')
        if not event_time:
            continue
        
        if not isinstance(event_time, datetime):
            event_time = datetime.fromisoformat(str(event_time))
        
        # Check if we're in pre-event blackout
        pre_event_start = event_time - timedelta(minutes=KS7_PRE_MINUTES)
        post_event_end = event_time + timedelta(minutes=KS7_POST_MINUTES)
        
        if pre_event_start <= current_time <= event_time:
            # Pre-event blackout
            blackout_start = pre_event_start
            blackout_end = event_time
            # Store pre-event data for R3
            pre_event_atr = event.get('pre_event_atr', 0.0)
            pre_event_price = event.get('pre_event_price', 0.0)
            break
        elif event_time < current_time <= post_event_end:
            # Post-event blackout
            blackout_start = event_time
            blackout_end = post_event_end
            break
    
    if blackout_start and blackout_end:
        return True, f"KS7: Event blackout active until {blackout_end}", {
            'blackout_start': blackout_start,
            'blackout_end': blackout_end,
            'pre_event_atr': pre_event_atr,
            'pre_event_price': pre_event_price,
            'event_name': next((e.get('name', 'Unknown') for e in upcoming_events if e.get('time') == event_time), 'Unknown')
        }
    
    return False, "NO_BLACKOUT", {}

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE KILL SWITCH CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class KillSwitchChecker:
    """
    Comprehensive kill switch checker for backtesting.
    """
    
    def __init__(self):
        self.consecutive_losses = 0
        self.ks4_reduced_trades_remaining = 0
        self.peak_equity = 0.0
        self.week_start_balance = 0.0
        self.day_start_balance = 0.0
        
    def check_all_kill_switches(
        self,
        state: Dict[str, Any],
        current_time: datetime,
        upcoming_events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run all kill switches and return comprehensive results.
        """
        results = {
            'trading_enabled': True,
            'triggered_switches': [],
            'warnings': [],
            'size_multiplier': 1.0,
            'state_updates': {}
        }
        
        if upcoming_events is None:
            upcoming_events = []
        
        equity = state.get('equity', state.get('balance', 0))
        daily_pnl = state.get('daily_pnl', 0)
        weekly_pnl = state.get('weekly_pnl', daily_pnl)
        current_spread = state.get('current_spread', 0)
        median_spread = state.get('median_spread_24h', 0)
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # KS1: Stop modification (checked during order modifications)
        # KS2: Spread guard
        ks2_triggered, ks2_reason = check_ks2_spread_guard(current_spread, median_spread)
        if ks2_triggered:
            results['triggered_switches'].append('KS2')
            results['warnings'].append(ks2_reason)
        
        # KS3: Daily loss limit
        ks3_triggered, ks3_reason = check_ks3_daily_loss(daily_pnl, self.day_start_balance)
        if ks3_triggered:
            results['triggered_switches'].append('KS3')
            results['state_updates']['trading_enabled'] = False
            results['state_updates']['shutdown_reason'] = ks3_reason
        
        # KS4: Loss streak with countdown
        ks4_triggered, ks4_reason, new_countdown = check_ks4_loss_streak(
            self.consecutive_losses, self.ks4_reduced_trades_remaining
        )
        if ks4_triggered:
            results['triggered_switches'].append('KS4')
            results['state_updates']['ks4_reduced_trades_remaining'] = new_countdown
            results['size_multiplier'] *= 0.5
        elif new_countdown != self.ks4_reduced_trades_remaining:
            results['state_updates']['ks4_reduced_trades_remaining'] = new_countdown
            if new_countdown > 0:
                results['size_multiplier'] *= 0.5
        
        # KS5: Weekly loss limit
        ks5_triggered, ks5_reason = check_ks5_weekly_loss(weekly_pnl, self.week_start_balance)
        if ks5_triggered:
            results['triggered_switches'].append('KS5')
            results['state_updates']['trading_enabled'] = False
            results['state_updates']['shutdown_reason'] = ks5_reason
        
        # KS6: Drawdown circuit breaker
        ks6_triggered, ks6_reason = check_ks6_drawdown(equity, self.peak_equity)
        if ks6_triggered:
            results['triggered_switches'].append('KS6')
            results['state_updates']['trading_enabled'] = False
            results['state_updates']['shutdown_reason'] = ks6_reason
        
        # KS7: Event blackout
        ks7_triggered, ks7_reason, ks7_data = check_ks7_event_blackout(current_time, upcoming_events)
        if ks7_triggered:
            results['triggered_switches'].append('KS7')
            results['warnings'].append(ks7_reason)
            results['state_updates'].update(ks7_data)
        
        # Update internal state
        self.ks4_reduced_trades_remaining = results['state_updates'].get('ks4_reduced_trades_remaining', self.ks4_reduced_trades_remaining)
        
        return results
    
    def update_trade_result(self, pnl: float) -> None:
        """Update consecutive losses counter."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def reset_daily(self, balance: float) -> None:
        """Reset daily state."""
        self.day_start_balance = balance
    
    def reset_weekly(self, balance: float) -> None:
        """Reset weekly state."""
        self.week_start_balance = balance
