"""
KS6 Auto-Reset Implementation for Backtesting Mode

This file contains the complete implementation of KS6 auto-reset functionality
that allows backtesting to continue after drawdown events while preserving
the emergency shutdown behavior for live trading.

Key Features:
- KS6 auto-resets only in backtest mode
- 24-hour cooldown after KS6 trigger
- Full event logging for post-analysis
- Equity peak reset (but not equity balance)
- Preserves live mode permanent shutdown behavior
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION ADDITIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
Add these to config.py:

# KS6 Auto-Reset Configuration (Backtest Only)
BACKTEST_KS6_AUTO_RESET = True        # Set False in live mode
BACKTEST_KS6_COOLDOWN_BARS = 96       # 96 x M15 bars = 24 hour cooldown
BACKTEST_MODE = True                  # Set False in live mode
"""

# ─────────────────────────────────────────────────────────────────────────────
# KS6 AUTO-RESET LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def should_enable_ks6_auto_reset() -> bool:
    """
    Check if KS6 auto-reset should be enabled.
    Only enabled in backtest mode with explicit flag.
    """
    try:
        import config
        return (getattr(config, 'BACKTEST_MODE', False) and 
                getattr(config, 'BACKTEST_KS6_AUTO_RESET', False))
    except ImportError:
        return False

def get_ks6_cooldown_bars() -> int:
    """Get KS6 cooldown period in bars."""
    try:
        import config
        return getattr(config, 'BACKTEST_KS6_COOLDOWN_BARS', 96)
    except ImportError:
        return 96

def handle_ks6_trigger_backtest(
    state: dict,
    bar_index: int,
    current_bar: dict,
    drawdown_pct: float
) -> dict:
    """
    Handle KS6 trigger in backtest mode with auto-reset logic.
    
    This function replaces permanent KS6 shutdown with auto-reset behavior
    while preserving all the important event data for analysis.
    
    Args:
        state: Current system state
        bar_index: Current bar index
        current_bar: Current bar data
        drawdown_pct: Current drawdown percentage
        
    Returns:
        Updated state after KS6 handling
    """
    
    if not should_enable_ks6_auto_reset():
        # Live mode - use existing permanent shutdown logic
        state['trading_enabled'] = False
        state['ks6_fired'] = True
        return state
    
    # Backtest mode - implement auto-reset logic
    
    # 1. Log KS6 event
    ks6_event = {
        'bar_index': bar_index,
        'date': current_bar.get('time', 'Unknown'),
        'current_equity': state.get('equity', 0),
        'drawdown_pct': drawdown_pct,
        'current_regime': state.get('regime', 'UNKNOWN'),
        'total_trades_so_far': len(state.get('closed_trades', [])),
        'peak_equity_before': state.get('peak_equity', 0),
        'equity_loss': state.get('peak_equity', 0) - state.get('equity', 0)
    }
    
    # Initialize ks6_events list if needed
    if 'ks6_events' not in state:
        state['ks6_events'] = []
    
    state['ks6_events'].append(ks6_event)
    
    # 2. Close all open positions at current bar close price
    # This simulates emergency liquidation
    open_positions = state.get('open_positions', {})
    for ticket, position in open_positions.items():
        # Create emergency close trade record
        emergency_close = {
            'ticket': ticket,
            'strategy': position.get('strategy', 'UNKNOWN'),
            'direction': position.get('direction', 'LONG'),
            'lot_size': position.get('lot_size', 0.01),
            'entry_price': position.get('entry_price', 0),
            'entry_time': position.get('entry_time', 'Unknown'),
            'exit_price': current_bar.get('close', 0),
            'exit_time': current_bar.get('time', 'Unknown'),
            'exit_type': 'KS6_EMERGENCY_CLOSE',
            'stop_price': position.get('stop_price', 0),
            'tp_price': position.get('tp_price', None),
            'pnl_net_dollars': 0,  # Will be calculated by execution engine
            'r_multiple': 0,
            'metadata': {
                'ks6_event': True,
                'original_exit_type': position.get('exit_type', 'UNKNOWN'),
                'emergency_close_bar': bar_index
            }
        }
        
        # Add to closed trades
        if 'closed_trades' not in state:
            state['closed_trades'] = []
        state['closed_trades'].append(emergency_close)
    
    # 3. Cancel all pending orders
    state['pending_orders'] = []
    
    # 4. Reset equity peak (new drawdown baseline)
    # CRITICAL: Reset peak to current equity, NOT to initial equity
    current_equity = state.get('equity', 0)
    state['peak_equity'] = current_equity
    
    # 5. Reset KS6 trigger flag
    state['ks6_fired'] = False
    
    # 6. Re-enable trading after cooldown
    state['trading_enabled'] = True
    
    # 7. Reset consecutive losses
    state['consecutive_losses'] = 0
    
    # 8. Reset daily P&L
    state['daily_pnl'] = 0.0
    
    # 9. Set cooldown period
    cooldown_bars = get_ks6_cooldown_bars()
    state['ks6_cooldown_until_bar'] = bar_index + cooldown_bars
    
    # 10. CRITICAL: Do NOT reset current_equity - losses are real
    # The equity balance remains unchanged, only the peak resets
    
    # 11. Reset position tracking
    state['open_positions'] = {}
    state['trend_family_occupied'] = False
    state['reversal_family_occupied'] = False
    state['independent_lanes_occupied'] = {}
    
    # 12. Log the event
    print(f"KS6 AUTO-RESET: Drawdown {drawdown_pct:.1f}% at bar {bar_index}")
    print(f"  - Equity: ${current_equity:,.2f} (loss: ${ks6_event['equity_loss']:,.2f})")
    print(f"  - Regime: {ks6_event['current_regime']}")
    print(f"  - Trades so far: {ks6_event['total_trades_so_far']}")
    print(f"  - Cooldown: {cooldown_bars} bars ({cooldown_bars/4:.1f} hours)")
    print(f"  - New peak equity: ${current_equity:,.2f}")
    
    return state

def check_ks6_cooldown(state: dict, bar_index: int) -> bool:
    """
    Check if we're in KS6 cooldown period.
    
    Args:
        state: Current system state
        bar_index: Current bar index
        
    Returns:
        True if in cooldown (should skip bar processing), False otherwise
    """
    if not should_enable_ks6_auto_reset():
        return False
    
    cooldown_until = state.get('ks6_cooldown_until_bar', -1)
    
    if bar_index < cooldown_until:
        # Still in cooldown
        remaining_bars = cooldown_until - bar_index
        if remaining_bars % 24 == 0:  # Log every 6 hours
            print(f"KS6 Cooldown: {remaining_bars} bars remaining ({remaining_bars/4:.1f} hours)")
        return True
    
    return False

def add_ks6_to_final_results(results: dict, state: dict) -> dict:
    """
    Add KS6 statistics to final backtest results.
    
    Args:
        results: Final backtest results
        state: Final system state
        
    Returns:
        Updated results with KS6 statistics
    """
    ks6_events = state.get('ks6_events', [])
    
    # Calculate KS6 statistics
    ks6_stats = {
        'ks6_total_count': len(ks6_events),
        'ks6_events': ks6_events,
        'ks6_auto_reset_enabled': should_enable_ks6_auto_reset(),
        'ks6_cooldown_bars': get_ks6_cooldown_bars() if should_enable_ks6_auto_reset() else 0
    }
    
    # Add regime analysis for KS6 events
    if ks6_events:
        regime_counts = {}
        for event in ks6_events:
            regime = event.get('current_regime', 'UNKNOWN')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        ks6_stats['ks6_regime_breakdown'] = regime_counts
        
        # Calculate average drawdown at KS6 events
        avg_drawdown = sum(event.get('drawdown_pct', 0) for event in ks6_events) / len(ks6_events)
        ks6_stats['ks6_avg_drawdown_pct'] = avg_drawdown
        
        # Calculate total equity loss from KS6 events
        total_equity_loss = sum(event.get('equity_loss', 0) for event in ks6_events)
        ks6_stats['ks6_total_equity_loss'] = total_equity_loss
    
    # Add to results
    results['ks6_analysis'] = ks6_stats
    
    return results

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINTS
# ─────────────────────────────────────────────────────────────────────────────

"""
INTEGRATION INSTRUCTIONS:

1. In config.py, add:
   BACKTEST_KS6_AUTO_RESET = True
   BACKTEST_KS6_COOLDOWN_BARS = 96
   BACKTEST_MODE = True

2. In state.py, add to build_initial_state():
   state['ks6_events'] = []
   state['ks6_cooldown_until_bar'] = -1

3. In backtest engine main loop, add at top:
   if check_ks6_cooldown(state, bar_index):
       continue

4. Replace KS6 trigger handling with:
   if ks6_triggered:
       state = handle_ks6_trigger_backtest(state, bar_index, current_bar, drawdown_pct)

5. In final results, add:
   results = add_ks6_to_final_results(results, state)

6. In risk_updated/kill_switches.py, modify check_ks6_drawdown():
   from backtest.risk.kill_switches import check_ks6_drawdown, _should_use_ks6_auto_reset
   if _should_use_ks6_auto_reset():
       # Return trigger info but don't set trading_enabled = False
       return True, f"KS6: Drawdown {drawdown_pct:.2f}% exceeds threshold {KS6_DRAWDOWN_LIMIT_PCT:.2f}%"
   else:
       # Live mode - permanent shutdown
       state['trading_enabled'] = False
       return True, f"KS6: Drawdown {drawdown_pct:.2f}% exceeds threshold {KS6_DRAWDOWN_LIMIT_PCT:.2f}%"
"""

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

def example_ks6_integration():
    """
    Example of how KS6 auto-reset integrates into backtest engine.
    """
    
    # Example state initialization
    state = {
        'equity': 10000,
        'peak_equity': 12000,
        'trading_enabled': True,
        'ks6_fired': False,
        'ks6_events': [],
        'ks6_cooldown_until_bar': -1,
        'open_positions': {},
        'pending_orders': [],
        'closed_trades': []
    }
    
    # Example bar processing loop
    for bar_index in range(1000):
        current_bar = {'time': f'2024-01-01 {bar_index:04d}', 'close': 2000.0}
        
        # Check cooldown first
        if check_ks6_cooldown(state, bar_index):
            continue
        
        # Check for KS6 trigger (simplified)
        current_equity = state['equity']
        peak_equity = state['peak_equity']
        drawdown_pct = (peak_equity - current_equity) / peak_equity * 100
        
        if drawdown_pct >= 20:  # KS6 threshold
            state = handle_ks6_trigger_backtest(state, bar_index, current_bar, drawdown_pct)
        
        # Continue with normal bar processing...
    
    # Generate final results
    results = {'summary': {'total_trades': 100}}
    results = add_ks6_to_final_results(results, state)
    
    print(f"KS6 Events: {results['ks6_analysis']['ks6_total_count']}")
    for event in results['ks6_analysis']['ks6_events']:
        print(f"  - {event['date']}: {event['drawdown_pct']:.1f}% drawdown in {event['current_regime']}")

if __name__ == "__main__":
    example_ks6_integration()
