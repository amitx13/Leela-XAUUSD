#!/usr/bin/env python3

from backtest.engine_enhanced import EnhancedBacktestEngine
from datetime import datetime

print('Testing enhanced engine run...')

# Test with minimal configuration
engine = EnhancedBacktestEngine(
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 2),  # Just 1 day
    initial_balance=1000.0,
    slippage_points=0.7,
    strategies=['S8_ATR_SPIKE'],  # Test with one strategy
    cache_dir='test_data'
)

try:
    # Try to run the backtest (will generate synthetic data)
    results = engine.run()
    print('✅ Engine run successful!')
    
    # Test results
    results.summary()
    print('✅ Generated {} trades'.format(len(results.trades)))
    print('✅ Final balance: ${:.2f}'.format(results.final_balance))
    
except Exception as e:
    print('❌ Error: {}'.format(e))
    import traceback
    traceback.print_exc()
