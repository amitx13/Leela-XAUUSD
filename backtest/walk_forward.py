"""
backtest/walk_forward_updated.py — Updated Walk-Forward Analysis

Walk-forward analysis for out-of-sample testing with rolling windows.
Integrates with existing analytics modules for comprehensive analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import logging

# Import existing analytics
try:
    from backtest.analytics.strategy_analytics import StrategyAnalytics
    from backtest.analytics.risk_analytics import RiskAnalytics
    from backtest.analytics.heat_maps import HeatMapGenerator
    from backtest.monte_carlo import EnhancedMonteCarlo
except ImportError:
    StrategyAnalytics = None
    RiskAnalytics = None
    HeatMapGenerator = None
    EnhancedMonteCarlo = None

# Import backtest engine
from backtest.engine import BacktestEngine

logger = logging.getLogger("backtest.walk_forward")

class WalkForwardAnalyzer:
    """
    Walk-forward analyzer for out-of-sample testing.
    
    Features:
    - Rolling window analysis
    - Out-of-sample performance tracking
    - Strategy stability analysis
    - Parameter optimization tracking
    - Integration with existing analytics
    """
    
    def __init__(self):
        self.results_history = []
    
    def run_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        train_months: int = 3,
        test_months: int = 1,
        strategies: Optional[List[str]] = None,
        initial_balance: float = 10000.0,
        slippage_points: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run comprehensive walk-forward analysis.
        
        Args:
            start_date: Overall start date
            end_date: Overall end date
            train_months: Training window in months
            test_months: Testing window in months
            strategies: List of strategies to test
            initial_balance: Starting balance
            slippage_points: Slippage in points
            
        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Starting walk-forward analysis: {train_months}m train, {test_months}m test")
        
        # Calculate window periods
        windows = self._calculate_windows(start_date, end_date, train_months, test_months)
        
        if not windows:
            return {'error': 'Insufficient data for walk-forward analysis'}
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        
        # Run analysis for each window
        window_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}: {train_start.date()} to {test_end.date()}")
            
            # Run training period
            train_result = self._run_backtest(
                train_start, train_end, strategies, initial_balance, slippage_points
            )
            
            # Run testing period
            test_result = self._run_backtest(
                test_start, test_end, strategies, 
                train_result.get('final_balance', initial_balance), slippage_points
            )
            
            # Analyze window
            window_analysis = self._analyze_window(
                train_result, test_result, train_start, test_end
            )
            
            window_results.append(window_analysis)
        
        # Compile overall results
        return self._compile_results(window_results, train_months, test_months)
    
    def _calculate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        train_months: int,
        test_months: int
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Calculate walk-forward windows."""
        windows = []
        current_start = start_date
        
        while True:
            train_end = current_start + timedelta(days=train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_months * 30)
            
            if test_end > end_date:
                break
            
            windows.append((current_start, train_end, test_start, test_end))
            current_start = test_start
        
        return windows
    
    def _run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        strategies: Optional[List[str]],
        balance: float,
        slippage_points: float
    ) -> Dict[str, Any]:
        """Run backtest for specified period."""
        try:
            engine = BacktestEngine(
                start_date=start_date,
                end_date=end_date,
                initial_balance=balance,
                slippage_points=slippage_points,
                strategies=strategies
            )
            
            return engine.run()
            
        except Exception as e:
            logger.error(f"Backtest failed for {start_date} to {end_date}: {e}")
            return {'error': str(e), 'start_date': start_date, 'end_date': end_date}
    
    def _analyze_window(
        self,
        train_result: Dict[str, Any],
        test_result: Dict[str, Any],
        test_start: datetime,
        test_end: datetime
    ) -> Dict[str, Any]:
        """Analyze individual walk-forward window."""
        window_analysis = {
            'test_start': test_start,
            'test_end': test_end,
            'train_result': train_result.get('summary', {}),
            'test_result': test_result.get('summary', {}),
            'stability_metrics': {}
        }
        
        # Calculate stability metrics
        train_pnl_pct = train_result.get('summary', {}).get('pnl_pct', 0)
        test_pnl_pct = test_result.get('summary', {}).get('pnl_pct', 0)
        
        window_analysis['stability_metrics'] = {
            'pnl_retention': test_pnl_pct / train_pnl_pct if train_pnl_pct != 0 else 0,
            'performance_drop': train_pnl_pct - test_pnl_pct,
            'positive_period': test_pnl_pct > 0,
            'win_rate_retention': (
                test_result.get('summary', {}).get('win_rate', 0) / 
                train_result.get('summary', {}).get('win_rate', 1)
            )
        }
        
        # Add analytics if available
        test_trades = test_result.get('trades', [])
        if test_trades and StrategyAnalytics:
            try:
                strategy_analytics = StrategyAnalytics(test_trades)
                window_analysis['strategy_analytics'] = strategy_analytics.get_performance_summary()
            except Exception as e:
                logger.warning(f"Strategy analytics failed: {e}")
        
        return window_analysis
    
    def _compile_results(
        self,
        window_results: List[Dict[str, Any]],
        train_months: int,
        test_months: int
    ) -> Dict[str, Any]:
        """Compile overall walk-forward results."""
        if not window_results:
            return {'error': 'No valid windows analyzed'}
        
        # Extract test results
        test_results = [w['test_result'] for w in window_results if 'test_result' in w]
        
        if not test_results:
            return {'error': 'No valid test results found'}
        
        # Calculate overall metrics
        total_pnl = sum(r.get('total_pnl', 0) for r in test_results)
        total_trades = sum(r.get('total_trades', 0) for r in test_results)
        total_wins = sum(r.get('winning_trades', 0) for r in test_results)
        
        # Stability analysis
        positive_periods = sum(1 for r in test_results if r.get('total_pnl', 0) > 0)
        consistency_score = (positive_periods / len(test_results)) * 100
        
        # Performance by period
        period_performance = []
        for i, result in enumerate(test_results):
            period_performance.append({
                'period': i + 1,
                'pnl_pct': result.get('pnl_pct', 0),
                'win_rate': result.get('win_rate', 0),
                'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0)
            })
        
        return {
            'success': True,
            'num_periods': len(window_results),
            'train_months': train_months,
            'test_months': test_months,
            'total_oos_pnl': total_pnl,
            'oos_win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'oos_sharpe': np.mean([r.get('sharpe_ratio', 0) for r in test_results]),
            'oos_max_dd': np.mean([r.get('max_drawdown_pct', 0) for r in test_results]),
            'positive_periods': positive_periods,
            'consistency_score': consistency_score,
            'period_performance': period_performance,
            'window_details': window_results
        }
