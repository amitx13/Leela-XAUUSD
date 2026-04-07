"""
backtest/analytics_updated.py — Updated Analytics Engine for Backtesting

Updated analytics engine with comprehensive performance analysis.
Integrates with existing analytics modules for enhanced functionality.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging

# Import existing analytics modules
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

logger = logging.getLogger("backtest.analytics")

class AnalyticsEngine:
    """
    Updated analytics engine matching live Truth Engine.
    """

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.summary_stats: Dict[str, Any] = {}

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of backtest results.
        Integrates with existing analytics modules when available.
        """
        self.trades = results.get('trades', [])
        self.equity_curve = results.get('equity_curve', [])
        self.summary_stats = results.get('summary', {})
        
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        # Base analysis (always available)
        analysis = {
            'basic_metrics': self._calculate_basic_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'strategy_analysis': self._analyze_strategies(),
            'regime_analysis': self._analyze_regimes(),
            'session_analysis': self._analyze_sessions(),
            'monthly_analysis': self._analyze_monthly_performance(),
            'edge_decay': self._analyze_edge_decay(),
            'conviction_analysis': self._analyze_conviction_levels(),
            'correlation_analysis': self._analyze_correlations(),
            'benchmark_comparison': self._compare_to_benchmark()
        }
        
        # Enhanced analytics (if modules available)
        if StrategyAnalytics:
            try:
                strategy_analytics = StrategyAnalytics(self.trades)
                analysis['enhanced_strategy'] = strategy_analytics.get_performance_summary()
            except Exception as e:
                logger.warning(f"Enhanced strategy analytics failed: {e}")
        
        if RiskAnalytics:
            try:
                risk_analytics = RiskAnalytics(self.trades, self.summary_stats.get('initial_balance', 10000))
                analysis['enhanced_risk'] = risk_analytics.get_portfolio_analysis()
            except Exception as e:
                logger.warning(f"Enhanced risk analytics failed: {e}")
        
        if HeatMapGenerator:
            try:
                heat_maps = HeatMapGenerator(self.trades)
                analysis['heat_maps'] = heat_maps.generate_all_heat_maps()
            except Exception as e:
                logger.warning(f"Heat map generation failed: {e}")
        
        return analysis

    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl_net_dollars'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl_net_dollars'] for t in self.trades)
        total_commission = sum(t['commission'] for t in self.trades)
        
        # Calculate expectancy
        wins = [t['pnl_net_dollars'] for t in self.trades if t['pnl_net_dollars'] > 0]
        losses = [abs(t['pnl_net_dollars']) for t in self.trades if t['pnl_net_dollars'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # R-multiple analysis
        r_multiples = [t['r_multiple'] for t in self.trades]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        # Trade duration analysis
        durations = []
        for trade in self.trades:
            entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
            exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
            duration = (exit_time - entry_time).total_seconds() / 3600  # hours
            durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'net_pnl': total_pnl - total_commission,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'avg_r_multiple': avg_r,
            'avg_trade_duration_hours': avg_duration,
            'profit_factor': sum(wins) / sum(losses) if losses else float('inf')
        }

    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics."""
        if not self.equity_curve:
            return {}
        
        # Extract equity values
        equity_values = [ep['equity'] for ep in self.equity_curve]
        initial_equity = equity_values[0] if equity_values else self.summary_stats.get('initial_balance', 10000)
        
        # Calculate drawdown
        peak = initial_equity
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        # Calculate returns
        returns = [equity_values[i] / equity_values[i-1] - 1 for i in range(1, len(equity_values))]
        
        # Sharpe ratio (annualized)
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = avg_return / std_return if std_return > 0 else 0
            sharpe_annual = sharpe * np.sqrt(252 * 24 * 12)  # M5 data: 252 days * 24 hours * 12 bars per hour
        else:
            sharpe_annual = 0
        
        # Sortino ratio (downside deviation only)
        if len(returns) > 1:
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                sortino = avg_return / downside_std if downside_std > 0 else 0
                sortino_annual = sortino * np.sqrt(252 * 24 * 12)
            else:
                sortino_annual = float('inf')
        else:
            sortino_annual = 0
        
        # VaR calculations
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
        else:
            var_95 = var_99 = 0
        
        # Calmar ratio (annual return / max drawdown)
        total_return = (equity_values[-1] - initial_equity) / initial_equity
        years = len(equity_curve) / (252 * 24 * 12)  # Approximate years of M5 data
        annual_return = total_return / years if years > 0 else 0
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        return {
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_duration_bars': max_dd_duration,
            'sharpe_ratio': sharpe_annual,
            'sortino_ratio': sortino_annual,
            'calmar_ratio': calmar,
            'var_95_pct': var_95 * 100,
            'var_99_pct': var_99 * 100,
            'annual_return_pct': annual_return * 100
        }

    def _analyze_strategies(self) -> Dict[str, Any]:
        """Analyze performance by strategy."""
        strategy_stats = {}
        
        # Group trades by strategy
        for trade in self.trades:
            strategy = trade['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trades': [],
                    'pnl': 0,
                    'wins': 0,
                    'losses': 0,
                    'r_multiples': []
                }
            
            strategy_stats[strategy]['trades'].append(trade)
            strategy_stats[strategy]['pnl'] += trade['pnl_net_dollars']
            strategy_stats[strategy]['r_multiples'].append(trade['r_multiple'])
            
            if trade['pnl_net_dollars'] > 0:
                strategy_stats[strategy]['wins'] += 1
            else:
                strategy_stats[strategy]['losses'] += 1
        
        # Calculate metrics for each strategy
        for strategy, stats in strategy_stats.items():
            total_trades = len(stats['trades'])
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            avg_r = np.mean(stats['r_multiples']) if stats['r_multiples'] else 0
            
            # Calculate expectancy
            wins = [t['pnl_net_dollars'] for t in stats['trades'] if t['pnl_net_dollars'] > 0]
            losses = [abs(t['pnl_net_dollars']) for t in stats['trades'] if t['pnl_net_dollars'] < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            strategy_stats[strategy].update({
                'total_trades': total_trades,
                'win_rate': win_rate * 100,
                'avg_r_multiple': avg_r,
                'expectancy': expectancy,
                'profit_factor': sum(wins) / sum(losses) if losses else float('inf')
            })
        
        return strategy_stats

    def _analyze_regimes(self) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        regime_stats = {}
        
        for trade in self.trades:
            regime = trade.get('regime_at_entry', 'UNKNOWN')
            if regime not in regime_stats:
                regime_stats[regime] = {
                    'trades': [],
                    'pnl': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            regime_stats[regime]['trades'].append(trade)
            regime_stats[regime]['pnl'] += trade['pnl_net_dollars']
            
            if trade['pnl_net_dollars'] > 0:
                regime_stats[regime]['wins'] += 1
            else:
                regime_stats[regime]['losses'] += 1
        
        # Calculate metrics
        for regime, stats in regime_stats.items():
            total_trades = len(stats['trades'])
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            
            regime_stats[regime].update({
                'total_trades': total_trades,
                'win_rate': win_rate * 100,
                'avg_pnl': stats['pnl'] / total_trades if total_trades > 0 else 0
            })
        
        return regime_stats

    def _analyze_sessions(self) -> Dict[str, Any]:
        """Analyze performance by trading session."""
        session_stats = {}
        
        for trade in self.trades:
            session = trade.get('session_at_entry', 'UNKNOWN')
            if session not in session_stats:
                session_stats[session] = {
                    'trades': [],
                    'pnl': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            session_stats[session]['trades'].append(trade)
            session_stats[session]['pnl'] += trade['pnl_net_dollars']
            
            if trade['pnl_net_dollars'] > 0:
                session_stats[session]['wins'] += 1
            else:
                session_stats[session]['losses'] += 1
        
        # Calculate metrics
        for session, stats in session_stats.items():
            total_trades = len(stats['trades'])
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            
            session_stats[session].update({
                'total_trades': total_trades,
                'win_rate': win_rate * 100,
                'avg_pnl': stats['pnl'] / total_trades if total_trades > 0 else 0
            })
        
        return session_stats

    def _analyze_monthly_performance(self) -> Dict[str, Any]:
        """Analyze monthly performance."""
        monthly_stats = {}
        
        for trade in self.trades:
            entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
            month_key = entry_time.strftime('%Y-%m')
            
            if month_key not in monthly_stats:
                monthly_stats[month_key] = {
                    'trades': [],
                    'pnl': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            monthly_stats[month_key]['trades'].append(trade)
            monthly_stats[month_key]['pnl'] += trade['pnl_net_dollars']
            
            if trade['pnl_net_dollars'] > 0:
                monthly_stats[month_key]['wins'] += 1
            else:
                monthly_stats[month_key]['losses'] += 1
        
        # Calculate metrics
        for month, stats in monthly_stats.items():
            total_trades = len(stats['trades'])
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            
            monthly_stats[month].update({
                'total_trades': total_trades,
                'win_rate': win_rate * 100,
                'avg_pnl': stats['pnl'] / total_trades if total_trades > 0 else 0
            })
        
        return monthly_stats

    def _analyze_edge_decay(self) -> Dict[str, Any]:
        """Analyze edge decay over time."""
        if len(self.trades) < 50:
            return {'error': 'Insufficient trades for edge decay analysis'}
        
        # Rolling window analysis
        window_size = 50
        rolling_expectancies = []
        trade_indices = []
        
        for i in range(window_size, len(self.trades) + 1):
            window_trades = self.trades[i-window_size:i]
            
            # Calculate expectancy for window
            wins = [t['pnl_net_dollars'] for t in window_trades if t['pnl_net_dollars'] > 0]
            losses = [abs(t['pnl_net_dollars']) for t in window_trades if t['pnl_net_dollars'] < 0]
            
            if wins and losses:
                win_rate = len(wins) / len(window_trades)
                avg_win = np.mean(wins)
                avg_loss = np.mean(losses)
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            else:
                expectancy = 0
            
            rolling_expectancies.append(expectancy)
            trade_indices.append(i)
        
        # Calculate decay metrics
        if len(rolling_expectancies) >= 2:
            initial_expectancy = rolling_expectancies[0]
            final_expectancy = rolling_expectancies[-1]
            decay_rate = (initial_expectancy - final_expectancy) / initial_expectancy if initial_expectancy != 0 else 0
            
            # Linear regression to detect trend
            x = np.arange(len(rolling_expectancies))
            y = np.array(rolling_expectancies)
            slope, intercept = np.polyfit(x, y, 1)
            
            return {
                'initial_expectancy': initial_expectancy,
                'final_expectancy': final_expectancy,
                'decay_rate_pct': decay_rate * 100,
                'trend_slope': slope,
                'rolling_expectancies': rolling_expectancies,
                'trade_indices': trade_indices
            }
        
        return {'error': 'Insufficient data for trend analysis'}

    def _analyze_conviction_levels(self) -> Dict[str, Any]:
        """Analyze performance by conviction level."""
        conviction_stats = {}
        
        for trade in self.trades:
            conviction = trade.get('conviction_level', 'STANDARD')
            if conviction not in conviction_stats:
                conviction_stats[conviction] = {
                    'trades': [],
                    'pnl': 0,
                    'wins': 0,
                    'losses': 0,
                    'r_multiples': []
                }
            
            conviction_stats[conviction]['trades'].append(trade)
            conviction_stats[conviction]['pnl'] += trade['pnl_net_dollars']
            conviction_stats[conviction]['r_multiples'].append(trade['r_multiple'])
            
            if trade['pnl_net_dollars'] > 0:
                conviction_stats[conviction]['wins'] += 1
            else:
                conviction_stats[conviction]['losses'] += 1
        
        # Calculate metrics
        for conviction, stats in conviction_stats.items():
            total_trades = len(stats['trades'])
            win_rate = stats['wins'] / total_trades if total_trades > 0 else 0
            avg_r = np.mean(stats['r_multiples']) if stats['r_multiples'] else 0
            
            conviction_stats[conviction].update({
                'total_trades': total_trades,
                'win_rate': win_rate * 100,
                'avg_r_multiple': avg_r,
                'avg_pnl': stats['pnl'] / total_trades if total_trades > 0 else 0
            })
        
        return conviction_stats

    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between strategies."""
        # Create strategy return matrix
        strategies = list(set(trade['strategy'] for trade in self.trades))
        if len(strategies) < 2:
            return {'error': 'Need at least 2 strategies for correlation analysis'}
        
        # Create daily returns by strategy
        strategy_returns = {}
        
        for strategy in strategies:
            strategy_trades = [t for t in self.trades if t['strategy'] == strategy]
            daily_returns = {}
            
            for trade in strategy_trades:
                entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                date_key = entry_time.strftime('%Y-%m-%d')
                
                if date_key not in daily_returns:
                    daily_returns[date_key] = 0
                daily_returns[date_key] += trade['pnl_net_dollars']
            
            # Convert to returns (percentage of initial balance)
            initial_balance = self.summary_stats.get('initial_balance', 10000)
            strategy_returns[strategy] = {
                date: pnl / initial_balance for date, pnl in daily_returns.items()
            }
        
        # Create correlation matrix
        all_dates = set()
        for returns in strategy_returns.values():
            all_dates.update(returns.keys())
        
        correlation_matrix = {}
        
        for strategy1 in strategies:
            correlation_matrix[strategy1] = {}
            returns1 = strategy_returns[strategy1]
            
            for strategy2 in strategies:
                returns2 = strategy_returns[strategy2]
                
                # Get common dates
                common_dates = all_dates & set(returns1.keys()) & set(returns2.keys())
                
                if len(common_dates) < 10:  # Need at least 10 data points
                    correlation_matrix[strategy1][strategy2] = 0
                    continue
                
                # Calculate correlation
                values1 = [returns1[date] for date in common_dates]
                values2 = [returns2[date] for date in common_dates]
                
                if len(values1) > 1 and len(values2) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlation_matrix[strategy1][strategy2] = correlation if not np.isnan(correlation) else 0
                else:
                    correlation_matrix[strategy1][strategy2] = 0
        
        return {
            'correlation_matrix': correlation_matrix,
            'strategies': strategies
        }

    def _compare_to_benchmark(self) -> Dict[str, Any]:
        """Compare performance to buy-and-hold benchmark."""
        if not self.equity_curve:
            return {'error': 'No equity curve data for benchmark comparison'}
        
        # Simple benchmark: buy and hold XAUUSD
        # This is simplified - in reality would need XAUUSD price data
        initial_equity = self.equity_curve[0]['equity']
        final_equity = self.equity_curve[-1]['equity']
        
        strategy_return = (final_equity - initial_equity) / initial_equity
        
        # Assume XAUUSD returned 15% over the period (placeholder)
        benchmark_return = 0.15
        
        alpha = strategy_return - benchmark_return
        
        return {
            'strategy_return_pct': strategy_return * 100,
            'benchmark_return_pct': benchmark_return * 100,
            'alpha_pct': alpha * 100,
            'outperformance': alpha > 0
        }
