"""
Enhanced Monte Carlo Simulation

Advanced Monte Carlo analysis with strategy breakdown and correlation analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger("backtest.enhanced_monte_carlo")


@dataclass
class MonteCarloResults:
    """Results from enhanced Monte Carlo simulation."""
    base_statistics: Dict[str, Any]
    strategy_breakdown: Dict[str, Dict[str, Any]]
    correlation_analysis: Dict[str, Any]
    regime_analysis: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    recommendations: List[str]


class EnhancedMonteCarlo:
    """
    Enhanced Monte Carlo simulation for risk-of-ruin analysis.
    
    Features:
    - Strategy-by-strategy breakdown
    - Correlation analysis between strategies
    - Regime-based performance analysis
    - Enhanced risk metrics
    - Actionable recommendations
    """
    
    def __init__(self, trades: List[Dict[str, Any]], config: Dict[str, Any]):
        self.trades = trades
        self.config = config
        self.rng = np.random.default_rng()
    
    def run_full_analysis(self, n_simulations: int = 10000) -> MonteCarloResults:
        """Run comprehensive Monte Carlo analysis."""
        logger.info(f"Running enhanced Monte Carlo with {n_simulations} simulations")
        
        # Extract trade data
        trade_data = self._prepare_trade_data()
        
        # Base Monte Carlo simulation
        base_stats = self._run_base_simulation(trade_data, n_simulations)
        
        # Strategy breakdown
        strategy_breakdown = self._analyze_strategies(trade_data, n_simulations)
        
        # Correlation analysis
        correlation_analysis = self._analyze_correlations(trade_data)
        
        # Regime analysis
        regime_analysis = self._analyze_regime_performance(trade_data)
        
        # Risk metrics
        risk_metrics = self._calculate_enhanced_risk_metrics(trade_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            base_stats, strategy_breakdown, correlation_analysis, risk_metrics
        )
        
        return MonteCarloResults(
            base_statistics=base_stats,
            strategy_breakdown=strategy_breakdown,
            correlation_analysis=correlation_analysis,
            regime_analysis=regime_analysis,
            risk_metrics=risk_metrics,
            recommendations=recommendations
        )
    
    def _prepare_trade_data(self) -> pd.DataFrame:
        """Prepare trade data for analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Convert timestamps
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Calculate returns
        if 'pnl' in df.columns:
            df['return_pct'] = df['pnl'] / df['pnl'].sum() * 100
        
        return df
    
    def _run_base_simulation(self, trade_data: pd.DataFrame, n_simulations: int) -> Dict[str, Any]:
        """Run base Monte Carlo simulation."""
        if trade_data.empty:
            return {"error": "No trades to simulate"}
        
        returns = trade_data['pnl'].values
        initial_balance = self.config.get('initial_balance', 10000)
        
        # Run simulations
        final_equities = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Randomize trade sequence
            shuffled_returns = self.rng.permutation(returns)
            
            # Calculate equity curve
            equity_curve = initial_balance + np.cumsum(shuffled_returns)
            final_equities.append(equity_curve[-1])
            
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_drawdowns.append(np.max(drawdown))
        
        # Calculate statistics
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            "probability_of_profit": np.mean(final_equities > initial_balance),
            "expected_return": np.mean(final_equities - initial_balance),
            "return_std": np.std(final_equities - initial_balance),
            "max_drawdown_mean": np.mean(max_drawdowns),
            "max_drawdown_std": np.std(max_drawdowns),
            "worst_case": np.min(final_equities),
            "best_case": np.max(final_equities),
            "risk_of_ruin": np.mean(final_equities < initial_balance * 0.5),
            "sharpe_ratio": np.mean(final_equities - initial_balance) / np.std(final_equities) if np.std(final_equities) > 0 else 0,
        }
    
    def _analyze_strategies(self, trade_data: pd.DataFrame, n_simulations: int) -> Dict[str, Any]:
        """Analyze each strategy separately."""
        if 'strategy' not in trade_data.columns:
            return {"error": "Strategy column not found"}
        
        strategy_results = {}
        
        for strategy in trade_data['strategy'].unique():
            strategy_trades = trade_data[trade_data['strategy'] == strategy]
            
            if len(strategy_trades) < 5:  # Skip strategies with too few trades
                continue
            
            returns = strategy_trades['pnl'].values
            
            # Monte Carlo for this strategy
            final_equities = []
            for _ in range(min(n_simulations // 10, 1000)):  # Fewer sims per strategy
                shuffled_returns = self.rng.permutation(returns)
                equity_curve = np.cumsum(shuffled_returns)
                final_equities.append(equity_curve[-1])
            
            strategy_results[strategy] = {
                "trade_count": len(strategy_trades),
                "mean_return": np.mean(returns),
                "return_std": np.std(returns),
                "win_rate": np.mean(returns > 0),
                "profit_factor": np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else float('inf'),
                "max_consecutive_wins": self._calculate_max_consecutive(returns, True),
                "max_consecutive_losses": self._calculate_max_consecutive(returns, False),
                "monte_carlo_stats": {
                    "probability_of_profit": np.mean(final_equities > 0),
                    "expected_value": np.mean(final_equities),
                    "volatility": np.std(final_equities),
                }
            }
        
        return strategy_results
    
    def _analyze_correlations(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between strategies."""
        if 'strategy' not in trade_data.columns:
            return {"error": "Strategy column not found"}
        
        # Create strategy return matrix
        strategy_returns = {}
        
        for strategy in trade_data['strategy'].unique():
            strategy_trades = trade_data[trade_data['strategy'] == strategy]
            if len(strategy_trades) >= 5:
                strategy_returns[strategy] = strategy_trades['pnl'].values
        
        if len(strategy_returns) < 2:
            return {"error": "Need at least 2 strategies for correlation analysis"}
        
        # Calculate correlation matrix
        strategies = list(strategy_returns.keys())
        n_strategies = len(strategies)
        correlation_matrix = np.zeros((n_strategies, n_strategies))
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Calculate correlation (need same length arrays)
                    min_len = min(len(strategy_returns[strategy1]), len(strategy_returns[strategy2]))
                    corr = np.corrcoef(
                        strategy_returns[strategy1][:min_len],
                        strategy_returns[strategy2][:min_len]
                    )[0, 1]
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        return {
            "strategies": strategies,
            "correlation_matrix": correlation_matrix.tolist(),
            "highest_correlation": self._find_highest_correlation(strategies, correlation_matrix),
            "lowest_correlation": self._find_lowest_correlation(strategies, correlation_matrix),
            "diversification_benefit": self._calculate_diversification_benefit(correlation_matrix),
        }
    
    def _analyze_regime_performance(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        if 'regime_at_entry' not in trade_data.columns:
            return {"error": "Regime column not found"}
        
        regime_results = {}
        
        for regime in trade_data['regime_at_entry'].unique():
            regime_trades = trade_data[trade_data['regime_at_entry'] == regime]
            
            if len(regime_trades) < 3:
                continue
            
            returns = regime_trades['pnl'].values
            
            regime_results[regime] = {
                "trade_count": len(regime_trades),
                "mean_return": np.mean(returns),
                "return_std": np.std(returns),
                "win_rate": np.mean(returns > 0),
                "profit_factor": np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else float('inf'),
                "regime_efficiency": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            }
        
        return regime_results
    
    def _calculate_enhanced_risk_metrics(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced risk metrics."""
        if trade_data.empty:
            return {"error": "No trade data"}
        
        returns = trade_data['pnl'].values
        
        # Basic metrics
        total_return = np.sum(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Enhanced metrics
        var_95 = np.percentile(returns, 5)  # 5% VaR
        var_99 = np.percentile(returns, 1)  # 1% VaR
        cvar_95 = np.mean(returns[returns <= var_95])  # Conditional VaR
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (if we have drawdown data)
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            "total_return": total_return,
            "mean_return": mean_return,
            "return_std": std_return,
            "sharpe_ratio": mean_return / std_return if std_return > 0 else 0,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "downside_deviation": downside_deviation,
            "skewness": self._calculate_skewness(returns),
            "kurtosis": self._calculate_kurtosis(returns),
        }
    
    def _generate_recommendations(
        self,
        base_stats: Dict[str, Any],
        strategy_breakdown: Dict[str, Any],
        correlation_analysis: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        if base_stats.get("risk_of_ruin", 0) > 0.1:
            recommendations.append("⚠️ High risk of ruin (>10%). Consider reducing position size or adding more strategies.")
        
        if base_stats.get("max_drawdown_mean", 0) > 0.2:
            recommendations.append("⚠️ High maximum drawdown (>20%). Implement stricter risk management.")
        
        # Strategy-based recommendations
        best_strategy = None
        worst_strategy = None
        best_performance = -float('inf')
        worst_performance = float('inf')
        
        for strategy, stats in strategy_breakdown.items():
            if isinstance(stats, dict) and 'mean_return' in stats:
                if stats['mean_return'] > best_performance:
                    best_performance = stats['mean_return']
                    best_strategy = strategy
                if stats['mean_return'] < worst_performance:
                    worst_performance = stats['mean_return']
                    worst_strategy = strategy
        
        if best_strategy and worst_strategy:
            recommendations.append(f"📈 Best performing strategy: {best_strategy} (avg return: {best_performance:.2f})")
            recommendations.append(f"📉 Worst performing strategy: {worst_strategy} (avg return: {worst_performance:.2f})")
        
        # Correlation-based recommendations
        if correlation_analysis.get("diversification_benefit", 0) < 0.3:
            recommendations.append("🔄 Low diversification benefit. Consider adding uncorrelated strategies.")
        
        # Risk-adjusted return recommendations
        if risk_metrics.get("sharpe_ratio", 0) < 0.5:
            recommendations.append("📊 Low Sharpe ratio (<0.5). Focus on improving risk-adjusted returns.")
        
        if risk_metrics.get("sortino_ratio", 0) < 0.7:
            recommendations.append("📉 Low Sortino ratio (<0.7). Reduce downside risk or improve upside capture.")
        
        return recommendations
    
    # Helper methods
    def _calculate_max_consecutive(self, returns: np.ndarray, wins: bool) -> int:
        """Calculate maximum consecutive wins or losses."""
        if wins:
            mask = returns > 0
        else:
            mask = returns < 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for is_consecutive in mask:
            if is_consecutive:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _find_highest_correlation(self, strategies: List[str], matrix: np.ndarray) -> Dict[str, Any]:
        """Find highest correlation (excluding diagonal)."""
        n = len(strategies)
        max_corr = -1
        best_pair = None
        
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] > max_corr:
                    max_corr = matrix[i, j]
                    best_pair = (strategies[i], strategies[j])
        
        return {
            "pair": best_pair,
            "correlation": max_corr
        }
    
    def _find_lowest_correlation(self, strategies: List[str], matrix: np.ndarray) -> Dict[str, Any]:
        """Find lowest correlation (excluding diagonal)."""
        n = len(strategies)
        min_corr = 1
        best_pair = None
        
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] < min_corr:
                    min_corr = matrix[i, j]
                    best_pair = (strategies[i], strategies[j])
        
        return {
            "pair": best_pair,
            "correlation": min_corr
        }
    
    def _calculate_diversification_benefit(self, correlation_matrix: np.ndarray) -> float:
        """Calculate diversification benefit (average off-diagonal correlation)."""
        n = correlation_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        # Calculate average off-diagonal correlation
        off_diagonal_sum = np.sum(correlation_matrix) - np.trace(correlation_matrix)
        off_diagonal_count = n * (n - 1)
        avg_correlation = off_diagonal_sum / off_diagonal_count if off_diagonal_count > 0 else 0
        
        # Diversification benefit = 1 - average correlation
        return 1 - avg_correlation
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown)
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((returns - mean) / std) ** 3)
        return skew
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        kurt = np.mean(((returns - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurt
