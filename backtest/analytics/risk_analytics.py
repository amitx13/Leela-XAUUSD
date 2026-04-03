"""
Risk Analytics

Advanced risk analysis and portfolio metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("backtest.risk_analytics")


class RiskAnalytics:
    """
    Advanced risk analytics for portfolio management.
    
    Features:
    - Portfolio heat map analysis
    - Correlation analysis between strategies
    - Risk contribution analysis
    - Stress testing scenarios
    - Portfolio optimization recommendations
    """
    
    def __init__(self, trades: List[Dict[str, Any]], initial_balance: float = 10000.0):
        self.trades = trades
        self.initial_balance = initial_balance
        self.df = self._prepare_trade_data()
        
        logger.info(f"Risk analytics initialized with {len(trades)} trades")
    
    def _prepare_trade_data(self) -> pd.DataFrame:
        """Prepare trade data for risk analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Convert timestamps
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Calculate equity curve
        df['equity_before'] = self.initial_balance + df['pnl'].cumsum().shift(1).fillna(self.initial_balance)
        df['equity_after'] = self.initial_balance + df['pnl'].cumsum()
        
        return df
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        if self.df.empty:
            return {"error": "No trade data available"}
        
        # Basic portfolio metrics
        total_trades = len(self.df)
        total_pnl = self.df['pnl'].sum()
        final_balance = self.initial_balance + total_pnl
        
        # Equity curve analysis
        self.df['cumulative_pnl'] = self.df['pnl'].cumsum()
        self.df['running_peak'] = self.df['cumulative_pnl'].expanding().max()
        self.df['drawdown'] = (self.df['running_peak'] - self.df['cumulative_pnl']) / self.df['running_peak']
        self.df['drawdown_pct'] = self.df['drawdown'] * 100
        
        max_drawdown = self.df['drawdown'].max()
        max_drawdown_pct = max_drawdown * 100
        
        # Calculate volatility metrics
        daily_returns = self._calculate_daily_returns()
        if daily_returns:
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
            downside_volatility = np.std([r for r in daily_returns if r < 0]) * np.sqrt(252)
        else:
            volatility = downside_volatility = 0
        
        # Risk-adjusted returns
        total_return = total_pnl / self.initial_balance
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        sortino_ratio = total_return / downside_volatility if downside_volatility > 0 else 0
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk calculations
        all_returns = self.df['pnl'] / self.initial_balance
        if len(all_returns) > 0:
            var_95 = np.percentile(all_returns, 5)  # 5% VaR
            var_99 = np.percentile(all_returns, 1)  # 1% VaR
            cvar_95 = np.mean(all_returns[all_returns <= var_95])  # Conditional VaR
        else:
            var_95 = var_99 = cvar_95 = 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'final_balance': final_balance,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'volatility_annualized': volatility,
            'downside_volatility': downside_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'avg_trade_size': self.df.get('lots', pd.Series([0.01])).mean(),
            'largest_loss': self.df['pnl'].min(),
            'largest_win': self.df['pnl'].max(),
        }
    
    def _calculate_daily_returns(self) -> List[float]:
        """Calculate daily returns for volatility analysis."""
        if self.df.empty or 'entry_time' not in self.df.columns:
            return []
        
        # Group trades by date
        self.df['entry_date'] = pd.to_datetime(self.df['entry_time']).dt.date
        daily_pnl = self.df.groupby('entry_date')['pnl'].sum()
        
        # Calculate daily returns
        daily_returns = []
        running_equity = self.initial_balance
        
        for date in sorted(daily_pnl.index):
            daily_return = daily_pnl[date] / running_equity
            daily_returns.append(daily_return)
            running_equity += daily_pnl[date]
        
        return daily_returns
    
    def calculate_risk_contribution(self) -> Dict[str, Any]:
        """Calculate risk contribution by strategy and regime."""
        if self.df.empty:
            return {"error": "No trade data available"}
        
        risk_contribution = {}
        
        # By strategy
        if 'strategy' in self.df.columns:
            strategy_risk = {}
            for strategy in self.df['strategy'].unique():
                strategy_trades = self.df[self.df['strategy'] == strategy]
                
                if len(strategy_trades) > 0:
                    strategy_pnl = strategy_trades['pnl'].sum()
                    strategy_var = strategy_trades['pnl'].var()
                    strategy_contribution = abs(strategy_pnl) / abs(self.df['pnl'].sum()) * 100
                    
                    strategy_risk[strategy] = {
                        'pnl': strategy_pnl,
                        'variance': strategy_var,
                        'contribution_pct': strategy_contribution,
                        'risk_adjusted_return': strategy_pnl / np.sqrt(strategy_var) if strategy_var > 0 else 0,
                        'trade_count': len(strategy_trades),
                    }
            
            risk_contribution['by_strategy'] = strategy_risk
        
        # By regime
        if 'regime_at_entry' in self.df.columns:
            regime_risk = {}
            for regime in self.df['regime_at_entry'].unique():
                regime_trades = self.df[self.df['regime_at_entry'] == regime]
                
                if len(regime_trades) > 0:
                    regime_pnl = regime_trades['pnl'].sum()
                    regime_var = regime_trades['pnl'].var()
                    regime_contribution = abs(regime_pnl) / abs(self.df['pnl'].sum()) * 100
                    
                    regime_risk[regime] = {
                        'pnl': regime_pnl,
                        'variance': regime_var,
                        'contribution_pct': regime_contribution,
                        'risk_adjusted_return': regime_pnl / np.sqrt(regime_var) if regime_var > 0 else 0,
                        'trade_count': len(regime_trades),
                    }
            
            risk_contribution['by_regime'] = regime_risk
        
        return risk_contribution
    
    def calculate_correlation_matrix(self) -> Dict[str, Any]:
        """Calculate correlation matrix between strategies."""
        if self.df.empty or 'strategy' not in self.df.columns:
            return {"error": "Strategy data not available"}
        
        # Create strategy return series
        strategy_returns = {}
        for strategy in self.df['strategy'].unique():
            strategy_trades = self.df[self.df['strategy'] == strategy]
            
            if len(strategy_trades) > 1:
                # Calculate strategy returns over time
                strategy_pnl = strategy_trades.sort_values('entry_time')['pnl']
                strategy_returns[strategy] = strategy_pnl.tolist()
        
        if len(strategy_returns) < 2:
            return {"error": "Need at least 2 strategies with multiple trades"}
        
        # Calculate correlation matrix
        strategies = list(strategy_returns.keys())
        n = len(strategies)
        correlation_matrix = np.zeros((n, n))
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Align returns by length
                    returns1 = strategy_returns[strategy1]
                    returns2 = strategy_returns[strategy2]
                    min_len = min(len(returns1), len(returns2))
                    
                    if min_len > 1:
                        corr = np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[i, j] = 0.0
        
        # Find highest and lowest correlations
        upper_triangle = np.triu(correlation_matrix, k=1)
        highest_corr = np.max(upper_triangle)
        lowest_corr = np.min(upper_triangle[upper_triangle > 0])
        
        return {
            'strategies': strategies,
            'correlation_matrix': correlation_matrix.tolist(),
            'highest_correlation': highest_corr,
            'lowest_correlation': lowest_corr,
            'average_correlation': np.mean(upper_triangle[upper_triangle > 0]) if np.any(upper_triangle > 0) else 0,
            'diversification_ratio': 1 - np.mean(upper_triangle[upper_triangle > 0]) if np.any(upper_triangle > 0) else 0,
        }
    
    def generate_stress_test_scenarios(self) -> Dict[str, Any]:
        """Generate stress test scenarios and analyze portfolio resilience."""
        if self.df.empty:
            return {"error": "No trade data available"}
        
        # Extract trade characteristics for stress testing
        avg_win = self.df[self.df['pnl'] > 0]['pnl'].mean() if len(self.df[self.df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(self.df[self.df['pnl'] < 0]['pnl'].mean()) if len(self.df[self.df['pnl'] < 0]) > 0 else 0
        win_rate = (self.df['pnl'] > 0).mean()
        
        scenarios = {}
        
        # Scenario 1: Doubled loss rate
        scenarios['doubled_loss_rate'] = {
            'description': 'Win rate drops by 50%',
            'impact': self._simulate_scenario_impact(win_rate * 0.5, avg_win, avg_loss),
            'probability': 'Low probability but possible during regime changes',
        }
        
        # Scenario 2: Tripled average loss
        scenarios['tripled_avg_loss'] = {
            'description': 'Average loss triples',
            'impact': self._simulate_scenario_impact(win_rate, avg_win, avg_loss * 3),
            'probability': 'Possible during high volatility periods',
        }
        
        # Scenario 3: Maximum drawdown doubles
        current_max_dd = self.df['drawdown'].max()
        scenarios['doubled_max_drawdown'] = {
            'description': 'Maximum drawdown doubles',
            'impact': self._simulate_scenario_impact(win_rate, avg_win, avg_loss, current_max_dd * 2),
            'probability': 'Market crash or strategy failure',
        }
        
        # Scenario 4: Consecutive losing streak
        max_consecutive_losses = self._calculate_max_consecutive_losses()
        scenarios['extended_losing_streak'] = {
            'description': 'Consecutive losses extend to 10 trades',
            'impact': self._simulate_scenario_impact(0.1, avg_win, avg_loss * 10),
            'probability': 'Strategy logic failure or adverse market conditions',
        }
        
        return scenarios
    
    def _simulate_scenario_impact(
        self, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float, 
        max_drawdown: float = 0
    ) -> Dict[str, Any]:
        """Simulate impact of stress scenario."""
        total_trades = len(self.df)
        
        # Calculate new metrics under stress
        stressed_pnl = (avg_win * win_rate * total_trades) - (avg_loss * (1 - win_rate) * total_trades)
        stressed_final_balance = self.initial_balance + stressed_pnl
        stressed_return = stressed_pnl / self.initial_balance
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'total_pnl': stressed_pnl,
            'final_balance': stressed_final_balance,
            'total_return': stressed_return,
            'impact_severity': 'HIGH' if abs(stressed_return) > 0.2 else 'MEDIUM' if abs(stressed_return) > 0.1 else 'LOW',
        }
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses from trade data."""
        if self.df.empty or 'pnl' not in self.df.columns:
            return 0
        
        losses = (self.df['pnl'] < 0).astype(int)
        consecutive = 0
        max_consecutive = 0
        
        for loss in losses:
            if loss:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if self.df.empty:
            return ["No trade data available for risk analysis"]
        
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        # Drawdown recommendations
        if portfolio_metrics.get('max_drawdown_pct', 0) > 20:
            recommendations.append("🚨 CRITICAL: Maximum drawdown exceeds 20%. Implement emergency position sizing limits.")
        
        elif portfolio_metrics.get('max_drawdown_pct', 0) > 15:
            recommendations.append("⚠️ High drawdown detected (>15%). Consider reducing position size by 25%.")
        
        # Sharpe ratio recommendations
        if portfolio_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("📉 Low Sharpe ratio (<0.5). Review strategy selection and risk management.")
        
        elif portfolio_metrics.get('sharpe_ratio', 0) > 2.0:
            recommendations.append("🚨 Excellent Sharpe ratio (>2.0). Consider increasing position size cautiously.")
        
        # Sortino ratio recommendations
        if portfolio_metrics.get('sortino_ratio', 0) < 0.3:
            recommendations.append("📊 Poor downside protection (Sortino <0.3). Implement tighter stop-losses.")
        
        # Volatility recommendations
        if portfolio_metrics.get('volatility_annualized', 0) > 0.3:
            recommendations.append("📈 High portfolio volatility (>30%). Consider reducing exposure during high volatility periods.")
        
        # VaR recommendations
        if portfolio_metrics.get('var_95', 0) < -0.05:  # More than 5% loss in 95% of cases
            recommendations.append("⚠️ High Value at Risk. Implement position sizing based on VaR.")
        
        # Correlation recommendations
        correlation_analysis = self.calculate_correlation_matrix()
        if correlation_analysis.get('average_correlation', 0) > 0.7:
            recommendations.append("🔄 High strategy correlation (>0.7). Consider adding uncorrelated strategies.")
        
        return recommendations
    
    def generate_comprehensive_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk analytics report."""
        return {
            'portfolio_metrics': self.calculate_portfolio_metrics(),
            'risk_contribution': self.calculate_risk_contribution(),
            'correlation_analysis': self.calculate_correlation_matrix(),
            'stress_test_scenarios': self.generate_stress_test_scenarios(),
            'recommendations': self.generate_risk_recommendations(),
        }
