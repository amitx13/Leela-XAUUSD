"""
Strategy Analytics

Comprehensive strategy-by-strategy performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("backtest.strategy_analytics")


class StrategyAnalytics:
    """
    Comprehensive strategy performance analytics.
    
    Features:
    - Strategy-by-strategy breakdown
    - Performance metrics by regime
    - Session-based analysis
    - Risk-adjusted returns
    - Win/loss distribution analysis
    - Trade duration analysis
    """
    
    def __init__(self, trades: List[Dict[str, Any]]):
        self.trades = trades
        self.df = self._prepare_trade_data()
        
        logger.info(f"Strategy analytics initialized with {len(trades)} trades")
    
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
        
        # Calculate additional metrics
        if 'entry_price' in df.columns and 'exit_price' in df.columns:
            df['price_change'] = df['exit_price'] - df['entry_price']
            df['price_change_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price'] * 100
        
        # Calculate trade duration
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['duration_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        
        # Calculate R-multiple if stop distance available
        if 'stop_distance' in df.columns and 'pnl' in df.columns:
            # Assume standard lot size for R calculation
            df['r_multiple'] = df['pnl'] / (df['stop_distance'] * 100)  # Rough approximation
        
        return df
    
    def generate_strategy_breakdown(self) -> Dict[str, Any]:
        """Generate comprehensive strategy-by-strategy breakdown."""
        if self.df.empty:
            return {"error": "No trade data available"}
        
        strategy_stats = {}
        
        for strategy in self.df['strategy'].unique():
            strategy_trades = self.df[self.df['strategy'] == strategy]
            
            if len(strategy_trades) == 0:
                continue
            
            # Basic metrics
            total_trades = len(strategy_trades)
            winning_trades = len(strategy_trades[strategy_trades['pnl'] > 0])
            losing_trades = len(strategy_trades[strategy_trades['pnl'] < 0])
            win_rate = winning_trades / total_trades
            
            # P&L metrics
            total_pnl = strategy_trades['pnl'].sum()
            gross_pnl = strategy_trades.get('pnl_gross', strategy_trades['pnl']).sum()
            total_commission = strategy_trades.get('commission', 0).sum()
            
            # Risk metrics
            avg_win = strategy_trades[strategy_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(strategy_trades[strategy_trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
            
            # Best and worst trades
            best_trade = strategy_trades.loc[strategy_trades['pnl'].idxmax()]
            worst_trade = strategy_trades.loc[strategy_trades['pnl'].idxmin()]
            
            # Duration analysis
            avg_duration = strategy_trades['duration_minutes'].mean()
            max_duration = strategy_trades['duration_minutes'].max()
            min_duration = strategy_trades['duration_minutes'].min()
            
            # Risk-adjusted metrics
            if 'r_multiple' in strategy_trades.columns:
                avg_r = strategy_trades['r_multiple'].mean()
                profit_factor = gross_pnl / abs(strategy_trades[strategy_trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
                
                # Sharpe-like ratio (simplified)
                returns = strategy_trades['pnl'] / strategy_trades['entry_price']  # Simplified
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                avg_r = 0
                profit_factor = 0
                sharpe = 0
            
            strategy_stats[strategy] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'gross_pnl': gross_pnl,
                'total_commission': total_commission,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade_pnl': best_trade['pnl'],
                'worst_trade_pnl': worst_trade['pnl'],
                'best_trade_r': best_trade.get('r_multiple', 0),
                'worst_trade_r': worst_trade.get('r_multiple', 0),
                'avg_duration': avg_duration,
                'max_duration': max_duration,
                'min_duration': min_duration,
                'avg_r_multiple': avg_r,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe,
                'avg_slippage': strategy_trades.get('slippage', pd.Series([0])).mean(),
                'partial_fills': strategy_trades.get('partial_fill', pd.Series([False])).sum(),
            }
        
        return strategy_stats
    
    def generate_regime_analysis(self) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        if self.df.empty or 'regime_at_entry' not in self.df.columns:
            return {"error": "Regime data not available"}
        
        regime_stats = {}
        
        for regime in self.df['regime_at_entry'].unique():
            regime_trades = self.df[self.df['regime_at_entry'] == regime]
            
            if len(regime_trades) == 0:
                continue
            
            # Regime-specific metrics
            total_trades = len(regime_trades)
            winning_trades = len(regime_trades[regime_trades['pnl'] > 0])
            win_rate = winning_trades / total_trades
            avg_pnl = regime_trades['pnl'].mean()
            total_pnl = regime_trades['pnl'].sum()
            
            regime_stats[regime] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'profit_factor': total_pnl / abs(regime_trades[regime_trades['pnl'] < 0]['pnl'].sum()) if len(regime_trades[regime_trades['pnl'] < 0]) > 0 else float('inf'),
                'avg_duration': regime_trades.get('duration_minutes', pd.Series([0])).mean(),
            }
        
        return regime_stats
    
    def generate_session_analysis(self) -> Dict[str, Any]:
        """Analyze performance by trading session."""
        if self.df.empty or 'session_at_entry' not in self.df.columns:
            return {"error": "Session data not available"}
        
        session_stats = {}
        
        # Map session to name
        def get_session_name(hour):
            if 0 <= hour < 13:
                return "Asian"
            elif 13 <= hour < 17:
                return "London"
            elif 17 <= hour < 22:
                return "NY"
            else:
                return "Off Hours"
        
        # Add session column
        if 'entry_time' in self.df.columns:
            self.df['session_name'] = self.df['entry_time'].dt.hour.apply(get_session_name)
        
        for session in self.df['session_name'].unique():
            session_trades = self.df[self.df['session_name'] == session]
            
            if len(session_trades) == 0:
                continue
            
            total_trades = len(session_trades)
            winning_trades = len(session_trades[session_trades['pnl'] > 0])
            win_rate = winning_trades / total_trades
            avg_pnl = session_trades['pnl'].mean()
            
            session_stats[session] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': session_trades['pnl'].sum(),
                'avg_duration': session_trades.get('duration_minutes', pd.Series([0])).mean(),
                'volatility': session_trades['price_change_pct'].std() if 'price_change_pct' in session_trades.columns else 0,
            }
        
        return session_stats
    
    def generate_risk_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive risk analysis."""
        if self.df.empty:
            return {"error": "No trade data available"}
        
        # Calculate running drawdown series
        self.df['cumulative_pnl'] = self.df['pnl'].cumsum()
        self.df['running_peak'] = self.df['cumulative_pnl'].expanding().max()
        self.df['drawdown'] = (self.df['running_peak'] - self.df['cumulative_pnl']) / self.df['running_peak']
        self.df['max_drawdown'] = self.df['drawdown'].expanding().max()
        
        # Risk metrics
        total_pnl = self.df['pnl'].sum()
        max_drawdown = self.df['max_drawdown'].max()
        
        # Calculate Value at Risk (VaR)
        returns = self.df['pnl']
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)  # 5% VaR
            var_99 = np.percentile(returns, 1)  # 1% VaR
            cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
        else:
            var_95 = var_99 = cvar_95 = 0
        
        # Calculate Calmar ratio
        calmar_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'downside_deviation': downside_deviation,
            'avg_trade_size': self.df.get('lots', pd.Series([0.01])).mean(),
            'largest_loss': self.df['pnl'].min(),
            'largest_win': self.df['pnl'].max(),
            'consecutive_losses': self._calculate_max_consecutive_losses(),
            'consecutive_wins': self._calculate_max_consecutive_wins(),
        }
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses."""
        if 'pnl' not in self.df.columns:
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
    
    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive wins."""
        if 'pnl' not in self.df.columns:
            return 0
        
        wins = (self.df['pnl'] > 0).astype(int)
        consecutive = 0
        max_consecutive = 0
        
        for win in wins:
            if win:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if self.df.empty:
            return ["No trade data available for analysis"]
        
        # Strategy recommendations
        strategy_breakdown = self.generate_strategy_breakdown()
        
        # Find best and worst performing strategies
        if isinstance(strategy_breakdown, dict) and strategy_breakdown:
            strategy_performance = {}
            for strategy, stats in strategy_breakdown.items():
                if isinstance(stats, dict) and 'sharpe_ratio' in stats:
                    strategy_performance[strategy] = stats['sharpe_ratio']
            
            if strategy_performance:
                best_strategy = max(strategy_performance, key=strategy_performance.get)
                worst_strategy = min(strategy_performance, key=strategy_performance.get)
                
                recommendations.append(f"🏆 Best performing strategy: {best_strategy} (Sharpe: {strategy_performance[best_strategy]:.3f})")
                recommendations.append(f"📉 Worst performing strategy: {worst_strategy} (Sharpe: {strategy_performance[worst_strategy]:.3f})")
        
        # Risk recommendations
        risk_analysis = self.generate_risk_analysis()
        
        if risk_analysis.get('max_drawdown_pct', 0) > 15:
            recommendations.append("⚠️ High maximum drawdown (>15%). Consider reducing position size or adding stop-loss protection.")
        
        if risk_analysis.get('consecutive_losses', 0) > 5:
            recommendations.append("📉 Long losing streak detected. Review strategy logic during adverse conditions.")
        
        if risk_analysis.get('calmar_ratio', 0) < 0.5:
            recommendations.append("📊 Low Calmar ratio (<0.5). Focus on improving risk-adjusted returns.")
        
        # Session recommendations
        session_analysis = self.generate_session_analysis()
        
        if isinstance(session_analysis, dict):
            asian_performance = session_analysis.get('Asian', {}).get('avg_pnl', 0)
            london_performance = session_analysis.get('London', {}).get('avg_pnl', 0)
            ny_performance = session_analysis.get('NY', {}).get('avg_pnl', 0)
            
            if asian_performance > london_performance and asian_performance > ny_performance:
                recommendations.append("🌅 Best performance in Asian session. Consider focusing on Asian session strategies.")
            elif london_performance > ny_performance:
                recommendations.append("🇬🇧 Best performance in London session. Consider focusing on London session strategies.")
            elif ny_performance > 0:
                recommendations.append("🇺🇸 Best performance in NY session. Consider focusing on NY session strategies.")
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        return {
            'strategy_breakdown': self.generate_strategy_breakdown(),
            'regime_analysis': self.generate_regime_analysis(),
            'session_analysis': self.generate_session_analysis(),
            'risk_analysis': self.generate_risk_analysis(),
            'recommendations': self.generate_recommendations(),
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        if self.df.empty:
            return {"error": "No data available"}
        
        return {
            'total_trades': len(self.df),
            'total_pnl': self.df['pnl'].sum(),
            'avg_pnl_per_trade': self.df['pnl'].mean(),
            'median_pnl': self.df['pnl'].median(),
            'std_pnl': self.df['pnl'].std(),
            'best_trade': self.df['pnl'].max(),
            'worst_trade': self.df['pnl'].min(),
            'win_rate': (self.df['pnl'] > 0).mean(),
            'avg_duration_hours': self.df.get('duration_minutes', pd.Series([0])).mean() / 60,
            'trades_per_month': len(self.df) / max(1, (self.df['exit_time'].max() - self.df['entry_time'].min()).days / 30),
        }
