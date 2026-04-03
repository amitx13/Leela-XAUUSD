"""
Heat Maps Analytics

Visual analytics for strategy performance by session and time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("backtest.heat_maps")


class HeatMapGenerator:
    """
    Heat map generator for visual strategy analytics.
    
    Features:
    - Strategy performance by hour of day
    - Session-based heat maps
    - Day-of-week analysis
    - Performance by regime and time
    - Visual heat map data generation
    """
    
    def __init__(self, trades: List[Dict[str, Any]]):
        self.trades = trades
        self.df = self._prepare_trade_data()
        
        logger.info(f"Heat map analytics initialized with {len(trades)} trades")
    
    def _prepare_trade_data(self) -> pd.DataFrame:
        """Prepare trade data for heat map analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        
        # Convert timestamps
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Add time-based columns
        df['entry_hour'] = df['entry_time'].dt.hour
        df['entry_day_of_week'] = df['entry_time'].dt.dayofweek
        df['entry_session'] = df['entry_hour'].apply(self._get_session_name)
        
        return df
    
    def _get_session_name(self, hour: int) -> str:
        """Map hour to session name."""
        if 0 <= hour < 13:
            return "Asian"
        elif 13 <= hour < 17:
            return "London"
        elif 17 <= hour < 22:
            return "NY"
        else:
            return "Off Hours"
    
    def generate_hourly_heatmap(self, metric: str = 'pnl') -> Dict[str, Any]:
        """Generate heat map of strategy performance by hour."""
        if self.df.empty:
            return {"error": "No trade data available"}
        
        # Create pivot table: hour vs strategy
        heatmap_data = pd.pivot_table(
            data=self.df,
            values=metric,
            index='entry_hour',
            columns='strategy',
            aggfunc='sum' if metric == 'count' else 'mean',
            fill_value=0
        )
        
        # Convert to numpy array for heat map
        heatmap_array = heatmap_data.values
        strategies = heatmap_data.columns.tolist()
        hours = list(range(24))
        
        # Calculate statistics
        hourly_totals = heatmap_array.sum(axis=1)
        hourly_performance = {}
        
        for hour in range(24):
            hour_data = {}
            for i, strategy in enumerate(strategies):
                value = heatmap_array[hour, i]
                hour_data[strategy] = {
                    'value': value,
                    'percentile': np.percentile(heatmap_array[:, i], 75) if np.any(heatmap_array[:, i] != 0) else 0,
                    'performance': 'HIGH' if value >= np.percentile(heatmap_array[:, i], 90) else 
                                   'MEDIUM' if value >= np.percentile(heatmap_array[:, i], 50) else 'LOW'
                }
            
            hourly_performance[hour] = {
                'total_pnl': hourly_totals[hour],
                'strategies': hour_data,
                'best_strategy': strategies[np.argmax(heatmap_array[hour])] if np.any(heatmap_array[hour] != 0) else None,
                'best_performance': np.max(heatmap_array[hour]) if np.any(heatmap_array[hour] != 0) else 0,
            }
        
        return {
            'heatmap_data': heatmap_array.tolist(),
            'strategies': strategies,
            'hours': hours,
            'hourly_performance': hourly_performance,
            'best_hour': np.argmax(hourly_totals) if np.any(hourly_totals != 0) else None,
            'worst_hour': np.argmin(hourly_totals) if np.any(hourly_totals != 0) else None,
        }
    
    def generate_session_heatmap(self, metric: str = 'pnl') -> Dict[str, Any]:
        """Generate heat map of strategy performance by session."""
        if self.df.empty or 'entry_session' not in self.df.columns:
            return {"error": "Session data not available"}
        
        # Create pivot table: session vs strategy
        heatmap_data = pd.pivot_table(
            data=self.df,
            values=metric,
            index='entry_session',
            columns='strategy',
            aggfunc='sum' if metric == 'count' else 'mean',
            fill_value=0
        )
        
        sessions = ['Asian', 'London', 'NY', 'Off Hours']
        heatmap_array = heatmap_data.values
        strategies = heatmap_data.columns.tolist()
        
        # Calculate session statistics
        session_totals = heatmap_array.sum(axis=1)
        session_performance = {}
        
        for i, session in enumerate(sessions):
            session_data = {}
            for j, strategy in enumerate(strategies):
                value = heatmap_array[i, j]
                session_data[strategy] = {
                    'value': value,
                    'percentile': np.percentile(heatmap_array[:, j], 75) if np.any(heatmap_array[:, j] != 0) else 0,
                    'performance': 'HIGH' if value >= np.percentile(heatmap_array[:, j], 90) else 
                                   'MEDIUM' if value >= np.percentile(heatmap_array[:, j], 50) else 'LOW'
                }
            
            session_performance[session] = {
                'total_pnl': session_totals[i],
                'strategies': session_data,
                'best_strategy': strategies[np.argmax(heatmap_array[i])] if np.any(heatmap_array[i] != 0) else None,
                'best_performance': np.max(heatmap_array[i]) if np.any(heatmap_array[i] != 0) else 0,
            }
        
        return {
            'heatmap_data': heatmap_array.tolist(),
            'strategies': strategies,
            'sessions': sessions,
            'session_performance': session_performance,
            'best_session': sessions[np.argmax(session_totals)] if np.any(session_totals != 0) else None,
            'worst_session': sessions[np.argmin(session_totals)] if np.any(session_totals != 0) else None,
        }
    
    def generate_dayofweek_heatmap(self, metric: str = 'pnl') -> Dict[str, Any]:
        """Generate heat map of strategy performance by day of week."""
        if self.df.empty or 'entry_day_of_week' not in self.df.columns:
            return {"error": "Day of week data not available"}
        
        # Create pivot table: day vs strategy
        heatmap_data = pd.pivot_table(
            data=self.df,
            values=metric,
            index='entry_day_of_week',
            columns='strategy',
            aggfunc='sum' if metric == 'count' else 'mean',
            fill_value=0
        )
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_array = heatmap_data.values
        strategies = heatmap_data.columns.tolist()
        
        # Calculate day statistics
        day_totals = heatmap_array.sum(axis=1)
        day_performance = {}
        
        for i, day in enumerate(days):
            day_data = {}
            for j, strategy in enumerate(strategies):
                value = heatmap_array[i, j]
                day_data[strategy] = {
                    'value': value,
                    'percentile': np.percentile(heatmap_array[:, j], 75) if np.any(heatmap_array[:, j] != 0) else 0,
                    'performance': 'HIGH' if value >= np.percentile(heatmap_array[:, j], 90) else 
                                  'MEDIUM' if value >= np.percentile(heatmap_array[:, j], 50) else 'LOW'
                }
            
            day_performance[day] = {
                'total_pnl': day_totals[i],
                'strategies': day_data,
                'best_strategy': strategies[np.argmax(heatmap_array[i])] if np.any(heatmap_array[i] != 0) else None,
                'best_performance': np.max(heatmap_array[i]) if np.any(heatmap_array[i] != 0) else 0,
            }
        
        return {
            'heatmap_data': heatmap_array.tolist(),
            'strategies': strategies,
            'days': days,
            'day_performance': day_performance,
            'best_day': days[np.argmax(day_totals)] if np.any(day_totals != 0) else None,
            'worst_day': days[np.argmin(day_totals)] if np.any(day_totals != 0) else None,
        }
    
    def generate_regime_heatmap(self, metric: str = 'pnl') -> Dict[str, Any]:
        """Generate heat map of strategy performance by regime."""
        if self.df.empty or 'regime_at_entry' not in self.df.columns:
            return {"error": "Regime data not available"}
        
        # Create pivot table: regime vs strategy
        heatmap_data = pd.pivot_table(
            data=self.df,
            values=metric,
            index='regime_at_entry',
            columns='strategy',
            aggfunc='sum' if metric == 'count' else 'mean',
            fill_value=0
        )
        
        regimes = ['NO_TRADE', 'UNSTABLE', 'RANGING_CLEAR', 'WEAK_TRENDING', 'NORMAL_TRENDING']
        heatmap_array = heatmap_data.values
        strategies = heatmap_data.columns.tolist()
        
        # Calculate regime statistics
        regime_totals = heatmap_array.sum(axis=1)
        regime_performance = {}
        
        for i, regime in enumerate(regimes):
            regime_data = {}
            for j, strategy in enumerate(strategies):
                value = heatmap_array[i, j]
                regime_data[strategy] = {
                    'value': value,
                    'percentile': np.percentile(heatmap_array[:, j], 75) if np.any(heatmap_array[:, j] != 0) else 0,
                    'performance': 'HIGH' if value >= np.percentile(heatmap_array[:, j], 90) else 
                                   'MEDIUM' if value >= np.percentile(heatmap_array[:, j], 50) else 'LOW'
                }
            
            regime_performance[regime] = {
                'total_pnl': regime_totals[i],
                'strategies': regime_data,
                'best_strategy': strategies[np.argmax(heatmap_array[i])] if np.any(heatmap_array[i] != 0) else None,
                'best_performance': np.max(heatmap_array[i]) if np.any(heatmap_array[i] != 0) else 0,
            }
        
        return {
            'heatmap_data': heatmap_array.tolist(),
            'strategies': strategies,
            'regimes': regimes,
            'regime_performance': regime_performance,
            'best_regime': regimes[np.argmax(regime_totals)] if np.any(regime_totals != 0) else None,
            'worst_regime': regimes[np.argmin(regime_totals)] if np.any(regime_totals != 0) else None,
        }
    
    def generate_visual_heatmap_data(self, heatmap_type: str = 'hourly') -> Dict[str, Any]:
        """Generate data suitable for heat map visualization."""
        if heatmap_type == 'hourly':
            return self.generate_hourly_heatmap()
        elif heatmap_type == 'session':
            return self.generate_session_heatmap()
        elif heatmap_type == 'dayofweek':
            return self.generate_dayofweek_heatmap()
        elif heatmap_type == 'regime':
            return self.generate_regime_heatmap()
        else:
            return {"error": f"Unknown heatmap type: {heatmap_type}"}
    
    def generate_optimization_insights(self) -> Dict[str, Any]:
        """Generate optimization insights based on heat map analysis."""
        insights = {}
        
        if self.df.empty:
            return {"error": "No trade data available"}
        
        # Best performing hours
        hourly_heatmap = self.generate_hourly_heatmap('pnl')
        if 'hourly_performance' in hourly_heatmap:
            best_hours = []
            for hour, perf in hourly_heatmap['hourly_performance'].items():
                if perf.get('best_performance', 0) > 0:
                    best_hours.append((hour, perf['best_performance'], perf.get('best_strategy')))
            
            # Sort by performance
            best_hours.sort(key=lambda x: x[1], reverse=True)
            insights['best_trading_hours'] = best_hours[:5]  # Top 5 hours
        
        # Best performing sessions
        session_heatmap = self.generate_session_heatmap('pnl')
        if 'session_performance' in session_heatmap:
            session_performance = [(session, perf['best_performance'], perf.get('best_strategy')) 
                             for session, perf in session_heatmap['session_performance'].items()
                             if perf.get('best_performance', 0) > 0]
            
            session_performance.sort(key=lambda x: x[1], reverse=True)
            insights['best_sessions'] = session_performance[:3]  # Top 3 sessions
        
        # Strategy optimization recommendations
        strategy_performance = {}
        for strategy in self.df['strategy'].unique():
            strategy_trades = self.df[self.df['strategy'] == strategy]
            if len(strategy_trades) > 0:
                strategy_pnl = strategy_trades['pnl'].sum()
                strategy_trades_count = len(strategy_trades)
                strategy_win_rate = (strategy_trades['pnl'] > 0).mean()
                
                strategy_performance[strategy] = {
                    'total_pnl': strategy_pnl,
                    'trade_count': strategy_trades_count,
                    'win_rate': strategy_win_rate,
                    'avg_pnl': strategy_pnl / strategy_trades_count,
                }
        
        # Find best and worst strategies
        if strategy_performance:
            best_strategy_data = max(strategy_performance.items(), key=lambda x: x[1]['total_pnl'])
            worst_strategy_data = min(strategy_performance.items(), key=lambda x: x[1]['total_pnl'])
            
            insights['strategy_ranking'] = {
                'best': best_strategy_data,
                'worst': worst_strategy_data,
                'recommendation': f"Focus on {best_strategy_data[0]} performance, consider reducing {worst_strategy_data[0]} exposure"
            }
        
        return insights
    
    def generate_comprehensive_heatmap_report(self) -> Dict[str, Any]:
        """Generate comprehensive heat map report."""
        return {
            'hourly_heatmap': self.generate_hourly_heatmap('pnl'),
            'session_heatmap': self.generate_session_heatmap('pnl'),
            'dayofweek_heatmap': self.generate_dayofweek_heatmap('pnl'),
            'regime_heatmap': self.generate_regime_heatmap('pnl'),
            'optimization_insights': self.generate_optimization_insights(),
            'recommendations': self._generate_heatmap_recommendations(),
        }
    
    def _generate_heatmap_recommendations(self) -> List[str]:
        """Generate recommendations based on heat map analysis."""
        recommendations = []
        
        if self.df.empty:
            return ["No trade data available for heat map analysis"]
        
        hourly_heatmap = self.generate_hourly_heatmap('count')
        session_heatmap = self.generate_session_heatmap('count')
        
        # Time-based recommendations
        if 'hourly_performance' in hourly_heatmap:
            best_hour = hourly_heatmap.get('best_hour')
            if best_hour is not None:
                recommendations.append(f"🕐 Best trading hour: {best_hour}:00. Consider increased activity during this period.")
        
        if 'session_performance' in session_heatmap:
            best_session = session_heatmap.get('best_session')
            if best_session is not None:
                recommendations.append(f"🌍 Best session: {best_session}. Consider focusing on {best_session} session strategies.")
        
        # Strategy diversity recommendations
        strategy_counts = self.df['strategy'].value_counts()
        if len(strategy_counts) > 1:
            concentration = strategy_counts.iloc[0] / len(self.df)  # Top strategy concentration
            if concentration > 0.5:  # More than 50% in one strategy
                recommendations.append("⚖️ High strategy concentration. Consider diversifying across multiple strategies.")
        
        # Performance improvement recommendations
        total_pnl = self.df['pnl'].sum()
        if total_pnl < 0:
            recommendations.append("📉 Negative overall performance. Review strategy parameters and market conditions.")
        
        return recommendations
