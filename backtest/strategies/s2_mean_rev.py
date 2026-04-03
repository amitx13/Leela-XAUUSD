"""
S2 Mean Reversion Strategy

Mean reversion based on overextended conditions.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S2MeanRev(BaseStrategy):
    """
    S2 Mean Reversion Strategy.
    
    Trades price extremes back to the mean.
    
    Logic:
    1. Detect overextended conditions (RSI extreme, price far from EMA)
    2. Wait for confirmation (price reversal)
    3. Enter on reversal with tight stops
    
    Constraints:
    - Trend family strategy
    - Max 1 trade per day
    - Blocked in NO_TRADE regime
    - Session restrictions (avoid news)
    """
    
    def get_strategy_name(self) -> str:
        return "S2_MEAN_REV"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S2 mean reversion conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if already fired today
        if state.s2_fired_today:
            return StrategyResult(orders, signals, state_updates)
        
        # Get indicators
        rsi_h1 = indicators.get("rsi_h1")
        ema20_h1 = indicators.get("ema20_h1")
        atr_h1 = indicators.get("atr_h1")
        
        if None in [rsi_h1, ema20_h1, atr_h1]:
            return StrategyResult(orders, signals, state_updates)
        
        # Get current price
        m5_bars = bar_data.get("M5", [])
        if not m5_bars:
            return StrategyResult(orders, signals, state_updates)
        
        current_price = m5_bars[-1]["close"]
        
        # Calculate distance from EMA
        ema_distance = abs(current_price - ema20_h1)
        ema_distance_pct = ema_distance / ema20_h1 * 100
        
        # Overextended conditions
        overextended_rsi = rsi_h1 > 70 or rsi_h1 < 30
        overextended_ema = ema_distance_pct > 0.5  # 0.5% from EMA
        
        if overextended_rsi and overextended_ema:
            # Determine direction (revert to mean)
            if rsi_h1 > 70:
                # Overbought - look for short
                direction = "SHORT"
                entry_price = current_price
                stop_price = current_price + (atr_h1 * 0.5)
            else:
                # Oversold - look for long
                direction = "LONG"
                entry_price = current_price
                stop_price = current_price - (atr_h1 * 0.5)
            
            # Calculate lot size
            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)
            
            if lot_size > 0:
                order = self.create_order(
                    direction=direction,
                    order_type="MARKET",
                    price=entry_price,
                    sl=stop_price,
                    lots=lot_size,
                    tag="s2_mean_reversion"
                )
                
                orders.append(order)
                
                # Update state
                state_updates.update({
                    "s2_fired_today": True,
                })
                
                signals.append({
                    "type": "S2_MEAN_REVERSION_PLACED",
                    "direction": direction,
                    "entry": entry_price,
                    "stop": stop_price,
                    "lots": lot_size,
                    "rsi": rsi_h1,
                    "ema_distance_pct": ema_distance_pct,
                    "time": current_time,
                })
                
                self.log_signal("MEAN_REVERSION_PLACED",
                               direction=direction,
                               entry=entry_price,
                               stop=stop_price,
                               lots=lot_size,
                               rsi=rsi_h1)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S2 has already fired today."""
        return state.s2_fired_today
    
    def _check_time_restrictions(
        self,
        state: SimulatedState,
        current_time: datetime
    ) -> tuple[bool, str]:
        """Check S2 time restrictions."""
        # Avoid major news hours
        hour = current_time.hour
        
        # Block during high-impact news windows
        if 8 <= hour <= 10:  # NY morning
            return False, "NEWS_BLACKOUT"
        
        # Block during London close/NY open overlap
        if 11 <= hour <= 12:
            return False, "VOLATILE_PERIOD"
        
        return True, "OK"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S2 daily counters."""
        state.s2_fired_today = False
