"""
S5 NY Session Compression Strategy

Trades compression patterns during NY session.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S5NyCompress(BaseStrategy):
    """
    S5 NY Session Compression Strategy.
    
    Trades compression patterns during NY session.
    
    Logic:
    1. Wait for NY session (overlap with London preferred)
    2. Detect compression (narrowing range after initial move)
    3. Trade breakout from compression
    4. Tight stops: 0.6× ATR
    
    Constraints:
    - Trend family strategy
    - Max 1 trade per day
    - NY session preferred
    - Blocked in NO_TRADE regime
    """
    
    def get_strategy_name(self) -> str:
        return "S5_NY_COMPRESS"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S5 compression conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if already fired today
        if state.s5_fired_today:
            return StrategyResult(orders, signals, state_updates)
        
        # Check session (NY preferred)
        session = state.current_session
        if session not in ["LONDON_NY_OVERLAP", "NY"]:
            return StrategyResult(orders, signals, state_updates)
        
        # Get indicators
        atr_m15 = indicators.get("atr_m15")
        if atr_m15 is None or atr_m15 <= 0:
            return StrategyResult(orders, signals, state_updates)
        
        # Get M15 bars for compression detection
        m15_bars = bar_data.get("M15", [])
        if len(m15_bars) < 4:
            return StrategyResult(orders, signals, state_updates)
        
        current_bar = m15_bars[-1]
        
        # Step 1: Detect initial move (first 2 hours of NY session)
        # NY starts at 13:00 UTC
        if current_time.hour < 13 or current_time.hour > 15:
            return StrategyResult(orders, signals, state_updates)
        
        # Get recent bars for analysis
        recent_bars = m15_bars[-4:]  # Last 4 bars (1 hour)
        
        # Calculate initial move and compression
        initial_high = max(bar["high"] for bar in recent_bars[:2])
        initial_low = min(bar["low"] for bar in recent_bars[:2])
        initial_range = initial_high - initial_low
        
        current_high = max(bar["high"] for bar in recent_bars[2:])
        current_low = min(bar["low"] for bar in recent_bars[2:])
        current_range = current_high - current_low
        
        # Step 2: Detect compression
        compression_threshold = atr_m15 * 0.8
        is_compressing = current_range < (initial_range * 0.6) and current_range < compression_threshold
        
        if is_compressing:
            # Compression detected - look for breakout
            breakout_threshold = compression_threshold * 0.5
            
            # Determine breakout direction
            if current_bar["close"] > initial_high:
                # Breakout up
                direction = "LONG"
                entry_price = current_bar["high"] + 0.3
                stop_price = current_low - (atr_m15 * 0.6)
                tp_price = entry_price + (atr_m15 * 1.2)
            elif current_bar["close"] < initial_low:
                # Breakout down
                direction = "SHORT"
                entry_price = current_bar["low"] - 0.3
                stop_price = current_high + (atr_m15 * 0.6)
                tp_price = entry_price - (atr_m15 * 1.2)
            else:
                # No clear breakout
                return StrategyResult(orders, signals, state_updates)
            
            # Step 3: Place breakout trade
            if "entry_price" in locals():
                # Calculate lot size
                base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
                lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)
                
                if lot_size > 0:
                    order = self.create_order(
                        direction=direction,
                        order_type="BUY_STOP" if direction == "LONG" else "SELL_STOP",
                        price=entry_price,
                        sl=stop_price,
                        tp=tp_price,
                        lots=lot_size,
                        expiry=current_time.replace(hour=18, minute=0, second=0),
                        tag="s5_ny_compression_breakout"
                    )
                    
                    orders.append(order)
                    
                    # Update state
                    state_updates.update({
                        "s5_fired_today": True,
                    })
                    
                    signals.append({
                        "type": "S5_COMPRESSION_BREAKOUT_PLACED",
                        "direction": direction,
                        "entry": entry_price,
                        "stop": stop_price,
                        "tp": tp_price,
                        "initial_range": initial_range,
                        "current_range": current_range,
                        "time": current_time,
                    })
                    
                    self.log_signal("COMPRESSION_BREAKOUT_PLACED",
                                   direction=direction,
                                   entry=entry_price,
                                   stop=stop_price,
                                   tp=tp_price,
                                   lots=lot_size)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S5 has already fired today."""
        return state.s5_fired_today
    
    def _check_time_restrictions(
        self,
        state: SimulatedState,
        current_time: datetime
    ) -> tuple[bool, str]:
        """Check S5 time restrictions."""
        # NY session preferred
        session = state.current_session
        if session not in ["LONDON_NY_OVERLAP", "NY"]:
            return False, "NOT_NY_SESSION"
        
        # Only trade during first 2 hours of NY (13:00-15:00 UTC)
        if current_time.hour < 13 or current_time.hour > 15:
            return False, "OUTSIDE_NY_WINDOW"
        
        return True, "OK"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S5 daily counters."""
        state.s5_fired_today = False
