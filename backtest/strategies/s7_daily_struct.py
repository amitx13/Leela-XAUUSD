"""
S7 Daily Structure Strategy

Daily structure breakout with ADX/DI trend filter.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S7DailyStruct(BaseStrategy):
    """
    S7 Daily Structure Strategy.
    
    Breakout from previous day's range with trend confirmation.
    
    Logic:
    1. Get previous day's high/low
    2. Apply ADX/DI trend filter
    3. Breakout: Price moves beyond previous day's range
    4. Place OCO orders in both directions
    
    Constraints:
    - Trend family strategy
    - Max 1 breakout per day
    - ADX > 20 AND DI+ > DI- for LONG, DI- > DI+ for SHORT
    - Cross-cancellation with S6
    - Blocked in NO_TRADE regime
    """
    
    def get_strategy_name(self) -> str:
        return "S7_DAILY_STRUCT"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S7 daily structure conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if already placed today
        if state.s7_placed_today:
            return StrategyResult(orders, signals, state_updates)
        
        # Get previous day's data
        if not state.s7_prev_day_high or not state.s7_prev_day_low:
            return StrategyResult(orders, signals, state_updates)
        
        # Get trend confirmation indicators
        adx_h4 = indicators.get("adx_h4")
        di_plus_h4 = indicators.get("di_plus_h4")
        di_minus_h4 = indicators.get("di_minus_h4")
        
        if None in [adx_h4, di_plus_h4, di_minus_h4]:
            return StrategyResult(orders, signals, state_updates)
        
        # Step 2: Apply ADX/DI trend filter
        trend_confirmed = False
        breakout_direction = None
        
        if adx_h4 > 20:  # Trend strength threshold (lower than S6)
            if di_plus_h4 > di_minus_h4:
                # Uptrend confirmed
                trend_confirmed = True
                breakout_direction = "LONG"
            elif di_minus_h4 > di_plus_h4:
                # Downtrend confirmed
                trend_confirmed = True
                breakout_direction = "SHORT"
        
        if not trend_confirmed:
            return StrategyResult(orders, signals, state_updates)
        
        # Step 3: Check for breakout conditions
        m5_bars = bar_data.get("M5", [])
        if not m5_bars:
            return StrategyResult(orders, signals, state_updates)
        
        current_price = m5_bars[-1]["close"]
        
        # Calculate breakout distances
        prev_range = state.s7_prev_day_high - state.s7_prev_day_low
        breakout_dist = prev_range * self.config.get("BREAKOUT_DIST_PCT", 0.15)
        
        breakout_up = current_price > (state.s7_prev_day_high + breakout_dist)
        breakout_down = current_price < (state.s7_prev_day_low - breakout_dist)
        
        if breakout_up or breakout_down:
            # Step 4: Place OCO orders (both directions)
            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)
            
            if lot_size > 0:
                # Long breakout order
                if breakout_up:
                    long_order = self.create_order(
                        direction="LONG",
                        order_type="BUY_STOP",
                        price=state.s7_prev_day_high + breakout_dist,
                        sl=state.s7_prev_day_low - 0.5,
                        lots=lot_size,
                        expiry=current_time.replace(hour=23, minute=55, second=0),
                        tag="s7_daily_breakout_long",
                        linked_tag="s7_daily_breakout_short"
                    )
                    orders.append(long_order)
                
                # Short breakout order
                if breakout_down:
                    short_order = self.create_order(
                        direction="SHORT",
                        order_type="SELL_STOP",
                        price=state.s7_prev_day_low - breakout_dist,
                        sl=state.s7_prev_day_high + 0.5,
                        lots=lot_size,
                        expiry=current_time.replace(hour=23, minute=55, second=0),
                        tag="s7_daily_breakout_short",
                        linked_tag="s7_daily_breakout_long"
                    )
                    orders.append(short_order)
                
                # Update state
                state_updates.update({
                    "s7_placed_today": True,
                    "s7_pending_buy_ticket": f"S7_BUY_{current_time.strftime('%Y%m%d_%H%M')}" if breakout_up else None,
                    "s7_pending_sell_ticket": f"S7_SELL_{current_time.strftime('%Y%m%d_%H%M')}" if breakout_down else None,
                })
                
                signals.append({
                    "type": "S7_DAILY_BREAKOUT_PLACED",
                    "direction": breakout_direction,
                    "prev_high": state.s7_prev_day_high,
                    "prev_low": state.s7_prev_day_low,
                    "prev_range": prev_range,
                    "adx": adx_h4,
                    "di_plus": di_plus_h4,
                    "di_minus": di_minus_h4,
                    "lots": lot_size,
                    "time": current_time,
                })
                
                self.log_signal("DAILY_BREAKOUT_PLACED",
                               direction=breakout_direction,
                               prev_high=state.s7_prev_day_high,
                               prev_low=state.s7_prev_day_low,
                               adx=adx_h4,
                               lots=lot_size)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S7 has already placed today."""
        return state.s7_placed_today
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S7 daily counters."""
        state.s7_placed_today = False
        state.s7_pending_buy_ticket = None
        state.s7_pending_sell_ticket = None
