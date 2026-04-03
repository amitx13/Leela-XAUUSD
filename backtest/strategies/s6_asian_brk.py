"""
S6 Asian Breakout Strategy

Asian session breakout with ADX/DI trend filter.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S6AsianBrk(BaseStrategy):
    """
    S6 Asian Breakout Strategy.
    
    Breakout from Asian session range with trend confirmation.
    
    Logic:
    1. Compute Asian range (00:00-05:30 UTC)
    2. Apply ADX/DI trend filter (must have trend confirmation)
    3. Breakout: Price moves > breakout_dist beyond range
    4. Place STOP orders in both directions (OCO)
    
    Constraints:
    - Trend family strategy
    - Max 1 breakout per day
    - ADX > 25 AND DI+ > DI- for LONG, DI- > DI+ for SHORT
    - Cross-cancellation with S7
    - Blocked in NO_TRADE regime
    """
    
    def get_strategy_name(self) -> str:
        return "S6_ASIAN_BRK"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S6 Asian breakout conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if already placed today
        if state.s6_placed_today:
            return StrategyResult(orders, signals, state_updates)
        
        # Check session (pre-London only)
        if current_time.hour >= 7:  # After 07:00 UTC
            return StrategyResult(orders, signals, state_updates)
        
        # Get Asian range data (00:00-05:30 UTC)
        m5_bars = bar_data.get("M5", [])
        if len(m5_bars) < 12:  # Need at least 1 hour of data
            return StrategyResult(orders, signals, state_updates)
        
        # Filter Asian session bars
        today = current_time.date()
        asian_bars = [
            bar for bar in m5_bars 
            if (bar["time"].date() == today and 
                0 <= bar["time"].hour < 6 or 
                (bar["time"].hour == 6 and bar["time"].minute <= 30))
        ]
        
        if len(asian_bars) < 12:
            return StrategyResult(orders, signals, state_updates)
        
        # Calculate Asian range
        asian_high = max(bar["high"] for bar in asian_bars)
        asian_low = min(bar["low"] for bar in asian_bars)
        asian_range = asian_high - asian_low
        
        min_range = self.config.get("MIN_RANGE_SIZE_PTS", 10)
        if asian_range < min_range:
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
        
        if adx_h4 > 25:  # Trend strength threshold
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
        current_price = m5_bars[-1]["close"]
        breakout_dist = asian_range * self.config.get("BREAKOUT_DIST_PCT", 0.12)
        
        breakout_up = current_price > (asian_high + breakout_dist)
        breakout_down = current_price < (asian_low - breakout_dist)
        
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
                        price=asian_high + breakout_dist,
                        sl=asian_low - 0.5,
                        lots=lot_size,
                        expiry=current_time.replace(hour=12, minute=0, second=0),
                        tag="s6_asian_breakout_long",
                        linked_tag="s6_asian_breakout_short"
                    )
                    orders.append(long_order)
                
                # Short breakout order
                if breakout_down:
                    short_order = self.create_order(
                        direction="SHORT",
                        order_type="SELL_STOP",
                        price=asian_low - breakout_dist,
                        sl=asian_high + 0.5,
                        lots=lot_size,
                        expiry=current_time.replace(hour=12, minute=0, second=0),
                        tag="s6_asian_breakout_short",
                        linked_tag="s6_asian_breakout_long"
                    )
                    orders.append(short_order)
                
                # Update state
                state_updates.update({
                    "s6_placed_today": True,
                    "s6_pending_buy_ticket": f"S6_BUY_{current_time.strftime('%Y%m%d_%H%M')}" if breakout_up else None,
                    "s6_pending_sell_ticket": f"S6_SELL_{current_time.strftime('%Y%m%d_%H%M')}" if breakout_down else None,
                })
                
                signals.append({
                    "type": "S6_ASIAN_BREAKOUT_PLACED",
                    "direction": breakout_direction,
                    "asian_high": asian_high,
                    "asian_low": asian_low,
                    "asian_range": asian_range,
                    "adx": adx_h4,
                    "di_plus": di_plus_h4,
                    "di_minus": di_minus_h4,
                    "lots": lot_size,
                    "time": current_time,
                })
                
                self.log_signal("ASIAN_BREAKOUT_PLACED",
                               direction=breakout_direction,
                               asian_high=asian_high,
                               asian_low=asian_low,
                               adx=adx_h4,
                               lots=lot_size)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S6 has already placed today."""
        return state.s6_placed_today
    
    def _check_time_restrictions(
        self,
        state: SimulatedState,
        current_time: datetime
    ) -> tuple[bool, str]:
        """Check S6 time restrictions."""
        # Pre-London only (before 07:00 UTC)
        if current_time.hour >= 7:
            return False, "AFTER_0700UTC"
        
        return True, "OK"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S6 daily counters."""
        state.s6_placed_today = False
        state.s6_pending_buy_ticket = None
        state.s6_pending_sell_ticket = None
