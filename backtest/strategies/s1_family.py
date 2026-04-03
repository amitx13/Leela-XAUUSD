"""
S1 Family Strategies

All S1 family strategies: S1 London Breakout + variants.
Trend family strategies - occupy trend_family_occupied.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S1LondonBrk(BaseStrategy):
    """
    S1 London Breakout Strategy.
    
    Classic London session breakout from Asian range.
    
    Logic:
    1. Compute Asian range (00:00-07:00 UTC)
    2. Wait for London session start (07:55 UTC)
    3. Breakout: Price moves > breakout_dist beyond range
    4. Place STOP order in breakout direction
    
    Constraints:
    - Trend family strategy (occupies trend_family_occupied)
    - Max 3 attempts per day
    - Blocked in NO_TRADE regime
    - Time kill: 18:00 UTC
    """
    
    def get_strategy_name(self) -> str:
        return "S1_LONDON_BRK"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S1 London breakout conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check session (London only)
        session = state.current_session
        if session not in ["LONDON", "LONDON_NY_OVERLAP"]:
            return StrategyResult(orders, signals, state_updates)
        
        # Check daily attempt limit
        if state.s1_family_attempts_today >= 3:
            return StrategyResult(orders, signals, state_updates)
        
        # Get Asian range data
        if not state.range_computed or state.range_size <= 0:
            return StrategyResult(orders, signals, state_updates)
        
        # Get current price
        m5_bars = bar_data.get("M5", [])
        if not m5_bars:
            return StrategyResult(orders, signals, state_updates)
        
        current_price = m5_bars[-1]["close"]
        
        # Calculate breakout distances
        breakout_dist = state.range_size * self.config.get("BREAKOUT_DIST_PCT", 0.12)
        
        # Check for breakout conditions
        breakout_up = current_price > (state.range_high + breakout_dist)
        breakout_down = current_price < (state.range_low - breakout_dist)
        
        if breakout_up:
            # Long breakout
            entry_price = state.range_high + breakout_dist
            stop_price = state.range_low - 0.5  # Small buffer below range
            
            # Calculate lot size
            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)
            
            if lot_size > 0:
                order = self.create_order(
                    direction="LONG",
                    order_type="BUY_STOP",
                    price=entry_price,
                    sl=stop_price,
                    lots=lot_size,
                    expiry=current_time.replace(hour=18, minute=0, second=0),
                    tag="s1_london_breakout_long"
                )
                
                orders.append(order)
                
                # Update state
                state_updates.update({
                    "s1_family_attempts_today": state.s1_family_attempts_today + 1,
                    "last_s1_direction": "LONG",
                })
                
                signals.append({
                    "type": "S1_BREAKOUT_LONG",
                    "entry": entry_price,
                    "stop": stop_price,
                    "lots": lot_size,
                    "time": current_time,
                })
                
                self.log_signal("BREAKOUT_LONG",
                               entry=entry_price,
                               stop=stop_price,
                               lots=lot_size)
        
        elif breakout_down:
            # Short breakout
            entry_price = state.range_low - breakout_dist
            stop_price = state.range_high + 0.5  # Small buffer above range
            
            # Calculate lot size
            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)
            
            if lot_size > 0:
                order = self.create_order(
                    direction="SHORT",
                    order_type="SELL_STOP",
                    price=entry_price,
                    sl=stop_price,
                    lots=lot_size,
                    expiry=current_time.replace(hour=18, minute=0, second=0),
                    tag="s1_london_breakout_short"
                )
                
                orders.append(order)
                
                # Update state
                state_updates.update({
                    "s1_family_attempts_today": state.s1_family_attempts_today + 1,
                    "last_s1_direction": "SHORT",
                })
                
                signals.append({
                    "type": "S1_BREAKOUT_SHORT",
                    "entry": entry_price,
                    "stop": stop_price,
                    "lots": lot_size,
                    "time": current_time,
                })
                
                self.log_signal("BREAKOUT_SHORT",
                               entry=entry_price,
                               stop=stop_price,
                               lots=lot_size)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check S1 daily attempt limit."""
        return state.s1_family_attempts_today >= 3
    
    def _check_time_restrictions(
        self,
        state: SimulatedState,
        current_time: datetime
    ) -> tuple[bool, str]:
        """Check S1 time restrictions."""
        # Time kill: 18:00 UTC
        if current_time.hour >= 18:
            return False, "TIME_KILL_1800UTC"
        
        # London session only
        session = state.current_session
        if session not in ["LONDON", "LONDON_NY_OVERLAP"]:
            return False, "NOT_LONDON_SESSION"
        
        return True, "OK"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S1 daily counters."""
        state.s1_family_attempts_today = 0
        state.s1f_attempts_today = 0
        state.s1b_pending_ticket = None
        state.s1d_ema_touched_today = False
        state.s1d_fired_today = False
        state.s1e_pyramid_done = False
        state.s1f_post_tk_active = False


class S1bFailedBrk(BaseStrategy):
    """
    S1b Failed Breakout Reversal Strategy.
    
    Trades failed breakout reversals back into the range.
    """
    
    def get_strategy_name(self) -> str:
        return "S1B_FAILED_BRK"
    
    def get_strategy_family(self) -> str:
        return "reversal"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S1b failed breakout conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if S1b already pending
        if state.s1b_pending_ticket:
            return StrategyResult(orders, signals, state_updates)
        
        # Need failed breakout condition
        if not state.failed_breakout_flag:
            return StrategyResult(orders, signals, state_updates)
        
        # Get current price and range data
        m5_bars = bar_data.get("M5", [])
        if not m5_bars or not state.range_computed:
            return StrategyResult(orders, signals, state_updates)
        
        current_price = m5_bars[-1]["close"]
        breakout_direction = state.failed_breakout_direction
        
        if not breakout_direction:
            return StrategyResult(orders, signals, state_updates)
        
        # Calculate reversal entry
        if breakout_direction == "LONG":
            # Failed long breakout - look for short reversal
            if current_price < state.range_high:
                entry_price = current_price
                stop_price = state.range_high + 0.5
                direction = "SHORT"
        else:
            # Failed short breakout - look for long reversal
            if current_price > state.range_low:
                entry_price = current_price
                stop_price = state.range_low - 0.5
                direction = "LONG"
        
        if 'direction' in locals():
            # Calculate lot size (reduced for reversal)
            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot * 0.8, state.size_multiplier)
            
            if lot_size > 0:
                order = self.create_order(
                    direction=direction,
                    order_type="MARKET",
                    price=entry_price,
                    sl=stop_price,
                    lots=lot_size,
                    tag="s1b_failed_breakout_reversal"
                )
                
                orders.append(order)
                
                # Update state
                state_updates.update({
                    "s1b_pending_ticket": f"S1B_{current_time.strftime('%Y%m%d_%H%M')}",
                    "failed_breakout_flag": False,
                    "failed_breakout_direction": None,
                })
                
                signals.append({
                    "type": "S1B_REVERSAL_PLACED",
                    "direction": direction,
                    "entry": entry_price,
                    "stop": stop_price,
                    "lots": lot_size,
                    "time": current_time,
                })
                
                self.log_signal("REVERSAL_PLACED",
                               direction=direction,
                               entry=entry_price,
                               stop=stop_price,
                               lots=lot_size)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """S1b has no specific daily limit."""
        return False
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S1b daily counters."""
        state.s1b_pending_ticket = None


class S1cStopHunt(BaseStrategy):
    """
    S1c Stop Hunt Detection Strategy.
    
    Detects and trades stop hunt patterns.
    """
    
    def get_strategy_name(self) -> str:
        return "S1C_STOP_HUNT"
    
    def get_strategy_family(self) -> str:
        return "reversal"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S1c stop hunt conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if already fired today
        if state.s3_fired_today:  # S3 uses same logic
            return StrategyResult(orders, signals, state_updates)
        
        # Get M5 bars for sweep detection
        m5_bars = bar_data.get("M5", [])
        if len(m5_bars) < 2:
            return StrategyResult(orders, signals, state_updates)
        
        current_bar = m5_bars[-1]
        previous_bar = m5_bars[-2]
        
        # Detect stop hunt (large down candle followed by reversal)
        candle_range = current_bar["high"] - current_bar["low"]
        atr_m15 = indicators.get("atr_m15", 20)
        
        # Stop hunt criteria: large down candle > 1.5× ATR
        is_sweep = (current_bar["low"] < previous_bar["low"] * 0.99 and 
                   candle_range > atr_m15 * 1.5)
        
        if is_sweep:
            # Store sweep info
            state_updates.update({
                "s3_sweep_candle_time": current_time,
                "s3_sweep_low": current_bar["low"],
                "s3_fired_today": True,
                "s3_sweep_direction": "SHORT",
            })
            
            signals.append({
                "type": "S1C_STOP_HUNT_DETECTED",
                "direction": "SHORT",
                "sweep_low": current_bar["low"],
                "candle_range": candle_range,
                "atr": atr_m15,
                "time": current_time,
            })
            
            self.log_signal("STOP_HUNT_DETECTED",
                           direction="SHORT",
                           sweep_low=current_bar["low"],
                           range=candle_range)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S1c has already fired today."""
        return state.s3_fired_today
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S1c daily counters."""
        state.s3_fired_today = False
        state.s3_sweep_candle_time = None
        state.s3_sweep_low = 0.0
        state.s3_sweep_direction = None


# Additional S1 family strategies would be implemented here:
# S1d: M5 Pullback Re-entries
# S1e: Pyramid Into Winners  
# S1f: Post-Time-Kill Re-entries

class S1dPyramid(BaseStrategy):
    """
    S1d: M5 Pullback Re-entries
    
    Logic: Add to winning positions on M5 pullbacks
    - Only when main S1 position is in profit
    - Pullback to EMA20 on M5
    - Add 50% of original position size
    - Max 2 pyramid additions per position
    """
    
    def get_strategy_name(self) -> str:
        return "S1D_PYRAMID"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}
        
        # Check if we have an open trend position
        if not state.open_position or not state.trend_family_occupied:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if we can pyramid (max 2 additions)
        pyramid_count = getattr(state, 's1d_pyramid_count', 0)
        if pyramid_count >= 2:
            return StrategyResult(orders, signals, state_updates)
        
        # Get current position details
        entry_price = state.entry_price
        current_price = self._get_current_price(bar_data)
        
        # Check if position is profitable (R > 0.5)
        if state.position_be_activated:  # BE activated means we have profit
            # Look for M5 pullback to EMA20
            m5_df = bar_data.get("M5")
            if len(m5_df) >= 20:
                ema20 = ta.ema(m5_df["close"], length=20).iloc[-1]
                
                # Pullback condition: price touches EMA20
                if abs(current_price - ema20) < 0.5:  # Within 0.5 points of EMA20
                    # Calculate pyramid position size (50% of original)
                    pyramid_size = state.original_lot_size * 0.5
                    
                    # Determine direction based on original position
                    direction = state.trend_trade_direction
                    
                    # Create pyramid order
                    if direction == "LONG":
                        pyramid_order = SimOrder(
                            strategy=self.get_strategy_name(),
                            direction="LONG",
                            order_type="MARKET",
                            price=current_price,
                            sl=current_price - (indicators.get("atr_m15", 20) * 1.5),
                            tp=current_price + (indicators.get("atr_m15", 20) * 2.0),
                            lots=pyramid_size,
                            expiry=current_time + timedelta(hours=4)
                        )
                    else:
                        pyramid_order = SimOrder(
                            strategy=self.get_strategy_name(),
                            direction="SHORT", 
                            order_type="MARKET",
                            price=current_price,
                            sl=current_price + (indicators.get("atr_m15", 20) * 1.5),
                            tp=current_price - (indicators.get("atr_m15", 20) * 2.0),
                            lots=pyramid_size,
                            expiry=current_time + timedelta(hours=4)
                        )
                    
                    orders.append(pyramid_order)
                    
                    # Update pyramid count
                    state_updates["s1d_pyramid_count"] = pyramid_count + 1
                    
                    signals.append({
                        "strategy": self.get_strategy_name(),
                        "type": "PYRAMID_ENTRY",
                        "direction": direction,
                        "price": current_price,
                        "size": pyramid_size,
                        "reason": "M5_PULLBACK_TO_EMA20"
                    })
        
        return StrategyResult(orders, signals, state_updates)


class S1ePyramid(BaseStrategy):
    """
    S1e: Pyramid Into Winners (Aggressive)
    
    Logic: Add to winning positions more aggressively
    - Only when position R > 1.0
    - Add on any minor pullback (no EMA requirement)
    - Add 75% of original position size
    - Max 3 pyramid additions per position
    """
    
    def get_strategy_name(self) -> str:
        return "S1E_PYRAMID"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}
        
        # Check if we have an open trend position
        if not state.open_position or not state.trend_family_occupied:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if we can pyramid (max 3 additions)
        pyramid_count = getattr(state, 's1e_pyramid_count', 0)
        if pyramid_count >= 3:
            return StrategyResult(orders, signals, state_updates)
        
        # Get current position details
        entry_price = state.entry_price
        current_price = self._get_current_price(bar_data)
        
        # Calculate current R
        if state.trend_trade_direction == "LONG":
            current_r = (current_price - entry_price) / indicators.get("atr_m15", 20)
        else:
            current_r = (entry_price - current_price) / indicators.get("atr_m15", 20)
        
        # Only pyramid if R > 1.0
        if current_r > 1.0:
            # Look for minor pullback (last 2 bars)
            m5_df = bar_data.get("M5")
            if len(m5_df) >= 3:
                last_close = m5_df["close"].iloc[-1]
                prev_close = m5_df["close"].iloc[-2]
                
                # Pullback condition: minor reversal
                if state.trend_trade_direction == "LONG":
                    pullback = last_close < prev_close  # Small pullback in uptrend
                else:
                    pullback = last_close > prev_close  # Small pullback in downtrend
                
                if pullback:
                    # Calculate pyramid position size (75% of original)
                    pyramid_size = state.original_lot_size * 0.75
                    
                    # Determine direction
                    direction = state.trend_trade_direction
                    
                    # Create pyramid order
                    if direction == "LONG":
                        pyramid_order = SimOrder(
                            strategy=self.get_strategy_name(),
                            direction="LONG",
                            order_type="MARKET",
                            price=current_price,
                            sl=current_price - (indicators.get("atr_m15", 20) * 1.2),
                            tp=current_price + (indicators.get("atr_m15", 20) * 2.5),
                            lots=pyramid_size,
                            expiry=current_time + timedelta(hours=3)
                        )
                    else:
                        pyramid_order = SimOrder(
                            strategy=self.get_strategy_name(),
                            direction="SHORT",
                            order_type="MARKET", 
                            price=current_price,
                            sl=current_price + (indicators.get("atr_m15", 20) * 1.2),
                            tp=current_price - (indicators.get("atr_m15", 20) * 2.5),
                            lots=pyramid_size,
                            expiry=current_time + timedelta(hours=3)
                        )
                    
                    orders.append(pyramid_order)
                    
                    # Update pyramid count
                    state_updates["s1e_pyramid_count"] = pyramid_count + 1
                    
                    signals.append({
                        "strategy": self.get_strategy_name(),
                        "type": "AGGRESSIVE_PYRAMID",
                        "direction": direction,
                        "price": current_price,
                        "size": pyramid_size,
                        "reason": "MINOR_PULLBACK_R>1.0"
                    })
        
        return StrategyResult(orders, signals, state_updates)


class S1fPostTk(BaseStrategy):
    """
    S1f: Post-Time-Kill Re-entries
    
    Logic: Re-enter positions after time kill
    - Only after time kill (18:00 UTC)
    - If original position was profitable
    - Wait for new breakout confirmation
    - Smaller position size (75% of original)
    - Only 1 re-entry per day
    """
    
    def get_strategy_name(self) -> str:
        return "S1F_POST_TK"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}
        
        # Only active after time kill (18:00 UTC)
        if current_time.hour < 18:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if we already re-entered today
        if getattr(state, 's1f_reentered_today', False):
            return StrategyResult(orders, signals, state_updates)
        
        # Check if we had a profitable position that was killed
        if not getattr(state, 's1f_killed_position_profitable', False):
            return StrategyResult(orders, signals, state_updates)
        
        # Check if trend family is available for re-entry
        if state.trend_family_occupied:
            return StrategyResult(orders, signals, state_updates)
        
        # Get M15 data for breakout confirmation
        m15_df = bar_data.get("M15")
        if len(m15_df) < 20:
            return StrategyResult(orders, signals, state_updates)
        
        current_price = self._get_current_price(bar_data)
        
        # Simple breakout logic (similar to S1 but simplified)
        # Use previous 15 bars as range
        recent_high = m15_df["high"].tail(15).max()
        recent_low = m15_df["low"].tail(15).min()
        
        breakout_up = current_price > recent_high
        breakout_down = current_price < recent_low
        
        if breakout_up or breakout_down:
            # Calculate position size (75% of original)
            re_entry_size = getattr(state, 's1f_original_size', 1.0) * 0.75
            
            # Determine direction
            direction = "LONG" if breakout_up else "SHORT"
            
            # Create re-entry order
            if direction == "LONG":
                re_entry_order = SimOrder(
                    strategy=self.get_strategy_name(),
                    direction="LONG",
                    order_type="STOP",
                    price=current_price + 0.5,  # Entry above current price
                    sl=recent_low - indicators.get("atr_m15", 20) * 0.5,
                    tp=current_price + indicators.get("atr_m15", 20) * 2.0,
                    lots=re_entry_size,
                    expiry=current_time + timedelta(hours=2)
                )
            else:
                re_entry_order = SimOrder(
                    strategy=self.get_strategy_name(),
                    direction="SHORT",
                    order_type="STOP",
                    price=current_price - 0.5,  # Entry below current price
                    sl=recent_high + indicators.get("atr_m15", 20) * 0.5,
                    tp=current_price - indicators.get("atr_m15", 20) * 2.0,
                    lots=re_entry_size,
                    expiry=current_time + timedelta(hours=2)
                )
            
            orders.append(re_entry_order)
            
            # Mark as re-entered today
            state_updates["s1f_reentered_today"] = True
            
            signals.append({
                "strategy": self.get_strategy_name(),
                "type": "POST_TK_REENTRY",
                "direction": direction,
                "price": current_price,
                "size": re_entry_size,
                "reason": "BREAKOUT_AFTER_TK"
            })
        
        return StrategyResult(orders, signals, state_updates)
    
    def _get_current_price(self, bar_data):
        """Get current price from bar data."""
        m5_df = bar_data.get("M5")
        if not m5_df.empty:
            return m5_df["close"].iloc[-1]
        return 0.0
