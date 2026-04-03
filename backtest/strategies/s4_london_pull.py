"""
S4 London Pullback Strategy

Pullback to EMA20 during London session.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S4LondonPull(BaseStrategy):
    """
    S4 London Pullback to EMA20 Strategy.
    
    Trades pullbacks to EMA20 during London session.
    
    Logic:
    1. Wait for established trend (price above/below EMA20)
    2. Wait for pullback to EMA20
    3. Enter on confirmation (bounce from EMA20)
    4. Stop: 0.8× ATR, TP: 1.5× ATR
    
    Constraints:
    - Trend family strategy
    - Max 1 trade per day
    - London session only
    - Blocked in NO_TRADE regime
    """
    
    def get_strategy_name(self) -> str:
        return "S4_LONDON_PULL"
    
    def get_strategy_family(self) -> str:
        return "trend"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S4 pullback conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if already fired today
        if state.s4_fired_today:
            return StrategyResult(orders, signals, state_updates)
        
        # Check session (London only)
        session = state.current_session
        if session not in ["LONDON", "LONDON_NY_OVERLAP"]:
            return StrategyResult(orders, signals, state_updates)
        
        # Get indicators
        ema20_h1 = indicators.get("ema20_h1")
        atr_h1 = indicators.get("atr_h1")
        
        if None in [ema20_h1, atr_h1]:
            return StrategyResult(orders, signals, state_updates)
        
        # Get H1 bars for trend analysis
        h1_bars = bar_data.get("H1", [])
        if len(h1_bars) < 3:
            return StrategyResult(orders, signals, state_updates)
        
        # Get current price
        m5_bars = bar_data.get("M5", [])
        if not m5_bars:
            return StrategyResult(orders, signals, state_updates)
        
        current_price = m5_bars[-1]["close"]
        
        # Step 1: Check for established trend
        # Need at least 3 H1 candles with consistent relationship to EMA20
        recent_h1 = h1_bars[-3:]
        
        if len(recent_h1) < 3:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if trend is established
        above_ema_count = sum(1 for bar in recent_h1 if bar["close"] > ema20_h1)
        below_ema_count = len(recent_h1) - above_ema_count
        
        if above_ema_count >= 2:
            # Established uptrend (price above EMA20)
            trend_direction = "LONG"
            pullback_threshold = ema20_h1 - (atr_h1 * 0.3)
        elif below_ema_count >= 2:
            # Established downtrend (price below EMA20)
            trend_direction = "SHORT"
            pullback_threshold = ema20_h1 + (atr_h1 * 0.3)
        else:
            # No clear trend
            return StrategyResult(orders, signals, state_updates)
        
        # Step 2: Check for pullback to EMA20
        if trend_direction == "LONG":
            # Looking for pullback down to EMA20
            if current_price <= pullback_threshold:
                # Pullback detected - look for bounce confirmation
                # Need price to move back above EMA20 for confirmation
                if current_price > ema20_h1 * 0.998:  # Small buffer
                    entry_price = current_price
                    stop_price = current_price - (atr_h1 * 0.8)
                    tp_price = current_price + (atr_h1 * 1.5)
                    
                    signals.append({
                        "type": "S4_PULLBACK_LONG_DETECTED",
                        "entry": entry_price,
                        "stop": stop_price,
                        "tp": tp_price,
                        "ema20": ema20_h1,
                        "time": current_time,
                    })
        else:
            # Looking for pullback up to EMA20
            if current_price >= pullback_threshold:
                # Pullback detected - look for bounce confirmation
                if current_price < ema20_h1 * 1.002:  # Small buffer
                    entry_price = current_price
                    stop_price = current_price + (atr_h1 * 0.8)
                    tp_price = current_price - (atr_h1 * 1.5)
                    
                    signals.append({
                        "type": "S4_PULLBACK_SHORT_DETECTED",
                        "entry": entry_price,
                        "stop": stop_price,
                        "tp": tp_price,
                        "ema20": ema20_h1,
                        "time": current_time,
                    })
        
        # Step 3: Place trade on confirmation
        if "entry_price" in locals():
            # Calculate lot size
            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)
            
            if lot_size > 0:
                order = self.create_order(
                    direction=trend_direction,
                    order_type="MARKET",
                    price=entry_price,
                    sl=stop_price,
                    tp=tp_price,
                    lots=lot_size,
                    tag="s4_london_pullback"
                )
                
                orders.append(order)
                
                # Update state
                state_updates.update({
                    "s4_fired_today": True,
                })
                
                self.log_signal("PULLBACK_TRADE_PLACED",
                               direction=trend_direction,
                               entry=entry_price,
                               stop=stop_price,
                               tp=tp_price,
                               lots=lot_size)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S4 has already fired today."""
        return state.s4_fired_today
    
    def _check_time_restrictions(
        self,
        state: SimulatedState,
        current_time: datetime
    ) -> tuple[bool, str]:
        """Check S4 time restrictions."""
        # London session only
        session = state.current_session
        if session not in ["LONDON", "LONDON_NY_OVERLAP"]:
            return False, "NOT_LONDON_SESSION"
        
        return True, "OK"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S4 daily counters."""
        state.s4_fired_today = False
