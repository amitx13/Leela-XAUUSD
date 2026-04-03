"""
S8 ATR Spike Strategy

F7: ATR Spike Trade — Flash spike continuation strategy.
Independent position lane - does not occupy trend family.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S8AtrSpike(BaseStrategy):
    """
    S8 ATR Spike Trade — Flash spike continuation strategy.
    
    Market structure: Flash spikes on XAUUSD frequently precede continuation moves,
    especially during London institutional sweeps. This strategy catches the
    confirmation candle that follows a spike.
    
    Logic:
    1. DETECT spike: M15 candle range > 1.5× ATR(14,H1) on same timeframe
    2. ARM: Store spike high/low, direction, arm_time
    3. CONFIRM: Next M15 candle closes past spike midpoint
    4. TRADE: STOP order in direction of continuation
    
    Constraints:
    - Independent lane (does NOT occupy trend_family)
    - Max 1 S8 per day
    - 0.5× base lot size (tight stops)
    - Does NOT fire when s8_fired_today = True
    - Blocked when S1 pending orders active (prevents double fills)
    """
    
    def get_strategy_name(self) -> str:
        return "S8_ATR_SPIKE"
    
    def get_strategy_family(self) -> str:
        return "independent"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S8 signal conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Get M15 data for spike detection
        m15_bars = bar_data.get("M15", [])
        if len(m15_bars) < 2:
            return StrategyResult(orders, signals, state_updates)
        
        current_bar = m15_bars[-1]
        previous_bar = m15_bars[-2]
        
        # Get current ATR for spike threshold
        atr_m15 = indicators.get("atr_m15")
        if atr_m15 is None or atr_m15 <= 0:
            return StrategyResult(orders, signals, state_updates)
        
        # Calculate spike threshold
        spike_threshold = atr_m15 * 1.5  # Spike = 1.5× ATR
        current_range = current_bar["high"] - current_bar["low"]
        
        # Step 1: Detect spike
        is_spike = current_range > spike_threshold
        
        if is_spike:
            # Determine spike direction
            if current_bar["close"] > current_bar["open"]:
                spike_direction = "LONG"
                spike_high = current_bar["high"]
                spike_low = current_bar["low"]
            else:
                spike_direction = "SHORT"
                spike_high = current_bar["high"]
                spike_low = current_bar["low"]
            
            # Step 2: Arm S8
            state_updates.update({
                "s8_armed": True,
                "s8_arm_time": current_time,
                "s8_spike_high": spike_high,
                "s8_spike_low": spike_low,
                "s8_spike_direction": spike_direction,
                "s8_confirmation_passed": False,
            })
            
            signals.append({
                "type": "S8_SPIKE_DETECTED",
                "direction": spike_direction,
                "range": current_range,
                "atr": atr_m15,
                "threshold": spike_threshold,
                "time": current_time,
            })
            
            self.log_signal("SPIKE_DETECTED", 
                           direction=spike_direction,
                           range=current_range,
                           atr=atr_m15)
        
        # Step 3: Check for confirmation (if armed)
        elif state.s8_armed and not state.s8_confirmation_passed:
            # Calculate spike midpoint
            spike_midpoint = (state.s8_spike_high + state.s8_spike_low) / 2
            
            # Check if current bar closes past midpoint in spike direction
            if state.s8_spike_direction == "LONG":
                confirmed = current_bar["close"] > spike_midpoint
                entry_price = current_bar["high"] + 0.5  # Small buffer above high
                stop_price = state.s8_spike_low - 0.5
            else:  # SHORT
                confirmed = current_bar["close"] < spike_midpoint
                entry_price = current_bar["low"] - 0.5  # Small buffer below low
                stop_price = state.s8_spike_high + 0.5
            
            if confirmed:
                # Step 4: Place trade
                state_updates["s8_confirmation_passed"] = True
                
                # Calculate lot size (0.5× base)
                base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
                regime_mult = state.size_multiplier
                lot_size = self.calculate_lot_size(state, base_lot * 0.5, regime_mult)
                
                if lot_size > 0:
                    # Create STOP order
                    order = self.create_order(
                        direction=state.s8_spike_direction,
                        order_type="BUY_STOP" if state.s8_spike_direction == "LONG" else "SELL_STOP",
                        price=entry_price,
                        sl=stop_price,
                        lots=lot_size,
                        tag="s8_spike_continuation"
                    )
                    
                    orders.append(order)
                    
                    # Update state
                    state_updates.update({
                        "s8_fired_today": True,
                        "s8_armed": False,
                        "s8_confirmation_passed": False,
                    })
                    
                    signals.append({
                        "type": "S8_TRADE_PLACED",
                        "direction": state.s8_spike_direction,
                        "entry": entry_price,
                        "stop": stop_price,
                        "lots": lot_size,
                        "time": current_time,
                    })
                    
                    self.log_signal("TRADE_PLACED",
                                   direction=state.s8_spike_direction,
                                   entry=entry_price,
                                   stop=stop_price,
                                   lots=lot_size)
        
        # Check arming window expiry (30 minutes)
        if state.s8_armed and state.s8_arm_time:
            elapsed = (current_time - state.s8_arm_time).total_seconds() / 60
            if elapsed > 30:  # 30 minute window
                state_updates.update({
                    "s8_armed": False,
                    "s8_arm_time": None,
                    "s8_spike_high": 0.0,
                    "s8_spike_low": 0.0,
                    "s8_spike_direction": None,
                    "s8_confirmation_passed": False,
                })
                
                signals.append({
                    "type": "S8_WINDOW_EXPIRED",
                    "elapsed_minutes": elapsed,
                    "time": current_time,
                })
                
                self.log_signal("WINDOW_EXPIRED", elapsed_minutes=elapsed)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S8 has already fired today."""
        return state.s8_fired_today
    
    def _check_independent_lane(self, state: SimulatedState) -> tuple[bool, str]:
        """Check S8 independent lane availability."""
        # Block if S8 already open
        if state.s8_open_ticket:
            return False, "S8_LANE_OCCUPIED"
        
        # Block if S1 pending orders active (prevents double fills)
        if state.s1_pending_buy_ticket or state.s1_pending_sell_ticket:
            return False, "S1_PENDING_ACTIVE"
        
        return True, "OK"
    
    def _handle_unstable_regime(self, state: SimulatedState) -> tuple[bool, str]:
        """S8 allowed in UNSTABLE with reduced risk."""
        # 0.4× regime multiplier × 0.5× S8 lot = 0.2× effective risk
        return True, "UNSTABLE_ALLOWED_REDUCED"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S8 daily counters."""
        state.s8_fired_today = False
        state.s8_armed = False
        state.s8_arm_time = None
        state.s8_spike_high = 0.0
        state.s8_spike_low = 0.0
        state.s8_spike_direction = None
        state.s8_confirmation_passed = False
