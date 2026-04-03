"""
S3 Stop Hunt Reversal Strategy

Detects stop hunts and trades the reversal.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S3StopHuntRev(BaseStrategy):
    """
    S3 Stop Hunt Reversal Strategy.
    
    Detects institutional stop hunts and trades the reversal.
    
    Logic:
    1. DETECT: Large spike candle that sweeps stops (1.5× ATR)
    2. ARM: Store sweep low/high and direction
    3. CONFIRM: Next candle confirms reversal (closes back inside range)
    4. TRADE: Market order in reversal direction
    
    Constraints:
    - Reversal family strategy
    - Max 1 trade per day
    - Allowed in UNSTABLE regime (stop hunts create volatility)
    - Tight stops (0.5× ATR)
    """
    
    def get_strategy_name(self) -> str:
        return "S3_STOP_HUNT_REV"
    
    def get_strategy_family(self) -> str:
        return "reversal"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S3 stop hunt conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Check if already fired today
        if state.s3_fired_today:
            return StrategyResult(orders, signals, state_updates)
        
        # Get M5 bars for sweep detection
        m5_bars = bar_data.get("M5", [])
        if len(m5_bars) < 2:
            return StrategyResult(orders, signals, state_updates)
        
        current_bar = m5_bars[-1]
        previous_bar = m5_bars[-2]
        
        # Get ATR for sweep threshold
        atr_m15 = indicators.get("atr_m15")
        if atr_m15 is None or atr_m15 <= 0:
            return StrategyResult(orders, signals, state_updates)
        
        # Calculate sweep threshold
        sweep_threshold = atr_m15 * 1.5  # Sweep = 1.5× ATR
        candle_range = current_bar["high"] - current_bar["low"]
        
        # Step 1: Detect sweep
        is_sweep_down = (current_bar["low"] < previous_bar["low"] * 0.99 and 
                        candle_range > sweep_threshold)
        is_sweep_up = (current_bar["high"] > previous_bar["high"] * 1.01 and 
                      candle_range > sweep_threshold)
        
        if is_sweep_down or is_sweep_up:
            # Step 2: Arm S3
            sweep_direction = "SHORT" if is_sweep_down else "LONG"
            sweep_level = current_bar["low"] if is_sweep_down else current_bar["high"]
            
            state_updates.update({
                "s3_sweep_candle_time": current_time,
                "s3_sweep_low": current_bar["low"],
                "s3_fired_today": False,  # Will be set on trade
                "s3_sweep_direction": sweep_direction,
            })
            
            signals.append({
                "type": "S3_SWEEP_DETECTED",
                "direction": sweep_direction,
                "sweep_level": sweep_level,
                "candle_range": candle_range,
                "atr": atr_m15,
                "time": current_time,
            })
            
            self.log_signal("SWEEP_DETECTED",
                           direction=sweep_direction,
                           sweep_level=sweep_level,
                           range=candle_range)
        
        # Step 3: Check for reversal confirmation
        elif state.s3_sweep_candle_time and not state.s3_fired_today:
            # Check if enough time has passed for confirmation
            elapsed = (current_time - state.s3_sweep_candle_time).total_seconds() / 60
            if elapsed < 5:  # Wait 5 minutes for confirmation
                return StrategyResult(orders, signals, state_updates)
            
            # Check for reversal confirmation
            sweep_direction = state.s3_sweep_direction
            current_price = current_bar["close"]
            
            if sweep_direction == "SHORT":
                # Looking for long reversal (price recovers)
                confirmed = current_price > (state.s3_sweep_low + atr_m15 * 0.2)
                if confirmed:
                    entry_price = current_price
                    stop_price = state.s3_sweep_low - 0.5
                    direction = "LONG"
            else:
                # Looking for short reversal (price rejects)
                confirmed = current_price < (state.s3_sweep_low + atr_m15 * 0.2)
                if confirmed:
                    entry_price = current_price
                    stop_price = state.s3_sweep_low + 0.5
                    direction = "SHORT"
            
            if confirmed:
                # Step 4: Place reversal trade
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
                        tag="s3_stop_hunt_reversal"
                    )
                    
                    orders.append(order)
                    
                    # Update state
                    state_updates.update({
                        "s3_fired_today": True,
                        "s3_sweep_candle_time": None,
                        "s3_sweep_low": 0.0,
                        "s3_sweep_direction": None,
                    })
                    
                    signals.append({
                        "type": "S3_REVERSAL_PLACED",
                        "direction": direction,
                        "entry": entry_price,
                        "stop": stop_price,
                        "lots": lot_size,
                        "elapsed_minutes": elapsed,
                        "time": current_time,
                    })
                    
                    self.log_signal("REVERSAL_PLACED",
                                   direction=direction,
                                   entry=entry_price,
                                   stop=stop_price,
                                   lots=lot_size,
                                   elapsed=elapsed)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if S3 has already fired today."""
        return state.s3_fired_today
    
    def _handle_unstable_regime(self, state: SimulatedState) -> tuple[bool, str]:
        """S3 allowed in UNSTABLE regime."""
        return True, "UNSTABLE_ALLOWED"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset S3 daily counters."""
        state.s3_fired_today = False
        state.s3_sweep_candle_time = None
        state.s3_sweep_low = 0.0
        state.s3_sweep_direction = None
