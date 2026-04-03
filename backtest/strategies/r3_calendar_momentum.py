"""
R3 Calendar Momentum Strategy

Post-event directional momentum strategy.
Independent position lane - does not occupy trend family.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class R3CalendarMomentum(BaseStrategy):
    """
    R3 Calendar Momentum — Post-event directional momentum strategy.
    
    Logic:
    1. ARM: High-impact economic event triggers arming
    2. WAIT: R3_ARMED_WINDOW_MIN delay (usually 5-15 min)
    3. EVALUATE: Check post-event price movement direction
    4. TRADE: Market order in direction of momentum
    
    Constraints:
    - Independent lane (does NOT occupy trend_family)
    - Max 1 R3 per event
    - 0.5× base lot size (tight stops)
    - Hard exit after 30 minutes
    - Blocked in NO_TRADE regime
    
    Entry: Market order in r3_direction (first M5 close after delay confirms direction)
    Stop: 0.5 × H1 ATR from entry (no structural anchor post-event)
    TP: 0.75 × H1 ATR from entry → 1.5:1 RR
    Hold: Max 30 min — checked separately in position management
    """
    
    def get_strategy_name(self) -> str:
        return "R3_CAL_MOMENTUM"
    
    def get_strategy_family(self) -> str:
        return "independent"
    
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate R3 signal conditions."""
        orders = []
        signals = []
        state_updates = {}
        
        # Check if can fire
        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)
        
        # Step 1: Check for economic events (arming trigger)
        # In backtest, this would come from historical event data
        # For now, we'll simulate event detection based on price movement
        m5_bars = bar_data.get("M5", [])
        if len(m5_bars) < 2:
            return StrategyResult(orders, signals, state_updates)
        
        # Simulate event detection (would be real event feed in live system)
        if self._detect_economic_event(state, m5_bars, current_time):
            # Arm R3
            state_updates.update({
                "r3_armed": True,
                "r3_arm_time": current_time,
                "r3_direction": None,  # Will be determined after delay
                "r3_fired_today": False,
            })
            
            # Store pre-event price for direction determination
            current_price = m5_bars[-1]["close"]
            state_updates["ks7_pre_event_price"] = current_price
            
            signals.append({
                "type": "R3_ARMED",
                "event_time": current_time,
                "pre_event_price": current_price,
            })
            
            self.log_signal("ARMED", event_time=current_time, price=current_price)
        
        # Step 2: Check if armed and window has passed
        elif state.r3_armed and not state.r3_fired_today:
            # Check arming window
            arm_time = state.r3_arm_time
            if arm_time is None:
                state_updates["r3_armed"] = False
                return StrategyResult(orders, signals, state_updates)
            
            elapsed_min = (current_time - arm_time).total_seconds() / 60
            armed_window = self.config.get("R3_ARMED_WINDOW_MIN", 10)
            
            if elapsed_min > armed_window:
                # Window expired
                state_updates.update({
                    "r3_armed": False,
                    "r3_arm_time": None,
                    "r3_direction": None,
                })
                
                signals.append({
                    "type": "R3_WINDOW_EXPIRED",
                    "elapsed_min": elapsed_min,
                })
                
                self.log_signal("WINDOW_EXPIRED", elapsed_min=elapsed_min)
            
            elif elapsed_min >= 5:  # Minimum 5 min wait before evaluation
                # Step 3: Evaluate post-event direction
                direction = self._evaluate_post_event_direction(state, m5_bars)
                
                if direction:
                    # Step 4: Place trade
                    atr_h1 = indicators.get("atr_h1")
                    if atr_h1 is None or atr_h1 <= 0:
                        return StrategyResult(orders, signals, state_updates)
                    
                    # Calculate entry, stop, TP
                    current_price = m5_bars[-1]["close"]
                    
                    if direction == "LONG":
                        stop_distance = atr_h1 * 0.5
                        tp_distance = atr_h1 * 0.75
                        stop_price = current_price - stop_distance
                        tp_price = current_price + tp_distance
                    else:  # SHORT
                        stop_distance = atr_h1 * 0.5
                        tp_distance = atr_h1 * 0.75
                        stop_price = current_price + stop_distance
                        tp_price = current_price - tp_distance
                    
                    # Calculate lot size (0.5× base)
                    base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
                    regime_mult = state.size_multiplier
                    lot_size = self.calculate_lot_size(state, base_lot * 0.5, regime_mult)
                    
                    if lot_size > 0:
                        # Create MARKET order
                        order = self.create_order(
                            direction=direction,
                            order_type="MARKET",
                            price=current_price,
                            sl=stop_price,
                            tp=tp_price,
                            lots=lot_size,
                            tag="r3_momentum"
                        )
                        
                        orders.append(order)
                        
                        # Update state
                        state_updates.update({
                            "r3_fired_today": True,
                            "r3_armed": False,
                            "r3_arm_time": None,
                            "r3_direction": direction,
                            "r3_entry_price": current_price,
                            "r3_stop_price": stop_price,
                            "r3_tp_price": tp_price,
                        })
                        
                        signals.append({
                            "type": "R3_TRADE_PLACED",
                            "direction": direction,
                            "entry": current_price,
                            "stop": stop_price,
                            "tp": tp_price,
                            "lots": lot_size,
                            "elapsed_min": elapsed_min,
                        })
                        
                        self.log_signal("TRADE_PLACED",
                                       direction=direction,
                                       entry=current_price,
                                       stop=stop_price,
                                       tp=tp_price,
                                       lots=lot_size)
        
        return StrategyResult(orders, signals, state_updates)
    
    def _detect_economic_event(
        self,
        state: SimulatedState,
        m5_bars: List[Dict[str, Any]],
        current_time: datetime
    ) -> bool:
        """
        Detect economic events.
        In backtest, this would use historical event data.
        For simulation, we'll use price movement patterns.
        """
        # Simple simulation: detect large price movements
        if len(m5_bars) < 2:
            return False
        
        current_bar = m5_bars[-1]
        previous_bar = m5_bars[-2]
        
        # Calculate price movement
        price_move = abs(current_bar["close"] - previous_bar["close"])
        atr_m15 = self.config.get("ATR_M15_DEFAULT", 20)  # Fallback ATR
        
        # Trigger if movement > 0.5× ATR
        if price_move > (atr_m15 * 0.5):
            # Don't trigger if already armed recently
            if state.r3_armed and state.r3_arm_time:
                elapsed = (current_time - state.r3_arm_time).total_seconds() / 60
                if elapsed < 30:  # 30 min cooldown
                    return False
            
            return True
        
        return False
    
    def _evaluate_post_event_direction(
        self,
        state: SimulatedState,
        m5_bars: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Evaluate post-event price direction."""
        if len(m5_bars) < 2:
            return None
        
        pre_event_price = state.ks7_pre_event_price
        if pre_event_price <= 0:
            return None
        
        current_price = m5_bars[-1]["close"]
        
        # Determine direction based on price movement
        if current_price > pre_event_price * 1.001:  # 0.1% threshold
            return "LONG"
        elif current_price < pre_event_price * 0.999:  # -0.1% threshold
            return "SHORT"
        
        return None
    
    def _check_daily_limit(self, state: SimulatedState) -> bool:
        """Check if R3 has already fired today."""
        return state.r3_fired_today
    
    def _check_independent_lane(self, state: SimulatedState) -> tuple[bool, str]:
        """Check R3 independent lane availability."""
        # Block if R3 already open
        if state.r3_open_ticket:
            return False, "R3_LANE_OCCUPIED"
        
        return True, "OK"
    
    def _handle_unstable_regime(self, state: SimulatedState) -> tuple[bool, str]:
        """R3 blocked in UNSTABLE regime."""
        return False, "UNSTABLE_BLOCKED"
    
    def _check_regime_gates(self, state: SimulatedState) -> tuple[bool, str]:
        """R3 specific regime gates."""
        regime = state.current_regime
        
        # NO_TRADE blocks R3
        if regime == "NO_TRADE":
            return False, "NO_TRADE"
        
        # UNSTABLE blocks R3
        if regime == "UNSTABLE":
            return False, "UNSTABLE_BLOCKED"
        
        return True, "OK"
    
    def reset_daily_counters(self, state: SimulatedState):
        """Reset R3 daily counters."""
        state.r3_fired_today = False
        state.r3_armed = False
        state.r3_arm_time = None
        state.r3_direction = None
        state.ks7_pre_event_price = 0.0
