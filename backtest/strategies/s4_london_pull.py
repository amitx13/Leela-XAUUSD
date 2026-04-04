"""
S4 London Pullback Strategy

Trades pullbacks during London session after initial breakout.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S4LondonPull(BaseStrategy):
    """S4 London Pullback Strategy."""

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
        orders = []
        signals = []
        state_updates = {}

        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        session = state.current_session
        if session not in ["LONDON", "LONDON_NY_OVERLAP"]:
            return StrategyResult(orders, signals, state_updates)

        if not state.range_computed or state.range_size <= 0:
            return StrategyResult(orders, signals, state_updates)

        m5_bars = bar_data.get("M5")
        current_bar  = self._get_bar(m5_bars, -1)
        previous_bar = self._get_bar(m5_bars, -2)
        if current_bar is None or previous_bar is None:
            return StrategyResult(orders, signals, state_updates)

        current_price  = float(current_bar["close"])
        previous_price = float(previous_bar["close"])
        atr_m15        = indicators.get("atr_m15", 20)

        above_range = current_price > state.range_high
        below_range = current_price < state.range_low

        if not (above_range or below_range):
            return StrategyResult(orders, signals, state_updates)

        if above_range:
            pullback = previous_price > current_price
            if pullback and current_price > state.range_high:
                direction   = "LONG"
                entry_price = current_price
                stop_price  = state.range_high - atr_m15 * 0.5
                tp_price    = current_price + atr_m15 * 2.0
            else:
                return StrategyResult(orders, signals, state_updates)
        else:
            pullback = previous_price < current_price
            if pullback and current_price < state.range_low:
                direction   = "SHORT"
                entry_price = current_price
                stop_price  = state.range_low + atr_m15 * 0.5
                tp_price    = current_price - atr_m15 * 2.0
            else:
                return StrategyResult(orders, signals, state_updates)

        base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
        lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)

        if lot_size > 0:
            order = self.create_order(
                direction=direction,
                order_type="MARKET",
                price=entry_price,
                sl=stop_price,
                tp=tp_price,
                lots=lot_size,
                expiry=current_time.replace(hour=18, minute=0, second=0),
                tag="s4_london_pullback"
            )
            orders.append(order)
            signals.append({
                "type":      "S4_PULLBACK_PLACED",
                "direction": direction,
                "entry":     entry_price,
                "stop":      stop_price,
                "tp":        tp_price,
                "lots":      lot_size,
                "time":      current_time,
            })
            self.log_signal("PULLBACK_PLACED", direction=direction,
                            entry=entry_price, stop=stop_price, lots=lot_size)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return False

    def reset_daily_counters(self, state: SimulatedState):
        pass
