"""
R3 Calendar Momentum Strategy

Trades momentum around high-impact calendar events.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class R3CalMomentum(BaseStrategy):
    """R3 Calendar Momentum Strategy."""

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
        orders = []
        signals = []
        state_updates = {}

        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        upcoming_events = indicators.get("upcoming_events", [])
        if not upcoming_events:
            return StrategyResult(orders, signals, state_updates)

        next_event = upcoming_events[0]
        event_time = next_event.get("time")
        if not event_time:
            return StrategyResult(orders, signals, state_updates)

        minutes_to_event = (event_time - current_time).total_seconds() / 60
        if not (5 <= minutes_to_event <= 30):
            return StrategyResult(orders, signals, state_updates)

        m5_bars = bar_data.get("M5")
        current_bar  = self._get_bar(m5_bars, -1)
        previous_bar = self._get_bar(m5_bars, -2)
        if current_bar is None or previous_bar is None:
            return StrategyResult(orders, signals, state_updates)

        current_price  = float(current_bar["close"])
        previous_price = float(previous_bar["close"])
        atr_m15        = indicators.get("atr_m15", 20)

        momentum_up   = current_price > previous_price + atr_m15 * 0.3
        momentum_down = current_price < previous_price - atr_m15 * 0.3

        if not (momentum_up or momentum_down):
            return StrategyResult(orders, signals, state_updates)

        direction   = "LONG" if momentum_up else "SHORT"
        entry_price = current_price
        stop_price  = (current_price - atr_m15 * 1.0) if momentum_up else (current_price + atr_m15 * 1.0)
        tp_price    = (current_price + atr_m15 * 2.5) if momentum_up else (current_price - atr_m15 * 2.5)

        base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
        lot_size = self.calculate_lot_size(state, base_lot * 1.2, state.size_multiplier)

        if lot_size > 0:
            order = self.create_order(
                direction=direction,
                order_type="MARKET",
                price=entry_price,
                sl=stop_price,
                tp=tp_price,
                lots=lot_size,
                expiry=event_time + timedelta(hours=1),
                tag="r3_calendar_momentum"
            )
            orders.append(order)
            signals.append({
                "type":             "R3_CAL_MOMENTUM",
                "direction":        direction,
                "entry":            entry_price,
                "stop":             stop_price,
                "tp":               tp_price,
                "lots":             lot_size,
                "event":            next_event.get("name", "UNKNOWN"),
                "minutes_to_event": minutes_to_event,
                "time":             current_time,
            })
            self.log_signal("CAL_MOMENTUM", direction=direction,
                            event=next_event.get("name", "UNKNOWN"),
                            minutes_to_event=minutes_to_event)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return False

    def _check_independent_lane(self, state: SimulatedState) -> tuple[bool, str]:
        return True, "OK"

    def reset_daily_counters(self, state: SimulatedState):
        pass
