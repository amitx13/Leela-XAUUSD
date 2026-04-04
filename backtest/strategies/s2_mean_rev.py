"""
S2 Mean Reversion Strategy

Mean reversion from extreme levels.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S2MeanRev(BaseStrategy):
    """S2 Mean Reversion Strategy."""

    def get_strategy_name(self) -> str:
        return "S2_MEAN_REV"

    def get_strategy_family(self) -> str:
        return "reversal"

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

        if state.s2_fired_today:
            return StrategyResult(orders, signals, state_updates)

        m5_bars = bar_data.get("M5")
        current_bar = self._get_bar(m5_bars, -1)
        if current_bar is None:
            return StrategyResult(orders, signals, state_updates)

        current_price = float(current_bar["close"])

        atr_m15 = indicators.get("atr_m15", 20)
        ema_h1  = indicators.get("ema_h1", current_price)

        deviation = abs(current_price - ema_h1)
        mean_rev_threshold = atr_m15 * self.config.get("MEAN_REV_ATR_MULT", 2.5)

        if deviation < mean_rev_threshold:
            return StrategyResult(orders, signals, state_updates)

        if current_price > ema_h1:
            direction   = "SHORT"
            entry_price = current_price
            stop_price  = current_price + atr_m15 * 1.5
            tp_price    = ema_h1
        else:
            direction   = "LONG"
            entry_price = current_price
            stop_price  = current_price - atr_m15 * 1.5
            tp_price    = ema_h1

        base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
        lot_size = self.calculate_lot_size(state, base_lot * 0.8, state.size_multiplier)

        if lot_size > 0:
            order = self.create_order(
                direction=direction,
                order_type="MARKET",
                price=entry_price,
                sl=stop_price,
                tp=tp_price,
                lots=lot_size,
                tag="s2_mean_reversion"
            )
            orders.append(order)
            state_updates["s2_fired_today"] = True
            signals.append({
                "type":      "S2_MEAN_REV_PLACED",
                "direction": direction,
                "entry":     entry_price,
                "stop":      stop_price,
                "tp":        tp_price,
                "lots":      lot_size,
                "deviation": deviation,
                "time":      current_time,
            })
            self.log_signal("MEAN_REV_PLACED", direction=direction,
                            entry=entry_price, stop=stop_price,
                            tp=tp_price, deviation=deviation)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return state.s2_fired_today

    def reset_daily_counters(self, state: SimulatedState):
        state.s2_fired_today = False
