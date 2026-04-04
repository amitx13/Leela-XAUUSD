"""
S8 ATR Spike Strategy

Trades momentum spikes with high ATR.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S8AtrSpike(BaseStrategy):
    """S8 ATR Spike Momentum Strategy."""

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
        orders = []
        signals = []
        state_updates = {}

        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        m5_bars = bar_data.get("M5")
        current_bar  = self._get_bar(m5_bars, -1)
        previous_bar = self._get_bar(m5_bars, -2)
        if current_bar is None or previous_bar is None:
            return StrategyResult(orders, signals, state_updates)

        atr_m15      = indicators.get("atr_m15", 20)
        candle_range = float(current_bar["high"]) - float(current_bar["low"])
        spike_threshold = atr_m15 * self.config.get("SPIKE_ATR_MULT", 2.0)

        if candle_range < spike_threshold:
            return StrategyResult(orders, signals, state_updates)

        current_close  = float(current_bar["close"])
        previous_close = float(previous_bar["close"])
        bullish_spike  = current_close > previous_close

        direction   = "LONG" if bullish_spike else "SHORT"
        entry_price = current_close
        stop_price  = (float(current_bar["low"]) - atr_m15 * 0.3) if bullish_spike else (float(current_bar["high"]) + atr_m15 * 0.3)
        tp_price    = (entry_price + atr_m15 * 1.5) if bullish_spike else (entry_price - atr_m15 * 1.5)

        base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
        lot_size = self.calculate_lot_size(state, base_lot * 0.75, state.size_multiplier)

        if lot_size > 0:
            order = self.create_order(
                direction=direction,
                order_type="MARKET",
                price=entry_price,
                sl=stop_price,
                tp=tp_price,
                lots=lot_size,
                expiry=current_time + timedelta(hours=2),
                tag="s8_atr_spike"
            )
            orders.append(order)
            signals.append({
                "type":         "S8_ATR_SPIKE",
                "direction":    direction,
                "entry":        entry_price,
                "stop":         stop_price,
                "tp":           tp_price,
                "lots":         lot_size,
                "candle_range": candle_range,
                "atr":          atr_m15,
                "time":         current_time,
            })
            self.log_signal("ATR_SPIKE", direction=direction,
                            entry=entry_price, candle_range=candle_range)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return False

    def reset_daily_counters(self, state: SimulatedState):
        pass
