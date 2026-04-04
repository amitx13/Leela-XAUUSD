"""
S5 NY Compression Strategy

Trades NY session compression breakouts.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S5NyCompress(BaseStrategy):
    """S5 NY Compression Breakout Strategy."""

    def get_strategy_name(self) -> str:
        return "S5_NY_COMPRESS"

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
        if session not in ["NEW_YORK", "LONDON_NY_OVERLAP"]:
            return StrategyResult(orders, signals, state_updates)

        m5_df = bar_data.get("M5")
        if m5_df is None or len(m5_df) < 12:
            return StrategyResult(orders, signals, state_updates)

        current_bar = self._get_bar(m5_df, -1)
        if current_bar is None:
            return StrategyResult(orders, signals, state_updates)

        recent_high = float(m5_df["high"].tail(12).max())
        recent_low  = float(m5_df["low"].tail(12).min())
        compress_range = recent_high - recent_low

        atr_m15 = indicators.get("atr_m15", 20)
        if compress_range > atr_m15 * self.config.get("COMPRESS_MAX_ATR", 0.8):
            return StrategyResult(orders, signals, state_updates)

        current_price = float(current_bar["close"])

        breakout_up   = current_price > recent_high
        breakout_down = current_price < recent_low

        if not (breakout_up or breakout_down):
            return StrategyResult(orders, signals, state_updates)

        direction   = "LONG" if breakout_up else "SHORT"
        entry_price = current_price
        stop_price  = (recent_low - atr_m15 * 0.5) if breakout_up else (recent_high + atr_m15 * 0.5)
        tp_price    = (current_price + atr_m15 * 2.0) if breakout_up else (current_price - atr_m15 * 2.0)

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
                expiry=current_time + timedelta(hours=4),
                tag="s5_ny_compression"
            )
            orders.append(order)
            signals.append({
                "type":           "S5_COMPRESSION_BRK",
                "direction":      direction,
                "entry":          entry_price,
                "stop":           stop_price,
                "tp":             tp_price,
                "lots":           lot_size,
                "compress_range": compress_range,
                "time":           current_time,
            })
            self.log_signal("COMPRESSION_BRK", direction=direction,
                            entry=entry_price, compress_range=compress_range)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return False

    def reset_daily_counters(self, state: SimulatedState):
        pass
