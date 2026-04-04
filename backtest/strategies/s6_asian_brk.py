"""
S6 Asian Breakout Strategy

Trades breakouts from the Asian session range.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S6AsianBrk(BaseStrategy):
    """S6 Asian Session Breakout Strategy."""

    def get_strategy_name(self) -> str:
        return "S6_ASIAN_BRK"

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

        # FIX Bug #5: set flag BEFORE evaluating to prevent duplicate orders
        if state.s6_placed_today:
            return StrategyResult(orders, signals, state_updates)

        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        session = state.current_session
        if session not in ["LONDON", "LONDON_NY_OVERLAP"]:
            return StrategyResult(orders, signals, state_updates)

        if not state.range_computed or state.range_size <= 0:
            return StrategyResult(orders, signals, state_updates)

        m5_bars = bar_data.get("M5")
        current_bar = self._get_bar(m5_bars, -1)
        if current_bar is None:
            return StrategyResult(orders, signals, state_updates)

        current_price = float(current_bar["close"])
        atr_m15       = indicators.get("atr_m15", 20)

        range_high = state.range_high
        range_low  = state.range_low

        buy_entry  = range_high + atr_m15 * 0.1
        sell_entry = range_low  - atr_m15 * 0.1
        buy_sl     = range_low  - atr_m15 * 0.3
        sell_sl    = range_high + atr_m15 * 0.3
        buy_tp     = buy_entry  + atr_m15 * 2.0
        sell_tp    = sell_entry - atr_m15 * 2.0

        base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
        lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)

        if lot_size > 0:
            state_updates["s6_placed_today"] = True  # set FIRST
            orders.append(self.create_order(
                direction="LONG", order_type="BUY_STOP",
                price=buy_entry, sl=buy_sl, tp=buy_tp,
                lots=lot_size,
                expiry=current_time.replace(hour=18, minute=0, second=0),
                tag="s6_asian_brk_long"
            ))
            orders.append(self.create_order(
                direction="SHORT", order_type="SELL_STOP",
                price=sell_entry, sl=sell_sl, tp=sell_tp,
                lots=lot_size,
                expiry=current_time.replace(hour=18, minute=0, second=0),
                tag="s6_asian_brk_short"
            ))
            signals.append({
                "type":       "S6_ASIAN_BRK_PLACED",
                "buy_entry":  buy_entry,
                "sell_entry": sell_entry,
                "lots":       lot_size,
                "time":       current_time,
            })
            self.log_signal("ASIAN_BRK_PLACED", buy_entry=buy_entry,
                            sell_entry=sell_entry, lots=lot_size)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return state.s6_placed_today

    def _check_independent_lane(self, state: SimulatedState) -> tuple[bool, str]:
        return True, "OK"

    def reset_daily_counters(self, state: SimulatedState):
        state.s6_placed_today = False
