"""
S7 Daily Structure Strategy

Trades daily structure breakouts.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S7DailyStruct(BaseStrategy):
    """S7 Daily Structure Breakout Strategy."""

    def get_strategy_name(self) -> str:
        return "S7_DAILY_STRUCT"

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

        # FIX Bug #5: set flag FIRST
        if state.s7_placed_today:
            return StrategyResult(orders, signals, state_updates)

        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        session = state.current_session
        if session not in ["LONDON", "LONDON_NY_OVERLAP", "NEW_YORK"]:
            return StrategyResult(orders, signals, state_updates)

        d1_df = bar_data.get("D1")
        prev_day = self._get_bar(d1_df, -1)
        if prev_day is None:
            return StrategyResult(orders, signals, state_updates)

        prev_high = float(prev_day["high"])
        prev_low  = float(prev_day["low"])

        m5_bars = bar_data.get("M5")
        current_bar = self._get_bar(m5_bars, -1)
        if current_bar is None:
            return StrategyResult(orders, signals, state_updates)

        atr_m15 = indicators.get("atr_m15", 20)

        buy_entry  = prev_high + atr_m15 * 0.1
        sell_entry = prev_low  - atr_m15 * 0.1
        buy_sl     = prev_high - atr_m15 * 1.0
        sell_sl    = prev_low  + atr_m15 * 1.0
        # FIX Bug #4: always set TP for S7
        buy_tp     = round(buy_entry  + atr_m15 * 2.0, 3)
        sell_tp    = round(sell_entry - atr_m15 * 2.0, 3)

        base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
        lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)

        if lot_size > 0:
            state_updates["s7_placed_today"] = True  # set FIRST
            orders.append(self.create_order(
                direction="LONG", order_type="BUY_STOP",
                price=buy_entry, sl=buy_sl, tp=buy_tp,
                lots=lot_size,
                expiry=current_time.replace(hour=23, minute=59, second=0),
                tag="s7_daily_struct_long"
            ))
            orders.append(self.create_order(
                direction="SHORT", order_type="SELL_STOP",
                price=sell_entry, sl=sell_sl, tp=sell_tp,
                lots=lot_size,
                expiry=current_time.replace(hour=23, minute=59, second=0),
                tag="s7_daily_struct_short"
            ))
            signals.append({
                "type":       "S7_DAILY_STRUCT_PLACED",
                "buy_entry":  buy_entry,
                "sell_entry": sell_entry,
                "prev_high":  prev_high,
                "prev_low":   prev_low,
                "lots":       lot_size,
                "time":       current_time,
            })
            self.log_signal("DAILY_STRUCT_PLACED", buy_entry=buy_entry,
                            sell_entry=sell_entry, lots=lot_size)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return state.s7_placed_today

    def _check_independent_lane(self, state: SimulatedState) -> tuple[bool, str]:
        return True, "OK"

    def reset_daily_counters(self, state: SimulatedState):
        state.s7_placed_today = False
