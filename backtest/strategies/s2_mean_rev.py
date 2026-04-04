"""
S2 Mean Reversion Strategy

Mean reversion based on overextended conditions.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S2MeanRev(BaseStrategy):
    """
    S2 Mean Reversion Strategy.

    Trades price extremes back to the mean.
    """

    def get_strategy_name(self) -> str:
        return "S2_MEAN_REV"

    def get_strategy_family(self) -> str:
        return "trend"

    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}

        can_fire, _ = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        if state.s2_fired_today:
            return StrategyResult(orders, signals, state_updates)

        rsi_h1   = indicators.get("rsi_h1")
        ema20_h1 = indicators.get("ema20_h1")
        atr_h1   = indicators.get("atr_h1")

        if None in [rsi_h1, ema20_h1, atr_h1]:
            return StrategyResult(orders, signals, state_updates)

        # FIX: .iloc[-1] via helper
        current_price = self._get_current_price(bar_data)
        if current_price == 0.0:
            return StrategyResult(orders, signals, state_updates)

        ema_distance     = abs(current_price - ema20_h1)
        ema_distance_pct = ema_distance / ema20_h1 * 100

        overextended_rsi = rsi_h1 > 70 or rsi_h1 < 30
        overextended_ema = ema_distance_pct > 0.5

        if overextended_rsi and overextended_ema:
            if rsi_h1 > 70:
                direction   = "SHORT"
                entry_price = current_price
                stop_price  = current_price + (atr_h1 * 0.5)
            else:
                direction   = "LONG"
                entry_price = current_price
                stop_price  = current_price - (atr_h1 * 0.5)

            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)

            if lot_size > 0:
                orders.append(self.create_order(
                    direction=direction, order_type="MARKET",
                    price=entry_price, sl=stop_price, lots=lot_size,
                    tag="s2_mean_reversion"
                ))
                state_updates["s2_fired_today"] = True
                signals.append({"type": "S2_MEAN_REVERSION_PLACED", "direction": direction, "entry": entry_price, "stop": stop_price, "lots": lot_size, "rsi": rsi_h1, "ema_distance_pct": ema_distance_pct, "time": current_time})
                self.log_signal("MEAN_REVERSION_PLACED", direction=direction, entry=entry_price, stop=stop_price, lots=lot_size, rsi=rsi_h1)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state):
        return state.s2_fired_today

    def _check_time_restrictions(self, state, current_time):
        hour = current_time.hour
        if 8  <= hour <= 10: return False, "NEWS_BLACKOUT"
        if 11 <= hour <= 12: return False, "VOLATILE_PERIOD"
        return True, "OK"

    def reset_daily_counters(self, state):
        state.s2_fired_today = False
