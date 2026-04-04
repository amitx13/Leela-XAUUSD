"""
S1 Family Strategies

All S1 family strategies: S1 London Breakout + variants.
Trend family strategies - occupy trend_family_occupied.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S1LondonBrk(BaseStrategy):
    """
    S1 London Breakout Strategy.

    Classic London session breakout from Asian range.
    """

    def get_strategy_name(self) -> str:
        return "S1_LONDON_BRK"

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

        if state.s1_family_attempts_today >= 3:
            return StrategyResult(orders, signals, state_updates)

        if not state.range_computed or state.range_size <= 0:
            return StrategyResult(orders, signals, state_updates)

        # FIX: use .iloc[-1] — bar_data["M5"] is a DataFrame, not a list
        m5_bars = bar_data.get("M5")
        current_bar = self._get_bar(m5_bars)
        if current_bar is None:
            return StrategyResult(orders, signals, state_updates)

        current_price = float(current_bar["close"])

        breakout_dist = state.range_size * self.config.get("BREAKOUT_DIST_PCT", 0.12)
        breakout_up   = current_price > (state.range_high + breakout_dist)
        breakout_down = current_price < (state.range_low  - breakout_dist)

        if breakout_up:
            entry_price = state.range_high + breakout_dist
            stop_price  = state.range_low  - 0.5
            base_lot    = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size    = self.calculate_lot_size(state, base_lot, state.size_multiplier)
            if lot_size > 0:
                orders.append(self.create_order(
                    direction="LONG", order_type="BUY_STOP",
                    price=entry_price, sl=stop_price, lots=lot_size,
                    expiry=current_time.replace(hour=18, minute=0, second=0),
                    tag="s1_london_breakout_long"
                ))
                state_updates.update({"s1_family_attempts_today": state.s1_family_attempts_today + 1, "last_s1_direction": "LONG"})
                signals.append({"type": "S1_BREAKOUT_LONG", "entry": entry_price, "stop": stop_price, "lots": lot_size, "time": current_time})
                self.log_signal("BREAKOUT_LONG", entry=entry_price, stop=stop_price, lots=lot_size)

        elif breakout_down:
            entry_price = state.range_low  - breakout_dist
            stop_price  = state.range_high + 0.5
            base_lot    = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size    = self.calculate_lot_size(state, base_lot, state.size_multiplier)
            if lot_size > 0:
                orders.append(self.create_order(
                    direction="SHORT", order_type="SELL_STOP",
                    price=entry_price, sl=stop_price, lots=lot_size,
                    expiry=current_time.replace(hour=18, minute=0, second=0),
                    tag="s1_london_breakout_short"
                ))
                state_updates.update({"s1_family_attempts_today": state.s1_family_attempts_today + 1, "last_s1_direction": "SHORT"})
                signals.append({"type": "S1_BREAKOUT_SHORT", "entry": entry_price, "stop": stop_price, "lots": lot_size, "time": current_time})
                self.log_signal("BREAKOUT_SHORT", entry=entry_price, stop=stop_price, lots=lot_size)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return state.s1_family_attempts_today >= 3

    def _check_time_restrictions(self, state, current_time):
        if current_time.hour >= 18:
            return False, "TIME_KILL_1800UTC"
        if state.current_session not in ["LONDON", "LONDON_NY_OVERLAP"]:
            return False, "NOT_LONDON_SESSION"
        return True, "OK"

    def reset_daily_counters(self, state: SimulatedState):
        state.s1_family_attempts_today = 0
        state.s1f_attempts_today       = 0
        state.s1b_pending_ticket       = None
        state.s1d_ema_touched_today    = False
        state.s1d_fired_today          = False
        state.s1e_pyramid_done         = False
        state.s1f_post_tk_active       = False


class S1bFailedBrk(BaseStrategy):
    """S1b Failed Breakout Reversal Strategy."""

    def get_strategy_name(self) -> str:
        return "S1B_FAILED_BRK"

    def get_strategy_family(self) -> str:
        return "reversal"

    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}

        can_fire, _ = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        if state.s1b_pending_ticket:
            return StrategyResult(orders, signals, state_updates)

        if not state.failed_breakout_flag:
            return StrategyResult(orders, signals, state_updates)

        # FIX: .iloc[-1]
        m5_bars = bar_data.get("M5")
        current_bar = self._get_bar(m5_bars)
        if current_bar is None or not state.range_computed:
            return StrategyResult(orders, signals, state_updates)

        current_price       = float(current_bar["close"])
        breakout_direction  = state.failed_breakout_direction

        if not breakout_direction:
            return StrategyResult(orders, signals, state_updates)

        direction = None
        if breakout_direction == "LONG" and current_price < state.range_high:
            entry_price = current_price
            stop_price  = state.range_high + 0.5
            direction   = "SHORT"
        elif breakout_direction == "SHORT" and current_price > state.range_low:
            entry_price = current_price
            stop_price  = state.range_low - 0.5
            direction   = "LONG"

        if direction:
            base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
            lot_size = self.calculate_lot_size(state, base_lot * 0.8, state.size_multiplier)
            if lot_size > 0:
                orders.append(self.create_order(
                    direction=direction, order_type="MARKET",
                    price=entry_price, sl=stop_price, lots=lot_size,
                    tag="s1b_failed_breakout_reversal"
                ))
                state_updates.update({
                    "s1b_pending_ticket": f"S1B_{current_time.strftime('%Y%m%d_%H%M')}",
                    "failed_breakout_flag": False,
                    "failed_breakout_direction": None,
                })
                signals.append({"type": "S1B_REVERSAL_PLACED", "direction": direction, "entry": entry_price, "stop": stop_price, "lots": lot_size, "time": current_time})
                self.log_signal("REVERSAL_PLACED", direction=direction, entry=entry_price, stop=stop_price, lots=lot_size)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state):
        return False

    def reset_daily_counters(self, state):
        state.s1b_pending_ticket = None


class S1cStopHunt(BaseStrategy):
    """S1c Stop Hunt Detection Strategy."""

    def get_strategy_name(self) -> str:
        return "S1C_STOP_HUNT"

    def get_strategy_family(self) -> str:
        return "reversal"

    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}

        can_fire, _ = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        if state.s3_fired_today:
            return StrategyResult(orders, signals, state_updates)

        # FIX: .iloc[-1] and .iloc[-2]
        m5_bars = bar_data.get("M5")
        current_bar  = self._get_bar(m5_bars, -1)
        previous_bar = self._get_bar(m5_bars, -2)
        if current_bar is None or previous_bar is None:
            return StrategyResult(orders, signals, state_updates)

        candle_range = float(current_bar["high"]) - float(current_bar["low"])
        atr_m15      = indicators.get("atr_m15", 20)

        is_sweep = (
            float(current_bar["low"]) < float(previous_bar["low"]) * 0.99
            and candle_range > atr_m15 * 1.5
        )

        if is_sweep:
            state_updates.update({
                "s3_sweep_candle_time": current_time,
                "s3_sweep_low":         float(current_bar["low"]),
                "s3_fired_today":       True,
                "s3_sweep_direction":   "SHORT",
            })
            signals.append({"type": "S1C_STOP_HUNT_DETECTED", "direction": "SHORT", "sweep_low": float(current_bar["low"]), "candle_range": candle_range, "atr": atr_m15, "time": current_time})
            self.log_signal("STOP_HUNT_DETECTED", direction="SHORT", sweep_low=float(current_bar["low"]), range=candle_range)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state):
        return state.s3_fired_today

    def reset_daily_counters(self, state):
        state.s3_fired_today         = False
        state.s3_sweep_candle_time   = None
        state.s3_sweep_low           = 0.0
        state.s3_sweep_direction     = None


class S1dPyramid(BaseStrategy):
    """S1d: M5 Pullback Re-entries."""

    def get_strategy_name(self) -> str:
        return "S1D_PYRAMID"

    def get_strategy_family(self) -> str:
        return "trend"

    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}

        if not state.open_position or not state.trend_family_occupied:
            return StrategyResult(orders, signals, state_updates)

        pyramid_count = getattr(state, "s1d_pyramid_count", 0)
        if pyramid_count >= 2:
            return StrategyResult(orders, signals, state_updates)

        current_price = self._get_current_price(bar_data)
        if current_price == 0.0:
            return StrategyResult(orders, signals, state_updates)

        if state.position_be_activated:
            # FIX: use .iloc[-1] via pandas_ta
            m5_df = bar_data.get("M5")
            if m5_df is not None and len(m5_df) >= 20:
                import pandas_ta as ta
                ema20 = ta.ema(m5_df["close"], length=20).iloc[-1]
                if abs(current_price - ema20) < 0.5:
                    pyramid_size = state.original_lot_size * 0.5
                    direction    = state.trend_trade_direction
                    atr          = indicators.get("atr_m15", 20)
                    if direction == "LONG":
                        order = SimOrder(strategy=self.get_strategy_name(), direction="LONG",  order_type="MARKET", price=current_price, sl=current_price - atr * 1.5, tp=current_price + atr * 2.0, lots=pyramid_size, expiry=current_time + timedelta(hours=4))
                    else:
                        order = SimOrder(strategy=self.get_strategy_name(), direction="SHORT", order_type="MARKET", price=current_price, sl=current_price + atr * 1.5, tp=current_price - atr * 2.0, lots=pyramid_size, expiry=current_time + timedelta(hours=4))
                    orders.append(order)
                    state_updates["s1d_pyramid_count"] = pyramid_count + 1
                    signals.append({"strategy": self.get_strategy_name(), "type": "PYRAMID_ENTRY", "direction": direction, "price": current_price, "size": pyramid_size, "reason": "M5_PULLBACK_TO_EMA20"})

        return StrategyResult(orders, signals, state_updates)


class S1ePyramid(BaseStrategy):
    """S1e: Pyramid Into Winners (Aggressive)."""

    def get_strategy_name(self) -> str:
        return "S1E_PYRAMID"

    def get_strategy_family(self) -> str:
        return "trend"

    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}

        if not state.open_position or not state.trend_family_occupied:
            return StrategyResult(orders, signals, state_updates)

        pyramid_count = getattr(state, "s1e_pyramid_count", 0)
        if pyramid_count >= 3:
            return StrategyResult(orders, signals, state_updates)

        current_price = self._get_current_price(bar_data)
        if current_price == 0.0:
            return StrategyResult(orders, signals, state_updates)

        entry_price = state.entry_price
        atr = indicators.get("atr_m15", 20)

        if state.trend_trade_direction == "LONG":
            current_r = (current_price - entry_price) / atr
        else:
            current_r = (entry_price - current_price) / atr

        if current_r > 1.0:
            # FIX: .iloc[-1] / .iloc[-2]
            m5_df = bar_data.get("M5")
            if m5_df is not None and len(m5_df) >= 3:
                last_close = float(m5_df["close"].iloc[-1])
                prev_close = float(m5_df["close"].iloc[-2])
                pullback = (last_close < prev_close) if state.trend_trade_direction == "LONG" else (last_close > prev_close)
                if pullback:
                    pyramid_size = state.original_lot_size * 0.75
                    direction    = state.trend_trade_direction
                    if direction == "LONG":
                        order = SimOrder(strategy=self.get_strategy_name(), direction="LONG",  order_type="MARKET", price=current_price, sl=current_price - atr * 1.2, tp=current_price + atr * 2.5, lots=pyramid_size, expiry=current_time + timedelta(hours=3))
                    else:
                        order = SimOrder(strategy=self.get_strategy_name(), direction="SHORT", order_type="MARKET", price=current_price, sl=current_price + atr * 1.2, tp=current_price - atr * 2.5, lots=pyramid_size, expiry=current_time + timedelta(hours=3))
                    orders.append(order)
                    state_updates["s1e_pyramid_count"] = pyramid_count + 1
                    signals.append({"strategy": self.get_strategy_name(), "type": "AGGRESSIVE_PYRAMID", "direction": direction, "price": current_price, "size": pyramid_size, "reason": "MINOR_PULLBACK_R>1.0"})

        return StrategyResult(orders, signals, state_updates)


class S1fPostTk(BaseStrategy):
    """S1f: Post-Time-Kill Re-entries."""

    def get_strategy_name(self) -> str:
        return "S1F_POST_TK"

    def get_strategy_family(self) -> str:
        return "trend"

    def evaluate(self, state, bar_data, current_time, indicators):
        orders = []
        signals = []
        state_updates = {}

        if current_time.hour < 18:
            return StrategyResult(orders, signals, state_updates)

        if getattr(state, "s1f_reentered_today", False):
            return StrategyResult(orders, signals, state_updates)

        if not getattr(state, "s1f_killed_position_profitable", False):
            return StrategyResult(orders, signals, state_updates)

        if state.trend_family_occupied:
            return StrategyResult(orders, signals, state_updates)

        m15_df = bar_data.get("M15")
        if m15_df is None or len(m15_df) < 20:
            return StrategyResult(orders, signals, state_updates)

        current_price = self._get_current_price(bar_data)
        if current_price == 0.0:
            return StrategyResult(orders, signals, state_updates)

        # FIX: use pandas tail / max / min — already correct, no [-1] needed here
        recent_high = float(m15_df["high"].tail(15).max())
        recent_low  = float(m15_df["low"].tail(15).min())

        breakout_up   = current_price > recent_high
        breakout_down = current_price < recent_low

        if breakout_up or breakout_down:
            re_entry_size = getattr(state, "s1f_original_size", 1.0) * 0.75
            direction     = "LONG" if breakout_up else "SHORT"
            atr           = indicators.get("atr_m15", 20)
            if direction == "LONG":
                order = SimOrder(strategy=self.get_strategy_name(), direction="LONG",  order_type="STOP", price=current_price + 0.5, sl=recent_low  - atr * 0.5, tp=current_price + atr * 2.0, lots=re_entry_size, expiry=current_time + timedelta(hours=2))
            else:
                order = SimOrder(strategy=self.get_strategy_name(), direction="SHORT", order_type="STOP", price=current_price - 0.5, sl=recent_high + atr * 0.5, tp=current_price - atr * 2.0, lots=re_entry_size, expiry=current_time + timedelta(hours=2))
            orders.append(order)
            state_updates["s1f_reentered_today"] = True
            signals.append({"strategy": self.get_strategy_name(), "type": "POST_TK_REENTRY", "direction": direction, "price": current_price, "size": re_entry_size, "reason": "BREAKOUT_AFTER_TK"})

        return StrategyResult(orders, signals, state_updates)
