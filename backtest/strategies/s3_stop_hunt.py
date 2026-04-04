"""
S3 Stop Hunt Reversal Strategy

Detects stop hunts and trades the reversal.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from .base_strategy import BaseStrategy, StrategyResult
from ..models import SimOrder, SimulatedState


class S3StopHuntRev(BaseStrategy):
    """
    S3 Stop Hunt Reversal Strategy.

    Detects institutional stop hunts and trades the reversal.

    Logic:
    1. DETECT: Large spike candle that sweeps stops (1.5x ATR)
    2. ARM: Store sweep low/high and direction
    3. CONFIRM: Next candle confirms reversal (closes back inside range)
    4. TRADE: Market order in reversal direction

    Constraints:
    - Reversal family strategy
    - Max 1 trade per day
    - Allowed in UNSTABLE regime (stop hunts create volatility)
    - Tight stops (0.5x ATR)
    """

    def get_strategy_name(self) -> str:
        return "S3_STOP_HUNT_REV"

    def get_strategy_family(self) -> str:
        return "reversal"

    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """Evaluate S3 stop hunt conditions."""
        orders = []
        signals = []
        state_updates = {}

        can_fire, reason = self.can_fire(state, current_time)
        if not can_fire:
            return StrategyResult(orders, signals, state_updates)

        if state.s3_fired_today:
            return StrategyResult(orders, signals, state_updates)

        m5_bars = bar_data.get("M5")
        current_bar  = self._get_bar(m5_bars, -1)
        previous_bar = self._get_bar(m5_bars, -2)
        if current_bar is None or previous_bar is None:
            return StrategyResult(orders, signals, state_updates)

        atr_m15 = indicators.get("atr_m15")
        if atr_m15 is None or atr_m15 <= 0:
            return StrategyResult(orders, signals, state_updates)

        sweep_threshold = atr_m15 * 1.5
        candle_range    = float(current_bar["high"]) - float(current_bar["low"])

        is_sweep_down = (
            float(current_bar["low"]) < float(previous_bar["low"]) * 0.99
            and candle_range > sweep_threshold
        )
        is_sweep_up = (
            float(current_bar["high"]) > float(previous_bar["high"]) * 1.01
            and candle_range > sweep_threshold
        )

        if is_sweep_down or is_sweep_up:
            sweep_direction = "SHORT" if is_sweep_down else "LONG"
            sweep_level     = float(current_bar["low"]) if is_sweep_down else float(current_bar["high"])

            state_updates.update({
                "s3_sweep_candle_time": current_time,
                "s3_sweep_low":         float(current_bar["low"]),
                "s3_fired_today":       False,
                "s3_sweep_direction":   sweep_direction,
            })
            signals.append({
                "type":         "S3_SWEEP_DETECTED",
                "direction":    sweep_direction,
                "sweep_level":  sweep_level,
                "candle_range": candle_range,
                "atr":          atr_m15,
                "time":         current_time,
            })
            self.log_signal("SWEEP_DETECTED", direction=sweep_direction,
                            sweep_level=sweep_level, range=candle_range)

        elif state.s3_sweep_candle_time and not state.s3_fired_today:
            elapsed = (current_time - state.s3_sweep_candle_time).total_seconds() / 60
            if elapsed < 5:
                return StrategyResult(orders, signals, state_updates)

            sweep_direction = state.s3_sweep_direction
            current_price   = float(current_bar["close"])
            confirmed       = False
            direction       = None
            entry_price     = 0.0
            stop_price      = 0.0

            if sweep_direction == "SHORT":
                confirmed = current_price > (state.s3_sweep_low + atr_m15 * 0.2)
                if confirmed:
                    entry_price = current_price
                    stop_price  = state.s3_sweep_low - 0.5
                    direction   = "LONG"
            else:
                confirmed = current_price < (state.s3_sweep_low + atr_m15 * 0.2)
                if confirmed:
                    entry_price = current_price
                    stop_price  = state.s3_sweep_low + 0.5
                    direction   = "SHORT"

            if confirmed and direction:
                base_lot = self.config.get("BASE_LOT_SIZE", 0.01)
                lot_size = self.calculate_lot_size(state, base_lot, state.size_multiplier)

                if lot_size > 0:
                    order = self.create_order(
                        direction=direction,
                        order_type="MARKET",
                        price=entry_price,
                        sl=stop_price,
                        lots=lot_size,
                        tag="s3_stop_hunt_reversal"
                    )
                    orders.append(order)

                    state_updates.update({
                        "s3_fired_today":      True,
                        "s3_sweep_candle_time": None,
                        "s3_sweep_low":         0.0,
                        "s3_sweep_direction":   None,
                    })
                    signals.append({
                        "type":            "S3_REVERSAL_PLACED",
                        "direction":       direction,
                        "entry":           entry_price,
                        "stop":            stop_price,
                        "lots":            lot_size,
                        "elapsed_minutes": elapsed,
                        "time":            current_time,
                    })
                    self.log_signal("REVERSAL_PLACED", direction=direction,
                                    entry=entry_price, stop=stop_price,
                                    lots=lot_size, elapsed=elapsed)

        return StrategyResult(orders, signals, state_updates)

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return state.s3_fired_today

    def _handle_unstable_regime(self, state: SimulatedState) -> tuple[bool, str]:
        return True, "UNSTABLE_ALLOWED"

    def reset_daily_counters(self, state: SimulatedState):
        state.s3_fired_today       = False
        state.s3_sweep_candle_time = None
        state.s3_sweep_low         = 0.0
        state.s3_sweep_direction   = None
