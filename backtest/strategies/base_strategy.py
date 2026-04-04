"""
Base Strategy Class

Foundation for all backtest strategies.
Provides common functionality and interface for strategy evaluation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import pandas as pd

from ..models import SimOrder, SimulatedState


@dataclass
class StrategyResult:
    """Result of strategy evaluation."""
    orders: List[SimOrder]
    signals: List[Dict[str, Any]]
    state_updates: Dict[str, Any]


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    Each strategy must implement:
    - evaluate(): Main signal evaluation logic
    - get_strategy_name(): Return strategy identifier
    - get_strategy_family(): Return strategy family for position management

    Strategies should mirror live system behavior exactly.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_name = self.get_strategy_name()
        self.strategy_family = self.get_strategy_family()

    # ------------------------------------------------------------------
    # DataFrame helpers — use these everywhere instead of df[-1]
    # ------------------------------------------------------------------

    @staticmethod
    def _get_bar(df, n: int = -1) -> Optional[pd.Series]:
        """
        Safe iloc accessor for a DataFrame row.

        Args:
            df:  pandas DataFrame (bar data) or None
            n:   row index — negative counts from end (-1 = last bar)

        Returns:
            pd.Series for that row, or None if df is empty / too short.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        if abs(n) > len(df):
            return None
        return df.iloc[n]

    def _get_current_price(self, bar_data: Dict[str, Any]) -> float:
        """
        Return the most recent M5 close price.

        Returns 0.0 if M5 data is unavailable so callers can guard with
        ``if current_price == 0.0: return ...``
        """
        bar = self._get_bar(bar_data.get("M5"))
        if bar is None:
            return 0.0
        return float(bar["close"])

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        state: SimulatedState,
        bar_data: Dict[str, Any],
        current_time: datetime,
        indicators: Dict[str, Any]
    ) -> StrategyResult:
        """
        Evaluate strategy conditions and generate signals.

        Args:
            state: Current simulation state
            bar_data: OHLCV data for relevant timeframes (values are DataFrames)
            current_time: Current timestamp
            indicators: Pre-calculated indicators (ADX, ATR, EMA, etc.)

        Returns:
            StrategyResult with orders, signals, and state updates
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the strategy name matching live system."""
        pass

    @abstractmethod
    def get_strategy_family(self) -> str:
        """Return strategy family: 'trend', 'reversal', or 'independent'."""
        pass

    # ------------------------------------------------------------------
    # Common firing checks
    # ------------------------------------------------------------------

    def can_fire(
        self,
        state: SimulatedState,
        current_time: datetime
    ) -> tuple[bool, str]:
        """
        Common firing checks for all strategies.

        Returns:
            (can_fire, reason) tuple
        """
        if not state.trading_enabled:
            return False, "TRADING_DISABLED"

        if self._check_daily_limit(state):
            return False, "DAILY_LIMIT_REACHED"

        can_fire_regime, regime_reason = self._check_regime_gates(state)
        if not can_fire_regime:
            return False, f"REGIME_{regime_reason}"

        can_fire_position, position_reason = self._check_position_family(state)
        if not can_fire_position:
            return False, f"POSITION_{position_reason}"

        can_fire_time, time_reason = self._check_time_restrictions(state, current_time)
        if not can_fire_time:
            return False, f"TIME_{time_reason}"

        return True, "OK"

    def _check_daily_limit(self, state: SimulatedState) -> bool:
        return False

    def _check_regime_gates(self, state: SimulatedState) -> tuple[bool, str]:
        regime = state.current_regime
        if regime == "NO_TRADE":
            return False, "NO_TRADE"
        if regime == "UNSTABLE":
            return self._handle_unstable_regime(state)
        return True, "OK"

    def _handle_unstable_regime(self, state: SimulatedState) -> tuple[bool, str]:
        return False, "UNSTABLE_BLOCKED"

    def _check_position_family(self, state: SimulatedState) -> tuple[bool, str]:
        family = self.strategy_family
        if family == "trend":
            if state.trend_family_occupied:
                return False, "TREND_FAMILY_OCCUPIED"
        elif family == "reversal":
            if state.reversal_family_occupied:
                return False, "REVERSAL_FAMILY_OCCUPIED"
        elif family == "independent":
            return self._check_independent_lane(state)
        return True, "OK"

    def _check_independent_lane(self, state: SimulatedState) -> tuple[bool, str]:
        return True, "OK"

    def _check_time_restrictions(
        self,
        state: SimulatedState,
        current_time: datetime
    ) -> tuple[bool, str]:
        session = state.current_session
        if session == "OFF_HOURS":
            return False, "OFF_HOURS"
        return True, "OK"

    # ------------------------------------------------------------------
    # Order creation and sizing
    # ------------------------------------------------------------------

    def create_order(
        self,
        direction: str,
        order_type: str,
        price: float,
        sl: float,
        tp: Optional[float] = None,
        lots: float = 0.01,
        expiry: Optional[datetime] = None,
        tag: str = "",
        linked_tag: str = ""
    ) -> SimOrder:
        """Create a SimOrder with strategy metadata."""
        return SimOrder(
            strategy=self.strategy_name,
            direction=direction,
            order_type=order_type,
            price=price,
            sl=sl,
            tp=tp,
            lots=lots,
            expiry=expiry,
            placed_time=datetime.utcnow(),
            tag=tag,
            linked_tag=linked_tag
        )

    def calculate_lot_size(
        self,
        state: SimulatedState,
        base_lot: float,
        regime_multiplier: float,
        strategy_multiplier: float = 1.0
    ) -> float:
        """Calculate position size with all multipliers."""
        if not state.trading_enabled:
            return 0.0
        size = base_lot * regime_multiplier * strategy_multiplier
        min_lot = self.config.get("MIN_LOT_SIZE", 0.01)
        size = max(size, min_lot)
        max_lot = self.config.get("MAX_LOT_SIZE", 1.0)
        size = min(size, max_lot)
        return round(size, 2)

    def log_signal(self, signal_type: str, **kwargs):
        import logging
        logger = logging.getLogger(f"backtest.{self.strategy_name}")
        logger.info(f"SIGNAL_{signal_type} {kwargs}")

    def update_state_counters(self, state: SimulatedState):
        pass

    def reset_daily_counters(self, state: SimulatedState):
        pass
