"""
Enhanced Backtest Engine

Complete rewrite using modular strategy system with all 13 strategies.
Mirrors live system behaviour exactly.

Fixes applied vs original broken version:
  1. Loop driven by BarBuffer.add_m5() — no clock-tick loop, real bar events
  2. MT5 data via EnhancedHistoricalDataFeed (no fake random bars)
  3. _manage_r3_position receives current_time parameter
  4. SL/TP exit logic is direction-aware (LONG and SHORT)
  5. exit_time uses simulation current_time, not datetime.utcnow()
  6. Session classification uses utils.session.get_session_for_datetime()
"""

import sys
import os
import logging
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from backtest.models import (
    SimOrder, SimPosition, TradeRecord, EquityPoint, SimulatedState,
)
from backtest.data_feed import BarBuffer
from backtest.data_feed import compute_atr_percentile  # reuse existing util
from backtest.data_feed_enhanced import EnhancedHistoricalDataFeed
from backtest.execution_simulator_enhanced import EnhancedExecutionSimulator
from backtest.strategies import STRATEGY_REGISTRY, ALL_STRATEGIES
from backtest.risk import run_all_kill_switches, check_position_limits
from backtest.results import BacktestResults

logger = logging.getLogger("backtest.enhanced_engine")

# Add parent dir to path for config imports
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    import config
except ImportError:
    config = None  # type: ignore


class EnhancedBacktestEngine:
    """
    Enhanced backtest engine with all 13 strategies and advanced features.

    Features:
    - Modular strategy system (all 13 strategies)
    - BarBuffer-driven loop — indicators only update on real TF completions
    - Enhanced regime classification (matches live system)
    - Independent position lanes (S8, R3)
    - Advanced risk management (all kill switches)
    - Position reconciliation and phantom order detection
    - Multi-timeframe support (M5, M15, H1, H4, D1)
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        slippage_points: float = 0.7,
        strategies: Optional[List[str]] = None,
        cache_dir: str = "backtest_data",
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.start_date   = start_date
        self.end_date     = end_date
        self.initial_balance = initial_balance
        self.slippage_points = slippage_points
        self.strategies   = strategies if strategies else ALL_STRATEGIES
        self.cache_dir    = cache_dir

        # Merge config overrides
        self.config: Dict[str, Any] = {**config.__dict__} if config else {}
        if config_override:
            self.config.update(config_override)

        # --- components ---
        self.data_feed    = EnhancedHistoricalDataFeed(cache_dir=cache_dir)
        self.bar_buffer   = BarBuffer()                        # ← reuse existing
        self.execution_sim = EnhancedExecutionSimulator(slippage_points)

        # --- state ---
        self.state = SimulatedState(
            balance=initial_balance,
            equity=initial_balance,
            peak_equity=initial_balance,
        )

        # --- strategy instances ---
        self.strategy_instances: Dict[str, Any] = {}
        for name in self.strategies:
            if name in STRATEGY_REGISTRY:
                self.strategy_instances[name] = STRATEGY_REGISTRY[name](self.config)

        # --- tracking ---
        self.trades:        List[TradeRecord]  = []
        self.equity_curve:  List[EquityPoint]  = []
        self.open_positions: List[SimPosition] = []
        self.pending_orders: List[SimOrder]    = []

        # --- indicator cache ---
        self._last_adx_h4:    float            = 25.0
        self._last_atr_pct_h1: float           = 50.0
        self._last_atr_h1_raw: float           = 20.0
        self._last_atr_m15:   float            = 5.0
        self._last_ema20_h1:  Optional[float]  = None
        self._last_rsi_h1:    Optional[float]  = None
        self._last_daily_atr: Optional[float]  = None

        # --- day tracking ---
        self._current_day: Optional[int] = None

        logger.info(f"Enhanced backtest engine initialised")
        logger.info(f"Strategies : {self.strategies}")
        logger.info(f"Period     : {start_date} to {end_date}")

    # ================================================================== #
    # PUBLIC — MAIN RUN                                                   #
    # ================================================================== #

    def run(self) -> BacktestResults:
        """Run the enhanced backtest. Drives loop from real M5 bar events."""
        logger.info("Starting enhanced backtest run")

        # Load all timeframes
        try:
            self.data_feed.load_data(
                self.start_date, self.end_date,
                timeframes=["M5", "M15", "H1", "H4", "D1"],
            )
        except RuntimeError as e:
            raise

        m5_df = self.data_feed.data.get("M5")
        if m5_df is None or m5_df.empty:
            raise RuntimeError("M5 data is empty after load — cannot run backtest.")

        total_bars   = len(m5_df)
        log_interval = max(total_bars // 20, 1)

        # Record initial equity
        self.equity_curve.append(EquityPoint(
            timestamp=self.start_date,
            equity=self.initial_balance,
        ))

        # ── MAIN LOOP: iterate M5 bars via BarBuffer ──────────────────── #
        for idx, row in m5_df.iterrows():
            bar          = row.to_dict()
            current_time = bar["time"]

            # Progress logging
            if (idx + 1) % log_interval == 0:
                pct = ((idx + 1) / total_bars) * 100
                logger.info(
                    f"Progress: {pct:.0f}% ({idx+1}/{total_bars}) "
                    f"| Equity: ${self.state.equity:,.2f} "
                    f"| Trades: {len(self.trades)}"
                )

            # Step 1 — feed bar into BarBuffer; get higher-TF completions
            completed = self.bar_buffer.add_m5(bar)

            # Step 2 — daily reset
            self._check_daily_reset(current_time)

            # Step 3 — session (uses same util as live system)
            self._update_session(current_time)

            # Step 4 — indicators only when their TF bar completes
            if completed.get("H1"):
                self._update_h1_indicators()
            if completed.get("H4"):
                self._update_h4_indicators()
            if completed.get("D1"):
                self._update_d1_indicators()
            if completed.get("M15"):
                self._update_m15_indicators()

            # Step 5 — regime (every M15 or on H4 completion)
            if completed.get("M15") or completed.get("H4"):
                self._update_regime(current_time)

            # Step 6 — process kill switches
            kill_results = run_all_kill_switches(
                self.state, current_time,
                self.data_feed.get_upcoming_events(current_time),
            )
            for key, value in kill_results.get("state_updates", {}).items():
                setattr(self.state, key, value)

            # Step 7 — process pending order fills
            self._process_pending_orders(bar, current_time)

            # Step 8 — manage open positions
            self._manage_positions(bar, current_time)

            # Step 9 — compute Asian range after 07:00 UTC
            if completed.get("M15"):
                self._update_range_data(current_time)

            # Step 10 — evaluate strategies on every M15
            if completed.get("M15") and kill_results.get("trading_enabled", True):
                self._evaluate_strategies(current_time)

            # Step 11 — record equity curve on every M15
            if completed.get("M15"):
                self._record_equity(current_time, bar["close"])

        # ── END OF LOOP ───────────────────────────────────────────────── #

        # Force-close any remaining open positions at last bar close
        if self.open_positions:
            last_bar = self.bar_buffer.get_last_bar("M5")
            if last_bar:
                self._close_all_positions(
                    last_bar["close"], last_bar["time"], "BACKTEST_END"
                )

        logger.info(
            f"Backtest complete: {len(self.trades)} trades, "
            f"final equity=${self.state.equity:,.2f}"
        )

        return BacktestResults(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.initial_balance,
            start_date=self.start_date,
            end_date=self.end_date,
            strategies=self.strategies,
        )

    # ================================================================== #
    # INTERNAL — SESSION / REGIME / INDICATORS                           #
    # ================================================================== #

    def _check_daily_reset(self, current_time: datetime) -> None:
        """Reset daily counters at midnight UTC."""
        day = current_time.timetuple().tm_yday
        if self._current_day is not None and day != self._current_day:
            self.state.s1_family_attempts_today = 0
            self.state.s1f_attempts_today       = 0
            self.state.s7_placed_today          = False
            self.state.s6_placed_today          = False
            self.state.s2_fired_today           = False
            self.state.daily_pnl                = 0.0
            self.state.daily_trades             = 0
            self.state.range_computed           = False
            logger.debug(f"Daily reset at {current_time}")
        self._current_day = day

    def _update_session(self, current_time: datetime) -> None:
        """Update session using the same util as the live system."""
        from utils.session import get_session_for_datetime
        self.state.current_session = get_session_for_datetime(current_time)

    def _update_h1_indicators(self) -> None:
        h1_df = self.bar_buffer.get_series("H1")
        if len(h1_df) < 20:
            return

        atr_mode   = self.config.get("ATR_MAMODE", "RMA")
        atr_period = self.config.get("ATR_PERIOD", 14)

        atr_series = ta.atr(h1_df["high"], h1_df["low"], h1_df["close"],
                            length=atr_period, mamode=atr_mode)
        if atr_series is not None and not atr_series.empty:
            vals = atr_series.dropna()
            if len(vals) > 0:
                self._last_atr_h1_raw  = float(vals.iloc[-1])
                self.state.last_atr_h1_raw  = self._last_atr_h1_raw
                self._last_atr_pct_h1  = compute_atr_percentile(
                    atr_series, self._last_atr_h1_raw
                )
                self.state.last_atr_pct_h1  = self._last_atr_pct_h1

        ema = ta.ema(h1_df["close"], length=20)
        if ema is not None and not ema.empty:
            v = ema.dropna()
            if len(v) > 0:
                self._last_ema20_h1 = float(v.iloc[-1])

        rsi = ta.rsi(h1_df["close"], length=14)
        if rsi is not None and not rsi.empty:
            v = rsi.dropna()
            if len(v) > 0:
                self._last_rsi_h1 = float(v.iloc[-1])

    def _update_h4_indicators(self) -> None:
        h4_df = self.bar_buffer.get_series("H4")
        if len(h4_df) < 30:
            return

        adx_df = ta.adx(h4_df["high"], h4_df["low"], h4_df["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            v = adx_df["ADX_14"].dropna()
            if len(v) > 0:
                self._last_adx_h4         = float(v.iloc[-1])
                self.state.last_adx_h4    = self._last_adx_h4

    def _update_d1_indicators(self) -> None:
        d1_df = self.bar_buffer.get_series("D1")
        if len(d1_df) < 16:
            return

        atr_mode = self.config.get("ATR_MAMODE", "RMA")
        atr_s    = ta.atr(d1_df["high"], d1_df["low"], d1_df["close"],
                          length=14, mamode=atr_mode)
        if atr_s is not None and not atr_s.empty:
            v = atr_s.dropna()
            if len(v) > 0:
                self._last_daily_atr = float(v.iloc[-1])

    def _update_m15_indicators(self) -> None:
        m15_df = self.bar_buffer.get_series("M15")
        if len(m15_df) < 20:
            return

        atr_mode = self.config.get("ATR_MAMODE", "RMA")
        atr_s    = ta.atr(m15_df["high"], m15_df["low"], m15_df["close"],
                          length=14, mamode=atr_mode)
        if atr_s is not None and not atr_s.empty:
            v = atr_s.dropna()
            if len(v) > 0:
                self._last_atr_m15        = float(v.iloc[-1])
                self.state.last_atr_m15   = self._last_atr_m15

    def _update_regime(self, current_time: datetime) -> None:
        """Update regime — mirrors live classify_regime() exactly."""
        spread     = self.data_feed.get_current_spread(current_time)
        avg_spread = sum(self.data_feed.spreads.values()) / len(self.data_feed.spreads) \
                     if self.data_feed.spreads else 25.0
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0

        has_event = len(self.data_feed.get_upcoming_events(current_time)) > 0

        no_trade_thresh  = self.config.get("ATR_PCT_NO_TRADE_THRESHOLD", 95)
        unstable_thresh  = self.config.get("ATR_PCT_UNSTABLE_THRESHOLD", 85)
        super_thresh     = self.config.get("ATR_PCT_SUPER_THRESHOLD", 55)
        ks2_spread_mult  = self.config.get("KS2_SPREAD_MULTIPLIER", 2.5)

        atr_pct = self._last_atr_pct_h1
        adx     = self._last_adx_h4
        session = self.state.current_session

        if atr_pct > no_trade_thresh or has_event or \
                session == "OFF_HOURS" or spread_ratio > ks2_spread_mult:
            regime, mult = "NO_TRADE", 0.0
        elif atr_pct > unstable_thresh or adx > 55 or (18 <= adx <= 20):
            regime, mult = "UNSTABLE", 0.4
        elif adx < 18:
            regime, mult = "RANGING_CLEAR", 0.7
        else:
            sess_mult = 1.0 if session in ("LONDON_NY_OVERLAP", "LONDON", "NY") else 0.7
            if adx > 35 and atr_pct > super_thresh:
                regime, mult = "NORMAL_TRENDING", round(1.0 * sess_mult, 3)
            elif adx > 26:
                regime, mult = "NORMAL_TRENDING", round(1.0 * sess_mult, 3)
            else:
                regime, mult = "WEAK_TRENDING", round(0.8 * sess_mult, 3)

        self.state.current_regime   = regime
        self.state.size_multiplier  = mult

    def _update_range_data(self, current_time: datetime) -> None:
        """Compute Asian range for S1/S3/S6 after 07:00 UTC."""
        if current_time.hour < 7 or self.state.range_computed:
            return

        m5_df = self.bar_buffer.get_series("M5")
        if m5_df.empty:
            return

        today = current_time.date()
        mask  = (
            (m5_df["time"].dt.date == today) &
            (m5_df["time"].dt.hour >= 0) &
            (m5_df["time"].dt.hour < 7)
        )
        asian = m5_df[mask]
        if len(asian) < 12:
            return

        rh = float(asian["high"].max())
        rl = float(asian["low"].min())
        rs = rh - rl

        min_range = self.config.get("MIN_RANGE_SIZE_PTS", 10)
        if rs < min_range:
            return

        self.state.range_high     = rh
        self.state.range_low      = rl
        self.state.range_size     = rs
        self.state.range_computed = True
        logger.debug(f"Asian range: {rl:.2f} - {rh:.2f} (size={rs:.2f})")

    # ================================================================== #
    # INTERNAL — ORDER PROCESSING                                         #
    # ================================================================== #

    def _process_pending_orders(
        self, bar: dict, current_time: datetime
    ) -> None:
        """Check pending orders for fills using the bar's OHLC."""
        if not self.pending_orders:
            return

        current_price = bar["close"]
        remaining: List[SimOrder] = []
        filled_tags: set          = set()

        for order in self.pending_orders:
            # Check expiry
            if order.expiry and current_time >= order.expiry:
                logger.debug(f"Order expired: {order.strategy} {order.direction}")
                continue

            fill = self.execution_sim.check_fill(order, current_price, current_time)

            if fill.filled:
                pos = self._create_position_from_order(order, fill, current_time)
                self.open_positions.append(pos)
                self._update_position_state(pos, "OPEN")

                if order.tag:
                    filled_tags.add(order.tag)

                # Update daily counters
                if pos.strategy == "S1_LONDON_BRK":
                    self.state.s1_family_attempts_today += 1
                elif pos.strategy == "S7_DAILY_STRUCT":
                    self.state.s7_placed_today = True
                elif pos.strategy == "S6_ASIAN_BRK":
                    self.state.s6_placed_today = True
                elif pos.strategy == "S2_MEAN_REV":
                    self.state.s2_fired_today = True

                logger.info(
                    f"Filled: {order.strategy} {order.direction} @ {fill.fill_price:.2f}"
                )
            else:
                remaining.append(order)

        # OCO: remove orders whose linked tag just filled
        new_remaining: List[SimOrder] = []
        for order in remaining:
            if order.linked_tag and order.linked_tag in filled_tags:
                logger.debug(
                    f"OCO cancel: {order.strategy} {order.direction} "
                    f"(linked={order.linked_tag})"
                )
                continue
            new_remaining.append(order)

        self.pending_orders = new_remaining

    # ================================================================== #
    # INTERNAL — POSITION MANAGEMENT                                      #
    # ================================================================== #

    def _manage_positions(
        self, bar: dict, current_time: datetime
    ) -> None:
        """Manage SL/TP, BE, trailing stop for all open positions."""
        close_price  = bar["close"]
        still_open:  List[SimPosition] = []

        for pos in self.open_positions:
            # --- check position limits ---
            can_manage, reason = check_position_limits(
                self.state, pos.direction, pos.lots
            )
            if not can_manage:
                logger.warning(f"Position mgmt blocked: {reason}")
                still_open.append(pos)
                continue

            # --- update MFE ---
            if pos.direction == "LONG":
                pos.max_favorable = max(pos.max_favorable, bar["high"] - pos.entry_price)
            else:
                pos.max_favorable = max(pos.max_favorable, pos.entry_price - bar["low"])

            r = pos.current_r(close_price)
            pos.max_r = max(pos.max_r, r)

            # --- SL / TP hit check (direction-aware) ---
            sl_hit = (
                (pos.direction == "LONG"  and bar["low"]  <= pos.current_sl) or
                (pos.direction == "SHORT" and bar["high"] >= pos.current_sl)
            )
            tp_hit = (
                pos.tp is not None and (
                    (pos.direction == "LONG"  and bar["high"] >= pos.tp) or
                    (pos.direction == "SHORT" and bar["low"]  <= pos.tp)
                )
            )

            if tp_hit:
                trade = self._build_trade_record(
                    pos, pos.tp, current_time, "TP"
                )
                self._record_trade(trade)
                continue
            if sl_hit:
                trade = self._build_trade_record(
                    pos, pos.current_sl, current_time, "SL"
                )
                self._record_trade(trade)
                continue

            # --- BE activation ---
            be_r = self.config.get("BE_ACTIVATION_R", 1.5)
            if not pos.be_activated and r >= be_r:
                pos.current_sl  = pos.entry_price
                pos.be_activated = True
                logger.debug(
                    f"BE activated: {pos.strategy} {pos.direction} @ {pos.entry_price:.2f}"
                )

            # --- ATR trailing stop (after BE) ---
            if pos.be_activated:
                trail_mult = self.config.get("ATR_TRAIL_MULTIPLIER", 2.5)
                trail_dist = self._last_atr_m15 * trail_mult
                if pos.direction == "LONG":
                    new_sl = close_price - trail_dist
                    if new_sl > pos.current_sl:
                        pos.current_sl = new_sl
                else:
                    new_sl = close_price + trail_dist
                    if new_sl < pos.current_sl:
                        pos.current_sl = new_sl

            still_open.append(pos)

        self.open_positions = still_open

    def _manage_r3_position(
        self,
        position: SimPosition,
        current_price: float,
        current_time: datetime,      # ← FIX: was missing, caused NameError
        indicators: Dict[str, Any],
    ) -> None:
        """Manage R3 position — hard exit after 30 minutes."""
        elapsed = (current_time - position.entry_time).total_seconds() / 60.0
        if elapsed >= 30:
            trade = self._build_trade_record(
                position, current_price, current_time, "R3_TIME_EXIT"
            )
            self._record_trade(trade)
            if position in self.open_positions:
                self.open_positions.remove(position)

    # ================================================================== #
    # INTERNAL — STRATEGY EVALUATION                                      #
    # ================================================================== #

    def _evaluate_strategies(self, current_time: datetime) -> None:
        """Evaluate all enabled strategies and queue new orders."""
        # Daily loss limit
        daily_loss_limit = self.config.get("KS3_DAILY_LOSS_LIMIT_PCT", -0.04)
        if self.state.balance > 0:
            if (self.state.daily_pnl / self.state.balance) <= daily_loss_limit:
                return

        for name in self.strategies:
            if name not in self.strategy_instances:
                continue

            strategy = self.strategy_instances[name]

            bar_data = {
                "M5":  self.bar_buffer.get_series("M5"),
                "M15": self.bar_buffer.get_series("M15"),
                "H1":  self.bar_buffer.get_series("H1"),
                "H4":  self.bar_buffer.get_series("H4"),
                "D1":  self.bar_buffer.get_series("D1"),
            }

            indicators = {
                "atr_h1":    self._last_atr_h1_raw,
                "atr_m15":   self._last_atr_m15,
                "atr_pct_h1": self._last_atr_pct_h1,
                "adx_h4":    self._last_adx_h4,
                "ema20_h1":  self._last_ema20_h1,
                "rsi_h1":    self._last_rsi_h1,
                "daily_atr": self._last_daily_atr,
            }

            result = strategy.evaluate(
                self.state, bar_data, current_time, indicators
            )

            for order in result.orders:
                self.pending_orders.append(order)
                logger.debug(
                    f"New order: {order.strategy} {order.direction} "
                    f"{order.order_type} @ {order.price:.2f}"
                )

            for key, value in result.state_updates.items():
                setattr(self.state, key, value)

        # Mark placed strategies
        for order in self.pending_orders:
            if order.strategy == "S7_DAILY_STRUCT":
                self.state.s7_placed_today = True
            elif order.strategy == "S6_ASIAN_BRK":
                self.state.s6_placed_today = True

    # ================================================================== #
    # INTERNAL — TRADE RECORDING / EQUITY                                 #
    # ================================================================== #

    def _build_trade_record(
        self,
        pos: SimPosition,
        exit_price: float,
        exit_time: datetime,          # ← FIX: simulation time, not utcnow()
        reason: str,
    ) -> TradeRecord:
        commission = self.config.get("COMMISSION_PER_LOT_ROUND_TRIP", 7.0) * pos.lots
        pnl_gross  = pos.unrealized_pnl(exit_price)
        pnl_net    = pnl_gross - commission

        stop_dist = abs(pos.entry_price - pos.stop_price_original)
        r_multiple = (pnl_gross / (stop_dist * 100.0 * pos.lots)) \
                     if (stop_dist > 0 and pos.lots > 0) else 0.0

        return TradeRecord(
            strategy=pos.strategy,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,          # ← simulation time
            lots=pos.lots,
            pnl=pnl_net,
            pnl_gross=pnl_gross,
            commission=commission,
            exit_reason=reason,
            max_r=pos.max_r,
            r_multiple=r_multiple,
            regime_at_entry=pos.regime_at_entry,
            regime_at_exit=self.state.current_regime,
        )

    def _record_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        self.state.balance     += trade.pnl
        self.state.daily_pnl   += trade.pnl
        self.state.daily_trades += 1

        if trade.pnl >= 0:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        logger.debug(
            f"Trade closed: {trade.strategy} {trade.direction} "
            f"P&L=${trade.pnl:+.2f} R={trade.r_multiple:+.2f} ({trade.exit_reason})"
        )

    def _record_equity(self, current_time: datetime, current_price: float) -> None:
        unrealized = sum(
            pos.unrealized_pnl(current_price) for pos in self.open_positions
        )
        equity = self.state.balance + unrealized
        self.state.equity      = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)

        dd = (self.state.peak_equity - equity) / self.state.peak_equity \
             if self.state.peak_equity > 0 else 0.0

        self.equity_curve.append(EquityPoint(
            timestamp=current_time,
            equity=equity,
            drawdown_pct=dd,
        ))

    # ================================================================== #
    # INTERNAL — HELPERS                                                   #
    # ================================================================== #

    def _create_position_from_order(
        self,
        order: SimOrder,
        fill,                       # FillResult
        current_time: datetime,
    ) -> SimPosition:
        return SimPosition(
            strategy=order.strategy,
            direction=order.direction,
            entry_price=fill.fill_price,
            entry_time=current_time,
            lots=order.lots,
            stop_price_original=order.sl,
            current_sl=order.sl,
            tp=order.tp,
            regime_at_entry=self.state.current_regime,
            be_activated=False,
            max_r=0.0,
            max_favorable=0.0,
        )

    def _update_position_state(self, position: SimPosition, action: str) -> None:
        """Update SimulatedState tracking fields based on position open/close."""
        if position.strategy == "S8_ATR_SPIKE":
            if action == "OPEN":
                self.state.s8_open_ticket           = position.strategy
                self.state.s8_entry_price           = position.entry_price
                self.state.s8_stop_price_original   = position.stop_price_original
                self.state.s8_stop_price_current    = position.current_sl
                self.state.s8_trade_direction       = position.direction
                self.state.s8_be_activated          = False
                self.state.s8_open_time_utc         = position.entry_time.isoformat()
            else:
                self.state.s8_open_ticket           = None
                self.state.s8_entry_price           = 0.0
                self.state.s8_stop_price_original   = 0.0
                self.state.s8_stop_price_current    = 0.0
                self.state.s8_trade_direction       = None
                self.state.s8_be_activated          = False
                self.state.s8_open_time_utc         = None

        elif position.strategy == "R3_CAL_MOMENTUM":
            if action == "OPEN":
                self.state.r3_open_ticket   = position.strategy
                self.state.r3_open_time     = position.entry_time
                self.state.r3_entry_price   = position.entry_price
                self.state.r3_stop_price    = position.current_sl
                self.state.r3_tp_price      = position.tp
            else:
                self.state.r3_open_ticket   = None
                self.state.r3_open_time     = None
                self.state.r3_entry_price   = 0.0
                self.state.r3_stop_price    = 0.0
                self.state.r3_tp_price      = 0.0

        else:
            # Trend-family
            if action == "OPEN":
                self.state.trend_family_occupied    = True
                self.state.trend_family_strategy    = position.strategy
                self.state.open_position            = position.strategy
                self.state.entry_price              = position.entry_price
                self.state.stop_price_original      = position.stop_price_original
                self.state.stop_price_current       = position.current_sl
                self.state.original_lot_size        = position.lots
                self.state.position_be_activated    = False
            else:
                self.state.trend_family_occupied    = False
                self.state.trend_family_strategy    = None
                self.state.open_position            = None
                self.state.entry_price              = 0.0
                self.state.stop_price_original      = 0.0
                self.state.stop_price_current       = 0.0
                self.state.original_lot_size        = 0.0
                self.state.position_be_activated    = False

    def _close_all_positions(
        self, price: float, time: datetime, reason: str
    ) -> None:
        """Force-close all open positions at end of backtest."""
        for pos in self.open_positions:
            trade = self._build_trade_record(pos, price, time, reason)
            self._record_trade(trade)
            self._update_position_state(pos, "CLOSE")
        self.open_positions = []

    def _reconcile_positions(self) -> None:
        """Log any ghost/orphan positions (mirrors live reconciliation)."""
        believed = set()
        for attr in ("open_position", "s8_open_ticket", "r3_open_ticket"):
            val = getattr(self.state, attr, None)
            if val:
                believed.add(val)

        actual = {pos.strategy for pos in self.open_positions}

        for ghost in (believed - actual):
            logger.critical(f"Ghost position detected: {ghost}")
        for orphan in (actual - believed):
            logger.warning(f"Orphan position detected: {orphan}")
