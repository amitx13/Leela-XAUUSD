"""
Enhanced Backtest Engine

Complete rewrite using modular strategy system with all 13 strategies.
Mirrors live system behaviour exactly.

Fixes applied (v2 — full review pass):
  1. Loop driven by BarBuffer.add_m5() — real bar events, no clock-tick loop
  2. MT5 data via EnhancedHistoricalDataFeed (no synthetic bars)
  3. _manage_r3_position() actually called from _manage_positions()            [BUG #3]
  4. _update_position_state("CLOSE") called after every SL/TP/TP fill          [BUG #4]
  5. DI+ / DI- written to state in _update_h4_indicators()                    [BUG #1]
  6. _check_daily_reset() resets ALL daily flags from SimulatedState           [BUG #2]
  7. TradeRecord created without non-existent max_r kwarg                      [BUG #8]
  8. compute_atr_percentile defined locally (not in data_feed.py)              [BUG #10]
  9. SUPER_TRENDING regime added to _update_regime()                           [BUG #9]
 10. _update_session() cached per-hour (not every M5)                          [BUG #7]
 11. _update_range_data() uses recent slice, not full BarBuffer series         [BUG #5]
 12. _evaluate_strategies() caps bar_data series to last 200 bars              [BUG #6]
 13. Progress logging uses enumerate(), not DataFrame index                    [BUG #12]
 14. _reconcile_positions() called at end of run()                             [BUG #13]
 15. Partial exit logic added to _manage_positions()                           [BUG #14]
 16. exit_time uses simulation current_time, not datetime.utcnow()
 17. Session classification uses utils.session.get_session_for_datetime()

Fixes applied (v3 — 0-trades diagnosis):
 18. S1 counter race: state_updates flushed per-strategy via setattr           [BUG-S1a]
 19. S6/S7 placed-today flags set BEFORE evaluate() guard, not post-loop       [BUG-S6S7]
 20. S4 daily fire guard added in s4_london_pull.py                            [BUG-S4a]

Fixes applied (v4 — fill accuracy + loop safety):
 21. MARKET/STOP/LIMIT orders use deterministic price-cross fill logic;
     bar high/low passed into check_fill() for accurate trigger detection      [BUG-FILL]
 22. _check_daily_reset() preserves consecutive_losses across midnight —
     only consecutive_m5_losses (intra-day micro-counter) resets               [BUG-KS5]
 23. _manage_r3_position() no longer calls open_positions.remove() inside
     the positions iteration loop; uses still_open list pattern                [BUG-R3LOOP]

Fixes applied (v5 — 0-trades root cause):
 24. execution_sim.submit_order() called alongside pending_orders.append() —
     simulator order_history was always empty so every check_fill() returned
     ORDER_NOT_FOUND immediately, silently blocking all fills                  [BUG-FILL-META]
 25. _update_session() + _update_regime() bootstrapped before main loop so
     initial regime is never stuck on NO_TRADE/OFF_HOURS default               [BUG-REGIME-STARTUP]
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
from backtest.data_feed_enhanced import EnhancedHistoricalDataFeed
from backtest.execution_simulator_enhanced import EnhancedExecutionSimulator
from backtest.strategies import STRATEGY_REGISTRY, ALL_STRATEGIES
from backtest.risk import run_all_kill_switches, check_position_limits
from backtest.results import BacktestResults

logger = logging.getLogger("backtest.enhanced_engine")

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    import config
except ImportError:
    config = None  # type: ignore


# ---------------------------------------------------------------------------
# LOCAL UTIL
# ---------------------------------------------------------------------------

def compute_atr_percentile(atr_series: pd.Series, current_atr: float) -> float:
    vals = atr_series.dropna().values
    if len(vals) == 0:
        return 50.0
    below = np.sum(vals <= current_atr)
    return float(below / len(vals) * 100.0)


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------


class EnhancedBacktestEngine:
    """
    Enhanced backtest engine with all 13 strategies and advanced features.
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
        self.start_date      = start_date
        self.end_date        = end_date
        self.initial_balance = initial_balance
        self.slippage_points = slippage_points
        self.strategies      = strategies if strategies else ALL_STRATEGIES
        self.cache_dir       = cache_dir

        self.config: Dict[str, Any] = {**config.__dict__} if config else {}
        if config_override:
            self.config.update(config_override)

        self.data_feed     = EnhancedHistoricalDataFeed(cache_dir=cache_dir)
        self.bar_buffer    = BarBuffer()
        self.execution_sim = EnhancedExecutionSimulator(slippage_points)

        self.state = SimulatedState(
            balance=initial_balance,
            equity=initial_balance,
            peak_equity=initial_balance,
        )

        self.strategy_instances: Dict[str, Any] = {}
        for name in self.strategies:
            if name in STRATEGY_REGISTRY:
                self.strategy_instances[name] = STRATEGY_REGISTRY[name](self.config)

        self.trades:         List[TradeRecord]  = []
        self.equity_curve:   List[EquityPoint]  = []
        self.open_positions: List[SimPosition]  = []
        self.pending_orders: List[SimOrder]     = []

        self._last_adx_h4:     float           = 25.0
        self._last_atr_pct_h1: float           = 50.0
        self._last_atr_h1_raw: float           = 20.0
        self._last_atr_m15:    float           = 5.0
        self._last_ema20_h1:   Optional[float] = None
        self._last_rsi_h1:     Optional[float] = None
        self._last_daily_atr:  Optional[float] = None

        self._current_day:       Optional[int] = None
        self._last_session_hour: int           = -1

        logger.info("Enhanced backtest engine initialised")
        logger.info(f"Strategies : {self.strategies}")
        logger.info(f"Period     : {start_date} to {end_date}")

    # ================================================================== #
    # PUBLIC
    # ================================================================== #

    def run(self) -> BacktestResults:
        logger.info("Starting enhanced backtest run")

        self.data_feed.load_data(
            self.start_date, self.end_date,
            timeframes=["M5", "M15", "H1", "H4", "D1"],
        )

        m5_df = self.data_feed.data.get("M5")
        if m5_df is None or m5_df.empty:
            raise RuntimeError("M5 data is empty after load — cannot run backtest.")

        total_bars   = len(m5_df)
        log_interval = max(total_bars // 20, 1)

        self.equity_curve.append(EquityPoint(
            timestamp=self.start_date,
            equity=self.initial_balance,
        ))

        # BUG-REGIME-STARTUP FIX: bootstrap session + regime from the actual
        # backtest start time so the initial regime is never stuck on the
        # OFF_HOURS / NO_TRADE dataclass default before the first M15 close.
        self._update_session(self.start_date)
        self._last_session_hour = self.start_date.hour
        self._update_regime(self.start_date)

        for bar_num, (_, row) in enumerate(m5_df.iterrows()):
            bar          = row.to_dict()
            current_time = bar["time"]

            if (bar_num + 1) % log_interval == 0:
                pct = ((bar_num + 1) / total_bars) * 100
                logger.info(
                    f"Progress: {pct:.0f}% ({bar_num+1}/{total_bars}) "
                    f"| Equity: ${self.state.equity:,.2f} "
                    f"| Trades: {len(self.trades)}"
                )

            completed = self.bar_buffer.add_m5(bar)

            self._check_daily_reset(current_time)

            if current_time.hour != self._last_session_hour:
                self._update_session(current_time)
                self._last_session_hour = current_time.hour

            if completed.get("H1"):
                self._update_h1_indicators()
            if completed.get("H4"):
                self._update_h4_indicators()
            if completed.get("D1"):
                self._update_d1_indicators()
            if completed.get("M15"):
                self._update_m15_indicators()

            if completed.get("M15") or completed.get("H4"):
                self._update_regime(current_time)

            kill_results = run_all_kill_switches(
                self.state, current_time,
                self.data_feed.get_upcoming_events(current_time),
            )
            for key, value in kill_results.get("state_updates", {}).items():
                setattr(self.state, key, value)

            self._process_pending_orders(bar, current_time)
            self._manage_positions(bar, current_time)

            if completed.get("M15"):
                self._update_range_data(current_time)

            if completed.get("M15") and kill_results.get("trading_enabled", True):
                self._evaluate_strategies(current_time)

            if completed.get("M15"):
                self._record_equity(current_time, bar["close"])

        if self.open_positions:
            last_bar = self.bar_buffer.get_last_bar("M5")
            if last_bar:
                self._close_all_positions(
                    last_bar["close"], last_bar["time"], "BACKTEST_END"
                )

        self._reconcile_positions()

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
    # SESSION / REGIME / INDICATORS
    # ================================================================== #

    def _check_daily_reset(self, current_time: datetime) -> None:
        """
        Reset daily counters at midnight.

        BUG-KS5 FIX: consecutive_losses is intentionally NOT reset here.
        It is a multi-day circuit breaker (KS5). The live system preserves
        it across midnight; resetting it would prevent KS5 from ever firing
        during a multi-day losing streak in the backtest.

        consecutive_m5_losses IS reset — it is an intra-day micro-counter
        used for short-burst halt logic only.
        """
        day = current_time.timetuple().tm_yday
        if self._current_day is not None and day != self._current_day:
            self.state.s1_family_attempts_today  = 0
            self.state.s1f_attempts_today        = 0
            self.state.s1d_ema_touched_today     = False
            self.state.s1d_fired_today           = False
            self.state.s1e_pyramid_done          = False
            self.state.s1f_post_tk_active        = False
            self.state.s1b_pending_ticket        = None
            self.state.s2_fired_today            = False
            self.state.s3_fired_today            = False
            self.state.s3_sweep_candle_time      = None
            self.state.s3_sweep_low              = 0.0
            self.state.s3_sweep_direction        = None
            self.state.s4_fired_today            = False
            self.state.s5_fired_today            = False
            self.state.s6_placed_today           = False
            self.state.s7_placed_today           = False
            self.state.s8_fired_today            = False
            self.state.s8_armed                  = False
            self.state.s8_arm_time               = None
            self.state.s8_spike_high             = 0.0
            self.state.s8_spike_low              = 0.0
            self.state.s8_spike_direction        = None
            self.state.s8_confirmation_passed    = False
            self.state.r3_fired_today            = False
            self.state.stop_hunt_detected        = False
            self.state.failed_breakout_flag      = False
            self.state.failed_breakout_direction = None
            # consecutive_m5_losses resets daily; consecutive_losses does NOT
            self.state.consecutive_m5_losses     = 0
            self.state.daily_pnl                 = 0.0
            self.state.daily_trades              = 0
            self.state.daily_commission_paid     = 0.0
            self.state.range_computed            = False
            self._last_session_hour              = -1
            logger.debug(f"Full daily reset at {current_time.date()}")
        self._current_day = day

    def _update_session(self, current_time: datetime) -> None:
        from utils.session import get_session_for_datetime
        self.state.current_session = get_session_for_datetime(current_time)

    def _update_h1_indicators(self) -> None:
        h1_df = self.bar_buffer.get_series("H1", count=200)
        if len(h1_df) < 20:
            return
        atr_mode   = self.config.get("ATR_MAMODE", "RMA")
        atr_period = self.config.get("ATR_PERIOD", 14)
        atr_series = ta.atr(h1_df["high"], h1_df["low"], h1_df["close"],
                            length=atr_period, mamode=atr_mode)
        if atr_series is not None and not atr_series.empty:
            vals = atr_series.dropna()
            if len(vals) > 0:
                self._last_atr_h1_raw      = float(vals.iloc[-1])
                self.state.last_atr_h1_raw = self._last_atr_h1_raw
                self._last_atr_pct_h1      = compute_atr_percentile(
                    atr_series, self._last_atr_h1_raw
                )
                self.state.last_atr_pct_h1 = self._last_atr_pct_h1
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
        h4_df = self.bar_buffer.get_series("H4", count=100)
        if len(h4_df) < 30:
            return
        adx_df = ta.adx(h4_df["high"], h4_df["low"], h4_df["close"], length=14)
        if adx_df is not None and "ADX_14" in adx_df.columns:
            v = adx_df["ADX_14"].dropna()
            if len(v) > 0:
                self._last_adx_h4      = float(v.iloc[-1])
                self.state.last_adx_h4 = self._last_adx_h4
            if "DMP_14" in adx_df.columns:
                v_dmp = adx_df["DMP_14"].dropna()
                if len(v_dmp) > 0:
                    self.state.last_di_plus_h4 = float(v_dmp.iloc[-1])
            if "DMN_14" in adx_df.columns:
                v_dmn = adx_df["DMN_14"].dropna()
                if len(v_dmn) > 0:
                    self.state.last_di_minus_h4 = float(v_dmn.iloc[-1])

    def _update_d1_indicators(self) -> None:
        d1_df = self.bar_buffer.get_series("D1", count=50)
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
        m15_df = self.bar_buffer.get_series("M15", count=100)
        if len(m15_df) < 20:
            return
        atr_mode = self.config.get("ATR_MAMODE", "RMA")
        atr_s    = ta.atr(m15_df["high"], m15_df["low"], m15_df["close"],
                          length=14, mamode=atr_mode)
        if atr_s is not None and not atr_s.empty:
            v = atr_s.dropna()
            if len(v) > 0:
                self._last_atr_m15      = float(v.iloc[-1])
                self.state.last_atr_m15 = self._last_atr_m15

    def _update_regime(self, current_time: datetime) -> None:
        spread     = self.data_feed.get_current_spread(current_time)
        avg_spread = (
            sum(self.data_feed.spreads.values()) / len(self.data_feed.spreads)
            if self.data_feed.spreads else 25.0
        )
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0
        has_event    = len(self.data_feed.get_upcoming_events(current_time)) > 0

        no_trade_thresh = self.config.get("ATR_PCT_NO_TRADE_THRESHOLD",  95)
        unstable_thresh = self.config.get("ATR_PCT_UNSTABLE_THRESHOLD",  85)
        super_thresh    = self.config.get("ATR_PCT_SUPER_THRESHOLD",     55)
        ks2_spread_mult = self.config.get("KS2_SPREAD_MULTIPLIER",      2.5)

        atr_pct = self._last_atr_pct_h1
        adx     = self._last_adx_h4
        session = self.state.current_session

        if (atr_pct > no_trade_thresh or has_event or
                session == "OFF_HOURS" or spread_ratio > ks2_spread_mult):
            regime, mult = "NO_TRADE", 0.0
        elif atr_pct > unstable_thresh or adx > 55 or (18 <= adx <= 20):
            regime, mult = "UNSTABLE", 0.4
        elif adx < 18:
            regime, mult = "RANGING_CLEAR", 0.7
        else:
            sess_mult = 1.0 if session in ("LONDON_NY_OVERLAP", "LONDON", "NY") else 0.7
            if adx > 35 and atr_pct > super_thresh:
                regime, mult = "SUPER_TRENDING", round(1.2 * sess_mult, 3)
            elif adx > 26:
                regime, mult = "NORMAL_TRENDING", round(1.0 * sess_mult, 3)
            else:
                regime, mult = "WEAK_TRENDING", round(0.8 * sess_mult, 3)

        self.state.current_regime  = regime
        self.state.size_multiplier = mult

    def _update_range_data(self, current_time: datetime) -> None:
        if current_time.hour < 7 or self.state.range_computed:
            return
        recent = self.data_feed.get_bars("M5", current_time, count=300)
        if not recent:
            return
        m5_df = pd.DataFrame(recent)
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
        if rs < self.config.get("MIN_RANGE_SIZE_PTS", 10):
            return
        self.state.range_high     = rh
        self.state.range_low      = rl
        self.state.range_size     = rs
        self.state.range_computed = True
        logger.debug(f"Asian range: {rl:.2f} – {rh:.2f} (size={rs:.2f})")

    # ================================================================== #
    # ORDER PROCESSING
    # ================================================================== #

    def _process_pending_orders(
        self, bar: dict, current_time: datetime
    ) -> None:
        if not self.pending_orders:
            return

        bar_high = float(bar.get("high", bar["close"]))
        bar_low  = float(bar.get("low",  bar["close"]))
        remaining:   List[SimOrder] = []
        filled_tags: set            = set()

        for order in self.pending_orders:
            if order.expiry and current_time >= order.expiry:
                logger.debug(f"Order expired: {order.strategy} {order.direction}")
                continue

            # BUG-FILL FIX: pass bar_high / bar_low for accurate price-cross detection
            fill = self.execution_sim.check_fill(
                order, bar["close"], current_time,
                bar_high=bar_high, bar_low=bar_low
            )

            if fill.filled:
                pos = self._create_position_from_order(order, fill, current_time)
                self.open_positions.append(pos)
                self._update_position_state(pos, "OPEN")

                if order.tag:
                    filled_tags.add(order.tag)

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
    # POSITION MANAGEMENT
    # ================================================================== #

    def _manage_positions(
        self, bar: dict, current_time: datetime
    ) -> None:
        close_price = bar["close"]
        still_open: List[SimPosition] = []

        for pos in self.open_positions:
            can_manage, reason = check_position_limits(
                self.state, pos.direction, pos.lots
            )
            if not can_manage:
                logger.warning(f"Position mgmt blocked: {reason}")
                still_open.append(pos)
                continue

            if pos.direction == "LONG":
                pos.max_favorable = max(pos.max_favorable, bar["high"] - pos.entry_price)
            else:
                pos.max_favorable = max(pos.max_favorable, pos.entry_price - bar["low"])

            r = pos.current_r(close_price)
            pos.max_r = max(pos.max_r, r)

            if pos.strategy == "R3_CAL_MOMENTUM":
                # BUG-R3LOOP FIX: pass still_open so removal happens via the
                # same list that replaces open_positions at end of loop.
                self._manage_r3_position(
                    pos, close_price, current_time, {}, still_open
                )
                continue  # _manage_r3_position handles whether pos is kept

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
                trade = self._build_trade_record(pos, pos.tp, current_time, "TP")
                self._record_trade(trade)
                self._update_position_state(pos, "CLOSE")
                continue
            if sl_hit:
                trade = self._build_trade_record(pos, pos.current_sl, current_time, "SL")
                self._record_trade(trade)
                self._update_position_state(pos, "CLOSE")
                continue

            partial_r = self.config.get("PARTIAL_EXIT_R", 1.0)
            if not pos.partial_done and r >= partial_r and pos.lots >= 0.02:
                half_lots = round(pos.lots / 2.0, 2)
                partial_exit_price = (
                    pos.entry_price + pos.stop_distance * partial_r
                    if pos.direction == "LONG"
                    else pos.entry_price - pos.stop_distance * partial_r
                )
                commission = self.config.get("COMMISSION_PER_LOT_ROUND_TRIP", 7.0) * half_lots
                pnl_gross  = (
                    (partial_exit_price - pos.entry_price) * half_lots * 100.0
                    if pos.direction == "LONG"
                    else (pos.entry_price - partial_exit_price) * half_lots * 100.0
                )
                partial_trade = TradeRecord(
                    strategy=pos.strategy,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=partial_exit_price,
                    entry_time=pos.entry_time,
                    exit_time=current_time,
                    lots=half_lots,
                    pnl=pnl_gross - commission,
                    pnl_gross=pnl_gross,
                    commission=commission,
                    exit_reason="PARTIAL",
                    r_multiple=partial_r,
                    regime_at_entry=pos.regime_at_entry,
                    regime_at_exit=self.state.current_regime,
                    stop_original=pos.stop_price_original,
                )
                self._record_trade(partial_trade)
                pos.lots         -= half_lots
                pos.partial_done  = True
                self.state.position_partial_done = True

            be_r = self.config.get("BE_ACTIVATION_R", 1.5)
            if not pos.be_activated and r >= be_r:
                pos.current_sl   = pos.entry_price
                pos.be_activated = True
                self.state.position_be_activated = True

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
        position:    SimPosition,
        current_price: float,
        current_time:  datetime,
        indicators:    Dict[str, Any],
        still_open:    List[SimPosition],
    ) -> None:
        """
        Manage R3 momentum position with 30-min time exit.

        BUG-R3LOOP FIX: appends to still_open (passed from caller) instead
        of calling self.open_positions.remove() inside the iteration loop,
        which could raise ValueError or skip positions.
        """
        elapsed = (current_time - position.entry_time).total_seconds() / 60.0
        if elapsed >= 30:
            trade = self._build_trade_record(
                position, current_price, current_time, "R3_TIME_EXIT"
            )
            self._record_trade(trade)
            self._update_position_state(position, "CLOSE")
            # Do NOT append to still_open — position is closed
        else:
            still_open.append(position)

    # ================================================================== #
    # STRATEGY EVALUATION
    # ================================================================== #

    def _evaluate_strategies(self, current_time: datetime) -> None:
        daily_loss_limit = self.config.get("KS3_DAILY_LOSS_LIMIT_PCT", -0.04)
        if self.state.balance > 0:
            if (self.state.daily_pnl / self.state.balance) <= daily_loss_limit:
                return

        for name in self.strategies:
            if name not in self.strategy_instances:
                continue

            strategy = self.strategy_instances[name]

            if name == "S6_ASIAN_BRK" and self.state.s6_placed_today:
                continue
            if name == "S7_DAILY_STRUCT" and self.state.s7_placed_today:
                continue

            bar_data = {
                "M5":  self.bar_buffer.get_series("M5",  count=100),
                "M15": self.bar_buffer.get_series("M15", count=200),
                "H1":  self.bar_buffer.get_series("H1",  count=200),
                "H4":  self.bar_buffer.get_series("H4",  count=100),
                "D1":  self.bar_buffer.get_series("D1",  count=50),
            }

            indicators = {
                "atr_h1":     self._last_atr_h1_raw,
                "atr_m15":    self._last_atr_m15,
                "atr_pct_h1": self._last_atr_pct_h1,
                "adx_h4":     self._last_adx_h4,
                "ema20_h1":   self._last_ema20_h1,
                "rsi_h1":     self._last_rsi_h1,
                "daily_atr":  self._last_daily_atr,
            }

            result = strategy.evaluate(
                self.state, bar_data, current_time, indicators
            )

            for order in result.orders:
                # BUG-FILL-META FIX: register the order with the execution
                # simulator's order_history so _get_meta() can find it via
                # identity lookup on the next fill check. Without this call
                # order_history stays empty and every check_fill() returns
                # ORDER_NOT_FOUND, silently blocking all fills forever.
                self.execution_sim.submit_order(order)
                self.pending_orders.append(order)
                logger.debug(
                    f"New order: {order.strategy} {order.direction} "
                    f"{order.order_type} @ {order.price:.2f}"
                )

            # Flush state_updates immediately so next strategy sees fresh counters
            for key, value in result.state_updates.items():
                setattr(self.state, key, value)

            if result.orders:
                if name == "S6_ASIAN_BRK":
                    self.state.s6_placed_today = True
                elif name == "S7_DAILY_STRUCT":
                    self.state.s7_placed_today = True

    # ================================================================== #
    # TRADE RECORDING / EQUITY
    # ================================================================== #

    def _build_trade_record(
        self,
        pos:        SimPosition,
        exit_price: float,
        exit_time:  datetime,
        reason:     str,
    ) -> TradeRecord:
        commission = self.config.get("COMMISSION_PER_LOT_ROUND_TRIP", 7.0) * pos.lots
        pnl_gross  = pos.unrealized_pnl(exit_price)
        pnl_net    = pnl_gross - commission
        stop_dist  = abs(pos.entry_price - pos.stop_price_original)
        r_multiple = (
            pnl_gross / (stop_dist * 100.0 * pos.lots)
            if (stop_dist > 0 and pos.lots > 0) else 0.0
        )
        return TradeRecord(
            strategy=pos.strategy,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            lots=pos.lots,
            pnl=pnl_net,
            pnl_gross=pnl_gross,
            commission=commission,
            exit_reason=reason,
            r_multiple=r_multiple,
            regime_at_entry=pos.regime_at_entry,
            regime_at_exit=self.state.current_regime,
            stop_original=pos.stop_price_original,
        )

    def _record_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        self.state.balance      += trade.pnl
        self.state.daily_pnl    += trade.pnl
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
        equity                 = self.state.balance + unrealized
        self.state.equity      = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        dd = (
            (self.state.peak_equity - equity) / self.state.peak_equity
            if self.state.peak_equity > 0 else 0.0
        )
        self.equity_curve.append(EquityPoint(
            timestamp=current_time,
            equity=equity,
            drawdown_pct=dd,
        ))

    # ================================================================== #
    # HELPERS
    # ================================================================== #

    def _create_position_from_order(
        self,
        order:        SimOrder,
        fill,
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
        if position.strategy == "S8_ATR_SPIKE":
            if action == "OPEN":
                self.state.s8_open_ticket         = position.strategy
                self.state.s8_entry_price         = position.entry_price
                self.state.s8_stop_price_original = position.stop_price_original
                self.state.s8_stop_price_current  = position.current_sl
                self.state.s8_trade_direction     = position.direction
                self.state.s8_be_activated        = False
                self.state.s8_open_time_utc       = position.entry_time.isoformat()
            else:
                self.state.s8_open_ticket         = None
                self.state.s8_entry_price         = 0.0
                self.state.s8_stop_price_original = 0.0
                self.state.s8_stop_price_current  = 0.0
                self.state.s8_trade_direction     = None
                self.state.s8_be_activated        = False
                self.state.s8_open_time_utc       = None
        elif position.strategy == "R3_CAL_MOMENTUM":
            if action == "OPEN":
                self.state.r3_open_ticket = position.strategy
                self.state.r3_open_time   = position.entry_time
                self.state.r3_entry_price = position.entry_price
                self.state.r3_stop_price  = position.current_sl
                self.state.r3_tp_price    = position.tp
            else:
                self.state.r3_open_ticket = None
                self.state.r3_open_time   = None
                self.state.r3_entry_price = 0.0
                self.state.r3_stop_price  = 0.0
                self.state.r3_tp_price    = 0.0
        else:
            if action == "OPEN":
                self.state.trend_family_occupied  = True
                self.state.trend_family_strategy  = position.strategy
                self.state.open_position          = position.strategy
                self.state.entry_price            = position.entry_price
                self.state.stop_price_original    = position.stop_price_original
                self.state.stop_price_current     = position.current_sl
                self.state.original_lot_size      = position.lots
                self.state.position_be_activated  = False
                self.state.position_partial_done  = False
            else:
                self.state.trend_family_occupied  = False
                self.state.trend_family_strategy  = None
                self.state.open_position          = None
                self.state.entry_price            = 0.0
                self.state.stop_price_original    = 0.0
                self.state.stop_price_current     = 0.0
                self.state.original_lot_size      = 0.0
                self.state.position_be_activated  = False
                self.state.position_partial_done  = False

    def _close_all_positions(
        self, price: float, time: datetime, reason: str
    ) -> None:
        for pos in list(self.open_positions):
            trade = self._build_trade_record(pos, price, time, reason)
            self._record_trade(trade)
            self._update_position_state(pos, "CLOSE")
        self.open_positions = []

    def _reconcile_positions(self) -> None:
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
