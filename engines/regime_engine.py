"""
engines/regime_engine.py — Layer 4: Regime Engine.

Gates EVERYTHING below it. No signal fires without regime permission.
This is the single most important safety layer in the system.

Six regime states with size multipliers:
  SUPER_TRENDING   1.5×  DXY-confirmed strong trend (SIZE-2: was 1.2×)
  NORMAL_TRENDING  1.0×  confirmed trend
  WEAK_TRENDING    0.8×  borderline, S1 only
  RANGING_CLEAR    0.7×  S2 only
  UNSTABLE         0.4×  reduce size, no re-entries
  NO_TRADE         0.0×  nothing fires

Fixes implemented:
  Fix 1  — Hysteresis: B→C→B does NOT count as 3 consecutive B readings
  B2     — ATR percentile uses explicit mamode='RMA' (Wilder's) — pinned
  ADD-2  — ADX 18–20 overlap comment: UNSTABLE takes precedence over RANGING
  G6     — APScheduler: coalesce=True, max_instances=1
  v1.1   — bootstrap_regime_from_history() replaces 45-min cold boot
  v1.1   — ADX bar fetch count raised to 100 (stabilise indicator)
  v1.1   — S6 pending orders cancelled on NO_TRADE transition
"""
import pytz
import numpy as np
import pandas as pd
import pandas_ta as ta
from enum import Enum
from datetime import datetime


import config
from utils.logger import log_event, log_warning
from db.persistence import persist_critical_state
from utils.mt5_client import get_mt5
from utils.session import get_current_session
from db.connection import execute_write


# ─────────────────────────────────────────────────────────────────────────────
# REGIME STATE ENUM
# ─────────────────────────────────────────────────────────────────────────────


class RegimeState(str, Enum):
    """
    str mixin ensures RegimeState.SUPER_TRENDING == "SUPER_TRENDING".
    Safe to store directly in DB without calling .value explicitly.
    """
    SUPER_TRENDING  = "SUPER_TRENDING"
    NORMAL_TRENDING = "NORMAL_TRENDING"
    WEAK_TRENDING   = "WEAK_TRENDING"
    RANGING_CLEAR   = "RANGING_CLEAR"
    UNSTABLE        = "UNSTABLE"
    NO_TRADE        = "NO_TRADE"


    @property
    def multiplier(self) -> float:
        return {
            RegimeState.SUPER_TRENDING:  1.5,  # SIZE-2 FIX: was 1.2 — confirmed strong trend deserves max edge
            RegimeState.NORMAL_TRENDING: 1.0,
            RegimeState.WEAK_TRENDING:   0.8,
            RegimeState.RANGING_CLEAR:   0.7,
            RegimeState.UNSTABLE:        0.4,
            RegimeState.NO_TRADE:        0.0,
        }[self]


    @property
    def is_trending(self) -> bool:
        return self in (
            RegimeState.SUPER_TRENDING,
            RegimeState.NORMAL_TRENDING,
            RegimeState.WEAK_TRENDING,
        )


    @property
    def allows_s1(self) -> bool:
        """S1 fires in any trending regime."""
        return self.is_trending


    @property
    def allows_s2(self) -> bool:
        """S2 fires in RANGING_CLEAR only."""
        return self == RegimeState.RANGING_CLEAR


    @property
    def allows_reentry(self) -> bool:
        """S1d re-entries only in SUPER or NORMAL."""
        return self in (RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING)


# Trending regimes that allow momentum cycling (spec Part 7)
MOMENTUM_CYCLING_REGIMES = {RegimeState.SUPER_TRENDING, RegimeState.NORMAL_TRENDING}


# Max M5 re-entries by regime (spec Part 6 — S1d)
MAX_M5_REENTRIES: dict[RegimeState, int] = {
    RegimeState.SUPER_TRENDING:  config.M5_REENTRY_MAX_SUPER,   # 8
    RegimeState.NORMAL_TRENDING: config.M5_REENTRY_MAX_NORMAL,  # 5
}


# ─────────────────────────────────────────────────────────────────────────────
# ATR PERCENTILE — B2 Fix (explicit RMA)
# ─────────────────────────────────────────────────────────────────────────────


def get_atr_percentile_h1(period: int = None,
                          lookback_days: int = None,
                          use_session_filter: bool = False,
                          session_filter: str = None) -> float | None:
    """
    B2 Fix: ATR percentile on H1 data using explicit Wilder's RMA smoothing.

    F1: Now supports EWMA-weighted percentile calculation.
    F2: Now supports session-normalized ATR percentile.

    EWMA weighting (λ=0.94):
      - Last ~20 bars carry ~65% of the ranking weight
      - Volatility clusters — recent bars are more relevant than 29-day-old bars
      - Equal-weight ranking gave 29-day-old bars the same weight as 2-hour-old bars

    Session normalization:
      - Filters historical bars by session before ranking
      - Asian ATR (8-14 pts) ranked separately from London ATR (16-30 pts)
      - Without this, Asian bars permanently suppress percentiles to 20-35th

    RULE: mamode='RMA' is pinned — never rely on pandas_ta defaults.
    Wilder's RMA = standard ATR as shown in TradingView and MT5.
    Any other mamode will produce different values and break calibration.

    ATR percentile reference (Phase 0 calibration — XAUUSD recalibrated):
      ~25th pct  ≈ quiet (8–12 pts)       → ATR_PCT_QUIET_REF         (25)
      ~55th pct  ≈ SUPER gate (~18–25pts) → ATR_PCT_SUPER_THRESHOLD   (55)
      ~85th pct  ≈ UNSTABLE (~27–35pts)   → ATR_PCT_UNSTABLE_THRESHOLD (85)
      ~95th pct  ≈ NO_TRADE (~49pts+)     → ATR_PCT_NO_TRADE_THRESHOLD (95)

    Note: XAUUSD is inherently high-volatility. 1% risk sizing is the real
    ATR protection — these thresholds are deliberately wide to avoid
    blocking tradeable sessions on gold.

    Returns None on data failure — caller treats as NO_TRADE.
    """
    if period is None:        period       = config.ATR_PERIOD
    if lookback_days is None: lookback_days = config.ATR_LOOKBACK_DAYS


    mt5  = get_mt5()
    bars = mt5.copy_rates_from_pos(
        config.SYMBOL, mt5.TIMEFRAME_H1, 0, lookback_days * 24
    )


    if bars is None or len(bars) < period + 1:
        log_warning("ATR_PCT_INSUFFICIENT_DATA",
                    bars_received=len(bars) if bars else 0,
                    bars_needed=period + 1)
        return None


    df = pd.DataFrame(bars)

    # Convert time to UTC for session filtering
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["hour_utc"] = df["time"].dt.hour

    # F2: Session filter — filter bars by session before ranking
    if use_session_filter and session_filter:
        df = _filter_bars_by_session(df, session_filter)
        if len(df) < 30:  # Fallback if session filter returns too few bars
            log_warning("ATR_SESSION_FILTER_TOO_SPARSE",
                        session=session_filter, bars_after_filter=len(df))
            # Fall back to non-filtered (but still EWMA)
            bars = mt5.copy_rates_from_pos(
                config.SYMBOL, mt5.TIMEFRAME_H1, 0, lookback_days * 24
            )
            df = pd.DataFrame(bars)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # B2 Fix: explicit mamode='RMA' — Wilder's smoothing
    # Never change this — must match TradingView and MT5 built-in ATR
    df["atr"] = ta.atr(
        df["high"], df["low"], df["close"],
        length=period,
        mamode=config.ATR_MAMODE  # 'RMA' — pinned in config
    )

    df.dropna(subset=["atr"], inplace=True)

    if df.empty:
        log_warning("ATR_PCT_AFTER_DROPNA_EMPTY")
        return None

    current_atr    = float(df["atr"].iloc[-1])
    historical_atr = df["atr"].values

    # F1: EWMA-weighted percentile calculation
    # λ=0.94 means last ~20 bars carry ~65% of ranking weight
    percentile = _ewma_percentile(historical_atr, current_atr)

    # Also store raw ATR in state for portfolio risk (F8 fix)
    # This is done in the calling function (regime_job) using the returned value

    log_event("ATR_PCT_CALCULATED",
              current_atr=round(current_atr, 4),
              percentile=round(percentile, 1),
              bars_used=len(df),
              ewma=True,
              session_filter=session_filter if use_session_filter else None)

    return percentile


def _ewma_percentile(historical_atr: np.ndarray, current_atr: float, lambda_decay: float = 0.94) -> float:
    """
    F1: EWMA-weighted percentile calculation.

    Uses exponentially weighted moving average to rank current ATR against
    historical ATR values. Recent bars have more weight than old bars.

    Formula:
      weight[i] = lambda^(n-1-i) for i in [0, n-1]
      normalized_weights = weights / sum(weights)
      ewma_pct = sum(normalized_weights * (historical_atr < current_atr).astype(float))

    λ=0.94: Last ~20 bars carry ~65% of ranking weight
    λ=0.96: Last ~25 bars carry ~65% of ranking weight

    This replaces equal-weight ranking where all bars counted equally.
    """
    n = len(historical_atr)

    # Compute EWMA weights: more recent bars have higher weight
    indices = np.arange(n)
    weights = np.power(lambda_decay, indices[::-1])  # Reversed so most recent has highest weight
    weights = weights / weights.sum()  # Normalize to sum to 1

    # Compute indicator: 1 if historical_atr[i] < current_atr, else 0
    below_current = (historical_atr < current_atr).astype(float)

    # EWMA-weighted percentile
    ewma_pct = float(np.dot(weights, below_current) * 100)

    return ewma_pct


def _filter_bars_by_session(df: pd.DataFrame, session: str) -> pd.DataFrame:
    """
    F2: Filter H1 bars by trading session.

    Session windows in UTC:
      - ASIAN:    22:00–07:00 UTC (Tokyo/Sydney open)
      - LONDON:   07:00–16:00 UTC (London session, includes overlap)
      - NY:        13:00–21:00 UTC (New York session)
      - OVERLAP:  13:00–16:00 UTC (London-NY overlap)

    Note: Uses UTC hour, which is stable year-round.
    London BST shifts are handled by MT5 server time, not UTC.
    """
    hour = df["hour_utc"]

    if session == "ASIAN":
        # 22:00–23:59 and 00:00–07:00 UTC
        mask = (hour >= 22) | (hour < 7)
    elif session == "LONDON":
        # 07:00–16:00 UTC
        mask = (hour >= 7) & (hour < 16)
    elif session == "NY":
        # 13:00–21:00 UTC
        mask = (hour >= 13) & (hour < 21)
    elif session == "OVERLAP":
        # 13:00–16:00 UTC (London-NY overlap)
        mask = (hour >= 13) & (hour < 16)
    else:
        # Unknown session — return all bars
        mask = np.ones(len(df), dtype=bool)

    return df[mask].copy()


def get_current_atr_m15(period: int = 14) -> float | None:
    """
    Returns the current ATR(14) on M15 bars using RMA.
    Used by position management for ATR trailing stop.
    Separate from get_atr_percentile_h1 — different timeframe, different purpose.
    """
    mt5  = get_mt5()
    bars = mt5.copy_rates_from_pos(
        config.SYMBOL, mt5.TIMEFRAME_M15, 0, 50
    )


    if bars is None or len(bars) < period + 1:
        log_warning("ATR_M15_INSUFFICIENT_DATA")
        return None


    df = pd.DataFrame(bars)
    df["atr"] = ta.atr(
        df["high"], df["low"], df["close"],
        length=period,
        mamode=config.ATR_MAMODE  # RMA — pinned
    )
    df.dropna(subset=["atr"], inplace=True)


    return float(df["atr"].iloc[-1]) if not df.empty else None


# ─────────────────────────────────────────────────────────────────────────────
# ADX H4
# ─────────────────────────────────────────────────────────────────────────────


def get_adx_h4(period: int = 14) -> float | None:
    """
    Returns the current ADX value from H4 bars.
    Uses pandas_ta adx() — returns ADX_{period} column.
    Returns None on data failure.

    v1.1: Fetch 100 bars (was period * 3 = 42). ADX14 needs enough history
    to stabilise Wilder's smoothing — 42 bars produced noisy early readings.
    """
    mt5 = get_mt5()

    # ── CHANGE 1: was `period * 3` (42 bars) — raised to 100 per v1.1 ────────
    bars = mt5.copy_rates_from_pos(
        config.SYMBOL, mt5.TIMEFRAME_H4, 0, 100
    )


    if bars is None or len(bars) < period * 2:
        log_warning("ADX_H4_INSUFFICIENT_DATA",
                    bars_received=len(bars) if bars else 0)
        return None


    df     = pd.DataFrame(bars)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=period)


    if adx_df is None or adx_df.empty:
        log_warning("ADX_H4_CALCULATION_FAILED")
        return None


    col_name = f"ADX_{period}"
    if col_name not in adx_df.columns:
        log_warning("ADX_H4_COLUMN_MISSING", columns=list(adx_df.columns))
        return None


    adx_val = float(adx_df[col_name].iloc[-1])


    if pd.isna(adx_val):
        log_warning("ADX_H4_VALUE_IS_NAN")
        return None


    log_event("ADX_H4_FETCHED", adx=round(adx_val, 2))
    return adx_val


def get_adx_h4_full(period: int = 14) -> tuple[float | None, float | None, float | None]:
    """
    CHANGE 9.1: Returns (adx, di_plus, di_minus) from H4 bars in a single fetch.
    Called from regime_job() to cache DI+/DI- in STATE for S6/S7 ADX trend filter.
    Avoids double-fetching H4 bars (get_adx_h4 + separate DI fetch).
    """
    mt5 = get_mt5()
    bars = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_H4, 0, 100)

    if bars is None or len(bars) < period * 2:
        log_warning("ADX_H4_FULL_INSUFFICIENT_DATA",
                    bars_received=len(bars) if bars else 0)
        return None, None, None

    df = pd.DataFrame(bars)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=period)

    if adx_df is None or adx_df.empty:
        return None, None, None

    col_adx = f"ADX_{period}"
    col_dmp = f"DMP_{period}"
    col_dmn = f"DMN_{period}"

    if col_adx not in adx_df.columns:
        return None, None, None

    adx_val = adx_df[col_adx].iloc[-1]
    if pd.isna(adx_val):
        return None, None, None

    di_plus  = adx_df[col_dmp].iloc[-1] if col_dmp in adx_df.columns else None
    di_minus = adx_df[col_dmn].iloc[-1] if col_dmn in adx_df.columns else None

    return (
        float(adx_val),
        float(di_plus)  if di_plus  is not None and not pd.isna(di_plus)  else None,
        float(di_minus) if di_minus is not None and not pd.isna(di_minus) else None,
    )


def get_adx_h4_slope(period: int = 14) -> tuple[float | None, bool]:
    """
    Returns (current_adx, is_increasing) using last 2 fully closed H4 bars.

    is_increasing = True if last closed ADX bar > second-to-last closed ADX bar.
    Used by S4: trend must be present (adx > 20) AND accelerating (increasing).

    Rule from spec: "Use the last two fully closed H4 bars for the ADX
    increasing check. Do not read ADX from the current forming H4 bar."
    - iloc[-1] = forming bar (skip)
    - iloc[-2] = last closed bar   → current_adx
    - iloc[-3] = previous closed bar → prev_adx
    """
    mt5 = get_mt5()
    bars = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_H4, 0, 100)

    if bars is None or len(bars) < 30:
        log_warning("ADX_SLOPE_INSUFFICIENT_DATA",
                    bars_received=len(bars) if bars else 0)
        return None, False

    df     = pd.DataFrame(bars)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=period)

    if adx_df is None or adx_df.empty:
        log_warning("ADX_SLOPE_CALCULATION_FAILED")
        return None, False

    col_name = f"ADX_{period}"
    if col_name not in adx_df.columns:
        log_warning("ADX_SLOPE_COLUMN_MISSING", columns=list(adx_df.columns))
        return None, False

    df["adx"] = adx_df[col_name]
    df.dropna(subset=["adx"], inplace=True)

    if len(df) < 4:
        log_warning("ADX_SLOPE_INSUFFICIENT_CLOSED_BARS")
        return None, False

    current_adx = float(df["adx"].iloc[-2])   # last fully closed H4 bar
    prev_adx    = float(df["adx"].iloc[-3])   # bar before that
    is_increasing = current_adx > prev_adx

    log_event("ADX_H4_SLOPE_CALCULATED",
              current=round(current_adx, 2),
              previous=round(prev_adx, 2),
              increasing=is_increasing)

    return current_adx, is_increasing


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CALCULATION
# ─────────────────────────────────────────────────────────────────────────────


def calculate_regime(
    adx_h4:         float,
    atr_pct_h1:     float,
    dxy_corr_50:    float,
    upcoming_events: bool,
    spread_ratio:   float,
    state:          dict = None,
) -> tuple[RegimeState, float]:
    """
    Core regime classification. Returns (RegimeState, effective_multiplier).
    effective_multiplier includes session adjustment.

    Thresholds from system_config (recalibrated for XAUUSD Phase 0):
      ATR_PCT_NO_TRADE_THRESHOLD  = 95   (was 90 — XAUUSD is inherently volatile)
      ATR_PCT_UNSTABLE_THRESHOLD  = 85   (was 70 — same reason)
      ATR_PCT_SUPER_THRESHOLD     = 55
      ADX overextended gate       = 55   (was 45 — gold trends to ADX 50–60 routinely)
      ADX unstable dead zone      = 18–20 (was 18–22, narrowed)
    """
    session = get_current_session()


    no_trade_thresh = config.ATR_PCT_NO_TRADE_THRESHOLD  # 95
    unstable_thresh = config.ATR_PCT_UNSTABLE_THRESHOLD  # 85
    super_thresh    = config.ATR_PCT_SUPER_THRESHOLD     # 55


    # ── HARD NO_TRADE ────────────────────────────────────────────────────────
    if atr_pct_h1 > no_trade_thresh:
        log_event("REGIME_NO_TRADE_ATR", atr_pct=round(atr_pct_h1, 1))
        return RegimeState.NO_TRADE, 0.0


    if upcoming_events:
        log_event("REGIME_NO_TRADE_EVENT_BLACKOUT")
        return RegimeState.NO_TRADE, 0.0


    if session == "OFF_HOURS":
        log_event("REGIME_NO_TRADE_OFF_HOURS")
        return RegimeState.NO_TRADE, 0.0


    if spread_ratio > config.KS2_SPREAD_MULTIPLIER:  # 2.5
        log_event("REGIME_NO_TRADE_SPREAD", spread_ratio=round(spread_ratio, 2))
        return RegimeState.NO_TRADE, 0.0


    # ── DXY MACRO BOOST ──────────────────────────────────────────────────────
    # Observation mode — only gates SUPER_TRENDING, logged per trade
    # F4: Check DXY stability before applying macro_boost
    # When DXY is whipsawing (high EWMA variance), correlation is noise
    dxy_stable = True
    if state is not None:
        dxy_variance = state.get("dxy_ewma_variance")
        if dxy_variance is not None:
            dxy_stable = dxy_variance <= config.DXY_VARIANCE_SPIKE_THRESHOLD

    macro_boost = (
        dxy_stable and
        dxy_corr_50 < config.DXY_CORR_SUPER_THRESHOLD  # -0.70
    )


    # ── UNSTABLE ─────────────────────────────────────────────────────────────
    # ADD-2: ADX 18–20 overlap with RANGING_CLEAR (adx < 18) is INTENTIONAL.
    # ADX in transition zone (18–20) = trend ambiguous, not confirmed ranging.
    # UNSTABLE takes precedence over RANGING here — evaluated first.
    # Do not reorder these blocks without a full regime review.
    # ADX > 55 gate: gold regularly trends to ADX 50–60; 45 was too tight.
    if atr_pct_h1 > unstable_thresh or adx_h4 > 55 or 18 <= adx_h4 <= 20:
        return RegimeState.UNSTABLE, 0.4


    # ── RANGING ──────────────────────────────────────────────────────────────
    if adx_h4 < 18:
        return RegimeState.RANGING_CLEAR, 0.7


    # ── TRENDING ─────────────────────────────────────────────────────────────
    # SIZE-3 FIX: Remove NY session penalty. NY has second highest liquidity
    # after London-NY overlap. Only penalize Asian/off-hours.
    session_mult = (
        1.0 if session in ("LONDON_NY_OVERLAP", "LONDON", "NY") else 0.7
    )


    if adx_h4 > 35 and atr_pct_h1 > super_thresh and macro_boost:
        return RegimeState.SUPER_TRENDING,  round(1.5 * session_mult, 3)  # SIZE-2 FIX: was 1.2
    elif adx_h4 > 35 and atr_pct_h1 > super_thresh:
        return RegimeState.NORMAL_TRENDING, round(1.0 * session_mult, 3)
    elif adx_h4 > 26:
        return RegimeState.NORMAL_TRENDING, round(1.0 * session_mult, 3)
    else:
        return RegimeState.WEAK_TRENDING,   round(0.8 * session_mult, 3)


# ─────────────────────────────────────────────────────────────────────────────
# HYSTERESIS — Fix 1 (corrected B→C→B logic)
# ─────────────────────────────────────────────────────────────────────────────


def apply_hysteresis(new_regime: RegimeState, state: dict) -> RegimeState:
    """
    Fix 1: Regime hysteresis — must see the SAME new regime for
    REGIME_HYSTERESIS_COUNT (3) consecutive readings before flipping.

    Corrected logic: B→C→B does NOT count as 3 consecutive B readings.
    If the pending target changes mid-sequence, the counter resets to 1.

    On flip to NO_TRADE: cancel_all_pending_orders() is called immediately.
    v1.1: S6 pending orders are also cancelled on NO_TRADE flip.
    """
    current = RegimeState(state["current_regime"])


    if new_regime != current:
        pending = state.get("pending_regime_state")


        if pending is not None and new_regime == RegimeState(pending):
            # Same pending target — increment counter
            state["consecutive_regime_readings"] += 1
        else:
            # Fix 1: Different target than pending — reset counter to 1
            # This is what prevents B→C→B from counting as 3× B
            state["pending_regime_state"]         = new_regime.value
            state["consecutive_regime_readings"]  = 1


        log_event("REGIME_PENDING",
                  pending=new_regime,
                  count=state["consecutive_regime_readings"],
                  required=config.REGIME_HYSTERESIS_COUNT)


        if state["consecutive_regime_readings"] >= config.REGIME_HYSTERESIS_COUNT:
            old_regime                           = state["current_regime"]
            state["current_regime"]              = new_regime.value
            state["consecutive_regime_readings"] = 0
            state["pending_regime_state"]        = None


            log_event("REGIME_CHANGED",
                      from_regime=old_regime,
                      to_regime=new_regime.value)


            # ── CHANGE 2: S6 pending also cancelled on NO_TRADE (v1.1) ───────
            if new_regime == RegimeState.NO_TRADE:
                _cancel_pending_on_no_trade()
                state["s1_pending_buy_ticket"]  = None
                state["s1_pending_sell_ticket"] = None
                # v1.1: S6 pending expire safely means cancel, not allow fill
                _cancel_s6_pending_on_no_trade(state)
                log_event("PENDING_CLEARED_ON_NO_TRADE")


    else:
        # New reading confirms current regime — reset pending state
        state["consecutive_regime_readings"] = 0
        state["pending_regime_state"]        = None


    return RegimeState(state["current_regime"])


def _cancel_pending_on_no_trade() -> None:
    """
    Deferred import of cancel_all_pending_orders to avoid circular import.
    Execution engine is built after regime engine.
    Called only when regime flips to NO_TRADE.
    """
    try:
        from engines.execution_engine import cancel_all_pending_orders
        cancel_all_pending_orders()
        log_event("PENDING_ORDERS_CANCELLED_ON_NO_TRADE")
    except ImportError:
        # Execution engine not yet built — safe during development
        log_event("CANCEL_PENDING_DEFERRED_EXEC_ENGINE_NOT_READY")
    except Exception as e:
        log_warning("CANCEL_PENDING_ON_NO_TRADE_FAILED", error=str(e))


def _cancel_s6_pending_on_no_trade(state: dict) -> None:
    """
    v1.1: Cancel S6 BUY STOP and SELL STOP on NO_TRADE transition.
    'Pending expire safely' in spec means cancel immediately on NO_TRADE —
    NOT allow a fill on a NO_TRADE day.
    """
    mt5 = get_mt5()
    for key in ("s6_pending_buy_ticket", "s6_pending_sell_ticket"):
        ticket = state.get(key)
        if ticket:
            try:
                result  = mt5.order_delete(ticket)
                retcode = result.retcode if result else "NONE"
                log_event("S6_PENDING_CANCELLED_NO_TRADE",
                          ticket=ticket, retcode=retcode)
            except Exception as e:
                log_warning("S6_CANCEL_ON_NO_TRADE_FAILED",
                            ticket=ticket, error=str(e))
            finally:
                state[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# STALENESS GUARD
# ─────────────────────────────────────────────────────────────────────────────


def get_safe_regime(state: dict) -> RegimeState:
    """
    Returns current regime only if the last calculation is fresh.
    If regime_calculated_at is older than REGIME_STALENESS_SEC (20 min),
    returns NO_TRADE unconditionally — stale regime is the same as no regime.

    Called by every signal engine gate before firing.
    """
    calc_at = state.get("regime_calculated_at")
    if calc_at is None:
        log_event("REGIME_STALE_NEVER_CALCULATED")
        return RegimeState.NO_TRADE


    age_sec = (datetime.now(pytz.utc) - calc_at).total_seconds()


    if age_sec > config.REGIME_STALENESS_SEC:
        log_event("REGIME_STALE",
                  age_seconds=int(age_sec),
                  limit=config.REGIME_STALENESS_SEC)
        return RegimeState.NO_TRADE


    return RegimeState(state["current_regime"])


# ─────────────────────────────────────────────────────────────────────────────
# REGIME LOG PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def persist_regime_log(
    adx_h4:        float,
    atr_pct_h1:    float,
    regime:        RegimeState,
    size_mult:     float,
    state:         dict,
    regime_age_sec: int = 0,
) -> None:
    """Stores one row to system_state.regime_log after every regime calculation."""
    execute_write(
        """INSERT INTO system_state.regime_log
           (timestamp, adx_h4, atr_pct_h1, session, regime_state,
            size_multiplier, dxy_corr, macro_boost, regime_age_seconds,
            combined_exposure_pct)
           VALUES
           (:ts, :adx, :atr_pct, :sess, :regime, :size_mult,
            :dxy_corr, :macro_boost, :age, :combined_exp)""",
        {
            "ts":          datetime.now(pytz.utc),
            "adx":         round(adx_h4, 4)     if adx_h4     else None,
            "atr_pct":     round(atr_pct_h1, 2) if atr_pct_h1 else None,
            "sess":        get_current_session(),
            "regime":      regime.value,
            "size_mult":   round(size_mult, 3),
            "dxy_corr":    round(state.get("dxy_corr_50", 0), 4),
            "macro_boost": state.get("macro_boost", False),
            "age":         regime_age_sec,
            # combined_exposure_pct: populated by execution engine once built
            # NULL here until then — not a data integrity issue
            "combined_exp": None,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL BOOTSTRAP — v1.1 (replaces 45-min cold boot)
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_regime_from_history(state: dict) -> bool:
    """
    v1.1 Phase 1 Step 2: Reads 100 H4 + 100 H1 bars and checks if the
    last 3 CLOSED bars agree on regime. Confirms instantly if they do.

    Rules (v1.1 — final):
      - Fetch 100 bars to STABILISE the indicator before reading.
        Never compute ADX from scratch on 6 bars (old spec was illustrative).
      - Read last 3 CLOSED bars only. iloc[-1] is the forming bar — skip it.
        Closed bars: iloc[-4], iloc[-3], iloc[-2].
      - If all 3 agree: consecutive_regime_readings = 3, confirmed instantly.
        Log WARM_START_REGIME_BOOTSTRAPPED.
      - If 2 of 3 agree (last two match): readings = 2, one live reading needed.
        Max 15 min wait before first live regime_job fires.
      - If < 2 agree: cold boot. 3 live readings needed, max 30 min.
      - Does NOT depend on DB state — fully market-driven.
      - Bootstrap completes in < 2 seconds.

    Returns True if regime confirmed without any live readings needed.
    Returns False if 1-2 live readings still required (normal startup path).
    """
    try:
        mt5 = get_mt5()

        # ── Fetch H4 bars (ADX) ───────────────────────────────────────────────
        h4_bars = mt5.copy_rates_from_pos(
            config.SYMBOL, mt5.TIMEFRAME_H4, 0, 100
        )
        if h4_bars is None or len(h4_bars) < 30:
            log_warning("BOOTSTRAP_INSUFFICIENT_H4_BARS",
                        count=len(h4_bars) if h4_bars else 0)
            return False

        # ── Fetch H1 bars (ATR percentile) ────────────────────────────────────
        h1_bars = mt5.copy_rates_from_pos(
            config.SYMBOL, mt5.TIMEFRAME_H1, 0, 100
        )
        if h1_bars is None or len(h1_bars) < 30:
            log_warning("BOOTSTRAP_INSUFFICIENT_H1_BARS",
                        count=len(h1_bars) if h1_bars else 0)
            return False

        df_h4 = pd.DataFrame(h4_bars)
        df_h1 = pd.DataFrame(h1_bars)

        # ── Compute ADX14 on all 100 H4 bars ─────────────────────────────────
        adx_result = ta.adx(df_h4["high"], df_h4["low"], df_h4["close"], length=14)
        if adx_result is None or adx_result.empty or "ADX_14" not in adx_result.columns:
            log_warning("BOOTSTRAP_ADX_CALCULATION_FAILED")
            return False

        df_h4["adx"] = adx_result["ADX_14"]

        # ── Compute ATR14 RMA on all 100 H1 bars ─────────────────────────────
        df_h1["atr"] = ta.atr(
            df_h1["high"], df_h1["low"], df_h1["close"],
            length=14, mamode=config.ATR_MAMODE  # RMA — pinned
        )

        df_h4.dropna(subset=["adx"], inplace=True)
        df_h1.dropna(subset=["atr"], inplace=True)

        if len(df_h4) < 4 or len(df_h1) < 4:
            log_warning("BOOTSTRAP_INSUFFICIENT_CLOSED_BARS_AFTER_DROPNA")
            return False

        # ── Read last 3 CLOSED bars (iloc[-1] = forming, skip it) ────────────
        adx_vals = [
            float(df_h4["adx"].iloc[-4]),
            float(df_h4["adx"].iloc[-3]),
            float(df_h4["adx"].iloc[-2]),
        ]

        # ATR percentile: for each bar, compute its pct against the full series
        full_atr_series = df_h1["atr"].values
        atr_closed = [
            float(df_h1["atr"].iloc[-4]),
            float(df_h1["atr"].iloc[-3]),
            float(df_h1["atr"].iloc[-2]),
        ]
        atr_pct_vals = [
            float((full_atr_series < v).mean() * 100)
            for v in atr_closed
        ]

        # ── Classify each of the 3 closed bars ───────────────────────────────
        def _classify(adx: float, atr_pct: float) -> str:
            if atr_pct > config.ATR_PCT_NO_TRADE_THRESHOLD:  return "NO_TRADE"
            if atr_pct > config.ATR_PCT_UNSTABLE_THRESHOLD:  return "UNSTABLE"
            if adx > 55 or 18 <= adx <= 20:                  return "UNSTABLE"
            if adx < 18:                                      return "RANGING_CLEAR"
            if adx > 35 and atr_pct > config.ATR_PCT_SUPER_THRESHOLD:
                return "SUPER_TRENDING"
            if adx > 26:                                      return "NORMAL_TRENDING"
            return "WEAK_TRENDING"

        regimes = [_classify(adx_vals[i], atr_pct_vals[i]) for i in range(3)]

        log_event("BOOTSTRAP_BAR_READINGS",
                  bar_minus3=regimes[0],
                  bar_minus2=regimes[1],
                  bar_minus1=regimes[2],
                  adx_vals=[round(v, 2) for v in adx_vals],
                  atr_pct_vals=[round(v, 1) for v in atr_pct_vals])

        now_utc = datetime.now(pytz.utc)

        # ── Case 1: All 3 agree — instant confirmation ────────────────────────
        if regimes[0] == regimes[1] == regimes[2]:
            state["current_regime"]              = regimes[0]
            state["consecutive_regime_readings"] = 3
            state["pending_regime_state"]        = None
            state["regime_calculated_at"]        = now_utc
            state["size_multiplier"]             = RegimeState(regimes[0]).multiplier
            log_event("WARM_START_REGIME_BOOTSTRAPPED",
                      regime=regimes[0],
                      adx=round(adx_vals[-1], 2),
                      atr_pct=round(atr_pct_vals[-1], 1))
            return True

        # ── Case 2: Last 2 agree — one live reading needed ────────────────────
        if regimes[1] == regimes[2]:
            state["current_regime"]              = regimes[2]
            state["consecutive_regime_readings"] = 2
            state["pending_regime_state"]        = regimes[2]
            state["regime_calculated_at"]        = now_utc
            state["size_multiplier"]             = RegimeState(regimes[2]).multiplier
            log_event("BOOTSTRAP_TWO_OF_THREE",
                      regime=regimes[2],
                      note="1 live reading needed — max 15 min wait")
            return False

        # ── Case 3: No agreement — cold boot ─────────────────────────────────
        log_event("BOOTSTRAP_NO_AGREEMENT",
                  regimes=str(regimes),
                  note="3 live readings needed — max 30 min wait")
        return False

    except Exception as e:
        log_warning("BOOTSTRAP_FAILED", error=str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REGIME JOB — APScheduler callable
# ─────────────────────────────────────────────────────────────────────────────


def regime_job(state: dict) -> None:
    """
    G6 Fix: APScheduler job — runs every 15 minutes.

    Called from main.py as:
        _safe_execute("regime", regime_job)
    which resolves to:
        regime_job()   ← NO, wait — regime_job takes state: dict

    Actually called via main.py's wrapper:
        def regime_job() -> None:   ← main.py wrapper, 0 args, uses global STATE
            regime_engine_job(STATE) ← calls THIS function with STATE

    So this function (regime_engine.regime_job) always receives STATE as arg.

    Full sequence:
    1. Fetch ADX H4 + ATR pct H1
    2. Read DXY corr from state (updated hourly by data engine)
    3. Check upcoming HIGH impact events
    4. Calculate current spread ratio
    5. Calculate raw regime
    6. Apply hysteresis (Fix 1)
    7. Update state + regime_calculated_at (size_mult = confirmed_regime.multiplier)
    8. Persist to regime_log
    9. Persist critical state (G8 Fix: on regime change)
    """
    from engines.data_engine import get_session_avg_spread


    # ── Step 1: indicators ───────────────────────────────────────────────────
    # CHANGE 9.1: Use get_adx_h4_full() to also capture DI+/DI- for S6/S7 filter
    adx_h4, di_plus_h4, di_minus_h4 = get_adx_h4_full()
    atr_pct_h1 = get_atr_percentile_h1()


    if adx_h4 is None or atr_pct_h1 is None:
        log_warning("REGIME_JOB_DATA_UNAVAILABLE",
                    adx_h4=adx_h4, atr_pct_h1=atr_pct_h1)
        # Data unavailable — force NO_TRADE conservatively
        state["current_regime"]       = RegimeState.NO_TRADE.value
        state["regime_calculated_at"] = datetime.now(pytz.utc)
        state["size_multiplier"]      = 0.0
        log_event("REGIME_FORCED_NO_TRADE_DATA_UNAVAILABLE")
        return


    # ── Step 2: DXY from state (updated every H1 by data engine job) ────────
    dxy_corr_50 = state.get("dxy_corr_50", config.DXY_CORR_NEUTRAL_FALLBACK)


    # ── Step 3: KS7 is placement-only (check_ks7 on place_order / pending helpers).
    # Do not force NO_TRADE here — existing S1/S6/S7 pendings stay grandfathered.
    has_events = False

    # ── Step 4: spread ratio ─────────────────────────────────────────────────
    mt5        = get_mt5()
    tick       = mt5.symbol_info_tick(config.SYMBOL)
    spread_now = 0.0
    if tick and config.CONTRACT_SPEC.get("point"):
        spread_now = (tick.ask - tick.bid) / config.CONTRACT_SPEC["point"]


    avg_spread   = get_session_avg_spread()
    spread_ratio = (spread_now / avg_spread) if avg_spread > 0 else 0.0


    # ── Step 5: raw regime calculation ───────────────────────────────────────
    new_regime, size_mult = calculate_regime(
        adx_h4          = adx_h4,
        atr_pct_h1      = atr_pct_h1,
        dxy_corr_50     = dxy_corr_50,
        upcoming_events = has_events,
        spread_ratio    = spread_ratio,
    )


    # ── Step 6: hysteresis (Fix 1) ───────────────────────────────────────────
    confirmed_regime = apply_hysteresis(new_regime, state)


    # ── Step 7: update state ─────────────────────────────────────────────────
    prev_calc      = state.get("regime_calculated_at")
    regime_age_sec = 0
    if prev_calc:
        regime_age_sec = int(
            (datetime.now(pytz.utc) - prev_calc).total_seconds()
        )


    state["regime_calculated_at"] = datetime.now(pytz.utc)
    state["size_multiplier"]      = confirmed_regime.multiplier
    state["macro_boost"]          = dxy_corr_50 < config.DXY_CORR_SUPER_THRESHOLD
    state["last_adx_h4"]          = adx_h4
    state["last_atr_pct_h1"]      = atr_pct_h1
    # CHANGE 9.1: Cache DI+/DI- for S6/S7 ADX trend filter
    if di_plus_h4 is not None:
        state["last_di_plus_h4"]  = di_plus_h4
    if di_minus_h4 is not None:
        state["last_di_minus_h4"] = di_minus_h4
    # F8 fix: also store raw ATR for portfolio risk (avoid re-fetching every M15 candle)
    # BUG-6 FIX: was referencing nonexistent local `df`. Now calls data_engine.
    from engines.data_engine import get_atr14_h1_rma as _get_raw_atr
    _raw_atr = _get_raw_atr()
    state["last_atr_h1_raw"] = _raw_atr if _raw_atr else 0.0


    log_event("REGIME_JOB_COMPLETE",
              adx_h4=round(adx_h4, 2),
              atr_pct=round(atr_pct_h1, 1),
              new_raw=new_regime.value,
              confirmed=confirmed_regime.value,
              size_mult=round(confirmed_regime.multiplier, 3),
              has_events=has_events,
              spread_ratio=round(spread_ratio, 2))


    # ── Step 8: regime log ───────────────────────────────────────────────────
    persist_regime_log(
        adx_h4=adx_h4,
        atr_pct_h1=atr_pct_h1,
        regime=confirmed_regime,
        size_mult=confirmed_regime.multiplier,
        state=state,
        regime_age_sec=regime_age_sec,
    )


    # ── Step 9: persist critical state on any regime change ──────────────────
    # G8 Fix: persist on every regime change (not just opens/closes)
    _persist_state_if_changed(confirmed_regime, state)


def _persist_state_if_changed(confirmed_regime: RegimeState, state: dict) -> None:
    """
    Persist critical state to DB when regime actually changes.
    Avoids writing on every 15-min tick when regime is stable.
    """
    try:
        from db.persistence import persist_critical_state
        if confirmed_regime.value != state.get("_last_persisted_regime"):
            persist_critical_state(state)
            state["_last_persisted_regime"] = confirmed_regime.value
            log_event("CRITICAL_STATE_PERSISTED_ON_REGIME_CHANGE",
                      regime=confirmed_regime.value)
    except ImportError:
        persist_critical_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE HELPERS (used by signal engine gates)
# ─────────────────────────────────────────────────────────────────────────────


def regime_allows_s1(state: dict) -> bool:
    return get_safe_regime(state).allows_s1


def regime_allows_s2(state: dict) -> bool:
    return get_safe_regime(state).allows_s2


def regime_allows_reentry(state: dict) -> bool:
    return get_safe_regime(state).allows_reentry


def get_max_m5_reentries(state: dict) -> int:
    regime = get_safe_regime(state)
    return MAX_M5_REENTRIES.get(regime, 0)   # ── CHANGE 3: removed extra ) ──
