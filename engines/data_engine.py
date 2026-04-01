"""
engines/data_engine.py — Layer 5: Data Engine.

Feeds everything below it. No layer above this one touches raw MT5 or
external APIs directly — all data flows through here.

Sources (spec Part 3):
  1. MT5 OHLCV         — M5, M15, H1, H4, real-time on candle close
  2. TLT/TIP ETF       — macro yield proxy, daily 09:00 IST (ADD-5)
  3. Economic Calendar — HorizonFX REST (primary) + hardcoded fallback (ADD-1)
  4. Spread Logger     — every 5 min, startup fallback (C1 Fix)
  5. Commission        — tracked per trade in Truth Engine
  6. DXY Correlation   — MT5 USDX primary / UUP daily fallback (B1 Fix)

RULE: CONTRACT_SPEC is populated from MT5 metadata on startup.
      Never hardcode tick_value, point_size, or lot_limits.
"""
import time
import pytz
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, date
from calendar import monthrange


import config
from utils.logger import log_event, log_warning
from utils.mt5_client import get_mt5
from utils.session import (
    get_current_session, get_london_local_time, get_ist_time, LONDON_TZ
)
from db.connection import execute_write, execute_query



# ─────────────────────────────────────────────────────────────────────────────
# 1 — CONTRACT SPEC
# ─────────────────────────────────────────────────────────────────────────────


def get_contract_spec(symbol: str = "XAUUSD") -> dict:
    """
    Pulls all contract parameters from MT5 symbol metadata on startup.
    RULE: Never hardcode tick_value, point_size, or lot_limits.
    Called once in initialize_system() — result stored in config.CONTRACT_SPEC.
    Prints all values for manual verification before first live trade.
    """
    mt5  = get_mt5()
    info = mt5.symbol_info(symbol)


    if info is None:
        raise RuntimeError(
            f"symbol_info({symbol}) returned None. "
            f"Verify symbol name in MT5 Market Watch. "
            f"Error: {mt5.last_error()}"
        )


    spec = {
        "symbol":       symbol,
        "point":        info.point,
        "tick_size":    info.trade_tick_size,
        "tick_value":   info.trade_tick_value,
        "volume_min":   info.volume_min,
        "volume_max":   info.volume_max,
        "volume_step":  info.volume_step,
        "contract_size": info.trade_contract_size,
        "digits":       info.digits,
        "currency_profit": info.currency_profit,
    }


    # Print for manual verification — required before first live trade
    print("\n── CONTRACT SPEC (manual verification required) ──────────────")
    for k, v in spec.items():
        print(f"  {k:<20} = {v}")
    print("──────────────────────────────────────────────────────────────\n")


    log_event("CONTRACT_SPEC_LOADED",
              symbol=symbol,
              tick_value=spec["tick_value"],
              volume_min=spec["volume_min"],
              volume_step=spec["volume_step"])


    return spec



# ─────────────────────────────────────────────────────────────────────────────
# 2 — OHLCV FETCHING + STORAGE
# ─────────────────────────────────────────────────────────────────────────────


# MT5 timeframe constants mapped to string labels
TF_MAP = {
    "M5":  1,    # mt5.TIMEFRAME_M5
    "M15": 3,    # mt5.TIMEFRAME_M15
    "H1":  16385, # mt5.TIMEFRAME_H1
    "H4":  16388, # mt5.TIMEFRAME_H4
    "D1":  16408, # mt5.TIMEFRAME_D1
}



def get_tf_constant(tf_str: str) -> int:
    """Returns MT5 timeframe integer constant for a given string."""
    mt5 = get_mt5()
    mapping = {
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    if tf_str not in mapping:
        raise ValueError(f"Unknown timeframe: {tf_str}")
    return mapping[tf_str]



def fetch_ohlcv(tf_str: str, count: int) -> pd.DataFrame | None:
    """
    Fetches the last `count` OHLCV bars from MT5 for XAUUSD.
    Returns a DataFrame with columns: time, open, high, low, close, tick_volume, spread.
    Returns None on failure.
    """
    mt5 = get_mt5()
    tf  = get_tf_constant(tf_str)


    bars = mt5.copy_rates_from_pos(config.SYMBOL, tf, 0, count)
    if bars is None or len(bars) == 0:
        log_warning("OHLCV_FETCH_FAILED", tf=tf_str, error=str(mt5.last_error()))
        return None


    # Materialise rpyc NetRef column-by-column (preserves field names).
    # list(bars) alone loses dtype field names — access named fields instead.
    df = pd.DataFrame({
        "time":        list(bars["time"]),
        "open":        list(bars["open"]),
        "high":        list(bars["high"]),
        "low":         list(bars["low"]),
        "close":       list(bars["close"]),
        "tick_volume": list(bars["tick_volume"]),
        "spread":      list(bars["spread"]),
    })
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df



def store_ohlcv(df: pd.DataFrame, tf_str: str) -> None:
    """
    Upserts OHLCV rows to market_data.ohlcv.
    Skips rows that already exist (unique index on timestamp+timeframe).
    """
    if df is None or df.empty:
        return


    rows = df.to_dict("records")
    sql = """
        INSERT INTO market_data.ohlcv
            (timestamp, timeframe, open, high, low, close, volume, spread)
        VALUES
            (:time, :tf, :open, :high, :low, :close, :vol, :spread)
        ON CONFLICT (timestamp, timeframe) DO NOTHING
    """
    for row in rows:
        execute_write(sql, {
            "time":   row["time"],
            "tf":     tf_str,
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  float(row["close"]),
            "vol":    int(row.get("tick_volume", 0)),
            "spread": float(row.get("spread", 0)),
        })



def fetch_and_store_ohlcv(tf_str: str = "M15", count: int = 100) -> pd.DataFrame | None:
    """Combined fetch + store. Called by APScheduler on candle close."""
    df = fetch_ohlcv(tf_str, count)
    if df is not None:
        store_ohlcv(df, tf_str)
        log_event("OHLCV_STORED", tf=tf_str, rows=len(df))
    return df



# ─────────────────────────────────────────────────────────────────────────────
# 3 — PRE-LONDON RANGE (G2 Fix — M15 candles, not H1)
# ─────────────────────────────────────────────────────────────────────────────


def get_m15_candles_in_london_window(start_hour: int, start_min: int,
                                      end_hour: int,   end_min: int) -> pd.DataFrame | None:
    """
    G2 Fix: Returns M15 candles strictly within the specified London local time window.

    Why M15 and not H1:
      An H1 candle starting at 07:00 London runs to 07:59:59 —
      it includes post-07:55 price action and contaminates the pre-London range.
      M15 candles let us cut off precisely at 07:55.

    Fetches last 48 M15 bars (~12 hours) then filters to London local window.
    """
    df = fetch_ohlcv("M15", count=48)
    if df is None or df.empty:
        log_warning("M15_CANDLES_FETCH_FAILED_FOR_RANGE")
        return None


    # Convert UTC timestamps to London local
    df["london_time"] = df["time"].dt.tz_convert(LONDON_TZ)
    df["london_hour"] = df["london_time"].dt.hour
    df["london_min"]  = df["london_time"].dt.minute


    # Filter: candle START time must be >= window start and < window end
    start_total = start_hour * 60 + start_min
    end_total   = end_hour   * 60 + end_min


    df["london_total_min"] = df["london_hour"] * 60 + df["london_min"]
    window = df[
        (df["london_total_min"] >= start_total) &
        (df["london_total_min"] <  end_total)
    ]


    return window if not window.empty else None



def calculate_pre_london_range() -> dict | None:
    """
    G2 Fix: Calculates pre-London session range using M15 candles.
    Called by APScheduler at London 07:55 local (DST-safe).

    ── CHANGE 2 (v1.1): Range window extended to 00:00–07:55 UTC ────────────
    Previous window: 04:00 London local → 07:55 London local
    New window:      00:00 UTC          → 07:55 UTC

    Why: Spec defines range as "Asian/pre-London high/low 00:00–07:55 GMT".
    04:00 London local = 03:00 UTC during BST — missed the first 3 hours of
    the Asian session every summer. Now filters on UTC directly to be
    DST-immune. get_m15_candles_in_london_window() is NOT used here because
    it filters on London local time — we need UTC-based filtering.

    Returns range dict consumed by S1 signal engine, or None if range invalid.
    """
    # Fetch 96 M15 bars (~24 hours) — enough to cover full 00:00–07:55 UTC
    df = fetch_ohlcv("M15", count=96)
    if df is None or df.empty:
        log_event("RANGE_CALC_NO_CANDLES")
        return None

    # df["time"] is already UTC (from fetch_ohlcv pd.to_datetime with utc=True)
    today_utc = datetime.now(pytz.utc).date()
    start_utc = pd.Timestamp(year=today_utc.year, month=today_utc.month,
                             day=today_utc.day, hour=0, minute=0,
                             tz=pytz.utc)
    end_utc   = pd.Timestamp(year=today_utc.year, month=today_utc.month,
                             day=today_utc.day, hour=7, minute=55,
                             tz=pytz.utc)

    candles = df[(df["time"] >= start_utc) & (df["time"] < end_utc)]

    if candles is None or candles.empty:
        log_event("RANGE_CALC_NO_CANDLES")
        return None


    range_high = float(candles["high"].max())
    range_low  = float(candles["low"].min())
    range_size = range_high - range_low


    if range_size < config.MIN_RANGE_SIZE_PTS:
        log_event("RANGE_TOO_NARROW",
                  range_size=round(range_size, 2),
                  min_required=config.MIN_RANGE_SIZE_PTS)
        return None


    result = {
        "range_high":     range_high,
        "range_low":      range_low,
        "range_size":     range_size,
        "breakout_dist":  range_size * config.BREAKOUT_DIST_PCT,   # 12%
        "hunt_threshold": range_size * config.HUNT_THRESHOLD_PCT,  # 8%
        "computed_at":    datetime.now(pytz.utc),
        "candle_count":   len(candles),
    }


    log_event("PRE_LONDON_RANGE_CALCULATED",
              high=round(range_high, 2),
              low=round(range_low, 2),
              size=round(range_size, 2),
              breakout_dist=round(result["breakout_dist"], 2))


    return result



# ─────────────────────────────────────────────────────────────────────────────
# 4 — SPREAD LOGGER + C1 FALLBACK
# ─────────────────────────────────────────────────────────────────────────────


def log_spread(state: dict) -> None:
    """
    Source 4: Logs current bid/ask spread to system_state.spread_log.
    Called by APScheduler every 5 minutes during market hours.

    C1 Fix: tracks spread_readings_count.
    After SPREAD_INIT_MIN_READINGS (6 readings = ~30 min),
    switches from hardcoded fallback averages to real rolling average.
    """
    mt5  = get_mt5()
    tick = mt5.symbol_info_tick(config.SYMBOL)


    if tick is None:
        log_warning("SPREAD_LOG_TICK_FAILED")
        return


    spread_pts = round((tick.ask - tick.bid) / config.CONTRACT_SPEC.get("point", 0.01), 1)
    session    = get_current_session()


    execute_write(
        """INSERT INTO system_state.spread_log (timestamp, spread_pts, session)
           VALUES (:ts, :sp, :sess)""",
        {"ts": datetime.now(pytz.utc), "sp": spread_pts, "sess": session}
    )


    # C1 Fix: increment reading counter, switch from fallback when ready
    state["spread_readings_count"] += 1
    if (state["spread_fallback_active"] and
            state["spread_readings_count"] >= config.SPREAD_INIT_MIN_READINGS):
        state["spread_fallback_active"]     = False
        state["session_spread_initialized"] = True
        log_event("SPREAD_FALLBACK_DEACTIVATED",
                  readings=state["spread_readings_count"])



def get_session_avg_spread(session: str | None = None) -> float:
    """
    Returns the rolling average spread for the current (or given) session.

    C1 Fix: If spread_fallback_active or fewer than 6 readings exist,
    returns hardcoded safe fallback value. Switches to real rolling
    average after 30 minutes (6 readings).

    This is the ONLY value used by KS2 spread guard.
    """
    if session is None:
        session = get_current_session()


    # Always try real rolling average first (last 12 readings = 1 hour)
    rows = execute_query(
        """SELECT AVG(spread_pts) as avg_spread, COUNT(*) as cnt
           FROM system_state.spread_log
           WHERE session = :sess
             AND timestamp > NOW() - INTERVAL '1 hour'""",
        {"sess": session}
    )


    if rows and rows[0]["cnt"] and int(rows[0]["cnt"]) >= config.SPREAD_INIT_MIN_READINGS:
        return float(rows[0]["avg_spread"])


    # C1 Fix: fallback to hardcoded safe values
    fallback_key = {
        "LONDON_NY_OVERLAP": "LONDON_NY",
        "LONDON":            "LONDON",
        "NY":                "NY",
        "OFF_HOURS":         "OFF_HOURS",
    }.get(session, "LONDON_NY")


    avg = config.SPREAD_FALLBACK_POINTS[fallback_key]
    log_event("SPREAD_USING_FALLBACK", session=session, fallback_avg=avg)
    return float(avg)


# ── CHANGE 4 (v1.1): New function — standardised 24h spread baseline ─────────
def get_avg_spread_last_24h() -> float:
    """
    v1.1: Returns average spread over the last 24 hours from spread_log.

    This is the SINGLE baseline used by both:
      - Pre-placement gate: block if current_spread > 1.2 × this value
      - KS2 hard gate:      kill switch if current_spread > 2.5 × this value

    Using the same baseline for both gates eliminates the inconsistency
    where KS2 used session-average but pre-placement used a different window.
    Falls back to get_session_avg_spread() if spread_log has insufficient data.
    """
    rows = execute_query(
        """SELECT AVG(spread_pts) as avg_spread, COUNT(*) as cnt
           FROM system_state.spread_log
           WHERE timestamp >= NOW() - INTERVAL '24 hours'
             AND spread_pts > 0""",
        {}
    )

    if rows and rows[0]["cnt"] and int(rows[0]["cnt"]) >= config.SPREAD_INIT_MIN_READINGS:
        val = float(rows[0]["avg_spread"])
        log_event("SPREAD_24H_BASELINE_USED", avg=round(val, 2),
                  readings=int(rows[0]["cnt"]))
        return val

    # Not enough 24h data yet (e.g. fresh system start) — fall back to session avg
    fallback = get_session_avg_spread()
    log_event("SPREAD_24H_FALLBACK_TO_SESSION", fallback=round(fallback, 2))
    return fallback


def get_spread_rolling_median(window_hours: int = 24) -> float:
    """
    F6: Returns MEDIAN (not average) spread over the last N hours.
    
    Median is more robust to spike contamination than average.
    When a 3-pt spread spike hits during London, it inflates the 24h average
    for hours, causing subsequent ratio calculations to appear "normal" when
    they're actually still elevated. Median resists this.

    Used by check_ks2_spread() for the upgraded spread gate.
    Falls back to get_avg_spread_last_24h() if insufficient data.
    """
    rows = execute_query(
        """SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY spread_pts) as median_spread,
                  COUNT(*) as cnt
           FROM system_state.spread_log
           WHERE timestamp >= NOW() - INTERVAL '1 hour' * :window_hours
             AND spread_pts > 0""",
        {"window_hours": window_hours}
    )

    if rows and rows[0]["cnt"] and int(rows[0]["cnt"]) >= config.SPREAD_INIT_MIN_READINGS:
        val = float(rows[0]["median_spread"])
        log_event("SPREAD_ROLLING_MEDIAN_USED", median=round(val, 2),
                  readings=int(rows[0]["cnt"]))
        return val

    # Fall back to average if not enough data for median
    fallback = get_avg_spread_last_24h()
    log_event("SPREAD_MEDIAN_FALLBACK_TO_AVG", fallback=round(fallback, 2))
    return fallback


def get_s1_preplacement_spread_baseline() -> float:
    """
    Pre-placement baseline for S1 London pendings only:
    AVG(spread_pts) from spread_log in [start,end] minute-of-day UTC (GMT wall),
    Mon–Fri (ISODOW 1–5), over the last N distinct UTC dates that have rows in that window.

    Threshold (1.2×) unchanged vs prior design — only the baseline source differs
    from flat get_avg_spread_last_24h().

    Falls back to get_avg_spread_last_24h() if fewer than SPREAD_INIT_MIN_READINGS rows.
    """
    sql = """
WITH eligible AS (
  SELECT sl.spread_pts AS spread_pts,
         (sl.timestamp AT TIME ZONE 'UTC')::date AS d,
         (EXTRACT(HOUR FROM sl.timestamp AT TIME ZONE 'UTC')::int * 60
          + EXTRACT(MINUTE FROM sl.timestamp AT TIME ZONE 'UTC')::int) AS min_of_day
  FROM system_state.spread_log sl
  WHERE sl.spread_pts > 0
    AND EXTRACT(ISODOW FROM sl.timestamp AT TIME ZONE 'UTC') BETWEEN 1 AND 5
    AND (EXTRACT(HOUR FROM sl.timestamp AT TIME ZONE 'UTC')::int * 60
         + EXTRACT(MINUTE FROM sl.timestamp AT TIME ZONE 'UTC')::int)
        >= :start_min
    AND (EXTRACT(HOUR FROM sl.timestamp AT TIME ZONE 'UTC')::int * 60
         + EXTRACT(MINUTE FROM sl.timestamp AT TIME ZONE 'UTC')::int)
        <= :end_min
),
last_n AS (
  SELECT DISTINCT e.d AS d
  FROM eligible e
  ORDER BY d DESC
  LIMIT :n_days
)
SELECT AVG(e.spread_pts) AS avg_spread, COUNT(*)::int AS cnt
FROM eligible e
WHERE e.d IN (SELECT d FROM last_n)
"""
    try:
        rows = execute_query(
            sql,
            {
                "start_min": config.S1_PREPLACEMENT_SPREAD_WINDOW_START_MIN_UTC,
                "end_min":   config.S1_PREPLACEMENT_SPREAD_WINDOW_END_MIN_UTC,
                "n_days":    config.S1_PREPLACEMENT_SPREAD_LOOKBACK_TRADING_DAYS,
            },
        )
    except Exception as e:
        log_warning("S1_PREPLACEMENT_BASELINE_QUERY_FAILED", error=str(e))
        return get_avg_spread_last_24h()

    if not rows or rows[0]["cnt"] is None:
        return get_avg_spread_last_24h()

    cnt = int(rows[0]["cnt"] or 0)
    if cnt < config.SPREAD_INIT_MIN_READINGS:
        fb = get_avg_spread_last_24h()
        log_event("S1_PREPLACEMENT_BASELINE_FALLBACK_COUNT",
                  cnt=cnt, need=config.SPREAD_INIT_MIN_READINGS, fallback=round(fb, 2))
        return fb

    val = float(rows[0]["avg_spread"] or 0.0)
    log_event("S1_PREPLACEMENT_BASELINE_USED",
              avg=round(val, 2), readings=cnt,
              window_utc_min=(config.S1_PREPLACEMENT_SPREAD_WINDOW_START_MIN_UTC,
                               config.S1_PREPLACEMENT_SPREAD_WINDOW_END_MIN_UTC),
              days=config.S1_PREPLACEMENT_SPREAD_LOOKBACK_TRADING_DAYS)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# 5 — DXY CORRELATION (B1 Fix — Timeframe Consistency)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_dxy_correlation(lookback: int = None) -> float:
    """
    B1 Fix: DXY/Gold correlation — timeframe consistency enforced strictly.

    PRIMARY path  : MT5 USDX H1 vs XAUUSD H1 — same feed, no tracking error.
    FALLBACK path : UUP daily vs XAUUSD D1   — MUST use daily gold to match.
                    NEVER mix hourly gold with daily UUP (B1 Fix).

    Status: OBSERVATION MODE — logged per trade, macro_boost flag only.
    Scheduled: every H1 candle close.
    """
    if lookback is None:
        lookback = config.DXY_CORR_LOOKBACK  # 50


    mt5  = get_mt5()
    used_fallback = False


    # ── PRIMARY: MT5 USDX H1 ───────────────────────────────────────────────
    dxy_h1 = mt5.copy_rates_from_pos(
        "USDX", mt5.TIMEFRAME_H1, 0, lookback + 1
    )


    if dxy_h1 is not None and len(dxy_h1) >= lookback:
        gold_h1     = mt5.copy_rates_from_pos(
            config.SYMBOL, mt5.TIMEFRAME_H1, 0, lookback + 1
        )
        gold_closes = [r["close"] for r in gold_h1][-lookback:]
        dxy_closes  = [r["close"] for r in dxy_h1][-lookback:]
        source      = "MT5_USDX_H1"


    else:
        # ── FALLBACK: UUP daily — MUST use D1 gold to match timeframe ──────
        # B1 Fix: timeframe mismatch check — never mix hourly gold with daily UUP
        log_event("DXY_PRIMARY_UNAVAILABLE_USING_UUP_FALLBACK")
        used_fallback = True


        try:
            import yfinance as yf
            uup = yf.download("UUP", period="60d", interval="1d", progress=False)
            if uup.empty:
                raise ValueError("UUP download returned empty")


            gold_d1 = mt5.copy_rates_from_pos(
                config.SYMBOL, mt5.TIMEFRAME_D1, 0, lookback + 1
            )
            # D1 gold to match daily UUP — never hourly (B1 Fix enforced here)
            gold_closes = [r["close"] for r in gold_d1][-lookback:]
            dxy_closes  = uup["Close"].values[-lookback:]
            source      = "UUP_D1_FALLBACK"


        except Exception as e:
            log_warning("DXY_FALLBACK_FAILED", error=str(e))
            log_event("DXY_RETURNING_NEUTRAL_FALLBACK",
                      value=config.DXY_CORR_NEUTRAL_FALLBACK)
            return config.DXY_CORR_NEUTRAL_FALLBACK


    if len(gold_closes) < lookback or len(dxy_closes) < lookback:
        log_warning("DXY_INSUFFICIENT_DATA",
                    gold_len=len(gold_closes), dxy_len=len(dxy_closes))
        return config.DXY_CORR_NEUTRAL_FALLBACK


    # Correlation on first differences (returns, not price levels)
    dxy_trim = dxy_closes[-len(gold_closes):]
    corr = float(
        np.corrcoef(np.diff(gold_closes), np.diff(dxy_trim))[0, 1]
    )


    # Handle NaN (e.g. all identical prices)
    if np.isnan(corr):
        log_warning("DXY_CORR_NAN_RETURNING_NEUTRAL")
        return config.DXY_CORR_NEUTRAL_FALLBACK


    log_event("DXY_CORR_CALCULATED",
              corr=round(corr, 4),
              source=source,
              lookback=lookback,
              used_fallback=used_fallback)


    return corr



def calculate_dxy_ewma_variance(lookback: int = 20) -> float:
    """
    F4: Calculate DXY EWMA variance to detect when dollar is whipsawing.

    When DXY is whipsawing (±0.8% hourly), the correlation computed from
    returns is statistical noise. This function detects that condition
    and stores it in state for use by the regime engine.

    Uses EWMA variance on DXY hourly returns:
      - EWMA variance > DXY_VARIANCE_SPIKE_THRESHOLD: DXY is unstable
      - When unstable: disable macro_boost regardless of correlation

    λ=0.94 (same as F1 ATR weighting): recent ~20 bars carry ~65% of weight.

    Returns float: EWMA variance of DXY returns (in squared returns units).
    Returns None on data failure (safe default = don't disable macro_boost).
    """
    mt5 = get_mt5()

    # Get DXY H1 bars
    dxy_h1 = mt5.copy_rates_from_pos(
        "USDX", mt5.TIMEFRAME_H1, 0, lookback + 1
    )

    if dxy_h1 is None or len(dxy_h1) < lookback:
        log_warning("DXY_EWMA_INSUFFICIENT_DATA",
                    bars_received=len(dxy_h1) if dxy_h1 else 0,
                    bars_needed=lookback)
        return None

    # Calculate DXY returns (first differences)
    dxy_closes = np.array([r["close"] for r in dxy_h1][-lookback:])
    returns = np.diff(dxy_closes)

    if len(returns) < lookback - 1:
        log_warning("DXY_EWMA_RETURNS_TOO_SHORT", len=len(returns))
        return None

    # EWMA variance calculation
    # λ=0.94: recent ~20 bars carry ~65% of weight
    lambda_decay = 0.94
    n = len(returns)

    # EWMA variance = sum of squared demeaned returns weighted by EWMA weights
    mean_return = np.mean(returns)
    squared_residuals = (returns - mean_return) ** 2

    # EWMA weights: most recent returns have highest weight
    indices = np.arange(n)
    weights = np.power(lambda_decay, indices[::-1])
    weights = weights / weights.sum()

    ewma_variance = float(np.dot(weights, squared_residuals))

    log_event("DXY_EWMA_VARIANCE",
              ewma_variance=round(ewma_variance, 8),
              variance_sqrt=round(np.sqrt(ewma_variance), 6),
              threshold=config.DXY_VARIANCE_SPIKE_THRESHOLD,
              is_unstable=ewma_variance > config.DXY_VARIANCE_SPIKE_THRESHOLD)

    return ewma_variance


def is_dxy_stable() -> bool:
    """
    F4: Returns True if DXY is stable (EWMA variance below threshold).
    Used by regime engine to decide whether to apply macro_boost.

    When DXY is unstable (whipsawing), macro_boost is disabled even if
    the correlation meets the threshold. This prevents false positive
    boosts based on noisy correlations during volatile dollar periods.
    """
    # BUG-7 FIX: STATE doesn't exist in regime_engine — it lives in main.py.
    # Instead, accept state as a parameter or use the shared state dict.
    # Since this is called from regime_engine which has access to state,
    # we import the state from main.py's global STATE dict.
    from main import STATE
    dxy_variance = STATE.get("dxy_ewma_variance", None)

    if dxy_variance is None:
        # No variance data yet — assume stable (don't block on missing data)
        return True

    threshold = config.DXY_VARIANCE_SPIKE_THRESHOLD
    is_stable = dxy_variance <= threshold

    if not is_stable:
        log_event("DXY_UNSTABLE_MACRO_BOOST_DISABLED",
                  variance=round(dxy_variance, 8),
                  threshold=threshold)

    return is_stable



# ─────────────────────────────────────────────────────────────────────────────
# 6 — ECONOMIC CALENDAR (ADD-1: HorizonFX + hardcoded fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _generate_hardcoded_events(target_date: date) -> list[dict]:
    """
    ADD-1: Generates approximate HIGH impact event windows from patterns.

    CRITICAL LIMITATION: Covers SCHEDULED releases only.
    Unscheduled Fed speeches are NOT catchable without live API.
    This is logged on every fallback activation.

    Returns list of event dicts compatible with economic_events schema.
    """
    ist = pytz.timezone("Asia/Kolkata")
    utc = pytz.utc
    events = []


    year  = target_date.year
    month = target_date.month


    def ist_to_utc(y, m, d, h, mi) -> datetime:
        dt_ist = ist.localize(datetime(y, m, d, h, mi, 0))
        return dt_ist.astimezone(utc)


    def first_weekday_of_month(y, m, weekday) -> int:
        """Returns day number of first occurrence of weekday (0=Mon) in month."""
        for day in range(1, 8):
            if date(y, m, day).weekday() == weekday:
                return day
        return 1


    _, last_day = monthrange(year, month)


    for pattern in config.HARDCODED_EVENT_PATTERNS:
        name = pattern["name"]


        if "day_of_month" in pattern:
            # Fixed approximate day (CPI, PPI, Retail Sales)
            day = min(pattern["day_of_month"], last_day)
        elif "week_of_month" in pattern and pattern["week_of_month"]:
            # First weekday of month (NFP = first Friday)
            base_day  = first_weekday_of_month(year, month, pattern["weekday"])
            week_offset = (pattern["week_of_month"] - 1) * 7
            day         = base_day + week_offset
            if day > last_day:
                continue
        else:
            # FOMC approximate — weekday pattern without fixed week
            # Find all occurrences of this weekday in the month
            weekday = pattern["weekday"]
            days    = [d for d in range(1, last_day + 1)
                       if date(year, month, d).weekday() == weekday]
            if not days:
                continue
            # Use the middle occurrence as approximate
            day = days[len(days) // 2]


        try:
            scheduled_utc = ist_to_utc(
                year, month, day,
                pattern["hour_ist"], pattern["minute_ist"]
            )
        except ValueError:
            continue


        events.append({
            "event_name":    name,
            "scheduled_utc": scheduled_utc,
            "impact_level":  pattern["impact"],
            "released_flag": None,   # NULL = unknown in fallback mode (ADD-1)
            "source":        "HARDCODED_FALLBACK",
            "fallback_used": True,
        })


    log_event("FALLBACK_CANNOT_COVER_UNSCHEDULED_FED",
              note="Hardcoded fallback active. Unscheduled Fed speeches not covered.")
    return events



def _fetch_horizonfx_events() -> list[dict] | None:
    """
    ADD-1: Primary calendar source — HorizonFX REST API.
    Returns list of raw event dicts or None on failure.
    Retries up to CALENDAR_PULL_RETRIES times with 5s backoff.
    """
    for attempt in range(1, config.CALENDAR_PULL_RETRIES + 1):
        try:
            resp = requests.get(
                config.HORIZONFX_BASE_URL,
                params={"impact": "high"},
                timeout=config.HORIZONFX_TIMEOUT_SEC,
            )
            if resp.status_code == 200:
                data = resp.json()
                log_event("HORIZONFX_CALENDAR_FETCHED",
                          event_count=len(data), attempt=attempt)
                return data
            else:
                log_warning("HORIZONFX_NON_200",
                            status=resp.status_code, attempt=attempt)
        except requests.exceptions.RequestException as e:
            log_warning("HORIZONFX_REQUEST_FAILED",
                        attempt=attempt, error=str(e))


        if attempt < config.CALENDAR_PULL_RETRIES:
            time.sleep(5)


    log_event("CALENDAR_FALLBACK_ACTIVE",
              reason="HorizonFX failed after all retries")
    return None



def _parse_horizonfx_event(raw: dict) -> dict | None:
    """
    Parses a single HorizonFX API response event into our schema format.
    Returns None if the event cannot be parsed (missing critical fields).
    Adjust field names here if HorizonFX changes their response format.
    """
    try:
        # HorizonFX field mapping — update if API schema changes
        raw_time = raw.get("datetime") or raw.get("time") or raw.get("date")
        if not raw_time:
            return None


        # Parse ISO 8601 or Unix timestamp
        if isinstance(raw_time, (int, float)):
            scheduled_utc = datetime.fromtimestamp(raw_time, tz=pytz.utc)
        else:
            scheduled_utc = datetime.fromisoformat(
                raw_time.replace("Z", "+00:00")
            )


        return {
            "event_name":    raw.get("title") or raw.get("event") or raw.get("name", "UNKNOWN"),
            "scheduled_utc": scheduled_utc,
            "impact_level":  "HIGH",
            "released_flag": raw.get("actual") is not None,  # has actual value = released
            "source":        "HORIZONFX",
            "fallback_used": False,
        }
    except Exception as e:
        log_warning("HORIZONFX_EVENT_PARSE_FAILED", error=str(e), raw=str(raw)[:100])
        return None



def fetch_economic_calendar() -> int:
    """
    ADD-1: Main calendar fetch job. Called daily at midnight IST by APScheduler.

    Flow:
      1. Try HorizonFX REST API (3 retries with backoff)
      2. On failure → generate from hardcoded pattern list
      3. Upsert all events to market_data.economic_events
      4. Log CALENDAR_FALLBACK_ACTIVE if fallback was used

    Returns count of events upserted.
    """
    raw_events = _fetch_horizonfx_events()
    today = get_ist_time().date()


    if raw_events is not None:
        # Parse live events
        events = [e for e in (_parse_horizonfx_event(r) for r in raw_events)
                  if e is not None]
    else:
        # ADD-1: fallback to hardcoded patterns for current + next month
        events  = _generate_hardcoded_events(today)
        # Also generate for next month if we're in the last week
        if today.day >= 24:
            next_m  = today.month % 12 + 1
            next_y  = today.year + (1 if today.month == 12 else 0)
            events += _generate_hardcoded_events(date(next_y, next_m, 1))


    if not events:
        log_warning("CALENDAR_NO_EVENTS_TO_STORE")
        return 0


    sql = """
        INSERT INTO market_data.economic_events
            (event_name, scheduled_utc, impact_level, released_flag,
             source, fallback_used)
        VALUES
            (:event_name, :scheduled_utc, :impact_level, :released_flag,
             :source, :fallback_used)
        ON CONFLICT DO NOTHING
    """
    count = 0
    for event in events:
        try:
            execute_write(sql, event)
            count += 1
        except Exception as e:
            log_warning("CALENDAR_EVENT_STORE_FAILED", error=str(e))


    log_event("CALENDAR_STORED", count=count,
              source=events[0]["source"] if events else "NONE")
    return count



def get_upcoming_events_within(minutes: int) -> list[dict]:
    """
    Returns HIGH impact events scheduled within the next `minutes` from now.
    Used by:
      - KS7 blackout guard (45 min pre-event)
      - Regime engine upcoming_events gate
      - Conviction level A+ check (90 min clear horizon)

    Returns empty list if no events found (safe default = trade permitted).
    """
    rows = execute_query(
        """SELECT event_name, scheduled_utc, source, fallback_used
           FROM market_data.economic_events
           WHERE impact_level = 'HIGH'
             AND scheduled_utc BETWEEN NOW() AND NOW() + :interval
             AND (released_flag IS NULL OR released_flag = FALSE)
           ORDER BY scheduled_utc ASC""",
        {"interval": f"{minutes} minutes"}
    )
    return rows



def get_minutes_since_last_event() -> float | None:
    """
    Returns minutes elapsed since the most recent HIGH impact event.
    Used by KS7 resume condition: must be >= KS7_POST_EVENT_MINUTES.
    Returns None if no recent events found.

    FIX (F5): Now checks all HIGH impact events in the window, not just
    those with released_flag = TRUE. This fixes the fallback mode bug where
    released_flag = NULL meant the post-event wait timer never triggered.
    """
    rows = execute_query(
        """SELECT scheduled_utc
           FROM market_data.economic_events
           WHERE impact_level = 'HIGH'
             AND scheduled_utc <= NOW() AT TIME ZONE 'UTC'
             AND scheduled_utc > NOW() AT TIME ZONE 'UTC' - INTERVAL '4 hours'
           ORDER BY scheduled_utc DESC
           LIMIT 1""",
        {}
    )
    if not rows:
        return None

    event_time = rows[0]["scheduled_utc"]
    if event_time.tzinfo is None:
        event_time = pytz.utc.localize(event_time)

    elapsed = (datetime.now(pytz.utc) - event_time).total_seconds() / 60
    return round(elapsed, 1)



# ─────────────────────────────────────────────────────────────────────────────
# F5: ECONOMIC SEVERITY SCORE (NEW)
# ─────────────────────────────────────────────────────────────────────────────

# Severity lookup — maps partial event name matches to severity score (0-100)
# Higher score = more dangerous = smaller lot size
# Score 100 = extreme (NFP/FOMC surprise)
# Score 0 = no significant events = normal operation
_EVENT_SEVERITY_MAP: dict[str, int] = {
    # Tier 1: Extreme (score 65-100)
    "nonfarm": 70,       # NFP — largest market mover
    "nfp": 70,
    "payroll": 70,
    "fomc": 65,          # Fed rate decision
    "fed rate": 65,
    "federal reserve": 65,
    "interest rate": 60, # Other rate decisions (BOE, ECB, BOJ)

    # Tier 2: High (score 50-65)
    "cpi": 55,           # Inflation
    "pce": 50,           # PCE deflator
    "ppi": 45,           # Producer prices
    "retail sales": 40,
    "gdp": 50,          # GDP reports
    "pmi": 35,          # Purchasing managers index
    "ism": 35,
    "employment": 40,
    "unemployment": 40,
    "consumer confidence": 30,
    "consumer sentiment": 30,
    "耐久財": 35,       # Japanese durable goods
    "trade balance": 25,
    "current account": 25,

    # Tier 3: Medium (score 20-35)
    " housing starts": 20,
    "building permits": 20,
    "existing home sales": 20,
    "new home sales": 20,
    "industrial production": 25,
    "capacity utilization": 20,
    "factory orders": 20,
    "business inventories": 15,

    # Tier 4: Lower (score 10-20) — still watch but less dangerous
    "philadelphia fed": 15,
    "ny empire": 15,
    "michigan": 15,
    "beige book": 10,
    "minutes": 10,       # FOMC/Central bank minutes
}


def _fuzzy_match_severity(event_name: str) -> int:
    """
    F5: Returns the highest matching severity score for an event name.
    Uses case-insensitive partial matching to handle variations like
    'Nonfarm Payrolls', 'Non-Farm Payrolls (NFP)', 'NFP', etc.
    """
    event_lower = event_name.lower().strip()
    best_score = 0

    for keyword, score in _EVENT_SEVERITY_MAP.items():
        # Use partial match to handle variations
        if keyword in event_lower:
            best_score = max(best_score, score)

    return best_score


def _decay_severity(minutes_away: float, base_score: int) -> float:
    """
    F5: Exponential decay of severity score over time.

    - Score is highest at t=0 (event is NOW)
    - Decays by ~50% every 15 minutes
    - At 60 minutes away, score is ~6% of original
    - After event passes, decay continues based on elapsed time

    Returns float score (0.0 - 100.0)
    """
    if minutes_away >= 0:
        # Pre-event: decay toward zero as we get further from event
        decay_rate = 0.046  # ln(0.5) / 15 ≈ 0.046
        return base_score * np.exp(-decay_rate * minutes_away)
    else:
        # Post-event: faster decay (news shock dissipates)
        decay_rate = 0.069  # ln(0.5) / 10 ≈ 0.069 (50% every 10 min)
        return base_score * np.exp(-decay_rate * abs(minutes_away))


def get_economic_severity_score(hours_ahead: int = 2) -> float:
    """
    F5: Calculate continuous economic event severity score.

    This replaces the binary KS7 block with a continuous score that:
    1. Looks at events in the next `hours_ahead` hours
    2. Assigns base severity based on event type (NFP > FOMC > CPI > PMI)
    3. Applies exponential decay based on time proximity
    4. Returns 0.0 if no significant events (safe default)

    Integration with lot sizing:
    - score >= 60: HARD HALT (no new entries)
    - score 30-59: severity_multiplier = 1.0 - (score / 100)
    - score < 30: normal operation (severity_multiplier = 1.0)

    Used by:
    - risk_engine.check_ks7_event_blackout() — lot sizing multiplier
    - calculate_lot_size() — severity_multiplier in lot chain

    Returns:
        float: severity score 0.0 to 100.0+
        (scores > 100 possible if multiple high-impact events stack)
    """
    # Get all HIGH impact events in the lookahead window
    rows = execute_query(
        """SELECT event_name, scheduled_utc, released_flag
           FROM market_data.economic_events
           WHERE impact_level = 'HIGH'
             AND scheduled_utc BETWEEN NOW() - INTERVAL '1 hour'
                                    AND NOW() + INTERVAL :interval
           ORDER BY scheduled_utc ASC""",
        {"interval": f"{hours_ahead} hours"}
    )

    if not rows:
        return 0.0

    now = datetime.now(pytz.utc)
    total_severity = 0.0

    for row in rows:
        event_name = row["event_name"]
        event_time = row["scheduled_utc"]

        if event_time.tzinfo is None:
            event_time = pytz.utc.localize(event_time)

        # Calculate minutes from now (negative = past event)
        minutes_away = (event_time - now).total_seconds() / 60

        # Get base severity from event type
        base_score = _fuzzy_match_severity(event_name)

        if base_score == 0:
            # Unknown event type — assume medium risk
            base_score = 25

        # Apply decay
        decayed_score = _decay_severity(minutes_away, base_score)

        # Apply release status penalty (fallback mode = less certain)
        # If released_flag is NULL (fallback mode), reduce score by 20%
        # because we don't know if the actual value has been released
        if row.get("released_flag") is None:
            decayed_score *= 0.8

        total_severity += decayed_score

    return round(total_severity, 1)


def get_severity_multiplier(severity_score: float) -> float:
    """
    F5: Convert severity score to a lot size multiplier.

    Rules:
    - score >= 60: HARD HALT → return 0.0 (caller must block trade)
    - score 30-59: scale linearly from 0.41 to 1.0
    - score < 30: normal operation → return 1.0

    The scaling is designed so that at worst (score=59), you're trading at 41%
    size. At score=30, you're at full size.
    """
    if severity_score >= 60:
        return 0.0  # HARD HALT
    elif severity_score >= 30:
        # Linear interpolation: score 59 → 0.41, score 30 → 1.0
        # multiplier = 1.0 - ((score - 30) / 29) * 0.59
        return round(1.0 - ((severity_score - 30) / 29) * 0.59, 3)
    else:
        return 1.0  # Normal operation



# ─────────────────────────────────────────────────────────────────────────────
# 7 — MACRO BIAS (ADD-5: macro_proxy_instrument tracked per row)
# ─────────────────────────────────────────────────────────────────────────────


def calculate_macro_bias() -> dict:
    """
    Part 5: Calculates macro bias from TLT or TIP ETF (ADD-5).

    The proxy instrument (TLT or TIP) is read from system_config at startup
    and stored in state["macro_proxy_instrument"]. Every row written to
    macro_signals includes the instrument used (ADD-5 audit trail).

    TLT slope threshold read from system_config (ADD-3 — Phase 0 calibration).
    Status: OBSERVATION MODE — logged per trade, not gating trades.
    Runs daily at 09:00 IST via APScheduler.

    Returns dict with bias label + all fields for DB insert.
    """
    proxy = getattr(config, "MACRO_PROXY_INSTRUMENT", "TLT")
    threshold = float(getattr(config, "TLT_SLOPE_THRESHOLD", config.TLT_SLOPE_THRESHOLD_DEFAULT))

    try:
        import yfinance as yf
        data   = yf.download(proxy, period="15d", interval="1d", progress=False)


        if data.empty or len(data) < 10:
            raise ValueError(f"{proxy} download returned insufficient data")


        closes = data["Close"].values.flatten()
        slope  = float(np.polyfit([0, 1, 2], closes[-3:], 1)[0])
        vs_ma  = float(closes[-1] / np.mean(closes[-10:]) - 1)


        # ADD-3: threshold comes from system_config (Phase 0 calibrated)
        if slope > threshold and vs_ma > 0:
            bias = "LONG_PERMITTED"
        elif slope < -threshold and vs_ma < 0:
            bias = "SHORT_PERMITTED"
        elif abs(slope) < threshold * 0.33:   # flat zone = both permitted
            bias = "BOTH_PERMITTED"
        else:
            bias = "NONE_PERMITTED"


    except Exception as e:
        log_warning("MACRO_BIAS_CALC_FAILED", proxy=proxy, error=str(e))
        bias   = "BOTH_PERMITTED"   # safe fallback — doesn't block anything
        slope  = 0.0
        vs_ma  = 0.0
        closes = []


    result = {
        "date":                   get_ist_time().date(),
        "proxy_close":            float(closes[-1]) if len(closes) else None,
        "proxy_3d_slope":         round(slope, 6),
        "proxy_vs_10d_ma":        round(vs_ma, 6),
        "macro_bias_label":       bias,
        "macro_proxy_instrument": proxy,   # ADD-5: written to every row
    }


    # Store to market_data.macro_signals
    execute_write(
        """INSERT INTO market_data.macro_signals
               (date, proxy_close, proxy_3d_slope, proxy_vs_10d_ma,
                macro_bias_label, macro_proxy_instrument)
           VALUES
               (:date, :proxy_close, :proxy_3d_slope, :proxy_vs_10d_ma,
                :macro_bias_label, :macro_proxy_instrument)
           ON CONFLICT (date) DO UPDATE SET
               proxy_close             = EXCLUDED.proxy_close,
               proxy_3d_slope          = EXCLUDED.proxy_3d_slope,
               proxy_vs_10d_ma         = EXCLUDED.proxy_vs_10d_ma,
               macro_bias_label        = EXCLUDED.macro_bias_label,
               macro_proxy_instrument  = EXCLUDED.macro_proxy_instrument""",
        result
    )


    log_event("MACRO_BIAS_CALCULATED",
              proxy=proxy,
              bias=bias,
              slope=round(slope, 4),
              vs_ma=round(vs_ma, 4),
              threshold_used=threshold)


    return result   # ── CHANGE 1: removed extra ) that was here ─────────────



# ─────────────────────────────────────────────────────────────────────────────
# 8 — ATR H1 VALUE (CHANGE 3 — new function used by signal engine)
# ─────────────────────────────────────────────────────────────────────────────


def get_atr14_h1_rma() -> float | None:
    """
    Returns the current ATR(14) VALUE on H1 bars using RMA (Wilder's smoothing).

    DISTINCT from get_atr_percentile_h1() in regime_engine.py:
      - regime_engine  → percentile (WHERE in the distribution is current ATR?)
      - this function  → raw value in price points (HOW LARGE is current ATR?)

    Used by:
      - signal_engine.evaluate_s1_signal()  — S1 stop = range_low - 5pts (Phase 1)
      - signal_engine.evaluate_s3_signal()  — stop = sweep_low - 0.5 × ATR14_H1
      - signal_engine.evaluate_s6_signal()  — stop = entry ± 0.5 × ATR14_H1
      - calibrate_atr.py                    — for percentile calibration display

    Fetches 50 H1 bars — enough to stabilise ATR14 RMA without over-fetching.
    """
    mt5  = get_mt5()
    bars = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_H1, 0, 50)

    if bars is None or len(bars) < 16:
        log_warning("ATR_H1_RMA_INSUFFICIENT_DATA",
                    bars_received=len(bars) if bars else 0)
        return None

    df = pd.DataFrame({
        "high":  list(bars["high"]),
        "low":   list(bars["low"]),
        "close": list(bars["close"]),
    })
    df["atr"] = ta.atr(
        df["high"], df["low"], df["close"],
        length=14,
        mamode=config.ATR_MAMODE  # 'RMA' — pinned, same as regime_engine
    )
    df.dropna(subset=["atr"], inplace=True)

    if df.empty:
        log_warning("ATR_H1_RMA_AFTER_DROPNA_EMPTY")
        return None

    val = float(df["atr"].iloc[-1])
    log_event("ATR_H1_RMA_FETCHED", value=round(val, 4))
    return val

# ─────────────────────────────────────────────────────────────────────────────
# 9 — S6 / S7 MARKET DATA (CHANGE 5 — new section for Phase 1 strategies)
# ─────────────────────────────────────────────────────────────────────────────

def get_asian_range() -> dict | None:
    """
    Fetches Asian session high/low: 00:00–05:30 UTC today.
    Called by s6_asian_range_job at 05:30 UTC.
    Returns None if range < S6_MIN_RANGE_PTS or no data.
    """
    mt5   = get_mt5()
    utc   = pytz.utc
    now   = datetime.now(utc)
    start = now.replace(hour=0,  minute=0,  second=0, microsecond=0)
    end   = now.replace(hour=5,  minute=30, second=0, microsecond=0)

    bars = mt5.copy_rates_range(
        config.SYMBOL, mt5.TIMEFRAME_M15, start, end
    )
    if bars is None or len(bars) < 4:
        log_warning("S6_ASIAN_RANGE_NO_DATA", bars=len(bars) if bars is not None else 0)
        return None

    import pandas as pd
    df     = pd.DataFrame(bars)
    high   = float(df["high"].max())
    low    = float(df["low"].min())
    rsize  = round(high - low, 2)

    if rsize < config.S6_MIN_RANGE_PTS:
        log_event("S6_SKIPPED_RANGE_TOO_SMALL",
                  range_size=rsize, min_required=config.S6_MIN_RANGE_PTS)
        return None

    tick          = mt5.symbol_info_tick(config.SYMBOL)
    spread        = round(tick.ask - tick.bid, 5) if tick else 0.0
    point         = config.CONTRACT_SPEC.get("point", 0.01)
    breakout_dist = round(config.S6_BREAKOUT_DIST_PTS * point, 3)

    log_event("S6_ASIAN_RANGE_CALCULATED",
              range_high=round(high, 3), range_low=round(low, 3),
              range_size=rsize, spread=spread)

    return {
        "range_high":    round(high, 3),
        "range_low":     round(low, 3),
        "range_size":    rsize,
        "breakout_dist": breakout_dist,
        "spread":        spread,
    }

def get_prev_day_ohlc() -> dict | None:
    """
    S7: Returns yesterday's OHLC from MT5 D1 bars.

    Called by midnight_reset_job to set prev_day_high and prev_day_low
    in state for the S7 range filter and pending order placement.

    copy_rates_from_pos(symbol, D1, pos=1, count=1):
      pos=0 → today (forming bar)
      pos=1 → yesterday (last closed D1 bar) ← we want this

    Returns None if data unavailable.
    """
    mt5  = get_mt5()
    bars = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_D1, 1, 1)

    if bars is None or len(bars) < 1:
        log_warning("S7_PREV_DAY_OHLC_UNAVAILABLE",
                    bars_received=len(bars) if bars else 0)
        return None

    row = bars[0]
    result = {
        "high":  float(row["high"]),
        "low":   float(row["low"]),
        "open":  float(row["open"]),
        "close": float(row["close"]),
    }

    log_event("S7_PREV_DAY_OHLC_FETCHED",
              high=round(result["high"], 3),
              low=round(result["low"], 3),
              range=round(result["high"] - result["low"], 2))

    return result

def get_daily_atr14() -> float | None:
    """
    S7: Returns ATR(14) on D1 bars using RMA — for the inside-day range filter.

    S7 rule: prev_day_range < 0.75 × daily_ATR14 → skip (inside day).
    Fetches 30 D1 bars to stabilise ATR14 before reading the last closed value.

    iloc[-1] is the forming daily bar — skip it.
    iloc[-2] is yesterday's closed bar — use this.
    """
    mt5  = get_mt5()
    bars = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_D1, 0, 30)

    if bars is None or len(bars) < 16:
        log_warning("S7_DAILY_ATR14_INSUFFICIENT_DATA",
                    bars_received=len(bars) if bars else 0)
        return None

    df = pd.DataFrame({
        "high":  list(bars["high"]),
        "low":   list(bars["low"]),
        "close": list(bars["close"]),
    })
    df["atr"] = ta.atr(
        df["high"], df["low"], df["close"],
        length=14,
        mamode=config.ATR_MAMODE  # RMA — pinned
    )
    df.dropna(subset=["atr"], inplace=True)

    if len(df) < 2:
        log_warning("S7_DAILY_ATR14_AFTER_DROPNA_INSUFFICIENT")
        return None

    # iloc[-2] = last closed D1 bar (iloc[-1] = forming today)
    val = float(df["atr"].iloc[-2])
    log_event("S7_DAILY_ATR14_FETCHED", value=round(val, 2))
    return val