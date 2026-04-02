"""
backtest/data_feed.py — Historical data providers for the backtesting framework.

HistoricalDataFeed:  Fetches M5 bars from MT5 or local cache (parquet/pickle).
BarBuffer:           Accumulates M5 bars into M15, H1, H4, D1 candles.
HistoricalSpreadFeed: Provides historical spread data.
HistoricalEventFeed:  Provides historical economic event data.

All timestamps are timezone-aware UTC (pytz.utc).
"""
import os
import pytz
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger("backtest.data_feed")


# ─────────────────────────────────────────────────────────────────────────────
# TIMEFRAME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# M5 bar count per higher timeframe
TF_M5_MULTIPLES = {
    "M15": 3,
    "H1":  12,
    "H4":  48,
    "D1":  288,
}

# Pandas resample rules for each timeframe
TF_RESAMPLE_RULE = {
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA FEED — M5 bars from MT5 or local cache
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalDataFeed:
    """
    Provides M5 OHLCV bars for a date range.

    Data source priority:
      1. Local parquet file (tools/collect_historical_data.py output)
      2. Local pickle file
      3. MT5 copy_rates_range() — requires live MT5 connection

    After initial fetch, data is cached in a pandas DataFrame.
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "XAUUSD",
        cache_dir: str = "backtest_data",
    ):
        self.start_date = start_date.replace(tzinfo=pytz.utc) if start_date.tzinfo is None else start_date
        self.end_date = end_date.replace(tzinfo=pytz.utc) if end_date.tzinfo is None else end_date
        self.symbol = symbol
        self.cache_dir = Path(cache_dir)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Load M5 data from best available source.
        Returns DataFrame with columns: time, open, high, low, close, tick_volume, spread
        """
        if self._df is not None:
            return self._df

        # Try local cache first
        df = self._try_load_local()
        if df is not None and len(df) > 0:
            logger.info(f"Loaded {len(df)} M5 bars from local cache")
        else:
            # Fall back to MT5
            df = self._fetch_from_mt5()
            if df is not None and len(df) > 0:
                logger.info(f"Fetched {len(df)} M5 bars from MT5")
            else:
                raise RuntimeError(
                    f"No M5 data available for {self.start_date} to {self.end_date}. "
                    "Run tools/collect_historical_data.py first, or ensure MT5 is connected."
                )

        # Filter to requested date range
        df = df[(df["time"] >= self.start_date) & (df["time"] <= self.end_date)].copy()
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        self._df = df
        logger.info(
            f"Data feed ready: {len(df)} M5 bars from "
            f"{df['time'].iloc[0]} to {df['time'].iloc[-1]}"
        )
        return df

    def iter_m5_bars(self) -> Iterator[dict]:
        """
        Yields M5 bars chronologically as dicts.
        Each dict has: time, open, high, low, close, tick_volume, spread
        """
        df = self.load()
        for _, row in df.iterrows():
            yield row.to_dict()

    def _try_load_local(self) -> Optional[pd.DataFrame]:
        """Try loading from local parquet or pickle file."""
        # Check project-relative paths
        base_dir = Path(__file__).parent.parent  # xauusd_algo/
        search_paths = [
            base_dir / self.cache_dir / f"{self.symbol}_M5.parquet",
            base_dir / self.cache_dir / f"{self.symbol}_M5.pkl",
            base_dir / "backtest_data" / f"{self.symbol}_M5.parquet",
            base_dir / "backtest_data" / f"{self.symbol}_M5.pkl",
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Loading local data from {path}")
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_pickle(path)
                df = self._normalize_df(df)
                return df

        return None

    def _fetch_from_mt5(self) -> Optional[pd.DataFrame]:
        """Fetch M5 data from MT5 using copy_rates_range()."""
        try:
            from utils.mt5_client import get_mt5
            mt5 = get_mt5()

            if not mt5.initialize():
                logger.error("MT5 initialize() failed")
                return None

            # MT5 copy_rates_range expects naive UTC datetimes
            start_naive = self.start_date.replace(tzinfo=None)
            end_naive = self.end_date.replace(tzinfo=None)

            bars = mt5.copy_rates_range(
                self.symbol,
                mt5.TIMEFRAME_M5,
                start_naive,
                end_naive,
            )

            if bars is None or len(bars) == 0:
                logger.warning("MT5 returned no M5 bars")
                return None

            df = pd.DataFrame(bars)
            df = self._normalize_df(df)
            return df

        except ImportError:
            logger.warning("MT5 client not available — cannot fetch live data")
            return None
        except Exception as e:
            logger.warning(f"MT5 fetch failed: {e}")
            return None

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame columns and ensure UTC timestamps.
        Handles both MT5 raw output (epoch seconds) and pre-processed data.
        """
        # Rename columns if needed (MT5 raw uses lowercase already)
        col_map = {
            "Time": "time", "Open": "open", "High": "high",
            "Low": "low", "Close": "close",
            "Volume": "tick_volume", "Tick_volume": "tick_volume",
            "Spread": "spread",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure required columns exist
        required = ["time", "open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add optional columns with defaults
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0
        if "spread" not in df.columns:
            df["spread"] = 0

        # Convert time to UTC datetime
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            # Assume epoch seconds (MT5 raw format)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        elif df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")

        return df


# ─────────────────────────────────────────────────────────────────────────────
# BAR BUFFER — Accumulates M5 into higher timeframes
# ─────────────────────────────────────────────────────────────────────────────


class BarBuffer:
    """
    Accumulates M5 bars and builds higher-timeframe candles (M15, H1, H4, D1).

    Usage:
        buf = BarBuffer()
        for bar in data_feed.iter_m5_bars():
            completed = buf.add_m5(bar)
            # completed = {"M15": bar_or_None, "H1": bar_or_None, ...}

        # Get full series for indicator computation
        h1_df = buf.get_series("H1")
    """

    def __init__(self):
        # Store all M5 bars
        self._m5_bars: list[dict] = []

        # Completed higher-TF bars
        self._completed: dict[str, list[dict]] = {
            "M15": [], "H1": [], "H4": [], "D1": [],
        }

        # Accumulator for current forming bar per timeframe
        self._forming: dict[str, list[dict]] = {
            "M15": [], "H1": [], "H4": [], "D1": [],
        }

        # Track last completed bar time per TF to detect boundaries
        self._last_boundary: dict[str, Optional[datetime]] = {
            "M15": None, "H1": None, "H4": None, "D1": None,
        }

    def add_m5(self, bar: dict) -> dict[str, Optional[dict]]:
        """
        Add an M5 bar and check if any higher-TF bars completed.

        Returns dict mapping timeframe -> completed bar (or None).
        """
        self._m5_bars.append(bar)
        bar_time = bar["time"]

        result: dict[str, Optional[dict]] = {}

        for tf in ("M15", "H1", "H4", "D1"):
            boundary = self._get_tf_boundary(bar_time, tf)

            if self._last_boundary[tf] is not None and boundary != self._last_boundary[tf]:
                # Boundary crossed — the forming bars constitute a completed candle
                if self._forming[tf]:
                    completed_bar = self._aggregate_bars(self._forming[tf], tf)
                    self._completed[tf].append(completed_bar)
                    result[tf] = completed_bar
                else:
                    result[tf] = None
                # Start new forming period
                self._forming[tf] = [bar]
            else:
                self._forming[tf].append(bar)
                result[tf] = None

            self._last_boundary[tf] = boundary

        return result

    def get_series(self, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        """
        Returns a DataFrame of completed bars for the given timeframe.

        Args:
            timeframe: "M5", "M15", "H1", "H4", "D1"
            count: Optional limit on number of most recent bars.

        Returns DataFrame with columns: time, open, high, low, close, tick_volume
        """
        if timeframe == "M5":
            bars = self._m5_bars
        elif timeframe in self._completed:
            bars = self._completed[timeframe]
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        if not bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

        if count is not None:
            bars = bars[-count:]

        df = pd.DataFrame(bars)
        # Ensure standard column names
        for col in ["time", "open", "high", "low", "close", "tick_volume"]:
            if col not in df.columns:
                if col == "tick_volume":
                    df[col] = 0
                else:
                    raise ValueError(f"Missing column {col} in {timeframe} series")
        return df

    def get_last_bar(self, timeframe: str) -> Optional[dict]:
        """Returns the most recently completed bar for the given timeframe."""
        if timeframe == "M5":
            return self._m5_bars[-1] if self._m5_bars else None
        bars = self._completed.get(timeframe, [])
        return bars[-1] if bars else None

    def get_m5_count(self) -> int:
        """Total M5 bars processed."""
        return len(self._m5_bars)

    @staticmethod
    def _get_tf_boundary(dt: datetime, tf: str) -> datetime:
        """
        Returns the period start boundary for a given datetime and timeframe.
        Used to detect when a higher-TF bar completes.
        """
        if tf == "M15":
            return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
        elif tf == "H1":
            return dt.replace(minute=0, second=0, microsecond=0)
        elif tf == "H4":
            return dt.replace(hour=(dt.hour // 4) * 4, minute=0, second=0, microsecond=0)
        elif tf == "D1":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unknown timeframe: {tf}")

    @staticmethod
    def _aggregate_bars(bars: list[dict], tf: str) -> dict:
        """Aggregate a list of M5 bars into a single OHLCV bar."""
        return {
            "time": bars[0]["time"],
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
            "tick_volume": sum(b.get("tick_volume", 0) for b in bars),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL SPREAD FEED
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalSpreadFeed:
    """
    Provides historical spread data for the simulation.

    Data sources (in priority order):
      1. Spread column from M5 bar data (MT5 stores spread per bar)
      2. Session-average fallback from config.SPREAD_FALLBACK_POINTS

    Spread is in POINTS (0.01 for XAUUSD).
    """

    def __init__(self, m5_df: Optional[pd.DataFrame] = None):
        self._spread_cache: dict[str, float] = {}
        self._m5_df = m5_df

        # Pre-compute session averages from M5 data if available
        if m5_df is not None and "spread" in m5_df.columns and m5_df["spread"].sum() > 0:
            self._build_spread_cache(m5_df)

    def _build_spread_cache(self, df: pd.DataFrame) -> None:
        """Pre-compute hourly average spreads from M5 data."""
        df = df.copy()
        if "hour" not in df.columns:
            df["hour"] = df["time"].dt.hour
        # Group by hour and compute mean spread
        hourly = df.groupby("hour")["spread"].mean()
        for hour, spread in hourly.items():
            self._spread_cache[f"hour_{hour}"] = float(spread)

    def get_spread_at(self, timestamp: datetime) -> float:
        """
        Returns spread in points at the given timestamp.

        Falls back to session-average from config if no historical data.
        """
        # Try hourly cache first
        hour_key = f"hour_{timestamp.hour}"
        if hour_key in self._spread_cache:
            return self._spread_cache[hour_key]

        # Fall back to session-based defaults from config
        from utils.session import get_session_for_datetime
        try:
            import config
        except ImportError:
            # Fallback defaults
            return 25.0

        session = get_session_for_datetime(timestamp)

        # Map session names to config keys
        session_map = {
            "ASIAN": "ASIAN",
            "LONDON": "LONDON",
            "LONDON_NY_OVERLAP": "LONDON_NY",
            "NY": "NY",
            "OFF_HOURS": "OFF_HOURS",
        }
        config_key = session_map.get(session, "OFF_HOURS")
        return config.SPREAD_FALLBACK_POINTS.get(config_key, 25.0)

    def get_avg_spread_24h(self) -> float:
        """Returns the overall average spread across all cached hours."""
        if self._spread_cache:
            return sum(self._spread_cache.values()) / len(self._spread_cache)
        return 25.0  # conservative default


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL EVENT FEED
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalEventFeed:
    """
    Provides historical economic event data for event blackout simulation.

    Uses config.HARDCODED_EVENT_PATTERNS to generate approximate event
    timestamps for the backtest period. This is sufficient for Phase 1
    backtesting — exact event times would require a historical calendar DB.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self._events: list[dict] = []
        self._build_event_calendar()

    def _build_event_calendar(self) -> None:
        """
        Generate approximate event timestamps from hardcoded patterns.
        Covers NFP, FOMC, CPI, PPI, Retail Sales.
        """
        try:
            import config
            patterns = config.HARDCODED_EVENT_PATTERNS
        except ImportError:
            patterns = []

        ist_tz = pytz.timezone("Asia/Kolkata")
        current = self.start_date.replace(day=1)

        while current <= self.end_date:
            year, month = current.year, current.month

            for pattern in patterns:
                event_time = self._resolve_pattern(pattern, year, month, ist_tz)
                if event_time and self.start_date <= event_time <= self.end_date:
                    self._events.append({
                        "name": pattern["name"],
                        "time": event_time,
                        "impact": pattern.get("impact", "HIGH"),
                    })

            # Next month
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        self._events.sort(key=lambda e: e["time"])
        logger.info(f"Built event calendar with {len(self._events)} events")

    @staticmethod
    def _resolve_pattern(
        pattern: dict, year: int, month: int, ist_tz
    ) -> Optional[datetime]:
        """Resolve a hardcoded event pattern to a specific datetime."""
        hour_ist = pattern.get("hour_ist", 18)
        minute_ist = pattern.get("minute_ist", 30)

        if "day_of_month" in pattern:
            day = pattern["day_of_month"]
            try:
                dt_ist = ist_tz.localize(
                    datetime(year, month, day, hour_ist, minute_ist)
                )
                return dt_ist.astimezone(pytz.utc)
            except ValueError:
                return None

        if "weekday" in pattern:
            weekday = pattern["weekday"]
            week = pattern.get("week_of_month")

            if week is not None:
                # Find the Nth weekday of the month
                import calendar
                cal = calendar.monthcalendar(year, month)
                count = 0
                for week_days in cal:
                    if week_days[weekday] != 0:
                        count += 1
                        if count == week:
                            day = week_days[weekday]
                            try:
                                dt_ist = ist_tz.localize(
                                    datetime(year, month, day, hour_ist, minute_ist)
                                )
                                return dt_ist.astimezone(pytz.utc)
                            except ValueError:
                                return None
            else:
                # Pattern without specific week — generate for each occurrence
                # (e.g., FOMC approximate — just use 3rd week Wednesday)
                import calendar
                cal = calendar.monthcalendar(year, month)
                count = 0
                for week_days in cal:
                    if week_days[weekday] != 0:
                        count += 1
                        if count == 3:  # default to 3rd occurrence
                            day = week_days[weekday]
                            try:
                                dt_ist = ist_tz.localize(
                                    datetime(year, month, day, hour_ist, minute_ist)
                                )
                                return dt_ist.astimezone(pytz.utc)
                            except ValueError:
                                return None

        return None

    def get_events_near(
        self, timestamp: datetime, window_minutes: int = 45
    ) -> list[dict]:
        """
        Returns list of HIGH-impact events within window_minutes of timestamp.
        Used for event blackout checks.
        """
        window = timedelta(minutes=window_minutes)
        return [
            e for e in self._events
            if abs((e["time"] - timestamp).total_seconds()) <= window.total_seconds()
        ]

    def has_upcoming_event(
        self, timestamp: datetime, pre_minutes: int = 45, post_minutes: int = 20
    ) -> bool:
        """
        Returns True if there's a HIGH-impact event within the blackout window.
        pre_minutes before event OR post_minutes after event.
        """
        for event in self._events:
            delta = (event["time"] - timestamp).total_seconds() / 60.0
            # Event is upcoming (within pre_minutes)
            if 0 < delta <= pre_minutes:
                return True
            # Event just happened (within post_minutes)
            if -post_minutes <= delta <= 0:
                return True
        return False
