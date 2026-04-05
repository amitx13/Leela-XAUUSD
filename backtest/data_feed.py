"""
backtest/data_feed.py — Historical data providers for the backtesting framework.

HistoricalDataFeed:   Fetches M5 bars from MT5 or local cache (parquet/pickle).
BarBuffer:            Accumulates M5 bars into M15, H1, H4, D1 candles.
HistoricalSpreadFeed: Provides historical spread data.
HistoricalEventFeed:  Provides historical economic event data.

  CRIT-1 FIX: HistoricalEventFeed now loads real event timestamps from
  backtest_data/events.csv when that file is present, falling back to
  the approximate pattern-based generator with a clear WARNING so the
  operator always knows which source is active.

  CSV schema (comma or tab separated, header row required):
    datetime_utc,name,impact
    2024-01-26 13:30:00,Non-Farm Payrolls,HIGH

  Run tools/fetch_events_history.py to build the CSV from Forex Factory.

All timestamps are timezone-aware UTC (pytz.utc).
"""
import os
import csv
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

TF_M5_MULTIPLES = {
    "M15": 3,
    "H1":  12,
    "H4":  48,
    "D1":  288,
}

TF_RESAMPLE_RULE = {
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}

# ---------------------------------------------------------------------------
# Default path for the real-events CSV (relative to repo root).
# Override by setting the env-var LEELA_EVENTS_CSV before running.
# ---------------------------------------------------------------------------
_DEFAULT_EVENTS_CSV = Path(__file__).parent.parent / "backtest_data" / "events.csv"


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
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "XAUUSD",
        cache_dir: str = "backtest_data",
    ):
        self.start_date = start_date.replace(tzinfo=pytz.utc) if start_date.tzinfo is None else start_date
        self.end_date   = end_date.replace(tzinfo=pytz.utc)   if end_date.tzinfo is None   else end_date
        self.symbol     = symbol
        self.cache_dir  = Path(cache_dir)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Load M5 data from best available source.
        Returns DataFrame with columns: time, open, high, low, close, tick_volume, spread
        """
        if self._df is not None:
            return self._df

        df = self._try_load_local()
        if df is not None and len(df) > 0:
            logger.info(f"Loaded {len(df)} M5 bars from local cache")
        else:
            df = self._fetch_from_mt5()
            if df is not None and len(df) > 0:
                logger.info(f"Fetched {len(df)} M5 bars from MT5")
            else:
                raise RuntimeError(
                    f"No M5 data available for {self.start_date} to {self.end_date}. "
                    "Run tools/collect_historical_data.py first, or ensure MT5 is connected."
                )

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
        """Yields M5 bars chronologically as dicts."""
        df = self.load()
        for _, row in df.iterrows():
            yield row.to_dict()

    def _try_load_local(self) -> Optional[pd.DataFrame]:
        base_dir = Path(__file__).parent.parent
        search_paths = [
            base_dir / self.cache_dir / f"{self.symbol}_M5.parquet",
            base_dir / self.cache_dir / f"{self.symbol}_M5.pkl",
            base_dir / "backtest_data" / f"{self.symbol}_M5.parquet",
            base_dir / "backtest_data" / f"{self.symbol}_M5.pkl",
        ]
        for path in search_paths:
            if path.exists():
                logger.info(f"Loading local data from {path}")
                df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
                return self._normalize_df(df)
        return None

    def _fetch_from_mt5(self) -> Optional[pd.DataFrame]:
        try:
            from utils.mt5_client import get_mt5
            mt5 = get_mt5()
            if not mt5.initialize():
                logger.error("MT5 initialize() failed")
                return None
            start_naive = self.start_date.replace(tzinfo=None)
            end_naive   = self.end_date.replace(tzinfo=None)
            bars = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M5, start_naive, end_naive)
            if bars is None or len(bars) == 0:
                logger.warning("MT5 returned no M5 bars")
                return None
            return self._normalize_df(pd.DataFrame(bars))
        except ImportError:
            logger.warning("MT5 client not available — cannot fetch live data")
            return None
        except Exception as e:
            logger.warning(f"MT5 fetch failed: {e}")
            return None

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "Time": "time", "Open": "open", "High": "high",
            "Low": "low", "Close": "close",
            "Volume": "tick_volume", "Tick_volume": "tick_volume",
            "Spread": "spread",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for col in ["time", "open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0
        if "spread" not in df.columns:
            df["spread"] = 0
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
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
    """

    def __init__(self):
        self._m5_bars: list[dict] = []
        self._completed: dict[str, list[dict]] = {"M15": [], "H1": [], "H4": [], "D1": []}
        self._forming:   dict[str, list[dict]] = {"M15": [], "H1": [], "H4": [], "D1": []}
        self._last_boundary: dict[str, Optional[datetime]] = {
            "M15": None, "H1": None, "H4": None, "D1": None,
        }

    def add_m5(self, bar: dict) -> dict[str, Optional[dict]]:
        """Add an M5 bar and return any newly completed higher-TF bars."""
        self._m5_bars.append(bar)
        bar_time = bar["time"]
        result: dict[str, Optional[dict]] = {}

        for tf in ("M15", "H1", "H4", "D1"):
            boundary = self._get_tf_boundary(bar_time, tf)
            if self._last_boundary[tf] is not None and boundary != self._last_boundary[tf]:
                if self._forming[tf]:
                    completed_bar = self._aggregate_bars(self._forming[tf], tf)
                    self._completed[tf].append(completed_bar)
                    result[tf] = completed_bar
                else:
                    result[tf] = None
                self._forming[tf] = [bar]
            else:
                self._forming[tf].append(bar)
                result[tf] = None
            self._last_boundary[tf] = boundary

        return result

    def get_series(self, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        """Returns a DataFrame of completed bars for the given timeframe."""
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
        for col in ["time", "open", "high", "low", "close", "tick_volume"]:
            if col not in df.columns:
                df[col] = 0 if col == "tick_volume" else (_ for _ in ()).throw(ValueError(f"Missing column {col}"))
        return df

    def get_last_bar(self, timeframe: str) -> Optional[dict]:
        if timeframe == "M5":
            return self._m5_bars[-1] if self._m5_bars else None
        bars = self._completed.get(timeframe, [])
        return bars[-1] if bars else None

    def get_m5_count(self) -> int:
        return len(self._m5_bars)

    @staticmethod
    def _get_tf_boundary(dt: datetime, tf: str) -> datetime:
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
        return {
            "time":        bars[0]["time"],
            "open":        bars[0]["open"],
            "high":        max(b["high"] for b in bars),
            "low":         min(b["low"]  for b in bars),
            "close":       bars[-1]["close"],
            "tick_volume": sum(b.get("tick_volume", 0) for b in bars),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL SPREAD FEED
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalSpreadFeed:
    """
    Provides historical spread data for the simulation.
    Spread is in POINTS (0.01 for XAUUSD).
    """

    def __init__(self, m5_df: Optional[pd.DataFrame] = None):
        self._spread_cache: dict[str, float] = {}
        self._m5_df = m5_df
        if m5_df is not None and "spread" in m5_df.columns and m5_df["spread"].sum() > 0:
            self._build_spread_cache(m5_df)

    def _build_spread_cache(self, df: pd.DataFrame) -> None:
        df = df.copy()
        if "hour" not in df.columns:
            df["hour"] = df["time"].dt.hour
        hourly = df.groupby("hour")["spread"].mean()
        for hour, spread in hourly.items():
            self._spread_cache[f"hour_{hour}"] = float(spread)

    def get_spread_at(self, timestamp: datetime) -> float:
        hour_key = f"hour_{timestamp.hour}"
        if hour_key in self._spread_cache:
            return self._spread_cache[hour_key]
        from utils.session import get_session_for_datetime
        try:
            import config
        except ImportError:
            return 25.0
        session = get_session_for_datetime(timestamp)
        session_map = {
            "ASIAN": "ASIAN", "LONDON": "LONDON",
            "LONDON_NY_OVERLAP": "LONDON_NY", "NY": "NY", "OFF_HOURS": "OFF_HOURS",
        }
        config_key = session_map.get(session, "OFF_HOURS")
        return config.SPREAD_FALLBACK_POINTS.get(config_key, 25.0)

    def get_avg_spread_24h(self) -> float:
        if self._spread_cache:
            return sum(self._spread_cache.values()) / len(self._spread_cache)
        return 25.0


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL EVENT FEED
# ─────────────────────────────────────────────────────────────────────────────


class HistoricalEventFeed:
    """
    Provides historical economic event data for event blackout simulation.

    CRIT-1 FIX — Real event timestamps vs. approximate pattern generator
    ─────────────────────────────────────────────────────────────────────
    Source priority:
      1. Real CSV  — backtest_data/events.csv (or LEELA_EVENTS_CSV env-var path)
                     Built by tools/fetch_events_history.py from Forex Factory.
                     When this file is present, ONLY its rows are used — the
                     pattern generator is skipped entirely.
      2. Patterns  — config.HARDCODED_EVENT_PATTERNS generates approximate
                     timestamps based on typical release schedules.
                     A WARNING is emitted so the operator knows data is
                     approximate. This is the fallback for historical periods
                     not yet covered by the CSV.

    CSV schema (comma or tab-separated, header required):
        datetime_utc,name,impact
        2024-01-26 13:30:00,Non-Farm Payrolls,HIGH
        2024-01-26 19:00:00,FOMC Statement,HIGH

    datetime_utc must be parseable by pd.to_datetime. Timezone-naive values
    are treated as UTC. Only rows where impact == "HIGH" (case-insensitive)
    are included in the blackout window checks.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date   = end_date
        self._events: list[dict] = []
        self._source: str = "UNKNOWN"   # "CSV" or "PATTERN" — logged at build time
        self._build_event_calendar()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_events_near(
        self, timestamp: datetime, window_minutes: int = 45
    ) -> list[dict]:
        """Returns list of HIGH-impact events within window_minutes of timestamp."""
        window = timedelta(minutes=window_minutes)
        return [
            e for e in self._events
            if abs((e["time"] - timestamp).total_seconds()) <= window.total_seconds()
        ]

    def has_upcoming_event(
        self, timestamp: datetime, pre_minutes: int = 45, post_minutes: int = 20
    ) -> bool:
        """
        Returns True if there is a HIGH-impact event within the blackout window.
        pre_minutes before OR post_minutes after the event.
        """
        for event in self._events:
            delta = (event["time"] - timestamp).total_seconds() / 60.0
            if 0 < delta <= pre_minutes:
                return True
            if -post_minutes <= delta <= 0:
                return True
        return False

    def had_recent_high_impact_event(
        self, timestamp: datetime, lookback_minutes: int = 30
    ) -> bool:
        """
        Returns True if a HIGH-impact event occurred within the last
        lookback_minutes before timestamp.

        Called by _evaluate_r3() in engine.py to detect the post-event
        momentum window for R3 (Calendar Momentum) trades.
        """
        for event in self._events:
            if event.get("impact", "HIGH") != "HIGH":
                continue
            seconds_ago = (timestamp - event["time"]).total_seconds()
            if 0 < seconds_ago <= lookback_minutes * 60:
                return True
        return False

    @property
    def source(self) -> str:
        """Returns 'CSV' if real data was loaded, 'PATTERN' if approximate."""
        return self._source

    # ── Internal builders ─────────────────────────────────────────────────────

    def _build_event_calendar(self) -> None:
        """
        Build self._events from the best available source.
        Sets self._source to 'CSV' or 'PATTERN' accordingly.
        """
        csv_path = Path(os.environ.get("LEELA_EVENTS_CSV", str(_DEFAULT_EVENTS_CSV)))

        if csv_path.exists():
            loaded = self._load_from_csv(csv_path)
            if loaded is not None:
                self._events = loaded
                self._source = "CSV"
                logger.info(
                    f"[CRIT-1 FIXED] Event feed loaded from REAL CSV: {csv_path} "
                    f"— {len(self._events)} HIGH-impact events in window "
                    f"({self.start_date.date()} → {self.end_date.date()})"
                )
                return
            else:
                logger.warning(
                    f"events.csv found at {csv_path} but failed to parse — "
                    "falling back to pattern generator."
                )

        # ── Fallback: approximate pattern-based generator ──────────────────
        logger.warning(
            "[CRIT-1 OPEN] HistoricalEventFeed is using the APPROXIMATE pattern "
            "generator — event blackouts will not match real release times exactly. "
            "Run tools/fetch_events_history.py to build backtest_data/events.csv "
            "and eliminate this gap."
        )
        self._source = "PATTERN"
        self._events = self._build_from_patterns()
        logger.info(
            f"Pattern-based event calendar: {len(self._events)} events "
            f"({self.start_date.date()} → {self.end_date.date()})"
        )

    def _load_from_csv(self, path: Path) -> Optional[list[dict]]:
        """
        Load events from a real CSV file.

        CSV must have columns: datetime_utc, name, impact
        Timezone-naive datetimes are assumed UTC.
        Only rows with impact == 'HIGH' (case-insensitive) are kept.
        Returns None if parsing fails.
        """
        try:
            # Auto-detect separator (comma or tab)
            with open(path, "r", encoding="utf-8") as f:
                sample = f.read(2048)
            sep = "\t" if sample.count("\t") > sample.count(",") else ","

            df = pd.read_csv(path, sep=sep, dtype=str)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Accept 'datetime_utc' or 'time' or 'date' as timestamp column
            ts_col = None
            for candidate in ("datetime_utc", "time", "datetime", "date"):
                if candidate in df.columns:
                    ts_col = candidate
                    break
            if ts_col is None:
                logger.error(f"events.csv has no recognised timestamp column. Columns: {df.columns.tolist()}")
                return None

            impact_col = "impact" if "impact" in df.columns else None
            name_col   = "name"   if "name"   in df.columns else None

            df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=["_ts"])

            # Filter to HIGH impact only
            if impact_col:
                df = df[df[impact_col].str.strip().str.upper() == "HIGH"]

            # Filter to backtest window
            start_utc = self.start_date
            end_utc   = self.end_date
            if start_utc.tzinfo is None:
                start_utc = pytz.utc.localize(start_utc)
            if end_utc.tzinfo is None:
                end_utc = pytz.utc.localize(end_utc)

            df = df[(df["_ts"] >= start_utc) & (df["_ts"] <= end_utc)]

            events = []
            for _, row in df.iterrows():
                events.append({
                    "name":   str(row[name_col]) if name_col else "HIGH_IMPACT",
                    "time":   row["_ts"].to_pydatetime(),
                    "impact": "HIGH",
                })

            events.sort(key=lambda e: e["time"])
            return events

        except Exception as exc:
            logger.error(f"Failed to parse events.csv ({path}): {exc}")
            return None

    def _build_from_patterns(self) -> list[dict]:
        """Approximate event calendar from config.HARDCODED_EVENT_PATTERNS."""
        try:
            import config
            patterns = config.HARDCODED_EVENT_PATTERNS
        except ImportError:
            patterns = []

        ist_tz  = pytz.timezone("Asia/Kolkata")
        current = self.start_date.replace(day=1)
        events: list[dict] = []

        while current <= self.end_date:
            year, month = current.year, current.month
            for pattern in patterns:
                event_time = self._resolve_pattern(pattern, year, month, ist_tz)
                if event_time and self.start_date <= event_time <= self.end_date:
                    events.append({
                        "name":   pattern["name"],
                        "time":   event_time,
                        "impact": pattern.get("impact", "HIGH"),
                    })
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        events.sort(key=lambda e: e["time"])
        return events

    @staticmethod
    def _resolve_pattern(
        pattern: dict, year: int, month: int, ist_tz
    ) -> Optional[datetime]:
        hour_ist   = pattern.get("hour_ist", 18)
        minute_ist = pattern.get("minute_ist", 30)

        if "day_of_month" in pattern:
            day = pattern["day_of_month"]
            try:
                dt_ist = ist_tz.localize(datetime(year, month, day, hour_ist, minute_ist))
                return dt_ist.astimezone(pytz.utc)
            except ValueError:
                return None

        if "weekday" in pattern:
            weekday = pattern["weekday"]
            week    = pattern.get("week_of_month")
            import calendar
            cal   = calendar.monthcalendar(year, month)
            count = 0
            target_week = week if week is not None else 3
            for week_days in cal:
                if week_days[weekday] != 0:
                    count += 1
                    if count == target_week:
                        day = week_days[weekday]
                        try:
                            dt_ist = ist_tz.localize(
                                datetime(year, month, day, hour_ist, minute_ist)
                            )
                            return dt_ist.astimezone(pytz.utc)
                        except ValueError:
                            return None
        return None
