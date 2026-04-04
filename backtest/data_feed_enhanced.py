"""
Enhanced Data Feed

Multi-timeframe support with event simulation and spread modeling.
Fetches real data from MT5 or local cache (parquet/pickle).
Falls back to MT5 copy_rates_range() — requires live MT5 connection.
NEVER generates synthetic/random bars.
"""

import os
import logging
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("backtest.enhanced_data_feed")

# MT5 timeframe string -> MT5 constant name
MT5_TF_MAP = {
    "M5":  "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1":  "TIMEFRAME_H1",
    "H4":  "TIMEFRAME_H4",
    "D1":  "TIMEFRAME_D1",
}


class EnhancedHistoricalDataFeed:
    """
    Enhanced historical data feed with multi-timeframe support.

    Data source priority (same as HistoricalDataFeed in data_feed.py):
      1. Local parquet  — backtest_data/XAUUSD_{TF}.parquet
      2. Local pickle   — backtest_data/XAUUSD_{TF}.pkl
      3. MT5 copy_rates_range() — requires live MT5 connection

    NEVER generates synthetic/random data.
    """

    def __init__(self, cache_dir: str = "backtest_data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data: Dict[str, pd.DataFrame] = {}   # timeframe -> DataFrame
        self.events: List[dict] = []
        self.spreads: Dict[str, pd.DataFrame] = {}

        logger.info(f"Enhanced data feed initialised, cache dir: {self.cache_dir}")

    # ------------------------------------------------------------------ #
    # PUBLIC                                                               #
    # ------------------------------------------------------------------ #

    def load_data(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframes: List[str] = None,
        symbol: str = "XAUUSD",
    ) -> None:
        """Load historical data for specified timeframes from cache or MT5."""
        if timeframes is None:
            timeframes = ["M5", "M15", "H1", "H4", "D1"]

        logger.info(f"Loading {timeframes} for {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")

        for tf in timeframes:
            self._load_timeframe(tf, start_date, end_date, symbol)

        # Build event calendar from config patterns (same as HistoricalEventFeed)
        self._build_event_calendar(start_date, end_date)

        # Build spread cache from M5 spread column (if available)
        if "M5" in self.data:
            self._build_spread_cache(self.data["M5"])

        logger.info("Data loading complete")

    def get_bars(
        self,
        timeframe: str,
        current_time: datetime,
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return up to `count` completed bars for `timeframe` up to current_time."""
        if timeframe not in self.data or self.data[timeframe].empty:
            return []

        df = self.data[timeframe]
        df_filtered = df[df["time"] <= current_time].tail(count)

        return df_filtered.to_dict("records")

    def get_upcoming_events(
        self,
        current_time: datetime,
        pre_minutes: int = 45,
        post_minutes: int = 20,
    ) -> List[dict]:
        """Return HIGH-impact events within the blackout window."""
        result = []
        for ev in self.events:
            delta = (ev["time"] - current_time).total_seconds() / 60.0
            if -post_minutes <= delta <= pre_minutes:
                result.append(ev)
        return result

    def get_current_spread(self, current_time: datetime) -> float:
        """Return spread in points for current_time."""
        hour_key = f"hour_{current_time.hour}"
        if hour_key in self.spreads:
            return self.spreads[hour_key]

        # Session-based fallback
        h = current_time.hour
        if 7 <= h < 12:
            return 18.0   # London open
        elif 12 <= h < 17:
            return 12.0   # London/NY overlap
        elif 17 <= h < 22:
            return 20.0   # NY
        else:
            return 30.0   # Off-hours

    # ------------------------------------------------------------------ #
    # PRIVATE — DATA LOADING                                              #
    # ------------------------------------------------------------------ #

    def _load_timeframe(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "XAUUSD",
    ) -> None:
        """
        Load one timeframe from best available source:
          1. parquet cache
          2. pickle cache
          3. MT5 live fetch
        Raises RuntimeError if no data is available.
        """
        # 1 & 2 — local cache
        for suffix, loader in [
            (".parquet", pd.read_parquet),
            (".pkl",     pd.read_pickle),
        ]:
            for candidate in [
                self.cache_dir / f"{symbol}_{timeframe}{suffix}",
                self.cache_dir / f"{symbol.lower()}_{timeframe.lower()}{suffix}",
                Path("backtest_data") / f"{symbol}_{timeframe}{suffix}",
            ]:
                if candidate.exists():
                    try:
                        df = loader(str(candidate))
                        df = self._normalize_df(df)
                        df = df[
                            (df["time"] >= start_date) &
                            (df["time"] <= end_date)
                        ].copy()
                        df.sort_values("time", inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        self.data[timeframe] = df
                        logger.info(
                            f"{timeframe}: loaded {len(df)} bars "
                            f"from {candidate.name}"
                        )
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load {candidate}: {e}")

        # 3 — MT5 live fetch
        df = self._fetch_from_mt5(timeframe, start_date, end_date, symbol)
        if df is not None and len(df) > 0:
            self.data[timeframe] = df
            logger.info(f"{timeframe}: fetched {len(df)} bars from MT5")
            return

        raise RuntimeError(
            f"No {timeframe} data available for {symbol} "
            f"({start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}). "
            "Run tools/collect_historical_data.py or ensure MT5 is connected."
        )

    def _fetch_from_mt5(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "XAUUSD",
    ) -> Optional[pd.DataFrame]:
        """Fetch bars from MT5 using copy_rates_range()."""
        try:
            from utils.mt5_client import get_mt5
            mt5 = get_mt5()
            if not mt5.initialize():
                logger.error("MT5 initialize() failed")
                return None

            tf_const_name = MT5_TF_MAP.get(timeframe)
            if tf_const_name is None:
                logger.error(f"Unknown timeframe: {timeframe}")
                return None

            tf_const = getattr(mt5, tf_const_name, None)
            if tf_const is None:
                logger.error(f"MT5 has no attribute {tf_const_name}")
                return None

            # MT5 expects naive UTC datetimes
            start_naive = start_date.replace(tzinfo=None)
            end_naive   = end_date.replace(tzinfo=None)

            bars = mt5.copy_rates_range(symbol, tf_const, start_naive, end_naive)

            if bars is None or len(bars) == 0:
                logger.warning(f"MT5 returned no bars for {timeframe}")
                return None

            df = pd.DataFrame(bars)
            df = self._normalize_df(df)
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        except ImportError:
            logger.warning("MT5 client not available — cannot fetch live data")
            return None
        except Exception as e:
            logger.warning(f"MT5 fetch failed for {timeframe}: {e}")
            return None

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise column names and ensure UTC-aware timestamps.
        Handles MT5 raw output (epoch seconds) and pre-processed CSVs.
        """
        col_map = {
            "Time": "time", "Open": "open", "High": "high",
            "Low":  "low",  "Close": "close",
            "Volume": "tick_volume", "Tick_volume": "tick_volume",
            "Spread": "spread",
            # CSV exports sometimes use these names
            "vol": "tick_volume", "volume": "tick_volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ["time", "open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0
        if "spread" not in df.columns:
            df["spread"] = 0

        # Convert time to UTC-aware datetime
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        elif df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")

        return df

    # ------------------------------------------------------------------ #
    # PRIVATE — EVENT CALENDAR                                            #
    # ------------------------------------------------------------------ #

    def _build_event_calendar(
        self, start_date: datetime, end_date: datetime
    ) -> None:
        """
        Generate approximate event timestamps from config.HARDCODED_EVENT_PATTERNS.
        Mirrors HistoricalEventFeed._build_event_calendar() exactly.
        """
        import calendar as cal_mod

        try:
            import config
            patterns = config.HARDCODED_EVENT_PATTERNS
        except (ImportError, AttributeError):
            patterns = []

        ist_tz = pytz.timezone("Asia/Kolkata")
        current = start_date.replace(day=1)
        events: List[dict] = []

        while current <= end_date:
            year, month = current.year, current.month
            for pattern in patterns:
                ev_time = self._resolve_event_pattern(
                    pattern, year, month, ist_tz, cal_mod
                )
                if ev_time and start_date <= ev_time <= end_date:
                    events.append({
                        "name":   pattern["name"],
                        "time":   ev_time,
                        "impact": pattern.get("impact", "HIGH"),
                    })
            if month == 12:
                current = current.replace(year=year + 1, month=1)
            else:
                current = current.replace(month=month + 1)

        events.sort(key=lambda e: e["time"])
        self.events = events
        logger.info(f"Event calendar built: {len(events)} events")

    @staticmethod
    def _resolve_event_pattern(
        pattern: dict, year: int, month: int, ist_tz, cal_mod
    ) -> Optional[datetime]:
        """Resolve a hardcoded event pattern to a UTC datetime."""
        hour_ist   = pattern.get("hour_ist", 18)
        minute_ist = pattern.get("minute_ist", 30)

        if "day_of_month" in pattern:
            try:
                dt_ist = ist_tz.localize(
                    datetime(year, month, pattern["day_of_month"], hour_ist, minute_ist)
                )
                return dt_ist.astimezone(pytz.utc)
            except ValueError:
                return None

        if "weekday" in pattern:
            weekday = pattern["weekday"]
            week    = pattern.get("week_of_month", 3)
            count   = 0
            for week_days in cal_mod.monthcalendar(year, month):
                if week_days[weekday] != 0:
                    count += 1
                    if count == week:
                        try:
                            dt_ist = ist_tz.localize(
                                datetime(
                                    year, month, week_days[weekday],
                                    hour_ist, minute_ist,
                                )
                            )
                            return dt_ist.astimezone(pytz.utc)
                        except ValueError:
                            return None

        return None

    # ------------------------------------------------------------------ #
    # PRIVATE — SPREAD CACHE                                              #
    # ------------------------------------------------------------------ #

    def _build_spread_cache(self, m5_df: pd.DataFrame) -> None:
        """
        Pre-compute hourly average spreads from M5 spread column.
        Falls back to session defaults if no spread data.
        """
        if "spread" not in m5_df.columns or m5_df["spread"].sum() == 0:
            return
        df = m5_df.copy()
        df["hour"] = df["time"].dt.hour
        hourly = df.groupby("hour")["spread"].mean()
        for hour, spread in hourly.items():
            self.spreads[f"hour_{hour}"] = float(spread)
        logger.info("Spread cache built from M5 data")
