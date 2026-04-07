"""
backtest/data_feed.py — Updated Data Feed for Backtesting

Updated to match current live data engine (engines/data_engine.py):
- MT5 OHLCV data fetching with proper timeframe constants
- Economic calendar integration (CSV-only; hardcoded fallback removed)
- Spread tracking with session-aware 24-hour baseline calculation
- DXY correlation analysis with robust CSV parsing
- TLT/TIP macro proxy data integration
- Contract specification fetching from MT5 metadata
- Session-aware data processing

Matches live data engine exactly for accurate backtesting.
"""
import os
import csv
import pytz
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List

logger = logging.getLogger("backtest.data_feed")

# ─────────────────────────────────────────────────────────────────────────────
# TIMEFRAME CONSTANTS - Match MT5 constants
# ─────────────────────────────────────────────────────────────────────────────

TF_M5_MULTIPLES = {
    "M15": 3,
    "H1": 12,
    "H4": 48,
    "D1": 288,
}

TF_RESAMPLE_RULE = {
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

# ---------------------------------------------------------------------------
# Default paths for data files
# ---------------------------------------------------------------------------
_DEFAULT_EVENTS_CSV = Path(__file__).parent.parent / "backtest_data" / "events_calendar.csv"
_DEFAULT_SPREAD_CSV = Path(__file__).parent.parent / "backtest_data" / "spreads_XAUUSD_M5.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT SPECIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_contract_spec(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    Get contract specifications from MT5 metadata.
    Matches live data engine exactly.
    """
    try:
        from utils.mt5_client import get_mt5
        mt5 = get_mt5()
        if not mt5.initialize():
            logger.error("MT5 initialize() failed for contract spec")
            return _get_fallback_contract_spec(symbol)
        
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"symbol_info({symbol}) returned None")
            return _get_fallback_contract_spec(symbol)
        
        spec = {
            "symbol": symbol,
            "point": info.point,
            "tick_size": info.trade_tick_size,
            "tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "contract_size": info.trade_contract_size,
            "digits": info.digits,
            "currency_profit": info.currency_profit,
        }
        
        logger.info(f"Contract spec loaded: {symbol} tick_value={spec['tick_value']} volume_min={spec['volume_min']}")
        return spec
        
    except Exception as e:
        logger.warning(f"Failed to get contract spec from MT5: {e}, using fallback")
        return _get_fallback_contract_spec(symbol)


def _get_fallback_contract_spec(symbol: str) -> Dict[str, Any]:
    """Fallback contract specifications for XAUUSD."""
    return {
        "symbol": symbol,
        "point": 0.01,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "volume_min": 0.01,
        "volume_max": 50.0,
        "volume_step": 0.01,
        "contract_size": 100.0,
        "digits": 2,
        "currency_profit": "USD",
    }

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA FEED
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalDataFeed:
    """
    Updated historical data feed matching live data engine.
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
        self.contract_spec = get_contract_spec(symbol)

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
        """Try to load from local cache with priority order."""
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
        """Fetch data from MT5 matching live data engine."""
        try:
            from utils.mt5_client import get_mt5
            mt5 = get_mt5()
            if not mt5.initialize():
                logger.error("MT5 initialize() failed")
                return None
            
            start_naive = self.start_date.replace(tzinfo=None)
            end_naive   = self.end_date.replace(tzinfo=None)
            
            tf_m5 = mt5.TIMEFRAME_M5
            bars = mt5.copy_rates_range(self.symbol, tf_m5, start_naive, end_naive)
            
            if bars is None or len(bars) == 0:
                logger.error("No bars returned from MT5")
                return None

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
            return self._normalize_df(df)
            
        except Exception as e:
            logger.error(f"MT5 fetch failed: {e}")
            return None

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to standard format."""
        if df.empty:
            return df
        
        required_cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 2.0 if col == "spread" else 0.0
        
        df = df.sort_values("time").reset_index(drop=True)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SPREAD FEED — session-aware synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalSpreadFeed:
    """
    Spread feed with session-aware synthetic generation.
    Loads from backtest_data/spreads_XAUUSD_M5.csv when present;
    otherwise generates realistic XAUUSD spreads by session.

    XAUUSD real-world spread profile:
      Asian (22-07 UTC):           2.5 – 5.0 pts  (low liquidity)
      London open (07:00-07:15):   1.3 – 2.0 pts  (tight at open)
      London core (07:15-13:00):   1.5 – 2.5 pts  (most liquid)
      NY overlap (13:00-17:00):    1.3 – 2.2 pts  (tightest)
      NY wind-down (17:00-20:00):  2.0 – 3.5 pts
      End of day (20:00-22:00):    2.5 – 5.0 pts
      Weekend:                     3.5 – 8.0 pts
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date.replace(tzinfo=pytz.utc)
        self.end_date   = end_date.replace(tzinfo=pytz.utc)
        self._spread_data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load spread data from CSV or generate session-aware synthetic data."""
        csv_path = _DEFAULT_SPREAD_CSV
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['time'])
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df = df[(df['time'] >= self.start_date) & (df['time'] <= self.end_date)]
            logger.info(f"Loaded spread data from CSV: {len(df)} records")
            self._spread_data = df
            return df
        
        logger.info("No spread CSV found — generating session-aware synthetic spreads")
        df = self._generate_synthetic_spread()
        self._spread_data = df
        return df

    def _generate_synthetic_spread(self) -> pd.DataFrame:
        """
        Generate session-aware XAUUSD spread data.
        Uses the same session profile as tools/generate_spread_csv.py.
        """
        np.random.seed(42)
        date_range = pd.date_range(self.start_date, self.end_date, freq='5min', tz='UTC')
        rows = []

        for t in date_range:
            h, m, wd = t.hour, t.minute, t.weekday()

            if wd >= 5:                           # Weekend
                spread = round(np.random.uniform(3.5, 8.0), 1)
            elif 22 <= h or h < 2:               # Late NY / early Asian
                spread = max(1.0, round(np.random.normal(3.5, 0.8), 1))
            elif 2 <= h < 6:                      # Asian core
                spread = max(1.0, round(np.random.normal(3.0, 0.7), 1))
            elif 6 <= h < 7:                      # Pre-London
                spread = max(1.0, round(np.random.normal(2.5, 0.5), 1))
            elif h == 7 and m < 15:              # London open spike
                spread = max(1.0, round(np.random.normal(1.6, 0.5), 1))
            elif 7 <= h < 9:                      # London early
                spread = max(1.0, round(np.random.normal(1.8, 0.4), 1))
            elif 9 <= h < 13:                     # London core
                spread = max(1.0, round(np.random.normal(1.7, 0.35), 1))
            elif 13 <= h < 14:                    # NY open
                spread = max(1.0, round(np.random.normal(1.5, 0.4), 1))
            elif 14 <= h < 17:                    # NY active
                spread = max(1.0, round(np.random.normal(1.8, 0.4), 1))
            elif 17 <= h < 19:                    # NY wind-down
                spread = max(1.0, round(np.random.normal(2.5, 0.6), 1))
            elif 19 <= h < 21:                    # End of day
                spread = max(1.0, round(np.random.normal(3.2, 0.7), 1))
            else:                                 # Late NY
                spread = max(1.0, round(np.random.normal(3.8, 0.8), 1))

            rows.append({'time': t, 'spread': spread})

        return pd.DataFrame(rows)

    def get_24h_median_spread(self, current_time: datetime) -> float:
        """Calculate 24-hour rolling median spread."""
        if self._spread_data is None:
            self._spread_data = self.load()
        
        cutoff_time = current_time - timedelta(hours=24)
        recent_spreads = self._spread_data[
            self._spread_data['time'] >= cutoff_time
        ]['spread']
        
        return float(recent_spreads.median()) if len(recent_spreads) > 0 else 2.0


# ─────────────────────────────────────────────────────────────────────────────
# ECONOMIC EVENT FEED — CSV-only; hardcoded fallback removed
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalEventFeed:
    """
    Economic event feed.

    CHANGE 1 — Hardcoded event generation has been removed entirely.
    When no events_calendar.csv exists:
      - load() returns [] and sets loaded_from_csv = False
      - get_upcoming_events() returns [] immediately
      - engine._ks7_enabled stays False → KS7 never fires
      - algo runs through all bars including news bars unfiltered

    To re-enable KS7, place a real events_calendar.csv in backtest_data/
    with columns: datetime_utc (or time), name, impact_level.
    """

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date.replace(tzinfo=pytz.utc)
        self.end_date   = end_date.replace(tzinfo=pytz.utc)
        self._events: Optional[List[Dict[str, Any]]] = None
        self.loaded_from_csv: bool = False  # engine checks this to gate KS7

    def load(self) -> List[Dict[str, Any]]:
        """
        Load events from CSV only.
        Returns [] when no CSV found — KS7 stays permanently disabled.
        """
        csv_path = _DEFAULT_EVENTS_CSV
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            time_col = 'datetime_utc' if 'datetime_utc' in df.columns else 'time'
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df[
                (df[time_col] >= self.start_date) &
                (df[time_col] <= self.end_date)
            ]
            events = []
            for rec in df.to_dict('records'):
                rec['time'] = rec.pop(time_col, rec.get('time'))
                events.append(rec)
            logger.info(f"Loaded {len(events)} events from CSV — KS7 enabled")
            self._events = events
            self.loaded_from_csv = True
            return events

        # No CSV — event system disabled
        logger.info(
            "No events CSV found at backtest_data/events_calendar.csv — "
            "event blackout (KS7) permanently disabled. "
            "Algo will trade through all news bars."
        )
        self.loaded_from_csv = False
        self._events = []
        return []

    def get_upcoming_events(
        self,
        current_time: datetime,
        minutes_ahead: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Returns upcoming events.
        Returns [] immediately when no CSV was loaded (KS7 never fires).
        """
        if self._events is None:
            self.load()

        # Fast-path: no CSV → no events ever
        if not self.loaded_from_csv:
            return []

        cutoff_time = current_time + timedelta(minutes=minutes_ahead)
        upcoming = [
            e for e in self._events
            if current_time <= e['time'] <= cutoff_time
        ]
        upcoming.sort(key=lambda x: x['time'])
        return upcoming


# ─────────────────────────────────────────────────────────────────────────────
# BAR BUFFER — multi-timeframe support
# ─────────────────────────────────────────────────────────────────────────────

class BarBuffer:
    """
    Updated bar buffer matching live market context.
    Accumulates M5 bars into M15, H1, H4, D1 candles.
    """

    def __init__(self):
        self.m5_bars: List[Dict[str, Any]] = []
        self.m15_bars: List[Dict[str, Any]] = []
        self.h1_bars: List[Dict[str, Any]] = []
        self.h4_bars: List[Dict[str, Any]] = []
        self.d1_bars: List[Dict[str, Any]] = []
        # BUG-11 FIX: persistent counter so modulo is correct even after buffer trim
        self._total_m5_count: int = 0

    def add_m5_bar(self, bar: Dict[str, Any]) -> None:
        """Add M5 bar and update higher timeframes."""
        self._total_m5_count += 1
        self.m5_bars.append(bar)

        max_m5 = 1000
        if len(self.m5_bars) > max_m5:
            self.m5_bars = self.m5_bars[-max_m5:]

        self._update_timeframes()

    def _update_timeframes(self) -> None:
        """Update M15, H1, H4, D1 from M5 data."""
        n = self._total_m5_count
        if n < 3:
            return

        if n % 3 == 0 and len(self.m5_bars) >= 3:
            self.m15_bars.append(self._create_ohlcv_from_m5(self.m5_bars[-3:]))

        if n % 12 == 0 and len(self.m5_bars) >= 12:
            self.h1_bars.append(self._create_ohlcv_from_m5(self.m5_bars[-12:]))

        if n % 48 == 0 and len(self.m5_bars) >= 48:
            self.h4_bars.append(self._create_ohlcv_from_m5(self.m5_bars[-48:]))

        if n % 288 == 0 and len(self.m5_bars) >= 288:
            self.d1_bars.append(self._create_ohlcv_from_m5(self.m5_bars[-288:]))

    def _create_ohlcv_from_m5(self, m5_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create OHLCV bar from M5 bars."""
        if not m5_bars:
            return {}
        return {
            'time':        m5_bars[-1]['time'],
            'open':        m5_bars[0]['open'],
            'high':        max(b['high'] for b in m5_bars),
            'low':         min(b['low']  for b in m5_bars),
            'close':       m5_bars[-1]['close'],
            'tick_volume': sum(b.get('tick_volume', 0) for b in m5_bars),
            'spread':      m5_bars[-1].get('spread', 0),
        }

    def get_latest_bars(self, timeframe: str, count: int = 1) -> List[Dict[str, Any]]:
        """Get latest bars for specified timeframe."""
        mapping = {
            'M5':  self.m5_bars,
            'M15': self.m15_bars,
            'H1':  self.h1_bars,
            'H4':  self.h4_bars,
            'D1':  self.d1_bars,
        }
        bars = mapping.get(timeframe, [])
        return bars[-count:] if count <= len(bars) else bars

    def get_dataframe(self, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        """Get DataFrame for specified timeframe."""
        attr = f"{timeframe.lower()}_bars"
        all_bars = getattr(self, attr, [])
        bars = self.get_latest_bars(timeframe, count or len(all_bars))
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# DXY FEED — robust CSV parsing (CHANGE 3: fixes KeyError: 'time')
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalDXYFeed:
    """
    Historical feed for DXY daily change to enable SUPER_TRENDING regime.

    CHANGE 3 — Rewritten load() to handle real dxy_daily.csv format:
      - Columns: Date,Open,High,Low,Close  (capital 'Date', quoted values)
      - Normalises all column names to lowercase
      - Auto-detects date column: 'date', 'time', 'datetime', 'datetime_utc'
      - Strips quotes and parses MM/DD/YYYY or ISO format automatically
      - Fixes KeyError: 'time' that crashed engine at bar 1
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        cache_dir: str = "backtest_data"
    ):
        self.start_date = start_date.replace(tzinfo=pytz.utc) if start_date.tzinfo is None else start_date
        self.end_date   = end_date.replace(tzinfo=pytz.utc)   if end_date.tzinfo is None   else end_date
        self.cache_dir  = Path(cache_dir)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load DXY daily CSV with robust column normalisation."""
        if self._df is not None:
            return self._df

        # Search in backtest_data/ first, then repo root
        candidates = [
            Path(__file__).parent.parent / self.cache_dir / "dxy_daily.csv",
            Path(__file__).parent.parent / "dxy_daily.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)

        if csv_path is None:
            logger.warning(
                "No DXY CSV found in backtest_data/ or repo root. "
                "SUPER_TRENDING regime will be disabled."
            )
            self._df = pd.DataFrame()
            return self._df

        df = pd.read_csv(csv_path)

        # ── Step 1: normalise column names to lowercase ──────────────────
        df.columns = [c.strip().lower() for c in df.columns]

        # ── Step 2: find the date column ─────────────────────────────────
        date_col = next(
            (c for c in df.columns if c in ('date', 'time', 'datetime', 'datetime_utc')),
            None
        )
        if date_col is None:
            logger.error(
                f"DXY CSV has no recognised date column. "
                f"Found columns: {list(df.columns)}"
            )
            self._df = pd.DataFrame()
            return self._df

        # ── Step 3: strip surrounding quotes, then parse date ────────────
        df[date_col] = (
            df[date_col]
            .astype(str)
            .str.strip('"')
            .str.strip("'")
        )
        df[date_col] = pd.to_datetime(
            df[date_col],
            infer_datetime_format=True,
            utc=True
        )

        # ── Step 4: strip quotes from price columns ───────────────────────
        for col in ('open', 'high', 'low', 'close'):
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip('"')
                    .str.strip("'")
                    .astype(float)
                )

        # ── Step 5: rename to canonical 'date' key ───────────────────────
        df = df.rename(columns={date_col: 'date'})
        df = df.sort_values('date').reset_index(drop=True)

        # ── Step 6: compute daily % change ───────────────────────────────
        df['daily_change'] = df['close'].pct_change() * 100
        df['daily_change'] = df['daily_change'].fillna(0.0)

        self._df = df
        logger.info(
            f"Loaded DXY data from {csv_path}: {len(df)} days "
            f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})"
        )
        return self._df

    def get_daily_change(self, bar_time: datetime) -> Optional[float]:
        """Return DXY daily % change for the most recent day on or before bar_time."""
        if self._df is None:
            self.load()

        if self._df.empty or 'daily_change' not in self._df.columns:
            return None

        mask = self._df['date'].dt.date <= bar_time.date()
        valid = self._df[mask]
        if valid.empty:
            return None

        return float(valid.iloc[-1]['daily_change'])
