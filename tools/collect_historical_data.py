"""
tools/collect_historical_data.py — Fetch and store historical OHLCV data from MT5.

Connects to MT5 via mt5linux rpyc bridge, fetches M5 OHLCV data for XAUUSD,
and stores it locally in parquet format for fast backtest loading.

Also fetches M15, H1, H4, D1 for cross-validation.

Usage:
    python -m tools.collect_historical_data --start 2025-01-01
    python -m tools.collect_historical_data --start 2025-01-01 --end 2026-03-31
    python -m tools.collect_historical_data --start 2025-01-01 --output-dir backtest_data
    python -m tools.collect_historical_data --start 2025-01-01 --format pickle

Run from the xauusd_algo/ directory:
    cd xauusd_algo && python -m tools.collect_historical_data --start 2025-01-01
"""
import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

import pytz
import pandas as pd

# Ensure xauusd_algo is on the path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

logger = logging.getLogger("tools.collect_historical_data")


TIMEFRAMES = {
    "M5":  None,   # Will be set from MT5 constants
    "M15": None,
    "H1":  None,
    "H4":  None,
    "D1":  None,
}


def fetch_mt5_data(
    symbol: str,
    timeframe_name: str,
    mt5_timeframe,
    start_date: datetime,
    end_date: datetime,
    mt5,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from MT5 for a given symbol and timeframe.

    Args:
        symbol: Trading symbol (e.g., "XAUUSD")
        timeframe_name: Human-readable name (e.g., "M5")
        mt5_timeframe: MT5 timeframe constant
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        mt5: MT5 connection object

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume, spread
    """
    logger.info(f"Fetching {symbol} {timeframe_name} from {start_date.date()} to {end_date.date()}...")

    # MT5 copy_rates_range expects naive datetimes (interpreted as UTC)
    start_naive = start_date.replace(tzinfo=None)
    end_naive = end_date.replace(tzinfo=None)

    bars = mt5.copy_rates_range(symbol, mt5_timeframe, start_naive, end_naive)

    if bars is None or len(bars) == 0:
        logger.warning(f"No {timeframe_name} data returned from MT5")
        return pd.DataFrame()

    df = pd.DataFrame(bars)

    # Normalize column names
    col_map = {
        "Time": "time", "Open": "open", "High": "high",
        "Low": "low", "Close": "close",
        "Volume": "tick_volume", "Tick_volume": "tick_volume",
        "Spread": "spread", "Real_volume": "real_volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Convert time to UTC datetime
    if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Ensure required columns
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    if "spread" not in df.columns:
        df["spread"] = 0

    # Keep only needed columns
    keep_cols = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info(f"  → {len(df)} {timeframe_name} bars fetched "
                f"({df['time'].iloc[0]} to {df['time'].iloc[-1]})")

    return df


def save_data(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_dir: Path,
    fmt: str = "parquet",
) -> None:
    """Save DataFrame to local file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        path = output_dir / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
    elif fmt == "pickle":
        path = output_dir / f"{symbol}_{timeframe}.pkl"
        df.to_pickle(path)
    elif fmt == "csv":
        path = output_dir / f"{symbol}_{timeframe}.csv"
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"  → Saved to {path} ({size_mb:.2f} MB)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect historical OHLCV data from MT5 for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--start", required=True, type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD",
        help="Trading symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="backtest_data",
        help="Output directory for data files (default: backtest_data)",
    )
    parser.add_argument(
        "--format", type=str, default="parquet",
        choices=["parquet", "pickle", "csv"],
        help="Output file format (default: parquet)",
    )
    parser.add_argument(
        "--timeframes", type=str, nargs="+",
        default=["M5", "M15", "H1", "H4", "D1"],
        help="Timeframes to fetch (default: M5 M15 H1 H4 D1)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=pytz.utc)
    except ValueError as e:
        print(f"Error: Invalid start date format. Use YYYY-MM-DD. ({e})")
        sys.exit(1)

    if args.end:
        try:
            end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=pytz.utc
            )
        except ValueError as e:
            print(f"Error: Invalid end date format. Use YYYY-MM-DD. ({e})")
            sys.exit(1)
    else:
        end_date = datetime.now(pytz.utc)

    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Historical Data Collection — Leela XAUUSD")
    logger.info("=" * 60)
    logger.info(f"Symbol:     {args.symbol}")
    logger.info(f"Period:     {start_date.date()} to {end_date.date()}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Output:     {output_dir} ({args.format})")
    logger.info("")

    # Connect to MT5
    try:
        from utils.mt5_client import get_mt5
        mt5 = get_mt5()

        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            print("\nError: Cannot connect to MT5. Ensure:")
            print("  1. MT5 terminal is running in Bottles/Wine")
            print("  2. mt5linux rpyc server is running")
            print(f"  3. Connection settings: {os.getenv('MT5_HOST', 'localhost')}:{os.getenv('MT5_PORT', '18812')}")
            sys.exit(1)

        logger.info("MT5 connected successfully")

        # Map timeframe names to MT5 constants
        tf_map = {
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }

    except ImportError:
        print("\nError: mt5linux package not available.")
        print("Install with: pip install mt5linux")
        sys.exit(1)
    except Exception as e:
        logger.error(f"MT5 connection failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)

    # Fetch and save data for each timeframe
    total_bars = 0
    for tf_name in args.timeframes:
        if tf_name not in tf_map:
            logger.warning(f"Unknown timeframe: {tf_name}, skipping")
            continue

        mt5_tf = tf_map[tf_name]
        df = fetch_mt5_data(
            args.symbol, tf_name, mt5_tf, start_date, end_date, mt5
        )

        if df.empty:
            logger.warning(f"No data for {tf_name} — skipping save")
            continue

        save_data(df, args.symbol, tf_name, output_dir, args.format)
        total_bars += len(df)

    logger.info("")
    logger.info(f"Collection complete: {total_bars} total bars across {len(args.timeframes)} timeframes")
    logger.info(f"Data saved to: {output_dir.absolute()}")

    # Verify data integrity
    logger.info("")
    logger.info("── Data Verification ──")
    for tf_name in args.timeframes:
        if args.format == "parquet":
            path = output_dir / f"{args.symbol}_{tf_name}.parquet"
        elif args.format == "pickle":
            path = output_dir / f"{args.symbol}_{tf_name}.pkl"
        else:
            path = output_dir / f"{args.symbol}_{tf_name}.csv"

        if path.exists():
            if args.format == "parquet":
                df = pd.read_parquet(path)
            elif args.format == "pickle":
                df = pd.read_pickle(path)
            else:
                df = pd.read_csv(path)

            logger.info(
                f"  {tf_name}: {len(df)} bars, "
                f"{df['time'].iloc[0]} → {df['time'].iloc[-1]}, "
                f"size={path.stat().st_size / 1024:.1f} KB"
            )
        else:
            logger.warning(f"  {tf_name}: FILE NOT FOUND at {path}")


if __name__ == "__main__":
    main()
