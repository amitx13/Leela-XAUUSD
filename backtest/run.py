"""
backtest/run.py — CLI entry point for the backtesting framework.

Usage:
    python -m backtest.run --start 2025-01-01 --end 2026-03-31
    python -m backtest.run --start 2025-01-01 --end 2026-03-31 --walk-forward
    python -m backtest.run --start 2025-01-01 --end 2026-03-31 --strategy S1_LONDON_BRK
    python -m backtest.run --start 2025-01-01 --end 2026-03-31 --strategy S1_LONDON_BRK --strategy S7_DAILY_STRUCT
    python -m backtest.run --start 2025-01-01 --end 2026-03-31 --balance 25000 --slippage 1.0
    python -m backtest.run --start 2025-01-01 --end 2026-03-31 --plot equity_curve.png
    python -m backtest.run --start 2025-01-01 --end 2026-03-31 --export trades.csv

Run from the xauusd_algo/ directory:
    cd xauusd_algo && python -m backtest.run --start 2025-01-01 --end 2026-03-31
"""
import sys
import os
import argparse
import logging
from datetime import datetime

import pytz

# Ensure xauusd_algo is on the path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


VALID_STRATEGIES = [
    "S1_LONDON_BRK",
    "S2_MEAN_REV",
    "S3_STOP_HUNT_REV",
    "S6_ASIAN_BRK",
    "S7_DAILY_STRUCT",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Leela XAUUSD Backtesting Framework — Phase 1B-1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtest.run --start 2025-01-01 --end 2026-03-31
  python -m backtest.run --start 2025-01-01 --end 2026-03-31 --walk-forward
  python -m backtest.run --start 2025-06-01 --end 2025-12-31 --strategy S1_LONDON_BRK
  python -m backtest.run --start 2025-01-01 --end 2026-03-31 --plot equity.png --export trades.csv
        """,
    )

    parser.add_argument(
        "--start", required=True, type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", required=True, type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0,
        help="Initial account balance in USD (default: 10000)",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.7,
        help="Slippage in price points (default: 0.7)",
    )
    parser.add_argument(
        "--strategy", action="append", default=None,
        choices=VALID_STRATEGIES,
        help="Strategy to include (can specify multiple). Default: all strategies.",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward analysis after backtest",
    )
    parser.add_argument(
        "--train-months", type=int, default=3,
        help="Walk-forward training window in months (default: 3)",
    )
    parser.add_argument(
        "--test-months", type=int, default=1,
        help="Walk-forward test window in months (default: 1)",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Save equity curve plot to file (e.g., equity.png)",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export trades to CSV file (e.g., trades.csv)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="backtest_data",
        help="Directory for cached historical data (default: backtest_data)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--monte-carlo", action="store_true",
        help="Run Risk-of-Ruin Monte Carlo simulation after backtest",
    )
    parser.add_argument(
        "--mc-sims", type=int, default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the backtest."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    """Main entry point for the backtest CLI."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("backtest.run")

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=pytz.utc)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=pytz.utc
        )
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. ({e})")
        sys.exit(1)

    if start_date >= end_date:
        print("Error: Start date must be before end date.")
        sys.exit(1)

    strategies = args.strategy if args.strategy else VALID_STRATEGIES

    logger.info(f"Backtest configuration:")
    logger.info(f"  Period:     {start_date.date()} to {end_date.date()}")
    logger.info(f"  Balance:    ${args.balance:,.2f}")
    logger.info(f"  Slippage:   {args.slippage} points")
    logger.info(f"  Strategies: {strategies}")
    logger.info(f"  Cache dir:  {args.cache_dir}")

    # Import and run engine
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_balance=args.balance,
        slippage_points=args.slippage,
        strategies=strategies,
        cache_dir=args.cache_dir,
    )

    try:
        results = engine.run()
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nTo collect historical data, run:")
        print("  python -m tools.collect_historical_data --start 2025-01-01")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        sys.exit(1)

    # Print summary
    print()
    results.summary()

    # Walk-forward analysis
    if args.walk_forward:
        print()
        results.walk_forward_report(
            train_months=args.train_months,
            test_months=args.test_months,
        )

    # Export trades
    if args.export:
        df = results.to_dataframe()
        if not df.empty:
            df.to_csv(args.export, index=False)
            print(f"\nTrades exported to {args.export}")
        else:
            print("\nNo trades to export.")

    # Plot equity curve
    if args.plot:
        results.plot_equity(save_path=args.plot)

    # Print monthly returns
    monthly = results.monthly_returns()
    if not monthly.empty:
        print("\n── MONTHLY RETURNS ──────────────────────────────────────")
        print(monthly.to_string(index=False))

    # Print exit reason breakdown
    exit_breakdown = results.exit_reason_breakdown()
    if not exit_breakdown.empty:
        print("\n── EXIT REASON BREAKDOWN ────────────────────────────────")
        print(exit_breakdown.to_string(index=False))

    # Monte Carlo risk-of-ruin simulation
    if args.monte_carlo:
        if not results.trades:
            print("\nNo trades for Monte Carlo simulation.")
        else:
            from backtest.monte_carlo import RiskOfRuinSimulator

            sim = RiskOfRuinSimulator.from_backtest_results(results)
            sim.run_full_report(n_simulations=args.mc_sims)


if __name__ == "__main__":
    main()
