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
    python -m backtest.run --start 2025-01-01 --end 2026-03-31 --sensitivity

Run from the repo root:
    cd xauusd_algo && python -m backtest.run --start 2025-01-01 --end 2026-03-31

Progress logging
────────────────
The engine emits an INFO-level line every 1 000 M5 bars showing:

    Progress    3000/46080  ( 6.5%)  bar=2025-02-14 09:25  equity=$10 241  ETA=4m 12s

These lines appear immediately because setup_logging() forces flush=True on the
StreamHandler.  When you redirect stdout/stderr to a file use --verbose or pipe
through `stdbuf -oL` to keep real-time output.

FIX-4 (time-period stability)
─────────────────────────────
profit_concentration_check() is called automatically after every run so the
stability report is always printed.  Pass --sensitivity to also run the FIX-3
parameter sweep (5 extra backtests — adds ~5× wall-clock time).
"""
import sys
import os
import argparse
import logging
from datetime import datetime

import pytz

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# strategies.py is the authoritative list — run.py does NOT define its own list
from backtest.strategies import ALL_STRATEGIES, STRATEGY_REGISTRY

VALID_STRATEGIES = ALL_STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leela XAUUSD Backtesting Framework — All 13 Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available strategies: {', '.join(ALL_STRATEGIES)}",
    )
    parser.add_argument("--start", required=True, type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--balance",  type=float, default=10000.0,
                        help="Initial account balance in USD (default: 10000)")
    parser.add_argument("--slippage", type=float, default=0.7,
                        help="Slippage in price points (default: 0.7)")
    parser.add_argument(
        "--strategy", action="append", default=None,
        choices=VALID_STRATEGIES, metavar="STRATEGY",
        help=f"Strategy to include (repeatable). Default: all. Choices: {', '.join(VALID_STRATEGIES)}",
    )
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward analysis after backtest")
    parser.add_argument("--train-months", type=int, default=3,
                        help="Walk-forward training window in months (default: 3)")
    parser.add_argument("--test-months",  type=int, default=1,
                        help="Walk-forward test window in months (default: 1)")
    parser.add_argument("--plot",      type=str, default=None,
                        help="Save equity curve plot to file (e.g., equity.png)")
    parser.add_argument("--export",    type=str, default=None,
                        help="Export trades to CSV file (e.g., trades.csv)")
    parser.add_argument("--cache-dir", type=str, default="backtest_data",
                        help="Directory for cached historical data (default: backtest_data)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose (DEBUG) logging")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Risk-of-Ruin Monte Carlo simulation after backtest")
    parser.add_argument("--mc-sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    # FIX-3: parameter sensitivity sweep flag
    parser.add_argument(
        "--sensitivity", action="store_true",
        help=(
            "Run FIX-3 parameter sensitivity sweep after the main backtest. "
            "Varies BREAKOUT_DIST_PCT ±20%% and ATR_PCT_UNSTABLE_THRESHOLD ±5 pts "
            "across 5 backtests and flags ⚠ OVERFIT if any variant drops >40%% P&L."
        ),
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logger with a StreamHandler that flushes immediately.

    Without flush=True, progress lines may be buffered when stdout is piped
    (e.g. tee, nohup) — they would all appear at once at the end instead of
    live.  force=True lets us reconfigure if basicConfig was already called
    by an imported module.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.flush = sys.stdout.flush          # ensure immediate flush per record
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any pre-existing handlers (e.g. from basicConfig in imports)
    root.handlers.clear()
    root.addHandler(handler)

    # Quieten noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("backtest.run")

    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=pytz.utc)
        end_date   = datetime.strptime(args.end,   "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=pytz.utc
        )
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. ({e})")
        sys.exit(1)

    if start_date >= end_date:
        print("Error: Start date must be before end date.")
        sys.exit(1)

    strategies = args.strategy if args.strategy else VALID_STRATEGIES

    logger.info("Backtest configuration:")
    logger.info(f"  Period:     {start_date.date()} to {end_date.date()}")
    logger.info(f"  Balance:    ${args.balance:,.2f}")
    logger.info(f"  Slippage:   {args.slippage} points")
    logger.info(f"  Strategies: {strategies}")
    logger.info(f"  Cache dir:  {args.cache_dir}")

    # ── engine.py is the single authoritative backtest engine ─────────────────
    # engine_enhanced.py / engine_enhanced_fixed.py are superseded and must NOT
    # be imported here — engine.py already contains all 13 strategies,
    # KS3/KS4/KS5/KS6, TLT macro bias proxy, commission, and weekly PnL.
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

    print()
    results.summary()

    # FIX-4: Time-period stability check — always run after every backtest.
    # Warns when >80% of gross profit is concentrated in ≤3 calendar months.
    print()
    BacktestEngine.profit_concentration_check(results)

    # Walk-forward analysis
    if args.walk_forward:
        print()
        results.walk_forward_report(
            train_months=args.train_months,
            test_months=args.test_months,
        )

    # FIX-3: Parameter sensitivity sweep (opt-in via --sensitivity flag)
    # Runs 5 extra backtests varying BREAKOUT_DIST_PCT ±20% and
    # ATR_PCT_UNSTABLE_THRESHOLD ±5 pts.  Prints ⚠ OVERFIT WARNING if any
    # variant drops >40% vs baseline — indicating the strategy is over-tuned.
    if args.sensitivity:
        print()
        logger.info("Running FIX-3 parameter sensitivity sweep (5 variants × full backtest)...")
        engine.run_sensitivity_test()

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

    # Monte Carlo simulation
    if args.monte_carlo:
        logger.info(f"Running Monte Carlo with {args.mc_sims} simulations...")
        try:
            from backtest.analytics import MonteCarlo
            mc = MonteCarlo(results.trades, {"initial_balance": args.balance})
            mc_results = mc.run(args.mc_sims)
            print("\n" + "=" * 60)
            print("MONTE CARLO RESULTS")
            print("=" * 60)
            print(f"Probability of Profit: {mc_results.get('probability_of_profit', 0):.1%}")
            print(f"Risk of Ruin:          {mc_results.get('risk_of_ruin', 0):.1%}")
            print(f"Expected Return:       ${mc_results.get('expected_return', 0):,.2f}")
            print("=" * 60)
        except ImportError:
            logger.warning("backtest.analytics not available — skipping Monte Carlo")

    # Monthly returns
    monthly = results.monthly_returns()
    if not monthly.empty:
        print("\n── MONTHLY RETURNS ──────────────────────────────────────")
        print(monthly.to_string(index=False))

    # Exit reason breakdown
    exit_breakdown = results.exit_reason_breakdown()
    if not exit_breakdown.empty:
        print("\n── EXIT REASON BREAKDOWN ────────────────────────────────")
        print(exit_breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
