"""
backtest/run_updated.py — Updated CLI Entry Point for Backtesting Framework

Updated to use the completely rebuilt backtest engine with:
- All 10 current strategies with correct naming
- Updated risk management thresholds (KS3: -7%, KS4: 4 losses, etc.)
- Phase-based progression and conviction levels
- Enhanced analytics with strategy breakdown
- Monte Carlo simulation integration
- Walk-forward analysis support
- Export capabilities for trades and equity curves

Usage examples:
    python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31
    python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31 --strategy S1_LONDON_BRK --strategy R3_CAL_MOMENTUM
    python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31 --walk-forward
    python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31 --monte-carlo --plot equity.png --export trades.csv
"""
import sys
import os
import argparse
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz

# Add parent directory to path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Import updated components
from backtest.strategies import ALL_STRATEGIES, STRATEGY_REGISTRY, STRATEGY_GROUPS
from backtest.engine import BacktestEngine
from backtest.monte_carlo import MonteCarloSimulator
from backtest.walk_forward import WalkForwardAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("backtest.run")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Leela XAUUSD Updated Backtesting Framework — All 10 Current Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies: {', '.join(ALL_STRATEGIES)}

Strategy groups:
  trend_family:   {', '.join(STRATEGY_GROUPS['trend_family'])}
  mean_reversion: {STRATEGY_GROUPS['mean_reversion'][0]}
  pattern:        {STRATEGY_GROUPS['pattern'][0]}
  pullback:       {', '.join(STRATEGY_GROUPS['pullback'])}
  oco_pairs:      {', '.join(STRATEGY_GROUPS['oco_pairs'])}
  independent:    {', '.join(STRATEGY_GROUPS['independent'])}
  phase1:         {', '.join(STRATEGY_GROUPS['phase1'])}
  phase2:         {', '.join(STRATEGY_GROUPS['phase2'])}

Examples:
  # Full system backtest
  python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31
  
  # Phase 1 strategies only
  python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31 --strategy-group phase1
  
  # Individual strategies
  python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31 --strategy S1_LONDON_BRK --strategy R3_CAL_MOMENTUM
  
  # With analysis
  python -m backtest.run_updated --start 2025-01-01 --end 2026-03-31 --monte-carlo --walk-forward --plot equity.png --export trades.csv
        """
    )
    
    # Date range
    parser.add_argument("--start", required=True, type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=str, help="End date YYYY-MM-DD")
    
    # Account settings
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Initial account balance in USD (default: 10000)")
    parser.add_argument("--slippage", type=float, default=0.7,
                        help="Slippage in price points (default: 0.7)")
    
    # Strategy selection
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument("--strategy", action="append", default=None,
                               choices=ALL_STRATEGIES, metavar="STRATEGY",
                               help="Individual strategy to include (repeatable)")
    strategy_group.add_argument("--strategy-group", type=str, choices=list(STRATEGY_GROUPS.keys()),
                               help="Strategy group to run")
    
    # Analysis options
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward analysis after backtest")
    parser.add_argument("--train-months", type=int, default=3,
                        help="Walk-forward training window in months (default: 3)")
    parser.add_argument("--test-months", type=int, default=1,
                        help="Walk-forward test window in months (default: 1)")
    
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo simulation after backtest")
    parser.add_argument("--mc-sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    
    # Output options
    parser.add_argument("--plot", type=str, default=None,
                        help="Save equity curve plot to file (e.g., equity.png)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export trades to CSV file (e.g., trades.csv)")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file (e.g., results.json)")
    
    # Data options
    parser.add_argument("--cache-dir", type=str, default="backtest_data",
                        help="Directory for cached historical data (default: backtest_data)")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose (DEBUG) logging")
    
    return parser.parse_args()

def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Quieten noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def validate_dates(start_str: str, end_str: str) -> tuple[datetime, datetime]:
    """Validate and parse date strings."""
    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=pytz.utc)
        end_date = datetime.strptime(end_str, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=pytz.utc
        )
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. ({e})")
    
    if start_date >= end_date:
        raise ValueError("Start date must be before end date.")
    
    # Check if dates are reasonable (not too far in past/future)
    now = datetime.now(pytz.utc)
    if end_date > now:
        logger.warning("End date is in the future. Results may be incomplete.")
    
    if start_date < now - timedelta(days=365*5):
        logger.warning("Start date is more than 5 years in the past. Data availability may be limited.")
    
    return start_date, end_date

def get_selected_strategies(args: argparse.Namespace) -> list[str]:
    """Get list of selected strategies based on arguments."""
    if args.strategy:
        return args.strategy
    elif args.strategy_group:
        return STRATEGY_GROUPS[args.strategy_group]
    else:
        return ALL_STRATEGIES

def save_results(results: dict, args: argparse.Namespace) -> None:
    """Save results to requested formats."""
    
    # Export trades to CSV
    if args.export:
        export_path = Path(args.export)
        if 'trades' in results and results['trades']:
            import pandas as pd
            df = pd.DataFrame(results['trades'])
            df.to_csv(export_path, index=False)
            logger.info(f"Trades exported to {export_path}")
        else:
            logger.warning("No trades to export")
    
    # Export results to JSON
    if args.json:
        json_path = Path(args.json)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results exported to {json_path}")
    
    # Generate plot
    if args.plot:
        plot_path = Path(args.plot)
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if 'equity_curve' in results and results['equity_curve']:
                # Extract equity data
                times = [datetime.fromisoformat(ep['time'].replace('Z', '+00:00')) for ep in results['equity_curve']]
                equity_values = [ep['equity'] for ep in results['equity_curve']]
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(times, equity_values, linewidth=2)
                ax.set_title(f"Equity Curve ({results.get('summary', {}).get('start_date', 'Unknown')} to {results.get('summary', {}).get('end_date', 'Unknown')})")
                ax.set_xlabel("Date")
                ax.set_ylabel("Equity ($)")
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
                # Add summary stats
                summary = results.get('summary', {})
                stats_text = f"Total P&L: ${summary.get('total_pnl', 0):,.2f} ({summary.get('pnl_pct', 0):.1f}%)\n"
                stats_text += f"Win Rate: {summary.get('win_rate', 0):.1f}% | Trades: {summary.get('total_trades', 0)}\n"
                stats_text += f"Max DD: {summary.get('max_drawdown_pct', 0):.1f}% | Sharpe: {summary.get('sharpe_ratio', 0):.2f}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Equity curve plot saved to {plot_path}")
            else:
                logger.warning("No equity curve data to plot")
                
        except ImportError:
            logger.error("Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")

def print_summary(results: dict) -> None:
    """Print formatted summary of results."""
    if 'error' in results:
        logger.error(f"Backtest failed: {results['error']}")
        return
    
    summary = results.get('summary', {})
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"Period:     {summary.get('start_date', 'Unknown')} to {summary.get('end_date', 'Unknown')}")
    print(f"Initial:    ${summary.get('initial_balance', 0):,.2f}")
    print(f"Final:      ${summary.get('final_balance', 0):,.2f}")
    print(f"Total P&L:  ${summary.get('total_pnl', 0):,.2f} ({summary.get('pnl_pct', 0):.1f}%)")
    print(f"Commission: ${summary.get('total_commission', 0):,.2f}")
    
    print(f"\nTrading Statistics:")
    print(f"  Total Trades:     {summary.get('total_trades', 0)}")
    print(f"  Winning Trades:   {summary.get('winning_trades', 0)}")
    print(f"  Losing Trades:    {summary.get('losing_trades', 0)}")
    print(f"  Win Rate:         {summary.get('win_rate', 0):.1f}%")
    print(f"  Expectancy:       ${summary.get('expectancy', 0):.2f}")
    
    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown:     {summary.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Sharpe Ratio:     {summary.get('sharpe_ratio', 0):.2f}")
    
    # Strategy breakdown
    strategy_perf = results.get('strategy_performance', {})
    if strategy_perf:
        print(f"\nStrategy Performance:")
        for strategy, perf in sorted(strategy_perf.items(), key=lambda x: x[1]['pnl'], reverse=True):
            print(f"  {STRATEGY_REGISTRY.get(strategy, strategy):25s} "
                  f"Trades: {perf['trades']:3d} | "
                  f"P&L: ${perf['pnl']:8.2f} | "
                  f"WR: {perf['win_rate']*100:5.1f}% | "
                  f"Avg R: {perf['avg_r']:5.2f}")
    
    print("="*60)

def run_monte_carlo(results: dict, args: argparse.Namespace) -> Optional[dict]:
    """Run Monte Carlo simulation if requested."""
    if not args.monte_carlo:
        return None
    
    logger.info("Running Monte Carlo simulation...")
    
    try:
        mc_simulator = MonteCarloSimulator()
        mc_results = mc_simulator.run_simulation(
            results.get('trades', []),
            args.mc_sims,
            results.get('summary', {}).get('initial_balance', 10000)
        )
        
        print_mc_summary(mc_results)
        return mc_results
        
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        return None

def run_walk_forward(results: dict, args: argparse.Namespace) -> Optional[dict]:
    """Run walk-forward analysis if requested."""
    if not args.walk_forward:
        return None
    
    logger.info("Running walk-forward analysis...")
    
    try:
        wf_analyzer = WalkForwardAnalyzer()
        wf_results = wf_analyzer.run_analysis(
            args.start,
            args.end,
            args.train_months,
            args.test_months,
            get_selected_strategies(args),
            args.balance,
            args.slippage
        )
        
        print_wf_summary(wf_results)
        return wf_results
        
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")
        return None

def print_mc_summary(mc_results: dict) -> None:
    """Print Monte Carlo results summary."""
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*60)
    
    print(f"Simulations Run:    {mc_results.get('num_simulations', 0):,}")
    print(f"Final Balance:")
    print(f"  Mean:             ${mc_results.get('final_balance_mean', 0):,.2f}")
    print(f"  Median:           ${mc_results.get('final_balance_median', 0):,.2f}")
    print(f"  Std Dev:          ${mc_results.get('final_balance_std', 0):,.2f}")
    print(f"  5th Percentile:   ${mc_results.get('final_balance_5th', 0):,.2f}")
    print(f"  95th Percentile:  ${mc_results.get('final_balance_95th', 0):,.2f}")
    
    print(f"\nRisk of Ruin:")
    print(f"  < 50% Capital:    {mc_results.get('risk_of_ruin_50', 0):.1f}%")
    print(f"  < 25% Capital:    {mc_results.get('risk_of_ruin_25', 0):.1f}%")
    print(f"  Complete Ruin:     {mc_results.get('risk_of_ruin_0', 0):.1f}%")
    
    print("="*60)

def print_wf_summary(wf_results: dict) -> None:
    """Print walk-forward results summary."""
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Walk-Forward Periods: {wf_results.get('num_periods', 0)}")
    print(f"Training Window:     {wf_results.get('train_months', 0)} months")
    print(f"Test Window:          {wf_results.get('test_months', 0)} months")
    
    print(f"\nOut-of-Sample Performance:")
    print(f"  Total P&L:          ${wf_results.get('total_oos_pnl', 0):,.2f}")
    print(f"  Win Rate:           {wf_results.get('oos_win_rate', 0):.1f}%")
    print(f"  Sharpe Ratio:       {wf_results.get('oos_sharpe', 0):.2f}")
    print(f"  Max Drawdown:       {wf_results.get('oos_max_dd', 0):.1f}%")
    
    print(f"\nStability Metrics:")
    print(f"  Positive Periods:   {wf_results.get('positive_periods', 0)}/{wf_results.get('num_periods', 0)}")
    print(f"  Consistency Score:  {wf_results.get('consistency_score', 0):.1f}%")
    
    print("="*60)

def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Validate dates
        start_date, end_date = validate_dates(args.start, args.end)
        
        # Get selected strategies
        strategies = get_selected_strategies(args)
        
        logger.info("Backtest configuration:")
        logger.info(f"  Period:     {start_date.date()} to {end_date.date()}")
        logger.info(f"  Balance:    ${args.balance:,.2f}")
        logger.info(f"  Slippage:   {args.slippage} points")
        logger.info(f"  Strategies: {', '.join(strategies)}")
        
        # Run backtest
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_balance=args.balance,
            slippage_points=args.slippage,
            strategies=strategies
        )
        
        results = engine.run()
        
        # Print summary
        print_summary(results)
        
        # Run additional analysis
        mc_results = run_monte_carlo(results, args)
        wf_results = run_walk_forward(results, args)
        
        # Combine results for export
        combined_results = results.copy()
        if mc_results:
            combined_results['monte_carlo'] = mc_results
        if wf_results:
            combined_results['walk_forward'] = wf_results
        
        # Save results
        save_results(combined_results, args)
        
        logger.info("Backtest completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
