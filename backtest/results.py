"""
backtest/results.py — Analytics and reporting for backtest results.

BacktestResults provides:
  - summary()           — overall and per-strategy metrics
  - compute_max_dd()    — maximum drawdown from equity curve
  - compute_sharpe()    — annualized Sharpe ratio
  - walk_forward_report() — rolling OOS performance analysis
  - to_dataframe()      — trades as pandas DataFrame
  - plot_equity()       — equity curve visualization (if matplotlib available)
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from backtest.models import TradeRecord, EquityPoint

logger = logging.getLogger("backtest.results")


class BacktestResults:
    """
    Analytics container for backtest output.

    Constructed by BacktestEngine.run() with all closed trades and equity curve.
    """

    def __init__(
        self,
        trades: list[TradeRecord],
        equity_curve: list[EquityPoint],
        initial_balance: float,
        start_date: datetime,
        end_date: datetime,
        strategies: list[str],
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_balance = initial_balance
        self.start_date = start_date
        self.end_date = end_date
        self.strategies = strategies

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Print and return a comprehensive summary of backtest results.
        Includes overall metrics and per-strategy breakdown.
        """
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("BACKTEST RESULTS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Period:          {self.start_date.date()} to {self.end_date.date()}")
        lines.append(f"Initial Balance: ${self.initial_balance:,.2f}")

        if self.equity_curve:
            final_eq = self.equity_curve[-1].equity
            lines.append(f"Final Equity:    ${final_eq:,.2f}")
            total_return = (final_eq - self.initial_balance) / self.initial_balance * 100
            lines.append(f"Total Return:    {total_return:+.2f}%")
        lines.append("")

        # Overall metrics
        overall = self._compute_metrics(self.trades)
        lines.append("── OVERALL ─────────────────────────────────────────────")
        lines.extend(self._format_metrics(overall))
        lines.append("")

        # Per-strategy breakdown
        strategy_trades: dict[str, list[TradeRecord]] = {}
        for trade in self.trades:
            strategy_trades.setdefault(trade.strategy, []).append(trade)

        for strat in sorted(strategy_trades.keys()):
            trades = strategy_trades[strat]
            metrics = self._compute_metrics(trades)
            lines.append(f"── {strat} ({len(trades)} trades) ──────────────────────")
            lines.extend(self._format_metrics(metrics))
            lines.append("")

        # Drawdown
        max_dd, max_dd_pct = self.compute_max_dd()
        lines.append(f"Max Drawdown:    ${max_dd:,.2f} ({max_dd_pct:.2f}%)")

        # Sharpe
        sharpe = self.compute_sharpe()
        lines.append(f"Sharpe Ratio:    {sharpe:.3f} (annualized)")

        lines.append("=" * 80)

        output = "\n".join(lines)
        print(output)
        return output

    def _compute_metrics(self, trades: list[TradeRecord]) -> dict:
        """Compute standard trading metrics for a list of trades."""
        if not trades:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "profit_factor": 0.0,
                "expectancy_r": 0.0,
                "avg_r": 0.0,
                "max_consecutive_losses": 0,
                "total_commission": 0.0,
            }

        pnls = [t.pnl for t in trades]
        r_multiples = [t.r_multiple for t in trades]
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for t in trades:
            if t.pnl <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        return {
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls) if pnls else 0.0,
            "avg_winner": np.mean([t.pnl for t in winners]) if winners else 0.0,
            "avg_loser": np.mean([t.pnl for t in losers]) if losers else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "expectancy_r": np.mean(r_multiples) if r_multiples else 0.0,
            "avg_r": np.mean(r_multiples) if r_multiples else 0.0,
            "max_consecutive_losses": max_consec,
            "total_commission": sum(t.commission for t in trades),
        }

    @staticmethod
    def _format_metrics(m: dict) -> list[str]:
        """Format metrics dict into display lines."""
        return [
            f"  Trades:          {m['total_trades']}  (W:{m['winners']} / L:{m['losers']})",
            f"  Win Rate:        {m['win_rate']:.1f}%",
            f"  Total P&L:       ${m['total_pnl']:+,.2f}",
            f"  Avg P&L:         ${m['avg_pnl']:+,.2f}",
            f"  Avg Winner:      ${m['avg_winner']:+,.2f}",
            f"  Avg Loser:       ${m['avg_loser']:+,.2f}",
            f"  Profit Factor:   {m['profit_factor']:.2f}",
            f"  Expectancy (R):  {m['expectancy_r']:+.3f}",
            f"  Max Consec Loss: {m['max_consecutive_losses']}",
            f"  Commission:      ${m['total_commission']:,.2f}",
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # DRAWDOWN
    # ─────────────────────────────────────────────────────────────────────────

    def compute_max_dd(self) -> tuple[float, float]:
        """
        Compute maximum drawdown from equity curve.

        Returns:
            (max_dd_dollars, max_dd_percent)
        """
        if not self.equity_curve:
            return 0.0, 0.0

        equities = [ep.equity for ep in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for eq in equities:
            peak = max(peak, eq)
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)

        return round(max_dd, 2), round(max_dd_pct * 100, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # SHARPE RATIO
    # ─────────────────────────────────────────────────────────────────────────

    def compute_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """
        Compute annualized Sharpe ratio from daily returns.

        Uses equity curve to compute daily returns, then annualizes
        with sqrt(252) (trading days per year).
        """
        if len(self.equity_curve) < 2:
            return 0.0

        # Build daily equity series
        eq_df = pd.DataFrame([
            {"time": ep.timestamp, "equity": ep.equity}
            for ep in self.equity_curve
        ])
        eq_df["time"] = pd.to_datetime(eq_df["time"])
        eq_df.set_index("time", inplace=True)

        # Resample to daily (last value per day)
        daily = eq_df.resample("1D").last().dropna()

        if len(daily) < 2:
            return 0.0

        # Daily returns
        returns = daily["equity"].pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualized Sharpe
        excess_return = returns.mean() - risk_free_rate / 252
        sharpe = (excess_return / returns.std()) * np.sqrt(252)

        return round(float(sharpe), 3)

    # ─────────────────────────────────────────────────────────────────────────
    # WALK-FORWARD ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def walk_forward_report(
        self,
        train_months: int = 3,
        test_months: int = 1,
    ) -> str:
        """
        Walk-forward analysis: split data into rolling train/test windows.

        For each window:
          - Train period: compute metrics (in-sample)
          - Test period: compute metrics (out-of-sample)
          - Compare IS vs OOS performance

        Returns formatted report string.
        """
        if not self.trades:
            return "No trades to analyze."

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("WALK-FORWARD ANALYSIS")
        lines.append(f"Train: {train_months} months | Test: {test_months} months")
        lines.append("=" * 80)

        window_start = self.start_date
        window_num = 0

        while window_start < self.end_date:
            train_end = window_start + timedelta(days=train_months * 30)
            test_end = train_end + timedelta(days=test_months * 30)

            if train_end >= self.end_date:
                break

            test_end = min(test_end, self.end_date)

            # Filter trades by period
            train_trades = [
                t for t in self.trades
                if window_start <= t.entry_time < train_end
            ]
            test_trades = [
                t for t in self.trades
                if train_end <= t.entry_time < test_end
            ]

            window_num += 1
            lines.append(
                f"\n── Window {window_num}: "
                f"Train {window_start.date()}→{train_end.date()} | "
                f"Test {train_end.date()}→{test_end.date()} ──"
            )

            train_metrics = self._compute_metrics(train_trades)
            test_metrics = self._compute_metrics(test_trades)

            lines.append(f"  IN-SAMPLE:  {train_metrics['total_trades']} trades, "
                         f"WR={train_metrics['win_rate']:.1f}%, "
                         f"E(R)={train_metrics['expectancy_r']:+.3f}, "
                         f"PF={train_metrics['profit_factor']:.2f}, "
                         f"P&L=${train_metrics['total_pnl']:+,.2f}")

            lines.append(f"  OUT-SAMPLE: {test_metrics['total_trades']} trades, "
                         f"WR={test_metrics['win_rate']:.1f}%, "
                         f"E(R)={test_metrics['expectancy_r']:+.3f}, "
                         f"PF={test_metrics['profit_factor']:.2f}, "
                         f"P&L=${test_metrics['total_pnl']:+,.2f}")

            # Degradation check
            if train_metrics['expectancy_r'] > 0 and test_metrics['expectancy_r'] > 0:
                retention = test_metrics['expectancy_r'] / train_metrics['expectancy_r'] * 100
                lines.append(f"  RETENTION:  {retention:.0f}% of IS expectancy")
            elif train_metrics['expectancy_r'] > 0:
                lines.append("  RETENTION:  ⚠ OOS expectancy negative — edge decay")

            # Advance window
            window_start = train_end

        lines.append("\n" + "=" * 80)

        output = "\n".join(lines)
        print(output)
        return output

    # ─────────────────────────────────────────────────────────────────────────
    # DATA EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to a pandas DataFrame for further analysis."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "strategy": t.strategy,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "lots": t.lots,
                "pnl": t.pnl,
                "pnl_gross": t.pnl_gross,
                "r_multiple": t.r_multiple,
                "exit_reason": t.exit_reason,
                "regime_at_entry": t.regime_at_entry,
                "regime_at_exit": t.regime_at_exit,
                "stop_original": t.stop_original,
                "commission": t.commission,
            })

        return pd.DataFrame(records)

    def equity_to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to a pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": ep.timestamp,
                "equity": ep.equity,
                "drawdown_pct": ep.drawdown_pct,
            }
            for ep in self.equity_curve
        ])

    def plot_equity(self, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve and drawdown.
        Requires matplotlib (optional dependency).

        Args:
            save_path: If provided, saves plot to file instead of showing.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib not installed — cannot plot equity curve")
            print("Install matplotlib for equity curve plotting: pip install matplotlib")
            return

        eq_df = self.equity_to_dataframe()
        if eq_df.empty:
            print("No equity data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})

        # Equity curve
        ax1.plot(eq_df["timestamp"], eq_df["equity"], color="steelblue", linewidth=1)
        ax1.axhline(y=self.initial_balance, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(
            f"Backtest Equity Curve — {self.start_date.date()} to {self.end_date.date()}"
        )
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2.fill_between(
            eq_df["timestamp"], -eq_df["drawdown_pct"] * 100, 0,
            color="salmon", alpha=0.5,
        )
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Equity curve saved to {save_path}")
        else:
            plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # PER-STRATEGY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def strategy_breakdown(self) -> dict[str, dict]:
        """Returns metrics dict per strategy."""
        result = {}
        strategy_trades: dict[str, list[TradeRecord]] = {}
        for trade in self.trades:
            strategy_trades.setdefault(trade.strategy, []).append(trade)

        for strat, trades in strategy_trades.items():
            result[strat] = self._compute_metrics(trades)

        return result

    def monthly_returns(self) -> pd.DataFrame:
        """Compute monthly P&L returns."""
        if not self.trades:
            return pd.DataFrame()

        df = self.to_dataframe()
        df["month"] = df["exit_time"].dt.to_period("M")
        monthly = df.groupby("month").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            avg_r=("r_multiple", "mean"),
        ).reset_index()
        monthly["month"] = monthly["month"].astype(str)
        return monthly

    def exit_reason_breakdown(self) -> pd.DataFrame:
        """Breakdown of trades by exit reason."""
        if not self.trades:
            return pd.DataFrame()

        df = self.to_dataframe()
        return df.groupby("exit_reason").agg(
            count=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_r=("r_multiple", "mean"),
        ).reset_index()
