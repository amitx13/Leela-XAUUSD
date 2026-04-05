"""
backtest/results.py — Analytics and reporting for backtest results.

BacktestResults provides:
  - summary()              — overall and per-strategy metrics
  - compute_max_dd()       — maximum drawdown from equity curve
  - compute_sharpe()       — annualized Sharpe (trade-based daily P&L) [FIXED]
  - compute_sortino()      — annualized Sortino ratio [NEW]
  - compute_calmar()       — CAGR / Max Drawdown ratio [NEW]
  - walk_forward_report()  — rolling OOS performance analysis
  - regime_breakdown()     — P&L / WR / E(R) split by entry regime [NEW]
  - duration_stats()       — avg/median trade duration by strategy [NEW]
  - rolling_expectancy()   — 20-trade rolling E(R) for edge stability [NEW]
  - monthly_returns()      — monthly P&L with win rate + streak data [ENHANCED]
  - to_dataframe()         — trades as pandas DataFrame
  - plot_equity()          — equity curve + drawdown visualisation
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
        self.trades        = trades
        self.equity_curve  = equity_curve
        self.initial_balance = initial_balance
        self.start_date    = start_date
        self.end_date      = end_date
        self.strategies    = strategies

    def _analysis_trades(self) -> list[TradeRecord]:
        """Exclude synthetic end-of-backtest liquidations from analytics metrics."""
        return [t for t in self.trades if t.exit_reason != "BACKTEST_END"]

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self) -> str:
        """Print and return a comprehensive summary of backtest results."""
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

        analysis_trades = self._analysis_trades()

        # Overall metrics
        overall = self._compute_metrics(analysis_trades)
        lines.append("── OVERALL ─────────────────────────────────────────────")
        lines.extend(self._format_metrics(overall))
        lines.append("")

        # Per-strategy breakdown
        strategy_trades: dict[str, list[TradeRecord]] = {}
        for trade in analysis_trades:
            strategy_trades.setdefault(trade.strategy, []).append(trade)

        for strat in sorted(strategy_trades.keys()):
            s_trades = strategy_trades[strat]
            metrics  = self._compute_metrics(s_trades)
            lines.append(f"── {strat} ({len(s_trades)} trades) ──────────────────────")
            lines.extend(self._format_metrics(metrics))
            lines.append("")

        # Risk metrics
        max_dd, max_dd_pct = self.compute_max_dd()
        sharpe   = self.compute_sharpe()
        sortino  = self.compute_sortino()
        calmar   = self.compute_calmar()

        lines.append(f"── RISK METRICS ────────────────────────────────────────")
        lines.append(f"  Max Drawdown:    ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
        lines.append(f"  Sharpe Ratio:    {sharpe:+.3f}  (annualised, trade-based daily P&L)")
        lines.append(f"  Sortino Ratio:   {sortino:+.3f}  (downside deviation)")
        lines.append(f"  Calmar Ratio:    {calmar:+.3f}  (CAGR / Max DD)")
        lines.append("=" * 80)

        output = "\n".join(lines)
        print(output)
        return output

    # =========================================================================
    # CORE METRICS
    # =========================================================================

    def _compute_metrics(self, trades: list[TradeRecord]) -> dict:
        """Compute standard trading metrics for a list of trades."""
        empty = {
            "total_trades": 0, "winners": 0, "losers": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
            "avg_winner": 0.0, "avg_loser": 0.0, "profit_factor": 0.0,
            "expectancy_r": 0.0, "avg_r": 0.0, "median_r": 0.0,
            "pct_positive_r": 0.0, "max_consecutive_losses": 0,
            "total_commission": 0.0,
        }
        if not trades:
            return empty

        pnls         = [t.pnl for t in trades]
        r_multiples  = [t.r_multiple for t in trades]
        winners      = [t for t in trades if t.pnl > 0]
        losers       = [t for t in trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss   = abs(sum(t.pnl for t in losers)) if losers else 0.0

        max_consec = cur = 0
        for t in trades:
            if t.pnl <= 0:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0

        return {
            "total_trades":           len(trades),
            "winners":                len(winners),
            "losers":                 len(losers),
            "win_rate":               len(winners) / len(trades) * 100,
            "total_pnl":              sum(pnls),
            "avg_pnl":                float(np.mean(pnls)),
            "avg_winner":             float(np.mean([t.pnl for t in winners])) if winners else 0.0,
            "avg_loser":              float(np.mean([t.pnl for t in losers]))  if losers  else 0.0,
            "profit_factor":          gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "expectancy_r":           float(np.mean(r_multiples)),
            "avg_r":                  float(np.mean(r_multiples)),
            "median_r":               float(np.median(r_multiples)),
            "pct_positive_r":         float(np.mean([r > 0 for r in r_multiples]) * 100),
            "max_consecutive_losses": max_consec,
            "total_commission":       sum(t.commission for t in trades),
        }

    @staticmethod
    def _format_metrics(m: dict) -> list[str]:
        lines = [
            f"  Trades:          {m['total_trades']}  (W:{m['winners']} / L:{m['losers']})",
            f"  Win Rate:        {m['win_rate']:.1f}%",
            f"  Total P&L:       ${m['total_pnl']:+,.2f}",
            f"  Avg P&L:         ${m['avg_pnl']:+,.2f}",
            f"  Avg Winner:      ${m['avg_winner']:+,.2f}",
            f"  Avg Loser:       ${m['avg_loser']:+,.2f}",
            f"  Profit Factor:   {m['profit_factor']:.2f}",
            f"  Expectancy (R):  {m['expectancy_r']:+.3f}  |  Median R: {m['median_r']:+.3f}",
            f"  %Trades +R:      {m['pct_positive_r']:.1f}%",
            f"  Max Consec Loss: {m['max_consecutive_losses']}",
            f"  Commission:      ${m['total_commission']:,.2f}",
        ]
        return lines

    # =========================================================================
    # DRAWDOWN
    # =========================================================================

    def compute_max_dd(self) -> tuple[float, float]:
        """Compute maximum drawdown from equity curve. Returns (dollars, percent)."""
        if not self.equity_curve:
            return 0.0, 0.0
        equities = [ep.equity for ep in self.equity_curve]
        peak = equities[0]
        max_dd = max_dd_pct = 0.0
        for eq in equities:
            peak = max(peak, eq)
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0.0
            max_dd     = max(max_dd, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)
        return round(max_dd, 2), round(max_dd_pct * 100, 2)

    # =========================================================================
    # RATIO METRICS  (Sharpe / Sortino / Calmar)
    # =========================================================================

    def _daily_pnl_series(self) -> pd.Series:
        """
        Build a tz-naive daily P&L series from closed trades.

        BUG FIX: the previous implementation preserved timezone info on
        exit_time when calling .dt.normalize(), producing a tz-aware index.
        pd.date_range(self.start_date.date(), ...) always returns a tz-naive
        index, so .reindex() found no matching keys and silently filled every
        row with NaN.  std(NaN series) == NaN → Sharpe fell through to the
        ``if std == 0`` guard and returned 0.0.

        Fix: strip tz from exit_time with .dt.tz_convert(None) before
        .dt.normalize() so both sides of .reindex() are always tz-naive.
        """
        analysis_trades = self._analysis_trades()
        if not analysis_trades:
            return pd.Series(dtype=float)

        df = self.to_dataframe(include_backtest_end=False)

        # Ensure exit_time is tz-naive before extracting the date key.
        exit_ts = pd.to_datetime(df["exit_time"])
        if exit_ts.dt.tz is not None:
            exit_ts = exit_ts.dt.tz_convert(None)
        df["date"] = exit_ts.dt.normalize()

        daily = df.groupby("date")["pnl"].sum()

        # Build a tz-naive index covering every calendar day in the backtest.
        start = pd.Timestamp(self.start_date.date())
        end   = pd.Timestamp(self.end_date.date())
        idx   = pd.date_range(start, end, freq="D")

        daily = daily.reindex(idx, fill_value=0.0)
        return daily

    def compute_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """
        Annualised Sharpe ratio using trade-based daily P&L.

        Each calendar day's return = sum(closed-trade P&L that day) / initial_balance.
        Days with no closed trades contribute a 0.0 return (they still count in the
        denominator — this is the correct treatment for a system that is "live" every
        trading day whether or not it fires a trade).

        Annualisation factor: sqrt(252) — standard for daily returns.
        """
        daily = self._daily_pnl_series()
        if len(daily) < 2:
            return 0.0

        returns = daily / self.initial_balance
        std = returns.std()
        if std == 0 or np.isnan(std):
            return 0.0

        excess = returns.mean() - risk_free_rate / 252
        return round(float(excess / std * np.sqrt(252)), 3)

    def compute_sortino(self, risk_free_rate: float = 0.0, mar: float = 0.0) -> float:
        """
        Annualised Sortino ratio.
        Uses downside deviation (returns below MAR) in denominator instead
        of total standard deviation — more relevant for a system with a hard
        daily loss limit (KS3) that truncates the downside.

        Args:
            mar: Minimum acceptable return per day (default 0 = don't lose money).
        """
        daily = self._daily_pnl_series()
        if len(daily) < 2:
            return 0.0

        returns   = daily / self.initial_balance
        downside  = returns[returns < mar] - mar
        if len(downside) == 0:
            return float("inf")

        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        excess = returns.mean() - risk_free_rate / 252
        return round(float(excess / downside_std * np.sqrt(252)), 3)

    def compute_calmar(self) -> float:
        """
        Calmar ratio = annualised CAGR / max drawdown.
        Primary metric for prop-firm-style evaluation.

        Returns 0.0 if backtest period < 1 month or max drawdown is 0.
        """
        _, max_dd_pct = self.compute_max_dd()
        if max_dd_pct == 0:
            return 0.0

        if not self.equity_curve:
            return 0.0

        final_eq = self.equity_curve[-1].equity
        days     = max((self.end_date - self.start_date).days, 1)
        years    = days / 365.25
        if years < (1 / 12):
            return 0.0

        cagr = (final_eq / self.initial_balance) ** (1 / years) - 1
        return round(float(cagr / (max_dd_pct / 100)), 3)

    # =========================================================================
    # WALK-FORWARD
    # =========================================================================

    def walk_forward_report(
        self,
        train_months: int = 3,
        test_months: int = 1,
    ) -> str:
        """
        Walk-forward analysis: rolling IS/OOS windows.

        NOTE: Parameters are NOT re-optimised per window (pure OOS split).
        Use this to check whether E(R) holds across time, not to optimise params.
        For true parameter robustness, run separate optimisation passes.
        """
        analysis_trades = self._analysis_trades()
        if not analysis_trades:
            return "No trades to analyze."

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("WALK-FORWARD ANALYSIS  (IS/OOS split — no param re-optimisation)")
        lines.append(f"Train: {train_months} months | Test: {test_months} months")
        lines.append("=" * 80)

        window_start = self.start_date
        window_num   = 0
        oos_retentions: list[float] = []

        while window_start < self.end_date:
            train_end = window_start + timedelta(days=train_months * 30)
            test_end  = train_end    + timedelta(days=test_months  * 30)

            if train_end >= self.end_date:
                break
            test_end = min(test_end, self.end_date)

            train_trades = [t for t in analysis_trades if window_start <= t.entry_time < train_end]
            test_trades  = [t for t in analysis_trades if train_end   <= t.entry_time < test_end]

            window_num += 1
            lines.append(
                f"\n── Window {window_num}: "
                f"Train {window_start.date()}→{train_end.date()} | "
                f"Test {train_end.date()}→{test_end.date()} ──"
            )

            tm = self._compute_metrics(train_trades)
            om = self._compute_metrics(test_trades)

            lines.append(
                f"  IN-SAMPLE:  {tm['total_trades']:>3} trades  "
                f"WR={tm['win_rate']:.1f}%  E(R)={tm['expectancy_r']:+.3f}  "
                f"Med R={tm['median_r']:+.3f}  PF={tm['profit_factor']:.2f}  "
                f"P&L=${tm['total_pnl']:+,.2f}"
            )
            lines.append(
                f"  OUT-SAMPLE: {om['total_trades']:>3} trades  "
                f"WR={om['win_rate']:.1f}%  E(R)={om['expectancy_r']:+.3f}  "
                f"Med R={om['median_r']:+.3f}  PF={om['profit_factor']:.2f}  "
                f"P&L=${om['total_pnl']:+,.2f}"
            )

            if tm["expectancy_r"] > 0:
                if om["expectancy_r"] > 0:
                    ret = om["expectancy_r"] / tm["expectancy_r"] * 100
                    oos_retentions.append(ret)
                    lines.append(f"  RETENTION:  {ret:.0f}% of IS expectancy  ✅")
                else:
                    oos_retentions.append(0.0)
                    lines.append("  RETENTION:  ⚠️  OOS expectancy negative — edge decay in this window")
            else:
                lines.append("  RETENTION:  n/a (IS expectancy ≤ 0)")

            window_start = train_end

        if oos_retentions:
            avg_ret = float(np.mean(oos_retentions))
            lines.append(f"\n── AVERAGE OOS RETENTION: {avg_ret:.0f}% across {len(oos_retentions)} windows")
            if avg_ret >= 70:
                lines.append("   ✅ Edge is robust across time periods.")
            elif avg_ret >= 40:
                lines.append("   🟡 Partial edge decay — consider regime filtering.")
            else:
                lines.append("   🔴 Significant OOS decay — system may be over-fit to IS data.")

        lines.append("\n" + "=" * 80)
        output = "\n".join(lines)
        print(output)
        return output

    # =========================================================================
    # REGIME BREAKDOWN  [NEW]
    # =========================================================================

    def regime_breakdown(self) -> pd.DataFrame:
        """
        Break down P&L, win rate, and E(R) by regime_at_entry.

        Answers: 'Which regimes are actually profitable? Am I losing money
        trading WEAK_TRENDING or RANGING_CLEAR regimes?'
        """
        analysis_trades = self._analysis_trades()
        if not analysis_trades:
            return pd.DataFrame()

        df = self.to_dataframe(include_backtest_end=False)
        rows = []
        for regime, grp in df.groupby("regime_at_entry"):
            winners = grp[grp["pnl"] > 0]
            rows.append({
                "regime":     regime,
                "trades":     len(grp),
                "win_rate":   len(winners) / len(grp) * 100,
                "total_pnl":  grp["pnl"].sum(),
                "avg_pnl":    grp["pnl"].mean(),
                "avg_r":      grp["r_multiple"].mean(),
                "median_r":   grp["r_multiple"].median(),
            })
        result = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
        return result.reset_index(drop=True)

    # =========================================================================
    # DURATION STATS  [NEW]
    # =========================================================================

    def duration_stats(self) -> pd.DataFrame:
        """
        Compute average and median trade duration (minutes) by strategy
        and by exit reason.

        Useful for spotting strategies that hold too long relative to their
        intended timeframe (e.g. R3 should be <30 min; S1 should be <4 hours).
        """
        analysis_trades = self._analysis_trades()
        if not analysis_trades:
            return pd.DataFrame()

        df = self.to_dataframe(include_backtest_end=False)
        df["duration_min"] = (
            pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])
        ).dt.total_seconds() / 60.0

        by_strat = (
            df.groupby("strategy")["duration_min"]
            .agg(avg_min="mean", median_min="median", max_min="max", count="count")
            .reset_index()
            .sort_values("avg_min", ascending=False)
        )
        return by_strat

    # =========================================================================
    # ROLLING EXPECTANCY  [NEW]
    # =========================================================================

    def rolling_expectancy(self, window: int = 20) -> pd.DataFrame:
        """
        Compute rolling E(R) over a sliding window of trades.

        A stable, consistently positive E(R) line indicates a robust edge.
        Oscillation around zero suggests regime-dependent or noisy edge.

        Args:
            window: Number of trades per rolling window (default 20).

        Returns:
            DataFrame with columns: trade_num, entry_time, rolling_e_r
        """
        analysis_trades = self._analysis_trades()
        if len(analysis_trades) < window:
            return pd.DataFrame()

        rows = []
        r_vals = [t.r_multiple for t in analysis_trades]
        times  = [t.entry_time for t in analysis_trades]

        for i in range(window - 1, len(r_vals)):
            window_r = r_vals[i - window + 1 : i + 1]
            rows.append({
                "trade_num":    i + 1,
                "entry_time":   times[i],
                "rolling_e_r":  float(np.mean(window_r)),
            })

        return pd.DataFrame(rows)

    # =========================================================================
    # ENHANCED MONTHLY RETURNS  [FIXED]
    # =========================================================================

    def monthly_returns(self) -> pd.DataFrame:
        """
        Monthly P&L breakdown.

        ENHANCED: now includes win_rate per month, month_profitable flag,
        and cumulative P&L — critical for spotting seasonal patterns and
        consecutive losing months.
        """
        analysis_trades = self._analysis_trades()
        if not analysis_trades:
            return pd.DataFrame()

        df = self.to_dataframe(include_backtest_end=False)
        df["month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M")

        rows = []
        for month, grp in df.groupby("month"):
            winners = grp[grp["pnl"] > 0]
            rows.append({
                "month":       str(month),
                "trades":      len(grp),
                "pnl":         grp["pnl"].sum(),
                "win_rate":    len(winners) / len(grp) * 100 if len(grp) > 0 else 0.0,
                "avg_r":       grp["r_multiple"].mean(),
                "median_r":    grp["r_multiple"].median(),
                "profitable":  grp["pnl"].sum() > 0,
            })

        monthly = pd.DataFrame(rows)
        monthly["cum_pnl"]      = monthly["pnl"].cumsum()
        profitable_months       = monthly["profitable"].sum()
        total_months            = len(monthly)

        logger.info(
            f"Monthly returns: {profitable_months}/{total_months} profitable months "
            f"({profitable_months/total_months*100:.0f}%)"
        )
        return monthly

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def to_dataframe(self, include_backtest_end: bool = True) -> pd.DataFrame:
        """Convert trades to a pandas DataFrame for further analysis."""
        trades = self.trades if include_backtest_end else self._analysis_trades()
        if not trades:
            return pd.DataFrame()
        records = []
        for t in trades:
            records.append({
                "strategy":        t.strategy,
                "direction":       t.direction,
                "entry_price":     t.entry_price,
                "exit_price":      t.exit_price,
                "entry_time":      t.entry_time,
                "exit_time":       t.exit_time,
                "lots":            t.lots,
                "pnl":             t.pnl,
                "pnl_gross":       t.pnl_gross,
                "r_multiple":      t.r_multiple,
                "exit_reason":     t.exit_reason,
                "regime_at_entry": t.regime_at_entry,
                "regime_at_exit":  t.regime_at_exit,
                "stop_original":   t.stop_original,
                "commission":      t.commission,
            })
        return pd.DataFrame(records)

    def equity_to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to a pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame([
            {"timestamp": ep.timestamp, "equity": ep.equity,
             "drawdown_pct": ep.drawdown_pct}
            for ep in self.equity_curve
        ])

    def strategy_breakdown(self) -> dict[str, dict]:
        """Returns metrics dict per strategy."""
        result: dict[str, dict] = {}
        strategy_trades: dict[str, list[TradeRecord]] = {}
        for trade in self._analysis_trades():
            strategy_trades.setdefault(trade.strategy, []).append(trade)
        for strat, s_trades in strategy_trades.items():
            result[strat] = self._compute_metrics(s_trades)
        return result

    def exit_reason_breakdown(self) -> pd.DataFrame:
        """Breakdown of trades by exit reason."""
        if not self.trades:
            return pd.DataFrame()
        df = self.to_dataframe(include_backtest_end=False)
        if df.empty:
            return pd.DataFrame()
        return df.groupby("exit_reason").agg(
            count=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_r=("r_multiple", "mean"),
        ).reset_index()

    # =========================================================================
    # EQUITY PLOT
    # =========================================================================

    def plot_equity(self, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve, drawdown, and rolling E(R).
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.gridspec as gridspec
        except ImportError:
            logger.warning("matplotlib not installed — cannot plot equity curve")
            return

        eq_df = self.equity_to_dataframe()
        if eq_df.empty:
            return

        roll_er = self.rolling_expectancy(window=20)

        n_rows  = 3 if not roll_er.empty else 2
        fig     = plt.figure(figsize=(15, 10))
        gs      = gridspec.GridSpec(
            n_rows, 1, height_ratios=[3, 1, 1][:n_rows], hspace=0.08
        )
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.plot(eq_df["timestamp"], eq_df["equity"],
                 color="steelblue", linewidth=1)
        ax1.axhline(y=self.initial_balance, color="gray",
                    linestyle="--", alpha=0.5, label="Initial balance")
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(
            f"Backtest  {self.start_date.date()} → {self.end_date.date()}  "
            f"| Return: {(eq_df['equity'].iloc[-1]-self.initial_balance)/self.initial_balance*100:+.1f}%  "
            f"| Sharpe: {self.compute_sharpe():+.2f}  "
            f"| Calmar: {self.compute_calmar():+.2f}"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        ax2.fill_between(eq_df["timestamp"],
                         -eq_df["drawdown_pct"] * 100, 0,
                         color="salmon", alpha=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=-12, color="red", linestyle=":",
                    alpha=0.7, label="KS6 limit (-12%)")
        ax2.legend(fontsize=8)

        if not roll_er.empty and n_rows == 3:
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.plot(roll_er["entry_time"], roll_er["rolling_e_r"],
                     color="darkorange", linewidth=1)
            ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax3.set_ylabel("Rolling E(R)\n(20 trades)")
            ax3.set_xlabel("Date")
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax2.set_xlabel("Date")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
