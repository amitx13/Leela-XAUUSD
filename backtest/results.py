"""
backtest/results.py — Backtest result analysis and reporting.

BacktestResults:
  - summary()              — human-readable performance summary
  - compute_max_dd()       — max drawdown (amount + %)
  - compute_sharpe()       — annualised Sharpe ratio
  - compute_sortino()      — annualised Sortino ratio
  - compute_calmar()       — Calmar ratio
  - walk_forward_report()  — IS/OOS split analysis
  - regime_breakdown()     — performance per regime
  - duration_stats()       — trade duration analysis
  - rolling_expectancy()   — rolling window expectancy
  - monthly_returns()      — monthly P&L with win rate + streak data [ENHANCED]
  - profit_concentration_check() — FIX-4: warn if >80% profit in ≤3 months [NEW]
  - to_dataframe()         — trades as pandas DataFrame
  - equity_to_dataframe()  — equity curve as pandas DataFrame
  - strategy_breakdown()   — per-strategy performance metrics
  - exit_reason_breakdown()— breakdown by exit reason
  - plot_equity()          — equity curve matplotlib chart

All timestamps are timezone-aware UTC (pytz.utc).
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy  as np
import pandas as pd

from backtest.models import TradeRecord, EquityPoint

logger = logging.getLogger("backtest.results")


class BacktestResults:
    """
    Aggregates and analyses results from a completed backtest run.
    """

    def __init__(
        self,
        trades:          list,
        equity_curve:    list,
        initial_balance: float = 10_000.0,
        final_balance:   float = 10_000.0,
        strategies:      list  = None,
        warmup_bars:     int   = 0,
    ):
        self.trades          = trades           # list[TradeRecord]
        self.equity_curve    = equity_curve     # list[EquityPoint]
        self.initial_balance = initial_balance
        self.final_balance   = final_balance
        self.strategies      = strategies or []
        self.warmup_bars     = warmup_bars

    def _analysis_trades(self) -> list:
        """Return only trades beyond the warmup period."""
        if self.warmup_bars <= 0:
            return self.trades
        if not self.equity_curve:
            return self.trades
        cutoff = self.equity_curve[self.warmup_bars].timestamp if len(self.equity_curve) > self.warmup_bars else None
        if cutoff is None:
            return self.trades
        return [t for t in self.trades if t.exit_time >= cutoff]

    def summary(self) -> str:
        """Return a human-readable performance summary string."""
        trades  = self._analysis_trades()
        winners = [t for t in trades if t.pnl > 0]
        losers  = [t for t in trades if t.pnl <= 0]

        net_pnl     = sum(t.pnl     for t in trades)
        gross_pnl   = sum(t.pnl_gross for t in trades)
        commission  = sum(t.commission for t in trades)
        win_rate    = len(winners) / len(trades) * 100 if trades else 0.0
        avg_win     = sum(t.pnl for t in winners) / len(winners) if winners else 0.0
        avg_loss    = sum(t.pnl for t in losers)  / len(losers)  if losers  else 0.0
        avg_r       = sum(t.r_multiple for t in trades) / len(trades) if trades else 0.0
        sharpe      = self.compute_sharpe()
        max_dd_amt, max_dd_pct = self.compute_max_dd()

        lines = [
            "=" * 60,
            "BACKTEST RESULTS SUMMARY",
            "=" * 60,
            f"  Trades          : {len(trades)}",
            f"  Win Rate        : {win_rate:.1f}%  ({len(winners)}W / {len(losers)}L)",
            f"  Net P&L         : ${net_pnl:>+10,.2f}",
            f"  Gross P&L       : ${gross_pnl:>+10,.2f}",
            f"  Commission paid : ${commission:>10,.2f}",
            f"  Avg Winner      : ${avg_win:>+10,.2f}",
            f"  Avg Loser       : ${avg_loss:>+10,.2f}",
            f"  Avg R-multiple  : {avg_r:>+8.3f}R",
            f"  Sharpe Ratio    : {sharpe:>8.3f}",
            f"  Max Drawdown    : ${max_dd_amt:,.2f}  ({max_dd_pct:.1f}%)",
            f"  Initial Balance : ${self.initial_balance:,.2f}",
            f"  Final Balance   : ${self.final_balance:,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _compute_metrics(self, trades: list) -> dict:
        if not trades:
            return {
                "n": 0, "win_rate": 0.0, "avg_r": 0.0, "median_r": 0.0,
                "avg_winner": 0.0, "avg_loser": 0.0, "profit_factor": 0.0,
                "net_pnl": 0.0,
            }
        winners = [t for t in trades if t.pnl > 0]
        losers  = [t for t in trades if t.pnl <= 0]
        r_vals  = [t.r_multiple for t in trades]

        gross_profit = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss   = abs(sum(t.pnl for t in losers)) if losers else 0.0

        return {
            "n":             len(trades),
            "win_rate":      len(winners) / len(trades) * 100,
            "avg_r":         float(np.mean(r_vals)),
            "median_r":      float(np.median(r_vals)),
            "avg_winner":    gross_profit / len(winners) if winners else 0.0,
            "avg_loser":     gross_loss   / len(losers)  if losers  else 0.0,
            "profit_factor": gross_profit / gross_loss   if gross_loss > 0 else float("inf"),
            "net_pnl":       sum(t.pnl for t in trades),
        }

    @staticmethod
    def _format_metrics(m: dict) -> list:
        return [
            f"  Trades        : {m['n']}",
            f"  Win Rate      : {m['win_rate']:.1f}%",
            f"  Avg R         : {m['avg_r']:+.3f}",
            f"  Median R      : {m['median_r']:+.3f}",
            f"  Avg Winner    : ${m['avg_winner']:,.2f}",
            f"  Avg Loser     : ${m['avg_loser']:,.2f}",
            f"  Profit Factor : {m['profit_factor']:.2f}",
            f"  Net P&L       : ${m['net_pnl']:+,.2f}",
        ]

    def compute_max_dd(self) -> tuple:
        """
        Compute maximum drawdown from the equity curve.
        Returns (max_dd_amount_usd, max_dd_pct).
        """
        if not self.equity_curve:
            return 0.0, 0.0

        equities = [ep.equity for ep in self.equity_curve]
        peak     = equities[0]
        max_dd   = 0.0
        max_dd_pct = 0.0

        for eq in equities:
            peak = max(peak, eq)
            dd   = peak - eq
            if dd > max_dd:
                max_dd     = dd
                max_dd_pct = dd / peak * 100 if peak > 0 else 0.0

        return round(max_dd, 2), round(max_dd_pct, 2)

    def _daily_pnl_series(self) -> pd.Series:
        """
        Build a daily P&L series from closed trades.
        Index is date, values are net P&L sums per day.
        """
        trades = self._analysis_trades()
        if not trades:
            return pd.Series(dtype=float)

        df = pd.DataFrame([
            {"date": t.exit_time.date(), "pnl": t.pnl}
            for t in trades
        ])
        daily = df.groupby("date")["pnl"].sum()

        # Fill missing trading days with 0
        if len(daily) > 1:
            idx   = pd.date_range(daily.index[0], daily.index[-1], freq="D")
            daily = daily.reindex(idx, fill_value=0.0)

        return daily

    def compute_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """
        Annualised Sharpe ratio using daily P&L returns.
        Returns 0.0 if insufficient data.
        """
        daily = self._daily_pnl_series()
        if len(daily) < 20:
            return 0.0

        # Convert to daily return %
        returns = daily / self.initial_balance
        excess  = returns - risk_free_rate / 252
        std     = excess.std()
        if std == 0:
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    def compute_sortino(self, risk_free_rate: float = 0.0, mar: float = 0.0) -> float:
        """
        Annualised Sortino ratio.
        Uses only downside deviation (returns below MAR).
        Returns 0.0 if insufficient data.
        """
        daily = self._daily_pnl_series()
        if len(daily) < 20:
            return 0.0

        returns  = daily / self.initial_balance
        excess   = returns - risk_free_rate / 252
        downside = excess[excess < mar]
        down_std = downside.std()
        if down_std == 0 or len(downside) == 0:
            return 0.0
        return float(excess.mean() / down_std * np.sqrt(252))

    def compute_calmar(self) -> float:
        """
        Calmar ratio = annualised return / max drawdown %.
        Returns 0.0 if drawdown is zero or data is insufficient.
        """
        daily = self._daily_pnl_series()
        if len(daily) < 20:
            return 0.0

        annual_return = (daily.sum() / self.initial_balance) * (252 / len(daily))
        _, max_dd_pct = self.compute_max_dd()
        if max_dd_pct == 0:
            return 0.0
        return round(annual_return / (max_dd_pct / 100), 3)

    def walk_forward_report(
        self,
        is_fraction:  float = 0.70,
        oos_fraction: float = 0.30,
    ) -> str:
        """
        Walk-forward IS/OOS split report.

        Splits trades into in-sample (first is_fraction) and
        out-of-sample (last oos_fraction) by exit time.

        Returns a formatted string comparing performance metrics.
        """
        trades = self._analysis_trades()
        if not trades:
            return "No trades to analyse."

        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        cutoff_idx    = int(len(sorted_trades) * is_fraction)
        is_trades     = sorted_trades[:cutoff_idx]
        oos_trades    = sorted_trades[cutoff_idx:]

        is_m  = self._compute_metrics(is_trades)
        oos_m = self._compute_metrics(oos_trades)

        lines = [
            "=" * 60,
            f"WALK-FORWARD REPORT  (IS={is_fraction:.0%} / OOS={oos_fraction:.0%})",
            "=" * 60,
            f"  IS  period: {is_trades[0].exit_time.date()} → {is_trades[-1].exit_time.date()}"
            if is_trades else "  IS  period: (no trades)",
            f"  OOS period: {oos_trades[0].exit_time.date()} → {oos_trades[-1].exit_time.date()}"
            if oos_trades else "  OOS period: (no trades)",
            "-" * 60,
            "  IN-SAMPLE",
        ] + self._format_metrics(is_m) + [
            "-" * 60,
            "  OUT-OF-SAMPLE",
        ] + self._format_metrics(oos_m) + [
            "=" * 60,
        ]

        # Degradation check
        if is_m["avg_r"] > 0 and oos_m["avg_r"] < is_m["avg_r"] * 0.6:
            lines.append("  ⚠  WARNING: OOS avg-R is <60% of IS avg-R — possible overfit.")
        if is_m["profit_factor"] > 1 and oos_m["profit_factor"] < 1:
            lines.append("  ⚠  WARNING: OOS profit factor < 1.0 — strategy loses money OOS.")

        return "\n".join(lines)

    def regime_breakdown(self) -> pd.DataFrame:
        """
        Performance breakdown by market regime at entry.

        Answers: 'Which regimes are actually profitable? Am I losing money
        in RANGING or TRENDING_WEAK regimes that I should filter out?'

        Returns a DataFrame with columns:
            regime, n, win_rate, avg_r, median_r, net_pnl, profit_factor
        """
        trades = self._analysis_trades()
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"regime": t.regime_at_entry, "pnl": t.pnl,
             "r": t.r_multiple, "win": t.pnl > 0}
            for t in trades
        ])

        rows = []
        for regime, grp in df.groupby("regime"):
            winners    = grp[grp["win"]]
            losers     = grp[~grp["win"]]
            gp         = winners["pnl"].sum()
            gl         = abs(losers["pnl"].sum())
            rows.append({
                "regime":        regime,
                "n":             len(grp),
                "win_rate":      round(len(winners) / len(grp) * 100, 1),
                "avg_r":         round(grp["r"].mean(), 3),
                "median_r":      round(grp["r"].median(), 3),
                "net_pnl":       round(grp["pnl"].sum(), 2),
                "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            })
        return pd.DataFrame(rows).sort_values("net_pnl", ascending=False)

    def duration_stats(self) -> pd.DataFrame:
        """
        Trade duration statistics.

        Returns DataFrame with columns:
            strategy, avg_duration_h, median_duration_h,
            max_duration_h, pct_open_24h
        """
        trades = self._analysis_trades()
        if not trades:
            return pd.DataFrame()

        rows = []
        strats = set(t.strategy for t in trades)
        for strat in sorted(strats):
            st = [t for t in trades if t.strategy == strat]
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600
                         for t in st]
            rows.append({
                "strategy":        strat,
                "n":               len(st),
                "avg_duration_h":  round(np.mean(durations), 2),
                "med_duration_h":  round(np.median(durations), 2),
                "max_duration_h":  round(max(durations), 2),
                "pct_open_24h":    round(sum(1 for d in durations if d >= 24)
                                         / len(durations) * 100, 1),
            })
        return pd.DataFrame(rows)

    def rolling_expectancy(self, window: int = 20) -> pd.DataFrame:
        """
        Rolling window expectancy (avg R per trade) over the trade sequence.

        Useful for spotting regime changes, skill decay, or mean-reversion
        in performance over time.

        Returns DataFrame:
            exit_time, rolling_avg_r, rolling_win_rate, rolling_net_pnl
        """
        trades = self._analysis_trades()
        if len(trades) < window:
            return pd.DataFrame()

        sorted_t = sorted(trades, key=lambda t: t.exit_time)
        rows     = []
        for i in range(window, len(sorted_t) + 1):
            window_trades = sorted_t[i - window:i]
            avg_r         = np.mean([t.r_multiple for t in window_trades])
            win_rate      = sum(1 for t in window_trades if t.pnl > 0) / window * 100
            net_pnl       = sum(t.pnl for t in window_trades)
            rows.append({
                "exit_time":       sorted_t[i - 1].exit_time,
                "rolling_avg_r":   round(avg_r, 4),
                "rolling_win_rate":round(win_rate, 1),
                "rolling_net_pnl": round(net_pnl, 2),
            })
        return pd.DataFrame(rows)

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
                "month":      str(month),
                "trades":     len(grp),
                "pnl":        grp["pnl"].sum(),
                "win_rate":   len(winners) / len(grp) * 100 if len(grp) > 0 else 0.0,
                "avg_r":      grp["r_multiple"].mean(),
                "median_r":   grp["r_multiple"].median(),
                "profitable": grp["pnl"].sum() > 0,
            })

        monthly = pd.DataFrame(rows)
        monthly["cum_pnl"]    = monthly["pnl"].cumsum()
        profitable_months     = monthly["profitable"].sum()
        total_months          = len(monthly)

        logger.info(
            f"Monthly returns: {profitable_months}/{total_months} profitable months "
            f"({profitable_months/total_months*100:.0f}%)"
        )
        return monthly

    def profit_concentration_check(self) -> dict:
        """
        FIX-4: Check whether profits are concentrated in ≤3 calendar months.

        Reuses monthly_returns() and emits a WARNING when the top-3 months
        account for >80% of total gross profit — indicating the backtest
        edge is not broadly distributed across time and therefore unlikely
        to be predictive in live trading.

        Call this after run() alongside monthly_returns() for a full
        time-period stability picture.

        Returns dict:
            top3_months          — [(year_month_str, pnl), ...]
            top3_pct_of_total    — float [0–1]
            total_profit_months  — int
            warning              — bool  (True = concentration detected)
        """
        monthly = self.monthly_returns()
        if monthly.empty:
            return {"warning": False, "top3_months": [],
                    "top3_pct_of_total": 0.0, "total_profit_months": 0}

        total_gross = monthly.loc[monthly["pnl"] > 0, "pnl"].sum()
        if total_gross <= 0:
            return {"warning": False, "top3_months": [],
                    "top3_pct_of_total": 0.0,
                    "total_profit_months": int((monthly["pnl"] > 0).sum())}

        top3      = monthly.nlargest(3, "pnl")
        top3_pct  = top3["pnl"].sum() / total_gross
        n_profit  = int((monthly["pnl"] > 0).sum())
        top3_list = list(zip(top3["month"].tolist(), top3["pnl"].round(2).tolist()))
        warning   = top3_pct > 0.80

        logger.warning(
            "TIME-PERIOD STABILITY: top-3 months = %.1f%% of total profit%s",
            top3_pct * 100,
            " — ⚠ CONCENTRATION RISK" if warning else " — OK",
        )
        return {
            "top3_months":        top3_list,
            "top3_pct_of_total":  round(top3_pct, 4),
            "total_profit_months": n_profit,
            "warning":            warning,
        }

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def to_dataframe(self, include_backtest_end: bool = True) -> pd.DataFrame:
        """Convert trades to a pandas DataFrame for further analysis."""
        trades = self._analysis_trades()
        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            rows.append({
                "strategy":      t.strategy,
                "direction":     t.direction,
                "entry_price":   t.entry_price,
                "exit_price":    t.exit_price,
                "entry_time":    t.entry_time,
                "exit_time":     t.exit_time,
                "lots":          t.lots,
                "pnl":           t.pnl,
                "pnl_gross":     t.pnl_gross,
                "r_multiple":    t.r_multiple,
                "exit_reason":   t.exit_reason,
                "regime_entry":  t.regime_at_entry,
                "regime_exit":   t.regime_at_exit,
                "commission":    t.commission,
            })
        df = pd.DataFrame(rows)
        if not include_backtest_end:
            df = df[df["exit_reason"] != "SESSION_CLOSE"]
        return df

    def equity_to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to a pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame([
            {"timestamp": ep.timestamp, "equity": ep.equity,
             "drawdown_pct": ep.drawdown_pct}
            for ep in self.equity_curve
        ])

    def strategy_breakdown(self) -> dict:
        """Per-strategy performance metrics."""
        trades = self._analysis_trades()
        breakdown = {}
        strats = set(t.strategy for t in trades)
        for strat in strats:
            st     = [t for t in trades if t.strategy == strat]
            m      = self._compute_metrics(st)
            breakdown[strat] = m
        return breakdown

    def exit_reason_breakdown(self) -> pd.DataFrame:
        """
        P&L and win rate broken down by exit reason
        (SL / TP / PARTIAL / BE / ATR_TRAIL / TIME_KILL / SESSION_CLOSE).
        """
        trades = self._analysis_trades()
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"reason": t.exit_reason, "pnl": t.pnl,
             "r": t.r_multiple, "win": t.pnl > 0}
            for t in trades
        ])
        rows = []
        for reason, grp in df.groupby("reason"):
            winners = grp[grp["win"]]
            rows.append({
                "exit_reason": reason,
                "n":           len(grp),
                "win_rate":    round(len(winners) / len(grp) * 100, 1),
                "avg_r":       round(grp["r"].mean(), 3),
                "net_pnl":     round(grp["pnl"].sum(), 2),
            })
        return pd.DataFrame(rows).sort_values("net_pnl", ascending=False)

    def plot_equity(
        self,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot equity curve using matplotlib.

        Args:
            save_path: If provided, saves the chart to this path (.png).
                       Otherwise calls plt.show().
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — cannot plot equity curve.")
            return

        eq_df = self.equity_to_dataframe()
        if eq_df.empty:
            logger.warning("No equity curve data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(eq_df["timestamp"], eq_df["equity"],
                 color="#2196F3", linewidth=1.2, label="Equity")
        ax1.axhline(y=self.initial_balance, color="gray",
                    linestyle="--", linewidth=0.8, alpha=0.7, label="Initial Balance")
        ax1.set_title("Backtest Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(eq_df["timestamp"], eq_df["drawdown_pct"],
                         color="#F44336", alpha=0.6)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Equity chart saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)
