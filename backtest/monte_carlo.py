"""
backtest/monte_carlo.py — Risk-of-Ruin Monte Carlo Simulator.

Built on top of the backtesting framework. Takes trade results and runs
randomized simulations to estimate drawdown probability distributions.

Usage:
    # After backtest:
    from backtest.monte_carlo import RiskOfRuinSimulator
    sim = RiskOfRuinSimulator(trade_results, initial_balance=10000)
    basic_report = sim.run_basic()
    clustered_report = sim.run_clustered(cluster_probability=0.3)

    # From live trade history:
    sim = RiskOfRuinSimulator.from_live_trades(initial_balance=10000)
    report = sim.run_clustered()
"""
from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

import config

logger = logging.getLogger("backtest.monte_carlo")

_LINE = "-" * 62
_BAR  = "=" * 62


class RiskOfRuinSimulator:
    """
    Monte Carlo simulator for risk-of-ruin analysis.

    Takes a list of trade P&L values (net of commission) and runs
    N simulations with randomized trade ordering to build a
    distribution of max drawdowns and final equity outcomes.
    """

    def __init__(
        self,
        trade_pnls: list[float],
        initial_balance: float = 10_000.0,
    ):
        if not trade_pnls:
            raise ValueError("trade_pnls must be a non-empty list of P&L values")
        self.trade_pnls = list(trade_pnls)
        self.initial_balance = initial_balance
        self.n_trades = len(trade_pnls)

        # Pre-compute win/loss classification for clustered mode
        self._wins = [p for p in self.trade_pnls if p >= 0]
        self._losses = [p for p in self.trade_pnls if p < 0]
        self._win_rate = len(self._wins) / self.n_trades if self.n_trades > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUCTORS
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_backtest_results(cls, results, initial_balance: Optional[float] = None) -> "RiskOfRuinSimulator":
        """
        Create simulator from BacktestResults object.

        Args:
            results: BacktestResults instance (has .trades list of TradeRecord).
            initial_balance: Override balance. Defaults to results.initial_balance.
        """
        pnls = [t.pnl for t in results.trades]
        balance = initial_balance if initial_balance is not None else results.initial_balance
        return cls(pnls, initial_balance=balance)

    @classmethod
    def from_live_trades(cls, initial_balance: float = 10_000.0) -> "RiskOfRuinSimulator":
        """
        Create simulator from live trade history in system_state.trades.

        Fetches all closed trades and extracts net P&L values.
        """
        from engines.truth_engine import _closed_trades_query

        trades = _closed_trades_query()
        if not trades:
            raise ValueError("No closed trades found in system_state.trades")

        pnls = [float(t["pnl_net_dollars"]) for t in trades if t.get("pnl_net_dollars") is not None]
        if not pnls:
            raise ValueError("No trades with pnl_net_dollars found")

        logger.info(f"Loaded {len(pnls)} closed trades from live history")
        return cls(pnls, initial_balance=initial_balance)

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION ENGINES
    # ─────────────────────────────────────────────────────────────────────────

    def _simulate_equity_path(self, pnl_sequence: list[float]) -> dict:
        """
        Walk through a P&L sequence and compute equity path metrics.

        Returns dict with max_drawdown_pct, final_equity, hit_ruin.
        """
        equity = self.initial_balance
        peak = equity
        max_dd_pct = 0.0
        hit_ruin = False

        for pnl in pnl_sequence:
            equity += pnl
            if equity <= 0:
                hit_ruin = True
                max_dd_pct = 1.0
                break
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return {
            "max_drawdown_pct": max_dd_pct,
            "final_equity": max(equity, 0.0),
            "hit_ruin": hit_ruin,
        }

    def run_basic(self, n_simulations: int = 10_000) -> dict:
        """
        Basic Monte Carlo: shuffle trade P&L values randomly.

        Each simulation randomly reorders all trades and walks the
        equity curve to find max drawdown.

        Returns:
            dict with keys: max_drawdowns, final_equities, ruin_count,
                           n_simulations, mode.
        """
        logger.info(f"Running basic Monte Carlo ({n_simulations} sims, {self.n_trades} trades)")

        max_drawdowns = []
        final_equities = []
        ruin_count = 0

        for _ in range(n_simulations):
            shuffled = self.trade_pnls.copy()
            random.shuffle(shuffled)
            result = self._simulate_equity_path(shuffled)
            max_drawdowns.append(result["max_drawdown_pct"])
            final_equities.append(result["final_equity"])
            if result["hit_ruin"]:
                ruin_count += 1

        return {
            "max_drawdowns": max_drawdowns,
            "final_equities": final_equities,
            "ruin_count": ruin_count,
            "n_simulations": n_simulations,
            "mode": "basic_shuffle",
        }

    def run_clustered(
        self,
        cluster_probability: float = 0.3,
        n_simulations: int = 10_000,
    ) -> dict:
        """
        Clustered Monte Carlo: models regime-dependent win/loss clustering.

        With probability `cluster_probability`, the next trade outcome
        matches the previous one (win→win, loss→loss). This captures
        the serial correlation that basic shuffle misses — losing streaks
        and winning streaks tend to cluster in real markets.

        Args:
            cluster_probability: Probability that next trade matches
                                 previous outcome type (0.0 = pure random,
                                 1.0 = perfect clustering).
            n_simulations: Number of simulations to run.

        Returns:
            dict with keys: max_drawdowns, final_equities, ruin_count,
                           n_simulations, mode, cluster_probability.
        """
        if not self._wins and not self._losses:
            raise ValueError("Need at least one win or loss trade for clustered simulation")

        logger.info(
            f"Running clustered Monte Carlo ({n_simulations} sims, "
            f"{self.n_trades} trades, cluster_p={cluster_probability})"
        )

        max_drawdowns = []
        final_equities = []
        ruin_count = 0

        for _ in range(n_simulations):
            sequence = self._generate_clustered_sequence(cluster_probability)
            result = self._simulate_equity_path(sequence)
            max_drawdowns.append(result["max_drawdown_pct"])
            final_equities.append(result["final_equity"])
            if result["hit_ruin"]:
                ruin_count += 1

        return {
            "max_drawdowns": max_drawdowns,
            "final_equities": final_equities,
            "ruin_count": ruin_count,
            "n_simulations": n_simulations,
            "mode": "clustered",
            "cluster_probability": cluster_probability,
        }

    def _generate_clustered_sequence(self, cluster_prob: float) -> list[float]:
        """
        Generate a trade sequence with win/loss clustering.

        First trade is drawn randomly based on historical win rate.
        Subsequent trades: with probability cluster_prob, draw from
        the same pool (win or loss) as the previous trade; otherwise
        draw from the opposite pool.
        """
        sequence = []
        prev_was_win = random.random() < self._win_rate

        for _ in range(self.n_trades):
            # Decide if this trade clusters with previous
            if random.random() < cluster_prob:
                # Cluster: same outcome type as previous
                use_wins = prev_was_win
            else:
                # No cluster: draw based on base win rate
                use_wins = random.random() < self._win_rate

            # Draw a random P&L from the appropriate pool
            if use_wins and self._wins:
                pnl = random.choice(self._wins)
            elif not use_wins and self._losses:
                pnl = random.choice(self._losses)
            elif self._wins:
                pnl = random.choice(self._wins)
            else:
                pnl = random.choice(self._losses)

            sequence.append(pnl)
            prev_was_win = pnl >= 0

        return sequence

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def analyze_results(results: dict) -> dict:
        """
        Compute and print statistics from simulation results.

        Args:
            results: dict returned by run_basic() or run_clustered().

        Returns:
            dict with computed statistics.
        """
        dd_array = np.array(results["max_drawdowns"])
        eq_array = np.array(results["final_equities"])
        n_sims = results["n_simulations"]
        ruin_count = results["ruin_count"]

        ks6_limit = getattr(config, "KS6_DRAWDOWN_LIMIT_PCT", 0.12)

        median_dd = float(np.median(dd_array))
        p95_dd = float(np.percentile(dd_array, 95))
        p99_dd = float(np.percentile(dd_array, 99))
        prob_hit_ks6 = float(np.mean(dd_array >= ks6_limit))
        prob_ruin = ruin_count / n_sims if n_sims > 0 else 0.0
        median_final_eq = float(np.median(eq_array))
        mean_final_eq = float(np.mean(eq_array))

        mode_label = results.get("mode", "unknown")
        cluster_p = results.get("cluster_probability")

        stats = {
            "mode": mode_label,
            "cluster_probability": cluster_p,
            "n_simulations": n_sims,
            "median_max_drawdown_pct": median_dd,
            "p95_max_drawdown_pct": p95_dd,
            "p99_max_drawdown_pct": p99_dd,
            "prob_hit_ks6": prob_hit_ks6,
            "ks6_limit_pct": ks6_limit,
            "prob_ruin": prob_ruin,
            "median_final_equity": median_final_eq,
            "mean_final_equity": mean_final_eq,
        }

        # Print report
        header = f"Monte Carlo Analysis — {mode_label}"
        if cluster_p is not None:
            header += f" (cluster_p={cluster_p})"

        print(f"\n{_BAR}")
        print(f"  {header}")
        print(_BAR)
        print(f"  Simulations:           {n_sims:,}")
        print(f"  Median Max Drawdown:   {median_dd:.2%}")
        print(f"  95th %ile Drawdown:    {p95_dd:.2%}")
        print(f"  99th %ile Drawdown:    {p99_dd:.2%}")
        print(f"  Prob(DD ≥ KS6 {ks6_limit:.0%}):  {prob_hit_ks6:.2%}")
        print(f"  Prob(Ruin):            {prob_ruin:.4%}")
        print(f"  Median Final Equity:   ${median_final_eq:,.2f}")
        print(f"  Mean Final Equity:     ${mean_final_eq:,.2f}")

        # Decision support
        if prob_hit_ks6 > 0.15:
            print(f"\n  ⚠️  WARNING: {prob_hit_ks6:.1%} probability of hitting KS6.")
            print(f"     Phase 2 scaling NOT recommended until this drops below 15%.")
            stats["phase2_recommendation"] = "HOLD"
        else:
            print(f"\n  ✅  KS6 risk acceptable ({prob_hit_ks6:.1%} < 15%).")
            stats["phase2_recommendation"] = "OK"

        print(_LINE)

        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # FULL REPORT
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_report(self, n_simulations: int = 10_000) -> dict:
        """
        Run basic + clustered (30% and 50%) simulations and print combined report.

        Returns:
            dict with keys: basic, clustered_30, clustered_50 — each containing
            the analyzed statistics.
        """
        print(f"\n{'#' * 62}")
        print(f"  RISK-OF-RUIN MONTE CARLO REPORT")
        print(f"  {self.n_trades} trades | ${self.initial_balance:,.2f} initial balance")
        print(f"  Win rate: {self._win_rate:.1%} | "
              f"Avg win: ${np.mean(self._wins):,.2f} | "
              f"Avg loss: ${np.mean(self._losses):,.2f}"
              if self._wins and self._losses else
              f"  Win rate: {self._win_rate:.1%}")
        print(f"{'#' * 62}")

        # Basic shuffle
        basic_results = self.run_basic(n_simulations=n_simulations)
        basic_stats = self.analyze_results(basic_results)

        # Clustered 30%
        clustered_30_results = self.run_clustered(
            cluster_probability=0.3, n_simulations=n_simulations
        )
        clustered_30_stats = self.analyze_results(clustered_30_results)

        # Clustered 50%
        clustered_50_results = self.run_clustered(
            cluster_probability=0.5, n_simulations=n_simulations
        )
        clustered_50_stats = self.analyze_results(clustered_50_results)

        # Summary comparison
        print(f"\n{_BAR}")
        print("  COMPARISON SUMMARY")
        print(_BAR)
        print(f"  {'Metric':<28} {'Basic':>10} {'Clust 30%':>10} {'Clust 50%':>10}")
        print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'Median Max DD':<28} "
              f"{basic_stats['median_max_drawdown_pct']:>9.2%} "
              f"{clustered_30_stats['median_max_drawdown_pct']:>9.2%} "
              f"{clustered_50_stats['median_max_drawdown_pct']:>9.2%}")
        print(f"  {'95th %ile DD':<28} "
              f"{basic_stats['p95_max_drawdown_pct']:>9.2%} "
              f"{clustered_30_stats['p95_max_drawdown_pct']:>9.2%} "
              f"{clustered_50_stats['p95_max_drawdown_pct']:>9.2%}")
        print(f"  {'P(DD ≥ KS6)':<28} "
              f"{basic_stats['prob_hit_ks6']:>9.2%} "
              f"{clustered_30_stats['prob_hit_ks6']:>9.2%} "
              f"{clustered_50_stats['prob_hit_ks6']:>9.2%}")
        print(f"  {'P(Ruin)':<28} "
              f"{basic_stats['prob_ruin']:>9.4%} "
              f"{clustered_30_stats['prob_ruin']:>9.4%} "
              f"{clustered_50_stats['prob_ruin']:>9.4%}")
        print(f"  {'Median Final Equity':<28} "
              f"${basic_stats['median_final_equity']:>8,.0f} "
              f"${clustered_30_stats['median_final_equity']:>8,.0f} "
              f"${clustered_50_stats['median_final_equity']:>8,.0f}")
        print(_LINE)

        # Overall recommendation based on worst case (clustered 50%)
        worst_ks6 = clustered_50_stats["prob_hit_ks6"]
        if worst_ks6 > 0.15:
            print(f"\n  🔴 OVERALL: Phase 2 scaling NOT recommended.")
            print(f"     Worst-case KS6 probability: {worst_ks6:.1%} (clustered 50%)")
        elif worst_ks6 > 0.10:
            print(f"\n  🟡 CAUTION: KS6 risk elevated at {worst_ks6:.1%} (clustered 50%).")
            print(f"     Consider conservative position sizing for Phase 2.")
        else:
            print(f"\n  🟢 CLEAR: KS6 risk acceptable across all scenarios.")
            print(f"     Phase 2 scaling can proceed.")

        return {
            "basic": basic_stats,
            "clustered_30": clustered_30_stats,
            "clustered_50": clustered_50_stats,
        }
