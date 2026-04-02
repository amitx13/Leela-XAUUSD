"""
engines/truth_engine.py - Layer 6: Truth Engine

OBSERVES everything. CHANGES nothing.

Spec refs:
  Part 11  - Full trade schema + Truth Engine mandate
  Part 9   - Phase gates (WR>45%, exp>+0.15R, maxDD<15%, Sharpe>1.0 Phase 3)
  ADD-3    - Conviction level observation (A+ vs STANDARD delta after 50 trades)
  COMP-3   - All logging via log_event(name, **kwargs)

Functions required by risk_engine.py:
  get_live_trade_count()        -> int
  get_max_drawdown_pct()        -> float
  get_rolling_sharpe(n)         -> float

CLI:
  python main.py --weekly       -> weekly_review_report() then exit
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pytz

import config
from db.connection import engine
from utils.logger import log_event, log_warning
from sqlalchemy import text

_LINE = "-" * 62
_BAR  = "=" * 62


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL QUERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _closed_trades_query(last_n: Optional[int] = None) -> list[dict]:
    """
    Return closed trades (exit_time IS NOT NULL) as list of dicts.
    Ordered by exit_time DESC. Optional last_n limit.
    """
    limit_clause = f"LIMIT {last_n}" if last_n else ""
    sql = text(f"""
        SELECT
            trade_id, signal_type, strategy_version,
            direction, entry_time, exit_time,
            entry_price, exit_price,
            stop_price_original, lot_size,
            r_multiple, pnl_gross_dollars AS gross_pnl,
            commission_entry, commission_exit, total_commission, pnl_net_dollars,
            regime_at_entry, session AS session_at_entry,
            macro_bias AS macro_bias_at_entry, macro_proxy_at_entry,
            dxy_corr_at_entry, conviction_level, signal_type,
            macro_boost_at_entry,
            partial_exit_done, be_activated,
            spread_at_entry, slippage_points,
            asian_range_size_pts AS range_size_at_entry
        FROM system_state.trades
        WHERE exit_time IS NOT NULL
        ORDER BY exit_time DESC
        {limit_clause}
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        log_warning("TRUTH_ENGINE_QUERY_FAILED", error=str(e))
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC - called by risk_engine.py
# ─────────────────────────────────────────────────────────────────────────────

def get_live_trade_count() -> int:
    """
    Count of fully closed trades.
    Used by calculate_lot_size() for Phase 1 vs Phase 2 risk scaling.
    Returns 0 on any DB error - safe fallback keeps system in Phase 1.
    """
    sql = text("SELECT COUNT(*) FROM system_state.trades WHERE exit_time IS NOT NULL")
    try:
        with engine.connect() as conn:
            result = conn.execute(sql).scalar()
        return int(result or 0)
    except Exception as e:
        log_warning("GET_LIVE_TRADE_COUNT_FAILED", error=str(e))
        return 0


def get_max_drawdown_pct() -> float:
    """
    Maximum drawdown percentage across all recorded performance rows.
    Used by Phase 2 gate: must stay below 15%.
    Returns 0.0 on error - conservative fallback.
    """
    sql = text("SELECT MAX(drawdown_pct) FROM system_state.performance WHERE drawdown_pct IS NOT NULL")
    try:
        with engine.connect() as conn:
            result = conn.execute(sql).scalar()
        return float(result or 0.0)
    except Exception as e:
        log_warning("GET_MAX_DD_FAILED", error=str(e))
        return 0.0


def get_rolling_sharpe(n: int = 50) -> float:
    """
    Per-trade Sharpe on last n closed trades:
        Sharpe = mean(R) / std(R)

    Annualisation not applied - this is a trade-series Sharpe.
    Used only for Phase 2 gate check (gate >= 0.8).
    Returns 0.0 if fewer than 10 trades or std == 0.
    """
    trades = _closed_trades_query(last_n=n)
    r_vals = [float(t["r_multiple"]) for t in trades if t["r_multiple"] is not None]

    if len(r_vals) < 10:
        return 0.0

    mean_r   = sum(r_vals) / len(r_vals)
    variance = sum((r - mean_r) ** 2 for r in r_vals) / len(r_vals)
    std_r    = math.sqrt(variance)

    if std_r == 0:
        return 0.0

    return round(mean_r / std_r, 4)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def _avg_r(trade_list: list[dict]) -> float:
    vals = [float(t["r_multiple"]) for t in trade_list if t["r_multiple"] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _breakdown_by(trades: list[dict], key: str) -> dict:
    groups: dict[str, list] = {}
    for t in trades:
        k = str(t.get(key, "UNKNOWN"))
        groups.setdefault(k, []).append(t)

    result = {}
    for label, group in groups.items():
        wins     = [t for t in group if t["r_multiple"] is not None and float(t["r_multiple"]) > 0]
        losses   = [t for t in group if t["r_multiple"] is not None and float(t["r_multiple"]) <= 0]
        r_vals   = [float(t["r_multiple"]) for t in group if t["r_multiple"] is not None]
        wr       = len(wins) / len(group) if group else 0.0
        avg_r    = sum(r_vals) / len(r_vals) if r_vals else 0.0
        exp      = sum(r_vals) / len(r_vals) if r_vals else 0.0  # Fix-2: mean(R)
        result[label] = {
            "trades": len(group), "win_rate": round(wr, 4),
            "avg_r": round(avg_r, 4), "expectancy": round(exp, 4),
        }
    return result


def _empty_stats() -> dict:
    return {
        "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0.0, "avg_r": 0.0, "expectancy": 0.0,
        "total_net_pnl": 0.0, "avg_commission": 0.0, "avg_slippage": 0.0,
        "by_signal_type": {}, "by_conviction": {}, "by_macro_boost": {},
    }


def get_trade_stats(last_n: Optional[int] = None) -> dict:
    """
    Full statistical breakdown.

    Returns:
        total_trades, wins, losses, win_rate, avg_r, expectancy,
        total_net_pnl, avg_commission, avg_slippage,
        by_signal_type, by_conviction, by_macro_boost
    """
    trades = _closed_trades_query(last_n=last_n)
    if not trades:
        return _empty_stats()

    wins   = [t for t in trades if t["r_multiple"] is not None and float(t["r_multiple"]) > 0]
    losses = [t for t in trades if t["r_multiple"] is not None and float(t["r_multiple"]) <= 0]
    r_vals = [float(t["r_multiple"]) for t in trades if t["r_multiple"] is not None]
    total  = len(trades)

    win_rate   = len(wins) / total if total else 0.0
    avg_r      = sum(r_vals) / len(r_vals) if r_vals else 0.0
    expectancy = sum(r_vals) / len(r_vals) if r_vals else 0.0  # Fix-2: mean(R)
    total_net = sum(float(t["pnl_net_dollars"] or 0) for t in trades)
    avg_comm   = sum(float(t["total_commission"] or 0) for t in trades) / total
    avg_slip   = sum(float(t["slippage_points"] or 0) for t in trades) / total

    return {
        "total_trades":   total,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(win_rate, 4),
        "avg_r":          round(avg_r, 4),
        "expectancy":     round(expectancy, 4),
        "total_net_pnl":  round(total_net, 2),
        "avg_commission": round(avg_comm, 4),
        "avg_slippage":   round(avg_slip, 2),
        "by_signal_type": _breakdown_by(trades, "signal_type"),
        "by_conviction":  _breakdown_by(trades, "conviction_level"),
        "by_macro_boost": _breakdown_by(trades, "macro_boost_at_entry"),
        "by_regime":      _breakdown_by(trades, "regime_at_entry"),
        "by_session":     _breakdown_by(trades, "session_at_entry"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE GATE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_phase_2_gate() -> tuple[bool, dict]:
    """
    Phase 1 -> Phase 2 promotion gate (spec Part 9):
        live_trades >= 50
        win_rate    >  0.45
        expectancy  >  +0.15R
        max_dd      <  0.15  (15%)

    Returns (eligible: bool, detail_dict).
    NEVER auto-promotes - human must update system_config.PHASE manually.
    """
    count  = get_live_trade_count()
    max_dd = get_max_drawdown_pct()
    stats  = get_trade_stats()

    gates = {
        "live_trades_50":   count >= config.PHASE_1_TRADE_GATE,
        "win_rate_45pct":   stats["win_rate"] > config.PHASE_2_WIN_RATE_MIN,
        "expectancy_0_15r": stats["expectancy"] > config.PHASE_2_EXPECTANCY_MIN,
        "max_dd_lt_15pct":  max_dd < config.PHASE_2_MAX_DD,
    }
    eligible = all(gates.values())
    detail   = {
        "eligible": eligible, "gates": gates,
        "live_trades": count, "win_rate": stats["win_rate"],
        "expectancy": stats["expectancy"], "max_dd": max_dd,
    }
    log_event("PHASE_2_GATE_CHECK",
              eligible=eligible, live_trades=count,
              win_rate=stats["win_rate"], expectancy=stats["expectancy"],
              max_dd=max_dd)
    return eligible, detail


# ─────────────────────────────────────────────────────────────────────────────
# CONVICTION + MACRO DELTA  (ADD-3 / observation mode)
# ─────────────────────────────────────────────────────────────────────────────

def _conviction_delta() -> Optional[float]:
    """
    A+ expectancy minus STANDARD expectancy.
    Returns None if either bucket < 10 trades.
    Spec: if delta > 8pp after 50 trades -> promote conviction to active modifier.
    """
    breakdown = get_trade_stats()["by_conviction"]
    a_plus    = breakdown.get("A_PLUS", {})
    standard  = breakdown.get("STANDARD", {})
    if a_plus.get("trades", 0) < 10 or standard.get("trades", 0) < 10:
        return None
    return round(a_plus["expectancy"] - standard["expectancy"], 4)


def _macro_boost_delta() -> Optional[float]:
    """
    macro_boost=True expectancy minus macro_boost=False expectancy.
    Returns None if either bucket < 10 trades.
    Spec: if never hits 8pp delta after 50 trades -> switch TLT -> TIP.
    """
    breakdown = get_trade_stats()["by_macro_boost"]
    boosted   = breakdown.get("True", {})
    plain     = breakdown.get("False", {})
    if boosted.get("trades", 0) < 10 or plain.get("trades", 0) < 10:
        return None
    return round(boosted["expectancy"] - plain["expectancy"], 4)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION ITEMS
# ─────────────────────────────────────────────────────────────────────────────

def _generate_action_items(count, stats, max_dd, conv_delta, macro_delta, ph2_ok) -> list[str]:
    items = []
    if count == 0:
        items.append("[PHASE 0] System running, waiting for first signal conditions")
        items.append("[PHASE 0] Monitor regime_log - confirm NO_TRADE transitions")
        items.append("[PHASE 0] Verify ATR calibration thresholds vs live market")
        return items
    if stats["win_rate"] < 0.35 and count >= 10:
        items.append("[RED] Win rate < 35% - do NOT increase risk - review signal logic")
    if stats["expectancy"] < 0:
        items.append("[RED] Negative expectancy - halt live trading pending review")
    if max_dd > 0.10:
        items.append(f"[WARN] Drawdown {max_dd*100:.1f}% - approaching 15% gate")
    if max_dd >= 0.08:
        items.append("[RED] KS6 circuit breaker active or imminent - written review required")
    if count >= config.PHASE_1_TRADE_GATE and ph2_ok:
        items.append("[PROMOTE] Phase 2 gate PASSED - set PHASE=2 in system_config after review")
    if conv_delta is not None and conv_delta > 0.08:
        items.append("[INFO] Conviction delta >8pp - A+ conditions measurably better")
    if macro_delta is not None and abs(macro_delta) < 0.08 and count >= 50:
        items.append("[INFO] Macro delta <8pp at 50 trades - consider switching TLT -> TIP")
    if not items:
        items.append("[OK] No action items - system operating normally")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY REVIEW REPORT  (python main.py --weekly)
# ─────────────────────────────────────────────────────────────────────────────

def weekly_review_report() -> None:
    """
    Full Truth Engine weekly review.
    Printed to stdout + logged. Called by main.py --weekly.

    Sections:
      1. Overall performance (win rate, expectancy, Sharpe)
      2. Phase gate status
      3. Per-signal-type breakdown
      4. Conviction delta (A+ vs STANDARD)
      5. Macro boost delta (observation mode)
      6. Last 10 trades
      7. Action items
    """
    ist         = pytz.timezone("Asia/Kolkata")
    now         = datetime.now(ist)
    count       = get_live_trade_count()
    stats       = get_trade_stats()
    max_dd      = get_max_drawdown_pct()
    sharpe      = get_rolling_sharpe(50)
    ph2_ok, ph2 = check_phase_2_gate()
    conv_delta  = _conviction_delta()
    macro_delta = _macro_boost_delta()
    recent      = _closed_trades_query(last_n=10)

    def gate(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print("")
    print("+" + _BAR + "+")
    print("|  TRUTH ENGINE -- WEEKLY REVIEW" + " " * 30 + "|")
    print("|  " + now.strftime("%Y-%m-%d  %H:%M IST") + " " * 40 + "|")
    print("+" + _BAR + "+")

    print("")
    print(_LINE)
    print(f"  OVERALL PERFORMANCE  (all {count} closed trades)")
    print(_LINE)
    if count == 0:
        print("  No closed trades yet - system in Phase 0 observation mode.")
    else:
        print(f"  Win rate        : {stats['win_rate']*100:5.1f}%   (gate >45.0%)")
        print(f"  Expectancy      : {stats['expectancy']:+.4f}R  (gate >+0.1500R)")
        print(f"  Avg R / trade   : {stats['avg_r']:+.4f}R")
        print(f"  Total net P&L   : ${stats['total_net_pnl']:,.2f}")
        print(f"  Max drawdown    : {max_dd*100:5.2f}%   (gate <15.00%)")
        print(f"  Rolling Sharpe  : {sharpe:.4f}    (Phase 3 gate >1.00)")
        print(f"  Avg commission  : ${stats['avg_commission']:.4f}/trade")
        print(f"  Avg slippage    : {stats['avg_slippage']:.2f} pts/trade")

    print("")
    print(_LINE)
    print("  PHASE GATE STATUS")
    print(_LINE)
    print(f"  Phase 0 -> 1 :  50 live trades required   [{count}/{config.PHASE_1_TRADE_GATE}]")
    if count >= config.PHASE_1_TRADE_GATE:
        g = ph2["gates"]
        print("  Phase 1 -> 2 :")
        print(f"    [{gate(g['win_rate_45pct'])}]  Win rate {stats['win_rate']*100:.1f}% > 45%")
        print(f"    [{gate(g['expectancy_0_15r'])}]  Expectancy {stats['expectancy']:+.4f}R > +0.15R")
        print(f"    [{gate(g['max_dd_lt_15pct'])}]  Max DD {max_dd*100:.2f}% < 15%")
        if ph2_ok:
            print("  >>> ELIGIBLE -- manually update PHASE in system_config <<<")
        else:
            print("  NOT YET ELIGIBLE")
    else:
        print(f"  Phase 1 -> 2 :  LOCKED (need {config.PHASE_1_TRADE_GATE} trades first)")

    print("")
    print(_LINE)
    print("  BY SIGNAL TYPE")
    print(_LINE)
    if not stats["by_signal_type"]:
        print("  No trades yet.")
    else:
        print(f"  {'Signal':<18} {'Trades':>6} {'WR%':>7} {'AvgR':>8} {'Expect':>8}")
        print(f"  {'-'*18} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")
        for sig, d in sorted(stats["by_signal_type"].items()):
            print(f"  {sig:<18} {d['trades']:>6} {d['win_rate']*100:>6.1f}% "
                  f"{d['avg_r']:>+8.4f} {d['expectancy']:>+8.4f}")

    print("")
    print(_LINE)
    print("  CONVICTION DELTA  (A+ vs STANDARD expectancy)")
    print(_LINE)
    if conv_delta is None:
        print("  Insufficient data -- need >=10 trades per bucket.")
        print("  A+/STANDARD split logged per trade for future analysis.")
    else:
        print(f"  Delta : {conv_delta:+.4f}R")
        if conv_delta > 0.08:
            print("  >8pp delta reached -- consider promoting conviction to active modifier")
        else:
            print("  Observation mode -- delta not yet significant")

    print("")
    print(_LINE)
    print("  MACRO BOOST DELTA  (TLT-assisted vs unassisted)")
    print(_LINE)
    if macro_delta is None:
        print("  Insufficient data -- need >=10 trades per bucket.")
    else:
        print(f"  Delta : {macro_delta:+.4f}R")
        if abs(macro_delta) < 0.08:
            print("  Delta <8pp -- consider switching proxy TLT -> TIP (real yields)")
        else:
            print("  Macro proxy showing meaningful signal")

    print("")
    print(_LINE)
    print(f"  LAST {min(10, len(recent))} TRADES")
    print(_LINE)
    if not recent:
        print("  No closed trades.")
    else:
        print(f"  {'Date':<12} {'Type':<16} {'Dir':<5} {'R':>7} {'Net$':>9} {'Conv':<12}")
        print(f"  {'-'*12} {'-'*16} {'-'*5} {'-'*7} {'-'*9} {'-'*12}")
        for t in recent:
            dt  = t["exit_time"].astimezone(ist).strftime("%m-%d %H:%M") if t["exit_time"] else "open"
            r   = f"{float(t['r_multiple']):+.2f}R" if t["r_multiple"] is not None else "   n/a"
            pnl = f"${float(t['pnl_net_dollars']):+.2f}" if t["pnl_net_dollars"] is not None else "    n/a"
            conv = str(t["conviction_level"] or "")
            print(f"  {dt:<12} {str(t['signal_type']):<16} {str(t['direction']):<5} "
                  f"{r:>7} {pnl:>9} {conv:<12}")

    print("")
    print(_LINE)
    print("  ACTION ITEMS")
    print(_LINE)
    for item in _generate_action_items(count, stats, max_dd, conv_delta, macro_delta, ph2_ok):
        print(f"  {item}")
    print("")
    print("+" + _BAR + "+")
    print("")

    log_event("WEEKLY_REVIEW_COMPLETE",
              total_trades=count,
              win_rate=stats["win_rate"],
              expectancy=stats["expectancy"],
              max_dd=max_dd,
              sharpe=sharpe,
              phase_2_eligible=ph2_ok)


def _ewma_win_rate(trades_ordered_oldest_first: list[dict], alpha: float = 0.05) -> float | None:
    """
    Exponentially weighted win rate.  alpha=0.05 gives half-life ≈ 14 trades.
    trades must be ordered oldest-first (exit_time ASC).
    Returns None if fewer than 10 trades.
    """
    if len(trades_ordered_oldest_first) < 10:
        return None
    n = len(trades_ordered_oldest_first)
    weights = [(1 - alpha) ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)
    weighted_wins = sum(
        w * (1.0 if t["outcome"] == "WIN" else 0.0)
        for w, t in zip(weights, trades_ordered_oldest_first)
    )
    return weighted_wins / total_weight


def get_conviction_delta() -> Optional[float]:
    """
    EXP-10 / 1A-1: Returns the EWMA win-rate delta (in decimal) between
    A_PLUS and OBSERVATION conviction trades.  Returns None if insufficient data.

    Uses exponentially weighted moving average (alpha=0.05, half-life ≈ 14 trades)
    so recent trades matter more than older ones — detects conviction decay faster
    than the previous equal-weighted AVG approach.

    Used by risk_engine to decide whether to promote conviction to active sizing.
    Requires at least 10 trades per bucket (enforced by _ewma_win_rate).
    """
    from db.connection import execute_query as _eq

    def _fetch_trades(conv_level: str) -> list[dict]:
        return _eq(
            """SELECT outcome, exit_time
               FROM system_state.trades
               WHERE exit_time IS NOT NULL
                 AND conviction_level = :conv_level
               ORDER BY exit_time ASC
               LIMIT 50""",
            {"conv_level": conv_level}
        )

    a_plus_trades = _fetch_trades("A_PLUS")
    obs_trades = _fetch_trades("OBSERVATION")

    a_plus_wr = _ewma_win_rate(a_plus_trades)
    obs_wr = _ewma_win_rate(obs_trades)

    if a_plus_wr is None or obs_wr is None:
        return None

    delta = a_plus_wr - obs_wr  # positive = A+ is better
    log_event("CONVICTION_DELTA_COMPUTED",
              a_plus_wr=round(a_plus_wr, 3),
              observation_wr=round(obs_wr, 3),
              delta_pp=round(delta * 100, 1),
              method="EWMA_alpha_0.05")
    return delta


# ─────────────────────────────────────────────────────────────────────────────
# EDGE DECAY MONITOR  (1A-3)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeDecayMonitor:
    """
    Detects degradation of the system's statistical edge by monitoring
    rolling win rate and expectancy across the last 100 and last 30 trades.

    Status levels:
        HEALTHY  — all metrics above warning thresholds
        WARNING  — one or more metrics below warning but above critical
        CRITICAL — one or more metrics below critical thresholds;
                   auto-reverts to Phase 1 risk as a safety measure

    Thresholds are read from config:
        EDGE_WARNING_EXPECTANCY   = 0.10   (below system min of 0.15R)
        EDGE_CRITICAL_EXPECTANCY  = 0.05   (near zero edge)
        EDGE_WARNING_WR           = 0.40   (below system min of 45%)
        EDGE_CRITICAL_WR          = 0.35   (severely degraded)
        EDGE_MIN_TRADES           = 30     (minimum trades for detection)
    """

    def __init__(self):
        self.status = "HEALTHY"
        self.alerts: list[str] = []
        self.actions: list[str] = []

    def check(self) -> dict:
        """
        Pull last 100 closed trades, compute rolling win rate and expectancy
        for the full window (100) and a fast window (30).

        Returns dict with keys: status, alerts, actions, metrics.
        """
        from db.connection import execute_query as _eq

        self.alerts = []
        self.actions = []
        self.status = "HEALTHY"

        trades = _eq(
            """SELECT outcome, r_multiple, exit_time
               FROM system_state.trades
               WHERE exit_time IS NOT NULL
                 AND r_multiple IS NOT NULL
               ORDER BY exit_time DESC
               LIMIT 100""",
            {}
        )

        if len(trades) < config.EDGE_MIN_TRADES:
            self.alerts.append(
                f"Insufficient trades for edge detection: {len(trades)}/{config.EDGE_MIN_TRADES}"
            )
            return self._result(trades_100={}, trades_30={})

        # ── Compute metrics for full window (up to 100 trades) ────────────
        metrics_100 = self._compute_metrics(trades)

        # ── Compute metrics for fast window (last 30 trades) ──────────────
        fast_trades = trades[:30]
        metrics_30 = self._compute_metrics(fast_trades)

        # ── Evaluate thresholds ───────────────────────────────────────────
        self._evaluate("100-trade", metrics_100)
        self._evaluate("30-trade", metrics_30)

        # ── If CRITICAL: auto-revert to Phase 1 risk ─────────────────────
        if self.status == "CRITICAL":
            self._auto_revert_phase_1()

        # ── Persist to edge_health_log ────────────────────────────────────
        self._persist(metrics_100, metrics_30)

        log_event("EDGE_DECAY_CHECK",
                  status=self.status,
                  wr_100=metrics_100.get("win_rate"),
                  exp_100=metrics_100.get("expectancy"),
                  wr_30=metrics_30.get("win_rate"),
                  exp_30=metrics_30.get("expectancy"),
                  alert_count=len(self.alerts))

        return self._result(metrics_100, metrics_30)

    @staticmethod
    def _compute_metrics(trades: list[dict]) -> dict:
        """Compute win rate and expectancy (mean R) for a list of trades."""
        if not trades:
            return {"win_rate": 0.0, "expectancy": 0.0, "trade_count": 0}

        r_vals = [float(t["r_multiple"]) for t in trades]
        wins = sum(1 for r in r_vals if r > 0)
        total = len(r_vals)

        win_rate = wins / total if total > 0 else 0.0
        expectancy = sum(r_vals) / total if total > 0 else 0.0

        return {
            "win_rate": round(win_rate, 4),
            "expectancy": round(expectancy, 4),
            "trade_count": total,
        }

    def _evaluate(self, window_label: str, metrics: dict) -> None:
        """Check metrics against warning/critical thresholds and update status."""
        wr = metrics.get("win_rate", 0.0)
        exp = metrics.get("expectancy", 0.0)

        # ── Critical checks ───────────────────────────────────────────────
        if exp < config.EDGE_CRITICAL_EXPECTANCY:
            self.status = "CRITICAL"
            self.alerts.append(
                f"CRITICAL [{window_label}]: expectancy {exp:+.4f}R "
                f"< {config.EDGE_CRITICAL_EXPECTANCY}R threshold"
            )
            self.actions.append("Auto-revert to Phase 1 risk (1.0% per trade)")

        if wr < config.EDGE_CRITICAL_WR:
            self.status = "CRITICAL"
            self.alerts.append(
                f"CRITICAL [{window_label}]: win rate {wr:.1%} "
                f"< {config.EDGE_CRITICAL_WR:.0%} threshold"
            )
            self.actions.append("Auto-revert to Phase 1 risk (1.0% per trade)")

        # ── Warning checks (only upgrade to WARNING, never downgrade from CRITICAL)
        if self.status != "CRITICAL":
            if exp < config.EDGE_WARNING_EXPECTANCY:
                self.status = "WARNING"
                self.alerts.append(
                    f"WARNING [{window_label}]: expectancy {exp:+.4f}R "
                    f"< {config.EDGE_WARNING_EXPECTANCY}R threshold"
                )
                self.actions.append("Review strategy parameters — edge may be degrading")

            if wr < config.EDGE_WARNING_WR:
                self.status = "WARNING"
                self.alerts.append(
                    f"WARNING [{window_label}]: win rate {wr:.1%} "
                    f"< {config.EDGE_WARNING_WR:.0%} threshold"
                )
                self.actions.append("Review recent trade quality and market conditions")

    def _auto_revert_phase_1(self) -> None:
        """Force system back to Phase 1 risk parameters as a safety measure."""
        from db.connection import execute_write as _ew
        try:
            _ew(
                """UPDATE public.system_config
                   SET value = '1', set_at = now(), set_by = 'EDGE_DECAY_AUTO'
                   WHERE key = 'BASE_RISK_PHASE'""",
                {}
            )
            self.actions.append("EXECUTED: Reverted BASE_RISK_PHASE to 1")
            log_warning("EDGE_DECAY_AUTO_REVERT",
                        message="Auto-reverted to Phase 1 risk due to CRITICAL edge decay")
        except Exception as e:
            log_warning("EDGE_DECAY_REVERT_FAILED", error=str(e))
            self.actions.append(f"FAILED to revert phase: {e}")

    def _persist(self, metrics_100: dict, metrics_30: dict) -> None:
        """Write check results to system_state.edge_health_log."""
        from db.connection import execute_write as _ew
        import json
        try:
            _ew(
                """INSERT INTO system_state.edge_health_log
                   (check_time, status,
                    wr_100, exp_100, trades_100,
                    wr_30, exp_30, trades_30,
                    alerts, actions)
                   VALUES (now(), :status,
                           :wr_100, :exp_100, :trades_100,
                           :wr_30, :exp_30, :trades_30,
                           :alerts, :actions)""",
                {
                    "status": self.status,
                    "wr_100": metrics_100.get("win_rate"),
                    "exp_100": metrics_100.get("expectancy"),
                    "trades_100": metrics_100.get("trade_count", 0),
                    "wr_30": metrics_30.get("win_rate"),
                    "exp_30": metrics_30.get("expectancy"),
                    "trades_30": metrics_30.get("trade_count", 0),
                    "alerts": json.dumps(self.alerts),
                    "actions": json.dumps(self.actions),
                }
            )
        except Exception as e:
            log_warning("EDGE_HEALTH_LOG_PERSIST_FAILED", error=str(e))

    def _result(self, trades_100: dict, trades_30: dict) -> dict:
        """Build the return dict."""
        return {
            "status": self.status,
            "alerts": list(self.alerts),
            "actions": list(self.actions),
            "metrics_100": trades_100,
            "metrics_30": trades_30,
        }


def daily_edge_check() -> dict:
    """
    Convenience function for daily edge health check.
    Called by the scheduler or weekly review.

    Returns dict with status, alerts, actions, and metrics.
    """
    monitor = EdgeDecayMonitor()
    result = monitor.check()

    for alert in result["alerts"]:
        log_warning("EDGE_DECAY_ALERT", message=alert)

    return result
