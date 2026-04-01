"""
db/persistence.py — Critical state persistence and restore.

Implements Fix 3 (system_state_persistent table), C4 Fix (daily vs rolling
state separation), B6 Fix (continuous peak_equity tracking), and G8 Fix
(persist on every open/close/KS/regime change).

Call persist_critical_state() on:
  - Every trade OPEN     (G8 Fix)
  - Every trade CLOSE
  - Every KS fire
  - Every regime change
  - Every peak_equity update

RULE: consecutive_losses does NOT reset at midnight.
      A losing streak spanning overnight is still a losing streak.
"""
import pytz
from datetime import datetime, timedelta
from decimal import Decimal

import config
from state import validate_state_keys
from utils.logger import log_event, log_warning
from utils.mt5_client import get_mt5
from engines.truth_engine import get_live_trade_count, get_max_drawdown_pct, get_rolling_sharpe
from db.connection import execute_write, execute_query, engine
from sqlalchemy import text


# ─────────────────────────────────────────────────────────────────────────────
# PERSIST 
# ─────────────────────────────────────────────────────────────────────────────

def persist_critical_state(state: dict) -> None:
    ist = pytz.timezone("Asia/Kolkata")
    state_date_today = datetime.now(ist).date()
    execute_write(
        """
        INSERT INTO system_state.system_state_persistent (
            state_date, saved_at,
            consecutive_m5_losses, s1_family_attempts_today,
            s1f_attempts_today, daily_net_pnl_pct,
            daily_commission_paid, consecutive_losses,
            peak_equity, failed_breakout_flag,
            failed_breakout_direction, last_s1_direction,
            stop_hunt_detected, trading_enabled,
            shutdown_reason, current_regime,
            regime_calculated_at, size_multiplier,
            ks4_reduced_trades_remaining, reversal_family_occupied,
            s1b_pending_ticket, s1d_ema_touched_today,
            s1d_fired_today, s3_sweep_candle_time,
            r3_fired_today, r3_armed,
            s4_ema_touched, s4_fired_today,
            s5_compression_confirmed, s5_fired_today,
            london_session_high, london_session_low, d1_atr_14,
            ks7_pre_event_price, spread_multiplier,
            spread_elevated_reading_count, dxy_ewma_variance
        ) VALUES (
            :state_date, :saved_at,
            :consecutive_m5_losses, :s1_family_attempts_today,
            :s1f_attempts_today, :daily_net_pnl_pct,
            :daily_commission_paid, :consecutive_losses,
            :peak_equity, :failed_breakout_flag,
            :failed_breakout_direction, :last_s1_direction,
            :stop_hunt_detected, :trading_enabled,
            :shutdown_reason, :current_regime,
            :regime_calculated_at, :size_multiplier,
            :ks4_reduced_trades_remaining, :reversal_family_occupied,
            :s1b_pending_ticket, :s1d_ema_touched_today,
            :s1d_fired_today, :s3_sweep_candle_time,
            :r3_fired_today, :r3_armed,
            :s4_ema_touched, :s4_fired_today,
            :s5_compression_confirmed, :s5_fired_today,
            :london_session_high, :london_session_low, :d1_atr_14,
            :ks7_pre_event_price, :spread_multiplier,
            :spread_elevated_reading_count, :dxy_ewma_variance
        )
        ON CONFLICT (state_date) DO UPDATE SET
            saved_at                    = EXCLUDED.saved_at,
            consecutive_m5_losses       = EXCLUDED.consecutive_m5_losses,
            s1_family_attempts_today    = EXCLUDED.s1_family_attempts_today,
            s1f_attempts_today          = EXCLUDED.s1f_attempts_today,
            daily_net_pnl_pct           = EXCLUDED.daily_net_pnl_pct,
            daily_commission_paid       = EXCLUDED.daily_commission_paid,
            consecutive_losses          = EXCLUDED.consecutive_losses,
            peak_equity                 = EXCLUDED.peak_equity,
            failed_breakout_flag        = EXCLUDED.failed_breakout_flag,
            failed_breakout_direction   = EXCLUDED.failed_breakout_direction,
            last_s1_direction           = EXCLUDED.last_s1_direction,
            stop_hunt_detected          = EXCLUDED.stop_hunt_detected,
            trading_enabled             = EXCLUDED.trading_enabled,
            shutdown_reason             = EXCLUDED.shutdown_reason,
            current_regime              = EXCLUDED.current_regime,
            regime_calculated_at        = EXCLUDED.regime_calculated_at,
            size_multiplier             = EXCLUDED.size_multiplier,
            ks4_reduced_trades_remaining = EXCLUDED.ks4_reduced_trades_remaining,
            reversal_family_occupied     = EXCLUDED.reversal_family_occupied,
            s1b_pending_ticket           = EXCLUDED.s1b_pending_ticket,
            s1d_ema_touched_today        = EXCLUDED.s1d_ema_touched_today,
            s1d_fired_today              = EXCLUDED.s1d_fired_today,
            s3_sweep_candle_time         = EXCLUDED.s3_sweep_candle_time,
            r3_fired_today               = EXCLUDED.r3_fired_today,
            r3_armed                     = EXCLUDED.r3_armed,
            s4_ema_touched              = EXCLUDED.s4_ema_touched,
            s4_fired_today               = EXCLUDED.s4_fired_today,
            s5_compression_confirmed    = EXCLUDED.s5_compression_confirmed,
            s5_fired_today               = EXCLUDED.s5_fired_today,
            london_session_high          = EXCLUDED.london_session_high,
            london_session_low           = EXCLUDED.london_session_low,
            d1_atr_14                    = EXCLUDED.d1_atr_14,
            ks7_pre_event_price          = EXCLUDED.ks7_pre_event_price,
            spread_multiplier            = EXCLUDED.spread_multiplier,
            spread_elevated_reading_count = EXCLUDED.spread_elevated_reading_count,
            dxy_ewma_variance            = EXCLUDED.dxy_ewma_variance
        """,
        {
            "state_date":                  state_date_today,
            "saved_at":                    datetime.now(pytz.utc),
            "consecutive_m5_losses":       state.get("consecutive_m5_losses", 0),
            "s1_family_attempts_today":    state.get("s1_family_attempts_today", 0),
            "s1f_attempts_today":          state.get("s1f_attempts_today", 0),
            "daily_net_pnl_pct":           state.get("daily_net_pnl_pct", 0.0),
            "daily_commission_paid":       state.get("daily_commission_paid", 0.0),
            "consecutive_losses":          state.get("consecutive_losses", 0),
            "peak_equity":                 state.get("peak_equity", 0.0),
            "failed_breakout_flag":        state.get("failed_breakout_flag", False),
            "failed_breakout_direction":   state.get("failed_breakout_direction"),
            "last_s1_direction":           state.get("last_s1_direction"),
            "stop_hunt_detected":          state.get("stop_hunt_detected", False),
            "trading_enabled":             state.get("trading_enabled", True),
            "shutdown_reason":             state.get("shutdown_reason"),
            "current_regime":              state.get("current_regime", "NO_TRADE"),
            "regime_calculated_at":        state.get("regime_calculated_at",
                                               datetime.now(pytz.utc)),
            "size_multiplier":             state.get("size_multiplier", 0.0),
            # ── v1.1 additions ────────────────────────────────────────────────
            "ks4_reduced_trades_remaining": state.get("ks4_reduced_trades_remaining", 0),
            "reversal_family_occupied":     state.get("reversal_family_occupied", False),
            "s1b_pending_ticket":           state.get("s1b_pending_ticket"),
            "s1d_ema_touched_today":        state.get("s1d_ema_touched_today", False),
            "s1d_fired_today":              state.get("s1d_fired_today", False),
            "s3_sweep_candle_time":         state.get("s3_sweep_candle_time"),
            # ── Phase 2 additions ───────────────────────────────────────────────
            "r3_fired_today":               state.get("r3_fired_today", False),
            "r3_armed":                     state.get("r3_armed", False),
            "s4_ema_touched":              state.get("s4_ema_touched", False),
            "s4_fired_today":               state.get("s4_fired_today", False),
            "s5_compression_confirmed":    state.get("s5_compression_confirmed", False),
            "s5_fired_today":               state.get("s5_fired_today", False),
            "london_session_high":          state.get("london_session_high", 0.0),
            "london_session_low":           state.get("london_session_low", 0.0),
            "d1_atr_14":                    state.get("d1_atr_14", 0.0),
            "ks7_pre_event_price":          state.get("ks7_pre_event_price", 0.0),
            "spread_multiplier":            state.get("spread_multiplier", 1.0),
            "spread_elevated_reading_count": state.get("spread_elevated_reading_count", 0),
            "dxy_ewma_variance":            state.get("dxy_ewma_variance", 0.0),
        }
    )
    log_event("CRITICAL_STATE_PERSISTED", date=str(state_date_today))

# ─────────────────────────────────────────────────────────────────────────────
# RESTORE — C4 Fix (daily vs rolling separation)
# ─────────────────────────────────────────────────────────────────────────────

def restore_critical_state(state: dict) -> None:
    """
    C4 Fix: Restores state on startup with strict daily/rolling separation.

    If today's row exists → load EVERYTHING (daily counters + rolling state).
    If no today's row   → load rolling state from yesterday only.
                           Daily counters stay at 0 (fresh day start).

    RULE: consecutive_losses loaded from yesterday if no today row.
          It does NOT reset to 0 on midnight — streak is continuous.
    """
    ist       = pytz.timezone("Asia/Kolkata")
    today     = datetime.now(ist).date()
    yesterday = today - timedelta(days=1)

    # Try today's row first
    rows = execute_query(
        """SELECT * FROM system_state.system_state_persistent
           WHERE state_date = :today
           ORDER BY saved_at DESC LIMIT 1""",
        {"today": today}
    )

    if rows:
        row = rows[0]
        # C4 Fix: today's row → load both daily counters and rolling state
        state["consecutive_losses"]       = int(row["consecutive_losses"] or 0)
        state["consecutive_m5_losses"]    = int(row["consecutive_m5_losses"] or 0)
        state["s1_family_attempts_today"] = int(row["s1_family_attempts_today"] or 0)
        state["s1f_attempts_today"]       = int(row["s1f_attempts_today"] or 0)
        state["daily_net_pnl_pct"]        = float(row["daily_net_pnl_pct"] or 0.0)
        state["daily_commission_paid"]    = float(row["daily_commission_paid"] or 0.0)
        state["peak_equity"]              = float(row["peak_equity"] or 0.0)
        state["failed_breakout_flag"]     = bool(row["failed_breakout_flag"])
        state["failed_breakout_direction"]= row["failed_breakout_direction"]
        state["last_s1_direction"]        = row["last_s1_direction"]
        state["stop_hunt_detected"]       = bool(row["stop_hunt_detected"])
        state["trading_enabled"]          = bool(row["trading_enabled"])
        state["shutdown_reason"]          = row["shutdown_reason"]
        state["current_regime"]           = row["current_regime"]
        state["regime_calculated_at"]     = row["regime_calculated_at"]
        state["size_multiplier"]          = row["size_multiplier"]
        for col, key in (
            ("ks4_reduced_trades_remaining", "ks4_reduced_trades_remaining"),
            ("reversal_family_occupied", "reversal_family_occupied"),
            ("s1b_pending_ticket", "s1b_pending_ticket"),
            ("s1d_ema_touched_today", "s1d_ema_touched_today"),
            ("s1d_fired_today", "s1d_fired_today"),
            ("s3_sweep_candle_time", "s3_sweep_candle_time"),
            # ── Phase 2 additions ───────────────────────────────────────────────
            ("r3_fired_today", "r3_fired_today"),
            ("r3_armed", "r3_armed"),
            ("s4_ema_touched", "s4_ema_touched"),
            ("s4_fired_today", "s4_fired_today"),
            ("s5_compression_confirmed", "s5_compression_confirmed"),
            ("s5_fired_today", "s5_fired_today"),
            ("london_session_high", "london_session_high"),
            ("london_session_low", "london_session_low"),
            ("d1_atr_14", "d1_atr_14"),
            ("ks7_pre_event_price", "ks7_pre_event_price"),
            ("spread_multiplier", "spread_multiplier"),
            ("spread_elevated_reading_count", "spread_elevated_reading_count"),
            ("dxy_ewma_variance", "dxy_ewma_variance"),
        ):
            if col not in row:
                continue
            val = row[col]
            if col == "reversal_family_occupied":
                val = bool(val)
            elif col == "ks4_reduced_trades_remaining":
                val = int(val or 0)
            elif col == "s1b_pending_ticket":
                val = int(val) if val is not None else None
            elif col in ("s1d_ema_touched_today", "s1d_fired_today"):
                val = bool(val)
            state[key] = val
        log_event("STATE_RESTORED_FROM_TODAY", date=str(today))

    else:
        # C4 Fix: no today row → fresh day for daily counters
        # Rolling state (consecutive_losses, peak_equity) from yesterday
        yesterday_rows = execute_query(
            """SELECT peak_equity, consecutive_losses
               FROM system_state.system_state_persistent
               WHERE state_date = :yesterday LIMIT 1""",
            {"yesterday": yesterday}
        )

        if yesterday_rows:
            y = yesterday_rows[0]
            # RULE: consecutive_losses carries over — streak is not reset by midnight
            state["consecutive_losses"] = int(y["consecutive_losses"] or 0)
            state["peak_equity"]        = float(y["peak_equity"] or 0.0)
            log_event("STATE_RESTORED_ROLLING_FROM_YESTERDAY",
                      consecutive_losses=state["consecutive_losses"],
                      peak_equity=state["peak_equity"])
        else:
            log_event("STATE_NO_HISTORICAL_ROW_FRESH_START")

        # Daily counters stay at 0 (already set by build_initial_state)
        log_event("STATE_FRESH_DAY_DAILY_COUNTERS_RESET", date=str(today))


# ─────────────────────────────────────────────────────────────────────────────
# PEAK EQUITY — B6 Fix (continuous tracking)
# ─────────────────────────────────────────────────────────────────────────────

def update_peak_equity(state: dict, live_equity: float | None = None) -> None:
    """
    B6 Fix + v1.1: Rolling 30-day peak from performance rows, merged with live equity.
    KS6 triggers if equity < peak × 0.92 (peak = max(state peak, 30d DB peak, live high).
    """
    if live_equity is not None and live_equity > 0:
        if live_equity > state.get("peak_equity", 0.0):
            state["peak_equity"] = live_equity
            log_event("PEAK_EQUITY_UPDATED_LIVE", peak=round(live_equity, 2))

    sql = text("""
        SELECT MAX(peak_equity)
        FROM system_state.performance
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
          AND peak_equity IS NOT NULL
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(sql).scalar()
        peak_30d = float(result or 0.0)
        if peak_30d > state.get("peak_equity", 0.0):
            state["peak_equity"] = peak_30d
            log_event("PEAK_EQUITY_UPDATED_30D", peak=round(peak_30d, 2))
    except Exception as e:
        log_warning("UPDATE_PEAK_EQUITY_FAILED", error=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS (Truth Engine queries — used by risk engine + weekly review)
# ─────────────────────────────────────────────────────────────────────────────

def get_weekly_net_pnl_pct() -> float:
    """Net P&L as % for the current calendar week (Mon–Sun IST)."""
    rows = execute_query(
        """SELECT SUM(pnl_net_dollars) as weekly_pnl
            FROM system_state.trades
            WHERE exit_time >= date_trunc('week', NOW() AT TIME ZONE 'Asia/Kolkata')
             AND exit_time IS NOT NULL""",
        {}
    )
    if not rows or rows[0]["weekly_pnl"] is None:
        return 0.0

    mt5  = get_mt5()
    info = mt5.account_info()
    if info is None or info.equity == 0:
        return 0.0

    return float(rows[0]["weekly_pnl"]) / float(info.equity)

# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY REVIEW — Fix 2 + Truth Engine
# ─────────────────────────────────────────────────────────────────────────────

def weekly_review(send_email: bool = True) -> dict:
    """
    Fix 2: Weekly Truth Engine review.
    Scheduled Sunday 08:00 IST via APScheduler.

    Fix 2 specifically: expectancy = df['r_multiple'].mean()
    NOT (WR × avg_win) - ((1-WR) × avg_loss) — that formula is biased.

    Checks promotion gates:
      Phase 2:    WR > 45%, expectancy > +0.15R, max_dd < 15%
      Macro bias: A+ vs STANDARD delta > 8pp after 50 trades → promote
      Proxy swap: TLT → TIP if macro never hits 8pp delta after 50 trades

    Returns summary dict for logging and email.
    """
    import numpy as np

    rows = execute_query(
        """SELECT
               r_multiple, pnl_net_dollars, pnl_gross_dollars, total_commission,
               regime_at_entry, conviction_level,
               macro_bias AS macro_bias_at_entry, macro_proxy_at_entry,
               signal_type, direction, entry_time, exit_time
           FROM system_state.trades
           WHERE exit_time IS NOT NULL
           ORDER BY exit_time DESC""",
        {}
    )

    if not rows:
        log_event("WEEKLY_REVIEW_NO_TRADES")
        return {}

    import pandas as pd
    df = pd.DataFrame(rows)
    df["r_multiple"]      = pd.to_numeric(df["r_multiple"],      errors="coerce")
    df["pnl_net_dollars"] = pd.to_numeric(df["pnl_net_dollars"], errors="coerce")  # ← fixed
    df["total_commission"]= pd.to_numeric(df["total_commission"], errors="coerce")

    total    = len(df)
    wins     = (df["r_multiple"] > 0).sum()
    win_rate = float(wins / total) if total > 0 else 0.0

    # Fix 2: expectancy = mean of R multiples (not the split formula)
    expectancy = float(df["r_multiple"].mean()) if not df.empty else 0.0
    total_net  = float(df["pnl_net_dollars"].sum())                                # ← fixed
    total_comm = float(df["total_commission"].sum())
    max_dd     = get_max_drawdown_pct()
    sharpe     = get_rolling_sharpe(min(total, 50))

    # ── By-regime breakdown ───────────────────────────────────────────────
    regime_stats = {}
    for regime, grp in df.groupby("regime_at_entry"):
        regime_stats[regime] = {
            "count":      len(grp),
            "win_rate":   round(float((grp["r_multiple"] > 0).mean()), 3),
            "expectancy": round(float(grp["r_multiple"].mean()), 3),
        }

    # ── Conviction delta (A+ vs STANDARD) ────────────────────────────────
    conviction_delta = 0.0
    if "conviction_level" in df.columns:
        aplus_rows    = df[df["conviction_level"] == "A_PLUS"]
        standard_rows = df[df["conviction_level"] == "STANDARD"]
        if len(aplus_rows) > 5 and len(standard_rows) > 5:
            aplus_wr         = float((aplus_rows["r_multiple"] > 0).mean())
            standard_wr      = float((standard_rows["r_multiple"] > 0).mean())
            conviction_delta = round((aplus_wr - standard_wr) * 100, 1)

    # ── Phase 2 gate check ────────────────────────────────────────────────
    phase_2_eligible = (
        total      >= config.PHASE_1_TRADE_GATE        and   # 50 trades
        win_rate    > config.PHASE_2_WIN_RATE_MIN       and  # >45%
        expectancy  > config.PHASE_2_EXPECTANCY_MIN     and  # >0.15R
        max_dd      < config.PHASE_2_MAX_DD                  # <15%
    )

    # ── Macro bias delta check ────────────────────────────────────────────
    macro_delta_pp = 0.0
    if "macro_bias_at_entry" in df.columns and total >= 50:
        aligned    = df[df["macro_bias_at_entry"].isin(
                        ["LONG_PERMITTED", "SHORT_PERMITTED"])]
        misaligned = df[df["macro_bias_at_entry"] == "NONE_PERMITTED"]
        if len(aligned) > 5 and len(misaligned) > 5:
            aligned_wr     = float((aligned["r_multiple"] > 0).mean())
            misaligned_wr  = float((misaligned["r_multiple"] > 0).mean())
            macro_delta_pp = round((aligned_wr - misaligned_wr) * 100, 1)

    summary = {
        "total_trades":        total,
        "win_rate":            round(win_rate, 3),
        "expectancy_r":        round(expectancy, 4),   # Fix 2
        "total_net_pnl":       round(total_net, 2),
        "total_commission":    round(total_comm, 2),
        "max_drawdown_pct":    round(max_dd, 4),
        "rolling_sharpe":      round(sharpe, 3),
        "phase_2_eligible":    phase_2_eligible,
        "conviction_delta_pp": conviction_delta,
        "macro_delta_pp":      macro_delta_pp,
        "regime_stats":        regime_stats,
    }

    # ── Print report ──────────────────────────────────────────────────────
    _print_weekly_report(summary)

    # ── Promotion alerts ─────────────────────────────────────────────────
    if phase_2_eligible:
        log_event("PHASE_2_ELIGIBLE",
                  win_rate=win_rate, expectancy=expectancy, max_dd=max_dd)
        if send_email:
            from utils.alerts import send_ks_alert
            send_ks_alert(
                "PHASE_2_ELIGIBLE",
                f"WR={win_rate:.1%} Exp={expectancy:.3f}R MaxDD={max_dd:.1%} "
                f"Trades={total} — manual review required before upgrading risk."
            )

    if total >= 50 and conviction_delta > config.MACRO_PROMOTE_DELTA_PP:
        log_event("CONVICTION_PROMOTE_A_PLUS", delta_pp=conviction_delta)

    if total >= 50 and macro_delta_pp > config.MACRO_PROMOTE_DELTA_PP:
        log_event("MACRO_BIAS_PROMOTE_TO_ACTIVE", delta_pp=macro_delta_pp)
    elif total >= 50 and abs(macro_delta_pp) < 2.0:
        log_event("MACRO_PROXY_SWAP_CANDIDATE",
                  note="TLT delta never >8pp. Consider switching to TIP ETF.")

    log_event("WEEKLY_REVIEW_COMPLETE",
              total=total, wr=round(win_rate, 3),
              exp=round(expectancy, 4), phase_2=phase_2_eligible)

    return summary

def _print_weekly_report(s: dict) -> None:
    print("\n══════════════════════════════════════════════════════")
    print("  WEEKLY TRUTH ENGINE REVIEW")
    print("══════════════════════════════════════════════════════")
    print(f"  Trades       : {s['total_trades']}")
    print(f"  Win Rate     : {s['win_rate']:.1%}")
    print(f"  Expectancy   : {s['expectancy_r']:+.4f}R   ← Fix 2: mean(r_multiple)")
    print(f"  Net P&L      : ${s['total_net_pnl']:,.2f}")
    print(f"  Commission   : ${s['total_commission']:,.2f}")
    print(f"  Max Drawdown : {s['max_drawdown_pct']:.2%}")
    print(f"  Sharpe       : {s['rolling_sharpe']:.3f}")
    print(f"  Conv. Delta  : {s['conviction_delta_pp']:+.1f}pp  (A+ vs STANDARD)")
    print(f"  Macro Delta  : {s['macro_delta_pp']:+.1f}pp  (aligned vs none)")
    print("──────────────────────────────────────────────────────")
    for regime, stats in s["regime_stats"].items():
        print(f"  {regime:<20} n={stats['count']:>3}  "
              f"WR={stats['win_rate']:.1%}  Exp={stats['expectancy']:+.3f}R")
    print("──────────────────────────────────────────────────────")
    if s["phase_2_eligible"]:
        print("  >>> PHASE 2 ELIGIBLE — review before upgrading risk <<<")
    else:
        print("  Phase 2: NOT YET ELIGIBLE")
    print("══════════════════════════════════════════════════════\n")
