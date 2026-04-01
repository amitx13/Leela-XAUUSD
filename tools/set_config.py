#!/usr/bin/env python3
"""
tools/set_config.py — Update system_config table from command line.
No Docker exec needed. Run from project root with venv active.

Usage:
    python tools/set_config.py KEY VALUE
    python tools/set_config.py ATR_PCT_NO_TRADE_THRESHOLD 99
    python tools/set_config.py --list          (show all current values)
    python tools/set_config.py --reset         (restore spec defaults)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import engine
from sqlalchemy import text

SPEC_DEFAULTS = {
    "TLT_SLOPE_THRESHOLD":          "0.15",
    "MACRO_PROXY_INSTRUMENT":       "TLT",
    "ATR_PCT_QUIET_REF":            "30",
    "ATR_PCT_SUPER_THRESHOLD":      "55",
    "ATR_PCT_UNSTABLE_THRESHOLD":   "85",
    "ATR_PCT_NO_TRADE_THRESHOLD":   "95",
    "BASE_RISK_PHASE":              "1",
    "STRATEGY_VERSION":             "V1",
    "PHASE":                        "0",
    "KS3_DAILY_LOSS_LIMIT":         "-0.030",
    "KS5_WEEKLY_LOSS_LIMIT":        "-0.080",
    "KS6_DRAWDOWN_PCT":             "0.08",
}

def list_config():
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT key, value, notes FROM public.system_config ORDER BY key")).fetchall()
    print("\n── system_config (live DB values) ─────────────────────────────────────")
    print(f"  {'KEY':<35} {'VALUE':<15} NOTES")
    print(f"  {'-'*35} {'-'*15} {'-'*30}")
    for r in rows:
        default = SPEC_DEFAULTS.get(r.key, "")
        flag = " ← MODIFIED" if default and str(r.value) != default else ""
        print(f"  {r.key:<35} {str(r.value):<15} {(r.notes or '')[:30]}{flag}")
    print()

def set_value(key: str, value: str):
    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE public.system_config SET value = :v WHERE key = :k"),
            {"k": key, "v": value}
        )
        if result.rowcount == 0:
            conn.execute(
                text("INSERT INTO public.system_config (key, value, notes) VALUES (:k, :v, 'set via CLI')"),
                {"k": key, "v": value}
            )
            print(f"  [INSERT] {key} = {value}")
        else:
            print(f"  [UPDATE] {key} = {value}")
    print(f"  Restart main.py for change to take effect.")

def reset_defaults():
    print("  Resetting all values to spec defaults...")
    with engine.begin() as conn:
        for key, value in SPEC_DEFAULTS.items():
            conn.execute(
                text("""
                    INSERT INTO public.system_config (key, value, notes)
                    VALUES (:k, :v, 'spec default')
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """),
                {"k": key, "v": value}
            )
            print(f"  [RESET] {key} = {value}")
    print("  Done. Restart main.py.")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--list":
        list_config()
    elif args[0] == "--reset":
        reset_defaults()
    elif len(args) == 2:
        set_value(args[0], args[1])
    else:
        print(__doc__)
        sys.exit(1)
