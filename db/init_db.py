"""
db/init_db.py — One-shot DB setup script.

Run ONCE during Phase 0 setup:
    python db/init_db.py

What it does:
    1. Creates schemas: market_data, system_state
    2. Creates all tables (CREATE TABLE IF NOT EXISTS)
    3. Seeds system_config with ADD-3 calibration placeholder
    4. Verifies all tables exist

Safe to re-run — all statements use IF NOT EXISTS.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.schema import create_all_schemas, migrate_total_commission, verify_schema
from db.connection import test_db_connection, engine
from sqlalchemy import text
from utils.logger import log_event


def seed_system_config() -> None:
    """ADD-3: Seed calibration values. ON CONFLICT DO NOTHING = safe to re-run."""
    seeds = [
        ("TLT_SLOPE_THRESHOLD",    "0.15",  "Phase 0 calibration gate — update after 50 trades"),
        ("PHASE",                  "0",     "System phase: 0=shadow, 1=live-micro, 2=live-full"),
        ("KS3_DAILY_LOSS_LIMIT",   "-0.030","Daily net PnL halt threshold (must be negative)"),
        ("KS5_WEEKLY_LOSS_LIMIT",  "-0.080","Weekly net PnL halt threshold (must be negative)"),
        ("KS6_DRAWDOWN_PCT",       "0.08",  "Drawdown circuit breaker: equity < peak*(1-this)"),
    ]
    with engine.connect() as conn:
        for key, value, desc in seeds:
            conn.execute(text("""
                INSERT INTO system_config (key, value, notes)
                VALUES (:key, :val, :desc)
                ON CONFLICT (key) DO NOTHING
            """), {"key": key, "val": value, "desc": desc})
        conn.commit()
    print(f"  ✅ system_config seeded ({len(seeds)} rows)")


def main():
    print("\n── XAUUSD Algo DB Init ───────────────────────────────────────────")

    print("\n[1/4] Testing DB connection...")
    if not test_db_connection():
        print("  ❌ Cannot connect. Check DATABASE_URL in .env and docker ps.")
        sys.exit(1)
    print("  ✅ Connected")

    print("\n[2/4] Creating schemas and tables...")
    create_all_schemas()
    migrate_total_commission()

    print("\n[3/4] Seeding system_config...")
    seed_system_config()

    print("\n[4/4] Verifying schema...")
    ok = verify_schema()
    if not ok:
        print("  ❌ Schema verification failed — check logs above")
        sys.exit(1)

    print("\n── Setup complete ✅ ────────────────────────────────────────────")
    print("Run: python main.py --checklist to verify readiness")


if __name__ == "__main__":
    main()
