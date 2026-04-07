#!/usr/bin/env python3
"""
tools/fetch_events_history.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-shot helper: downloads HIGH-impact economic events from Forex Factory
for a specified date window and writes them to backtest_data/events.csv.

Once this CSV exists, HistoricalEventFeed in backtest/data_feed.py will
automatic load it (CRIT-1 fix) instead of using the approximate
pattern-based generator — giving exact event blackout fidelity.

Usage:
    python tools/fetch_events_history.py --start 2023-01-01 --end 2024-12-31
    python tools/fetch_events_history.py --start 2023-01-01 --end 2024-12-31 \\
        --out custom/path/events.csv

The script tries Forex Factory's public JSON endpoint. If that fails (rate
limit or network block), it falls back to writing a stub CSV with instructions
so you can populate it manually from a broker calendar export.

Output CSV schema:
    datetime_utc,name,impact
    2024-01-26 13:30:00,Non-Farm Payrolls,HIGH
"""
import sys
import time
import argparse
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_events")

# Forex Factory's JSON calendar endpoint (unofficial but stable since 2018)
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# HIGH-impact currency filters — we only care about USD and XAU movers
HIGH_IMPACT_CURRENCIES = {"USD", "XAU", "ALL"}

DEFAULT_OUT = Path(__file__).parent.parent / "backtest_data" / "events.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Fetch Forex Factory events to CSV")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="End date   YYYY-MM-DD")
    p.add_argument("--out",   default=str(DEFAULT_OUT), help="Output CSV path")
    p.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between weekly requests (default 2.0)"
    )
    return p.parse_args()


def fetch_ff_week(week_start: datetime) -> list[dict]:
    """
    Fetch FF calendar JSON for the ISO week containing week_start.
    Returns list of raw event dicts or empty list on failure.
    """
    if requests is None:
        logger.error("'requests' package not installed. Run: pip install requests")
        return []

    # FF public endpoint — week parameter is Monday of the ISO week
    monday = week_start - timedelta(days=week_start.weekday())
    url    = f"https://nfs.faireconomy.media/ff_calendar_{monday.strftime('%Y-%m-%d')}.json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        # Fallback to this-week endpoint
        resp2 = requests.get(FF_URL, timeout=15)
        if resp2.status_code == 200:
            return resp2.json()
        logger.warning(f"FF returned HTTP {resp.status_code} for week {monday.date()}")
        return []
    except Exception as exc:
        logger.warning(f"FF request failed for week {monday.date()}: {exc}")
        return []


def events_to_rows(raw_events: list[dict]) -> list[dict]:
    """
    Filter and normalise raw FF event dicts to {datetime_utc, name, impact}.
    Only HIGH impact + USD/XAU currency events are kept.
    """
    rows = []
    for ev in raw_events:
        impact = str(ev.get("impact", "")).strip().upper()
        if impact != "HIGH":
            continue
        currency = str(ev.get("currency", "")).strip().upper()
        if currency not in HIGH_IMPACT_CURRENCIES:
            continue
        raw_dt = ev.get("date") or ev.get("datetime") or ""
        try:
            # FF uses ISO 8601 with timezone offset
            import dateutil.parser
            dt_utc = dateutil.parser.parse(raw_dt).utctimetuple()
            dt_str = datetime(*dt_utc[:6]).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        rows.append({
            "datetime_utc": dt_str,
            "name":         str(ev.get("title", ev.get("name", "HIGH_IMPACT"))).strip(),
            "impact":       "HIGH",
        })
    return rows


def write_stub_csv(out_path: Path, start: datetime, end: datetime) -> None:
    """
    Write a stub CSV with example rows and instructions when FF is unreachable.
    The operator can populate it manually from a broker calendar export.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime_utc", "name", "impact"])
        writer.writeheader()
        writer.writerow({
            "datetime_utc": "# STUB — replace with real event data",
            "name": "See https://www.forexfactory.com/calendar",
            "impact": "HIGH",
        })
        # Example rows so the operator knows the format
        writer.writerow({"datetime_utc": "2024-01-26 13:30:00", "name": "Non-Farm Payrolls", "impact": "HIGH"})
        writer.writerow({"datetime_utc": "2024-01-31 19:00:00", "name": "FOMC Statement",     "impact": "HIGH"})
    logger.warning(
        f"Could not fetch real event data. Stub CSV written to {out_path}. "
        f"Please populate it manually for the window {start.date()} → {end.date()}."
    )


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end,   "%Y-%m-%d")
    out   = Path(args.out)

    if requests is None:
        logger.error("requests not installed. Run: pip install requests python-dateutil")
        write_stub_csv(out, start, end)
        sys.exit(1)

    try:
        import dateutil  # noqa: F401
    except ImportError:
        logger.error("python-dateutil not installed. Run: pip install python-dateutil")
        write_stub_csv(out, start, end)
        sys.exit(1)

    logger.info(f"Fetching HIGH-impact events {start.date()} → {end.date()}")

    all_rows: list[dict] = []
    current  = start
    weeks_fetched = 0
    weeks_failed  = 0

    while current <= end:
        raw = fetch_ff_week(current)
        if raw:
            rows = events_to_rows(raw)
            # Only keep events in [start, end]
            for r in rows:
                try:
                    dt = datetime.strptime(r["datetime_utc"], "%Y-%m-%d %H:%M:%S")
                    if start <= dt <= end:
                        all_rows.append(r)
                except ValueError:
                    pass
            weeks_fetched += 1
        else:
            weeks_failed += 1

        current += timedelta(weeks=1)
        if current <= end:
            time.sleep(args.delay)

    if not all_rows:
        logger.warning(f"No events fetched ({weeks_failed} failed weeks). Writing stub.")
        write_stub_csv(out, start, end)
        sys.exit(0)

    # Deduplicate and sort
    seen = set()
    unique_rows = []
    for r in all_rows:
        key = (r["datetime_utc"], r["name"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)
    unique_rows.sort(key=lambda x: x["datetime_utc"])

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime_utc", "name", "impact"])
        writer.writeheader()
        writer.writerows(unique_rows)

    logger.info(
        f"Written {len(unique_rows)} HIGH-impact events to {out} "
        f"({weeks_fetched} weeks OK, {weeks_failed} weeks failed)"
    )


if __name__ == "__main__":
    main()
