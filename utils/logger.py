"""
utils/logger.py — Structured KEY=VALUE log wrapper (COMP-3).

RULE: Never call logging.info("free text") directly anywhere in the codebase.
Always use log_event(event_name, **kwargs) so every log line is parseable.
Format: TIMESTAMP | LEVEL | EVENT=name KEY=value KEY=value ...
"""
import logging
import pytz
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Configure once at module level
os.makedirs("logs", exist_ok=True)
_logger = logging.getLogger("xauusd_algo")
_logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                          datefmt="%Y-%m-%dT%H:%M:%S")

_fh = RotatingFileHandler("logs/xauusd.log", maxBytes=10_000_000, backupCount=5)
_fh.setFormatter(_fmt)
_logger.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
_logger.addHandler(_sh)


def _format_kwargs(kwargs: dict) -> str:
    """Format kwargs as KEY=value pairs on a single line."""
    return " ".join(f"{k}={v}" for k, v in kwargs.items())


def log_event(event: str, **kwargs) -> None:
    """
    Standard structured log line.
    Example: log_event("KS3_FIRED", daily_pnl_pct=-0.016, equity=9840.0)
    Output:  2026-03-22T18:00:00 | INFO | EVENT=KS3_FIRED daily_pnl_pct=-0.016 equity=9840.0
    """
    parts = f"EVENT={event}"
    if kwargs:
        parts += " " + _format_kwargs(kwargs)
    _logger.info(parts)


def log_critical(event: str, **kwargs) -> None:
    """For emergency shutdowns and unrecoverable errors."""
    parts = f"EVENT={event}"
    if kwargs:
        parts += " " + _format_kwargs(kwargs)
    _logger.critical(parts)


def log_warning(event: str, **kwargs) -> None:
    parts = f"EVENT={event}"
    if kwargs:
        parts += " " + _format_kwargs(kwargs)
    _logger.warning(parts)


def log_execution_error(retcode: int, **kwargs) -> None:
    log_warning("log_execution_error", retcode=retcode, **kwargs)
