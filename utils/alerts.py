"""
utils/alerts.py — SMTP alert dispatch (COMP-1).

Rules:
  - All KS-level alerts (KS3, KS5, KS6, EMERGENCY) use send_ks_alert().
  - KS1, KS2, KS4, KS7 are routine — log_event() only, no email.
  - SMTP timeout=10 enforced — never use blocking default (COMP-1).
  - Alert send failure does NOT halt the system. Trade protection takes priority.
  - test_smtp_connection() must pass in pre_session_checklist (ADD-1 compensating).
"""
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pytz
import config
from utils.logger import log_event, log_warning


# KS names that trigger email alerts (others are log-only)
ALERT_KS_NAMES = {"KS3", "KS5", "KS6", "EMERGENCY_SHUTDOWN"}


def send_ks_alert(ks_name: str, details: str) -> bool:
    """
    Send kill-switch alert email.
    Returns True if sent successfully, False if failed.
    System continues operating regardless of return value.
    """
    if ks_name not in ALERT_KS_NAMES:
        log_event("KS_ALERT_SKIPPED_NOT_EMAIL_LEVEL", ks_name=ks_name)
        return False
    
    if not config.SMTP_USER or not config.ALERT_RECIPIENT:
        log_warning("KS_ALERT_SMTP_NOT_CONFIGURED", ks_name=ks_name)
        return False

    now_ist = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")

    msg = EmailMessage()
    msg["Subject"] = f"[XAUUSD] {ks_name} FIRED"
    msg["From"]    = config.SMTP_USER
    msg["To"]      = config.ALERT_RECIPIENT
    msg.set_content(
        f"Kill switch : {ks_name}\n"
        f"Details     : {details}\n"
        f"Time        : {now_ist}\n"
        f"Action      : Review Truth Engine before restarting.\n"
        f"\n"
        f"Do NOT restart without completing written review."
    )

    try:
        # COMP-1: timeout=10 — never block indefinitely
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT,
                          timeout=config.SMTP_TIMEOUT_SEC) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASS)
            server.send_message(msg)
        log_event("KS_ALERT_SENT", ks_name=ks_name)
        return True

    except (smtplib.SMTPException, TimeoutError, OSError) as e:
        # Alert failure is logged but does NOT stop the system.
        # Trade protection (hard stops, KS shutdown) continues regardless.
        log_warning("KS_ALERT_SEND_FAILED", ks_name=ks_name, error=str(e))
        return False


def test_smtp_connection() -> bool:
    """
    Pre-session checklist item (compensating control for apprise rejection).
    Verifies SMTP is reachable before session starts.
    """
    if not config.SMTP_USER or not config.ALERT_RECIPIENT:
        log_warning("SMTP_TEST_SKIPPED_NOT_CONFIGURED")
        return True  # not configured = not required — don't block session

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT,
                          timeout=config.SMTP_TIMEOUT_SEC) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASS)
        log_event("SMTP_TEST_PASSED")
        return True
    except Exception as e:
        log_warning("SMTP_TEST_FAILED", error=str(e))
        return False
