"""
utils/mt5_client.py — MT5 singleton connection.

Single point of truth for the mt5linux rpyc connection.
All engines import get_mt5() — never construct MetaTrader5() elsewhere.

Architecture note:
  mt5linux wraps the real MetaTrader5 package running inside Bottles/Wine
  via rpyc. The object behaves identically to the native MT5 Python API.
  host/port set in config.py and read from .env.
"""
import time
from mt5linux import MetaTrader5
import config
from utils.logger import log_event, log_critical

_mt5: MetaTrader5 | None = None


def get_mt5() -> MetaTrader5:
    """
    Returns the shared MT5 connection object.
    Does NOT call initialize() — that is done once in initialize_system().
    Safe to call from any engine at any time after init.
    """
    global _mt5
    if _mt5 is None:
        _mt5 = MetaTrader5(host=config.MT5_HOST, port=config.MT5_PORT)
    return _mt5


def ensure_mt5_connected(max_retries: int = 3) -> bool:
    """
    Lightweight liveness check with auto-reconnect and exponential backoff.
    Returns True if initialize() + terminal_info() both succeed.
    On failure, resets the connection and retries up to max_retries times.
    Backoff: 1s, 2s, 4s between attempts.
    """
    # Import here to avoid circular imports (alerts → mt5_client)
    from utils.alerts import send_ks_alert

    for attempt in range(max_retries):
        try:
            mt5 = get_mt5()
            ok = mt5.initialize()
            if ok:
                info = mt5.terminal_info()
                if info is not None:
                    if attempt > 0:
                        log_event("MT5_RECONNECTED_SUCCESS", attempt=attempt + 1)
                    return True

            # initialize() failed or terminal_info() is None
            log_event("MT5_CONNECT_ATTEMPT_FAILED",
                      attempt=attempt + 1,
                      error=str(mt5.last_error()))

        except Exception as e:
            log_event("MT5_CONNECT_EXCEPTION", attempt=attempt + 1, error=str(e))

        # Reset connection and wait before retrying (exponential backoff: 1s, 2s, 4s)
        reset_mt5_connection()
        time.sleep(2 ** attempt)

    # All retries exhausted
    log_critical("MT5_CONNECTION_LOST_PERMANENTLY", attempts=max_retries)
    from utils.alerts import send_ks_alert
    send_ks_alert("EMERGENCY_SHUTDOWN", "MT5 unreachable after multiple reconnect attempts.")
    return False


def reset_mt5_connection() -> None:
    """
    Force a fresh connection object on next get_mt5() call.
    Called only by emergency_shutdown after mt5.shutdown().
    """
    global _mt5
    _mt5 = None
    log_event("MT5_CONNECTION_RESET")
