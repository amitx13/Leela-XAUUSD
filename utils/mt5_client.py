"""
utils/mt5_client.py — MT5 singleton connection.

Single point of truth for the mt5linux rpyc connection.
All engines import get_mt5() — never construct MetaTrader5() elsewhere.

Architecture note:
  mt5linux wraps the real MetaTrader5 package running inside Bottles/Wine
  via rpyc. The object behaves identically to the native MT5 Python API.
  host/port set in config.py and read from .env.
"""
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


def ensure_mt5_connected() -> bool:
    """
    Lightweight liveness check used in heartbeat and pre-session checklist.
    Returns True if initialize() + terminal_info() both succeed.
    initialize() is idempotent — safe to call on every check.
    """
    try:
        mt5  = get_mt5()
        ok   = mt5.initialize()          # must be called before any API method
        if not ok:
            log_event("MT5_INITIALIZE_FAILED",
                      error=str(mt5.last_error()))
            return False
        info = mt5.terminal_info()
        alive = info is not None
        if not alive:
            log_event("MT5_TERMINAL_INFO_NONE")
        return alive
    except Exception as e:
        log_event("MT5_CONNECTION_CHECK_FAILED", error=str(e))
        return False


def reset_mt5_connection() -> None:
    """
    Force a fresh connection object on next get_mt5() call.
    Called only by emergency_shutdown after mt5.shutdown().
    """
    global _mt5
    _mt5 = None
    log_event("MT5_CONNECTION_RESET")
