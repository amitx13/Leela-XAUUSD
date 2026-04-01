"""
db/connection.py — SQLAlchemy engine + session factory.

Pool settings (spec Part 10):
  pool_size=5, max_overflow=10, pool_pre_ping=True
  pool_pre_ping: tests connection before use — prevents silent dead-connection errors
  pool_recycle=3600: recycle after 1 hour (avoids PostgreSQL idle timeout)

Connection string uses 127.0.0.1 explicitly (arch decision):
  Forces TCP over Unix socket — required for Docker PostgreSQL container.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import config
from utils.logger import log_event, log_warning

engine = create_engine(
    config.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,    # test connection before handing to caller
    pool_recycle=3600,     # recycle after 1 hour
    echo=False,            # set True temporarily for SQL debug only
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for DB sessions. Always commits or rolls back cleanly."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        log_warning("DB_SESSION_ERROR", error=str(e))
        raise
    finally:
        session.close()


def test_db_connection() -> bool:
    """Pre-session checklist check. Returns True if PostgreSQL is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log_warning("DB_CONNECTION_FAILED", error=str(e))
        return False


def execute_query(sql: str, params: tuple = ()) -> list:
    """
    Raw query helper for simple reads. Use ORM for writes.
    Returns list of row mappings.
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        return [dict(row._mapping) for row in result]


def execute_write(sql: str, params: dict) -> None:
    """Raw write helper for upserts not easily expressed in ORM."""
    with engine.begin() as conn:
        conn.execute(text(sql), params)

def get_config_value(key: str) -> str | None:
    """
    Reads a single value from system_state.system_config by key.
    Returns the value string, or None if the key doesn't exist.
    Used by: phase gate, KS4 countdown, correlation check counter.
    Never raises — returns None on any DB error.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT value FROM system_state.system_config WHERE key = :key"),
                {"key": key}
            )
            row = result.fetchone()
            return row[0] if row else None
    except Exception as e:
        log_warning("GET_CONFIG_VALUE_FAILED", key=key, error=str(e))
        return None