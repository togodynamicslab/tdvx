"""SQLite connection management with context manager."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """
    Get the database file path and ensure the directory exists.

    Returns:
        Path to the database file
    """
    db_path = Path(__file__).parent.parent.parent / "data" / "tdvx.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_db_connection():
    """
    Context manager for SQLite connections with auto-commit/rollback.

    Yields:
        sqlite3.Connection with row_factory set to sqlite3.Row

    Example:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM api_keys")
            results = cursor.fetchall()
    """
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row  # Access columns by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def execute_query(query: str, params: tuple = ()) -> List[sqlite3.Row]:
    """
    Execute a SELECT query and return results.

    Args:
        query: SQL SELECT query
        params: Query parameters (tuple)

    Returns:
        List of sqlite3.Row objects

    Example:
        results = execute_query("SELECT * FROM api_keys WHERE id = ?", (1,))
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()


def execute_insert(query: str, params: tuple = ()) -> int:
    """
    Execute an INSERT query and return the last row ID.

    Args:
        query: SQL INSERT query
        params: Query parameters (tuple)

    Returns:
        Last inserted row ID

    Example:
        key_id = execute_insert(
            "INSERT INTO api_keys (name, key_hash) VALUES (?, ?)",
            ("My Key", "hash123")
        )
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.lastrowid


def execute_update(query: str, params: tuple = ()) -> int:
    """
    Execute an UPDATE or DELETE query and return affected rows count.

    Args:
        query: SQL UPDATE or DELETE query
        params: Query parameters (tuple)

    Returns:
        Number of affected rows

    Example:
        affected = execute_update(
            "UPDATE api_keys SET is_active = 0 WHERE id = ?",
            (1,)
        )
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.rowcount
