"""Database initialization and schema management."""

import logging
from app.database.connection import get_db_connection, get_db_path

logger = logging.getLogger(__name__)

# SQL schema for api_keys table
CREATE_API_KEYS_TABLE = """
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    rate_limit_per_hour INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    revoked_at TIMESTAMP,
    revoked_reason TEXT
);
"""

# SQL schema for api_key_usage table (rate limiting tracking)
CREATE_API_KEY_USAGE_TABLE = """
CREATE TABLE IF NOT EXISTS api_key_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id INTEGER NOT NULL,
    hour_window TEXT NOT NULL,
    request_count INTEGER DEFAULT 0,
    last_request_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE,
    UNIQUE(api_key_id, hour_window)
);
"""

# Indexes for performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_key_hash ON api_keys(key_hash);",
    "CREATE INDEX IF NOT EXISTS idx_active ON api_keys(is_active);",
    "CREATE INDEX IF NOT EXISTS idx_created ON api_keys(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_usage_window ON api_key_usage(api_key_id, hour_window);",
    "CREATE INDEX IF NOT EXISTS idx_usage_time ON api_key_usage(last_request_at);",
]


def initialize_database() -> None:
    """
    Initialize the database schema.

    Creates all tables and indexes. Safe to call multiple times
    (uses IF NOT EXISTS).

    Raises:
        sqlite3.Error: If database initialization fails
    """
    logger.info(f"Initializing database at: {get_db_path()}")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            logger.info("Creating api_keys table...")
            cursor.execute(CREATE_API_KEYS_TABLE)

            logger.info("Creating api_key_usage table...")
            cursor.execute(CREATE_API_KEY_USAGE_TABLE)

            # Create indexes
            logger.info("Creating indexes...")
            for index_sql in CREATE_INDEXES:
                cursor.execute(index_sql)

            conn.commit()

        logger.info("Database initialization complete!")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def cleanup_old_usage_records(days: int = 30) -> int:
    """
    Clean up old usage records for maintenance.

    Args:
        days: Delete records older than this many days (default: 30)

    Returns:
        Number of deleted records

    Example:
        deleted = cleanup_old_usage_records(days=7)
        logger.info(f"Cleaned up {deleted} old usage records")
    """
    from app.database.connection import execute_update

    query = """
    DELETE FROM api_key_usage
    WHERE datetime(last_request_at) < datetime('now', '-' || ? || ' days')
    """

    try:
        affected = execute_update(query, (days,))
        logger.info(f"Cleaned up {affected} usage records older than {days} days")
        return affected
    except Exception as e:
        logger.error(f"Failed to cleanup old usage records: {e}")
        raise
