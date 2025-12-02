"""Rate limiting logic for API keys."""

from datetime import datetime
from typing import Tuple, Dict, Any
import logging

from app.database.connection import execute_query, execute_update

logger = logging.getLogger(__name__)


def get_current_hour_window() -> str:
    """
    Get the current hour window for rate limiting.

    Returns:
        String in format 'YYYY-MM-DD HH:00' (UTC)

    Example:
        window = get_current_hour_window()
        # Returns: "2025-12-02 15:00"
    """
    return datetime.utcnow().strftime("%Y-%m-%d %H:00")


def get_next_hour_reset_time() -> str:
    """
    Get the ISO timestamp for the next hour (when rate limit resets).

    Returns:
        ISO format timestamp string

    Example:
        reset_time = get_next_hour_reset_time()
        # Returns: "2025-12-02T16:00:00Z"
    """
    from datetime import timedelta
    next_hour = (datetime.utcnow() + timedelta(hours=1)).replace(
        minute=0, second=0, microsecond=0
    )
    return next_hour.isoformat() + "Z"


def check_rate_limit(api_key_id: int, limit: int) -> Tuple[bool, int, int]:
    """
    Check if a request is within the rate limit for an API key.

    Args:
        api_key_id: ID of the API key
        limit: Maximum requests per hour

    Returns:
        Tuple of (is_allowed, current_count, remaining)
        - is_allowed: True if request is within limit
        - current_count: Current number of requests this hour
        - remaining: Number of requests remaining (0 if limit exceeded)

    Example:
        is_allowed, current, remaining = check_rate_limit(key_id=123, limit=100)
        if not is_allowed:
            print(f"Rate limit exceeded! Used: {current}/{limit}")
        else:
            print(f"{remaining} requests remaining this hour")
    """
    hour_window = get_current_hour_window()

    query = """
    SELECT request_count
    FROM api_key_usage
    WHERE api_key_id = ? AND hour_window = ?
    """

    try:
        results = execute_query(query, (api_key_id, hour_window))

        current_count = results[0]["request_count"] if results else 0
        is_allowed = current_count < limit
        remaining = max(0, limit - current_count)

        return is_allowed, current_count, remaining

    except Exception as e:
        logger.error(f"Failed to check rate limit: {e}")
        # On error, allow the request (fail open)
        return True, 0, limit


def increment_usage(api_key_id: int) -> None:
    """
    Increment the usage counter for the current hour.

    Uses UPSERT pattern (INSERT or UPDATE) to handle both new
    and existing hour windows.

    Args:
        api_key_id: ID of the API key

    Raises:
        sqlite3.Error: If database operation fails

    Example:
        increment_usage(key_id=123)
    """
    hour_window = get_current_hour_window()

    # SQLite UPSERT syntax
    query = """
    INSERT INTO api_key_usage (api_key_id, hour_window, request_count, last_request_at)
    VALUES (?, ?, 1, CURRENT_TIMESTAMP)
    ON CONFLICT(api_key_id, hour_window)
    DO UPDATE SET
        request_count = request_count + 1,
        last_request_at = CURRENT_TIMESTAMP
    """

    try:
        execute_update(query, (api_key_id, hour_window))
    except Exception as e:
        logger.error(f"Failed to increment usage: {e}")
        raise


def get_current_usage(api_key_id: int) -> Dict[str, Any]:
    """
    Get current hour's usage statistics for an API key.

    Args:
        api_key_id: ID of the API key

    Returns:
        Dictionary with usage information:
        - hour_window: Current hour window
        - request_count: Number of requests this hour
        - last_request_at: Timestamp of last request

    Example:
        usage = get_current_usage(key_id=123)
        print(f"Used {usage['request_count']} requests this hour")
    """
    hour_window = get_current_hour_window()

    query = """
    SELECT hour_window, request_count, last_request_at
    FROM api_key_usage
    WHERE api_key_id = ? AND hour_window = ?
    """

    try:
        results = execute_query(query, (api_key_id, hour_window))

        if not results:
            return {
                "hour_window": hour_window,
                "request_count": 0,
                "last_request_at": None
            }

        row = results[0]
        return {
            "hour_window": row["hour_window"],
            "request_count": row["request_count"],
            "last_request_at": row["last_request_at"]
        }

    except Exception as e:
        logger.error(f"Failed to get current usage: {e}")
        return {
            "hour_window": hour_window,
            "request_count": 0,
            "last_request_at": None
        }


def get_usage_stats(api_key_id: int, hours: int = 24) -> Dict[str, Any]:
    """
    Get usage statistics for the last N hours.

    Args:
        api_key_id: ID of the API key
        hours: Number of hours to look back (default: 24)

    Returns:
        Dictionary with statistics:
        - total_requests: Total requests in period
        - hourly_breakdown: List of (hour_window, request_count) tuples
        - peak_hour: Hour with most requests
        - peak_count: Number of requests in peak hour

    Example:
        stats = get_usage_stats(key_id=123, hours=24)
        print(f"Total requests (24h): {stats['total_requests']}")
        print(f"Peak hour: {stats['peak_hour']} ({stats['peak_count']} requests)")
    """
    query = """
    SELECT hour_window, request_count
    FROM api_key_usage
    WHERE api_key_id = ?
      AND datetime(last_request_at) >= datetime('now', '-' || ? || ' hours')
    ORDER BY hour_window DESC
    """

    try:
        results = execute_query(query, (api_key_id, hours))

        if not results:
            return {
                "total_requests": 0,
                "hourly_breakdown": [],
                "peak_hour": None,
                "peak_count": 0
            }

        hourly_breakdown = [
            (row["hour_window"], row["request_count"])
            for row in results
        ]

        total_requests = sum(count for _, count in hourly_breakdown)

        # Find peak hour
        peak_hour, peak_count = max(
            hourly_breakdown,
            key=lambda x: x[1],
            default=(None, 0)
        )

        return {
            "total_requests": total_requests,
            "hourly_breakdown": hourly_breakdown,
            "peak_hour": peak_hour,
            "peak_count": peak_count
        }

    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        return {
            "total_requests": 0,
            "hourly_breakdown": [],
            "peak_hour": None,
            "peak_count": 0
        }
