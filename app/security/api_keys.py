"""API key generation, validation, and CRUD operations."""

import secrets
import hashlib
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import logging

from app.database.connection import execute_query, execute_insert, execute_update

logger = logging.getLogger(__name__)

# API key format configuration
KEY_PREFIX = "sk_tdvx_"
KEY_LENGTH = 32


def generate_api_key() -> Tuple[str, str, str]:
    """
    Generate a new API key with cryptographically secure randomness.

    Returns:
        Tuple of (full_key, key_hash, key_prefix)
        - full_key: Complete key to give to user (shown only once)
        - key_hash: SHA-256 hash for storage
        - key_prefix: First 12 characters for display/logs

    Example:
        full_key, key_hash, key_prefix = generate_api_key()
        # full_key: "sk_tdvx_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        # key_prefix: "sk_tdvx_a1b2"
    """
    random_part = secrets.token_urlsafe(KEY_LENGTH)[:KEY_LENGTH]
    full_key = f"{KEY_PREFIX}{random_part}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:12]  # "sk_tdvx_" + 4 chars

    return full_key, key_hash, key_prefix


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for comparison with stored hashes.

    Args:
        api_key: Full API key string

    Returns:
        SHA-256 hex digest

    Example:
        key_hash = hash_api_key("sk_tdvx_abc123...")
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_api_key(
    name: str,
    rate_limit_per_hour: int,
    description: Optional[str] = None
) -> Tuple[int, str, str]:
    """
    Create a new API key in the database.

    Args:
        name: Descriptive name for the key
        rate_limit_per_hour: Request limit per hour
        description: Optional description

    Returns:
        Tuple of (key_id, full_key, key_prefix)

    Raises:
        sqlite3.IntegrityError: If key hash collision (extremely unlikely)
        sqlite3.Error: For other database errors

    Example:
        key_id, full_key, prefix = create_api_key(
            name="Production API",
            rate_limit_per_hour=1000,
            description="Main production key"
        )
        # Save full_key securely - it won't be shown again!
    """
    full_key, key_hash, key_prefix = generate_api_key()

    query = """
    INSERT INTO api_keys (key_hash, key_prefix, name, description, rate_limit_per_hour)
    VALUES (?, ?, ?, ?, ?)
    """

    try:
        key_id = execute_insert(query, (key_hash, key_prefix, name, description, rate_limit_per_hour))
        logger.info(f"Created API key: {key_prefix} ({name})")
        return key_id, full_key, key_prefix
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise


def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Validate an API key and return key information if valid.

    Args:
        api_key: Full API key to validate

    Returns:
        Dictionary with key information if valid and active, None otherwise
        Dict contains: id, key_prefix, name, rate_limit_per_hour, is_active, created_at

    Example:
        key_info = validate_api_key("sk_tdvx_abc123...")
        if key_info:
            print(f"Valid key: {key_info['name']}")
            print(f"Rate limit: {key_info['rate_limit_per_hour']}/hour")
        else:
            print("Invalid or revoked key")
    """
    if not api_key or not api_key.startswith(KEY_PREFIX):
        return None

    key_hash = hash_api_key(api_key)

    query = """
    SELECT id, key_prefix, name, rate_limit_per_hour, is_active, created_at
    FROM api_keys
    WHERE key_hash = ? AND is_active = 1
    """

    try:
        results = execute_query(query, (key_hash,))

        if not results:
            return None

        row = results[0]
        return {
            "id": row["id"],
            "key_prefix": row["key_prefix"],
            "name": row["name"],
            "rate_limit_per_hour": row["rate_limit_per_hour"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"]
        }
    except Exception as e:
        logger.error(f"Failed to validate API key: {e}")
        return None


def revoke_api_key(key_id: int, reason: Optional[str] = None) -> bool:
    """
    Revoke an API key (mark as inactive).

    Args:
        key_id: ID of the key to revoke
        reason: Optional reason for revocation

    Returns:
        True if key was revoked, False if key not found or already revoked

    Example:
        success = revoke_api_key(123, reason="Security breach")
        if success:
            print("Key revoked successfully")
    """
    query = """
    UPDATE api_keys
    SET is_active = 0, revoked_at = CURRENT_TIMESTAMP, revoked_reason = ?
    WHERE id = ? AND is_active = 1
    """

    try:
        rows_affected = execute_update(query, (reason, key_id))

        if rows_affected > 0:
            logger.info(f"Revoked API key ID {key_id}: {reason or 'No reason provided'}")
            return True
        else:
            logger.warning(f"API key ID {key_id} not found or already revoked")
            return False
    except Exception as e:
        logger.error(f"Failed to revoke API key: {e}")
        raise


def list_api_keys(include_revoked: bool = False) -> List[Dict[str, Any]]:
    """
    List all API keys.

    Args:
        include_revoked: If True, include revoked keys in the list

    Returns:
        List of dictionaries with key information

    Example:
        keys = list_api_keys(include_revoked=False)
        for key in keys:
            print(f"{key['name']}: {key['key_prefix']}")
    """
    if include_revoked:
        query = """
        SELECT id, key_prefix, name, description, rate_limit_per_hour,
               created_at, last_used_at, is_active, revoked_at, revoked_reason
        FROM api_keys
        ORDER BY created_at DESC
        """
        params = ()
    else:
        query = """
        SELECT id, key_prefix, name, description, rate_limit_per_hour,
               created_at, last_used_at, is_active, revoked_at, revoked_reason
        FROM api_keys
        WHERE is_active = 1
        ORDER BY created_at DESC
        """
        params = ()

    try:
        results = execute_query(query, params)

        return [
            {
                "id": row["id"],
                "key_prefix": row["key_prefix"],
                "name": row["name"],
                "description": row["description"],
                "rate_limit_per_hour": row["rate_limit_per_hour"],
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"],
                "is_active": bool(row["is_active"]),
                "revoked_at": row["revoked_at"],
                "revoked_reason": row["revoked_reason"]
            }
            for row in results
        ]
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise


def get_api_key_details(key_id: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific API key.

    Args:
        key_id: ID of the key

    Returns:
        Dictionary with key information or None if not found

    Example:
        details = get_api_key_details(123)
        if details:
            print(f"Name: {details['name']}")
            print(f"Rate limit: {details['rate_limit_per_hour']}/hour")
    """
    query = """
    SELECT id, key_prefix, name, description, rate_limit_per_hour,
           created_at, last_used_at, is_active, revoked_at, revoked_reason
    FROM api_keys
    WHERE id = ?
    """

    try:
        results = execute_query(query, (key_id,))

        if not results:
            return None

        row = results[0]
        return {
            "id": row["id"],
            "key_prefix": row["key_prefix"],
            "name": row["name"],
            "description": row["description"],
            "rate_limit_per_hour": row["rate_limit_per_hour"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "is_active": bool(row["is_active"]),
            "revoked_at": row["revoked_at"],
            "revoked_reason": row["revoked_reason"]
        }
    except Exception as e:
        logger.error(f"Failed to get API key details: {e}")
        raise


def update_last_used(key_id: int) -> None:
    """
    Update the last_used_at timestamp for an API key.

    Args:
        key_id: ID of the key

    Example:
        update_last_used(123)
    """
    query = """
    UPDATE api_keys
    SET last_used_at = CURRENT_TIMESTAMP
    WHERE id = ?
    """

    try:
        execute_update(query, (key_id,))
    except Exception as e:
        logger.error(f"Failed to update last_used_at: {e}")
        # Non-critical, don't raise


def update_rate_limit(key_id: int, new_limit: int) -> bool:
    """
    Update the rate limit for an API key.

    Args:
        key_id: ID of the key
        new_limit: New rate limit (requests per hour)

    Returns:
        True if updated, False if key not found or revoked

    Example:
        success = update_rate_limit(123, 2000)
        if success:
            print("Rate limit updated")
    """
    query = """
    UPDATE api_keys
    SET rate_limit_per_hour = ?
    WHERE id = ? AND is_active = 1
    """

    try:
        rows_affected = execute_update(query, (new_limit, key_id))

        if rows_affected > 0:
            logger.info(f"Updated rate limit for key ID {key_id} to {new_limit}/hour")
            return True
        else:
            logger.warning(f"API key ID {key_id} not found or already revoked")
            return False
    except Exception as e:
        logger.error(f"Failed to update rate limit: {e}")
        raise
