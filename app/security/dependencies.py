"""FastAPI dependencies for authentication."""

from fastapi import Header, HTTPException, status
from typing import Optional, Annotated
import logging

from app.config import settings
from app.security.api_keys import validate_api_key, update_last_used
from app.security.rate_limiter import (
    check_rate_limit,
    increment_usage,
    get_next_hour_reset_time
)

logger = logging.getLogger(__name__)


class APIKeyInfo:
    """Container for authenticated API key information."""

    def __init__(self, key_id: int, name: str, rate_limit: int):
        """
        Initialize API key info.

        Args:
            key_id: Database ID of the key
            name: Descriptive name of the key
            rate_limit: Rate limit (requests per hour)
        """
        self.id = key_id
        self.name = name
        self.rate_limit = rate_limit


async def verify_api_key(
    authorization: Annotated[Optional[str], Header()] = None
) -> APIKeyInfo:
    """
    FastAPI dependency to verify API key from Authorization header.

    Expected format: 'Bearer sk_tdvx_xxx'

    Args:
        authorization: Authorization header value

    Returns:
        APIKeyInfo object with authenticated key information

    Raises:
        HTTPException 401: If key is missing, invalid, or revoked
        HTTPException 429: If rate limit is exceeded

    Example:
        @app.get("/protected", dependencies=[Depends(verify_api_key)])
        async def protected_endpoint():
            return {"message": "Access granted"}
    """
    # Check if authentication is enabled
    if not settings.enable_api_key_auth:
        # If disabled, create a dummy key info
        return APIKeyInfo(key_id=0, name="Auth Disabled", rate_limit=999999)

    # 1. Check for Authorization header
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 2. Parse Bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format. Expected: 'Bearer <api_key>'",
            headers={"WWW-Authenticate": "Bearer"}
        )

    api_key = parts[1]

    # 3. Validate API key
    key_info = validate_api_key(api_key)
    if not key_info:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 4. Check rate limit
    is_allowed, current_count, remaining = check_rate_limit(
        key_info["id"],
        key_info["rate_limit_per_hour"]
    )

    if not is_allowed:
        logger.warning(
            f"Rate limit exceeded for {key_info['key_prefix']} ({key_info['name']})"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {key_info['rate_limit_per_hour']}/hour",
            headers={
                "X-RateLimit-Limit": str(key_info["rate_limit_per_hour"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": get_next_hour_reset_time(),
                "Retry-After": "3600"  # Retry after 1 hour
            }
        )

    # 5. Increment usage counter
    try:
        increment_usage(key_info["id"])
    except Exception as e:
        logger.error(f"Failed to increment usage: {e}")
        # Don't block request if counter fails

    # 6. Update last_used timestamp (async, non-blocking)
    try:
        update_last_used(key_info["id"])
    except Exception as e:
        logger.error(f"Failed to update last_used: {e}")
        # Don't block request if update fails

    # Log successful authentication
    logger.info(
        f"Authenticated: {key_info['key_prefix']} ({key_info['name']}) "
        f"- {remaining - 1} requests remaining"
    )

    return APIKeyInfo(
        key_id=key_info["id"],
        name=key_info["name"],
        rate_limit=key_info["rate_limit_per_hour"]
    )


async def verify_master_key(
    authorization: Annotated[Optional[str], Header()] = None
) -> None:
    """
    FastAPI dependency to verify Master API key for admin endpoints.

    Expected format: 'Bearer <master_key>'

    Args:
        authorization: Authorization header value

    Raises:
        HTTPException 401: If master key is missing or invalid

    Example:
        @app.post("/api/keys", dependencies=[Depends(verify_master_key)])
        async def create_key():
            return {"message": "Admin access granted"}
    """
    # 1. Check for Authorization header
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 2. Parse Bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format. Expected: 'Bearer <master_key>'",
            headers={"WWW-Authenticate": "Bearer"}
        )

    provided_key = parts[1]

    # 3. Compare with configured master key
    if provided_key != settings.master_api_key:
        logger.warning("Invalid master key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid master API key",
            headers={"WWW-Authenticate": "Bearer"}
        )

    logger.info("Master key validated - admin access granted")


def verify_api_key_for_websocket(api_key: Optional[str]) -> APIKeyInfo:
    """
    Verify API key for WebSocket connections (query parameter).

    Unlike HTTP endpoints, WebSocket handshakes don't support custom headers
    in browsers, so the key must be passed as a query parameter.

    Args:
        api_key: API key from query parameter

    Returns:
        APIKeyInfo object with authenticated key information

    Raises:
        HTTPException 401: If key is missing, invalid, or revoked
        HTTPException 429: If rate limit is exceeded

    Example:
        @app.websocket("/ws/transcribe")
        async def websocket_endpoint(
            websocket: WebSocket,
            api_key: Optional[str] = Query(None)
        ):
            key_info = verify_api_key_for_websocket(api_key)
            await websocket.accept()
            # ... handle WebSocket connection
    """
    # Check if authentication is enabled
    if not settings.enable_api_key_auth:
        return APIKeyInfo(key_id=0, name="Auth Disabled", rate_limit=999999)

    # 1. Check for API key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing api_key query parameter"
        )

    # 2. Validate API key
    key_info = validate_api_key(api_key)
    if not key_info:
        logger.warning("Invalid API key attempt (WebSocket)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key"
        )

    # 3. Check rate limit
    is_allowed, current_count, remaining = check_rate_limit(
        key_info["id"],
        key_info["rate_limit_per_hour"]
    )

    if not is_allowed:
        logger.warning(
            f"Rate limit exceeded for {key_info['key_prefix']} ({key_info['name']}) (WebSocket)"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {key_info['rate_limit_per_hour']}/hour"
        )

    # 4. Increment usage counter
    try:
        increment_usage(key_info["id"])
    except Exception as e:
        logger.error(f"Failed to increment usage (WebSocket): {e}")

    # 5. Update last_used timestamp
    try:
        update_last_used(key_info["id"])
    except Exception as e:
        logger.error(f"Failed to update last_used (WebSocket): {e}")

    logger.info(
        f"Authenticated (WebSocket): {key_info['key_prefix']} ({key_info['name']}) "
        f"- {remaining - 1} requests remaining"
    )

    return APIKeyInfo(
        key_id=key_info["id"],
        name=key_info["name"],
        rate_limit=key_info["rate_limit_per_hour"]
    )
