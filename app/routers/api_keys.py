"""API key management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import logging

from app.models.api_key import (
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    APIKeyListItem,
    APIKeyDetailResponse,
    UpdateRateLimitRequest,
    RevokeAPIKeyRequest
)
from app.security.dependencies import verify_master_key
from app.security.api_keys import (
    create_api_key,
    list_api_keys,
    get_api_key_details,
    update_rate_limit,
    revoke_api_key
)
from app.security.rate_limiter import get_current_usage, get_usage_stats

logger = logging.getLogger(__name__)

# Create router with master key authentication for all endpoints
router = APIRouter(
    prefix="/api/keys",
    tags=["API Keys Management"],
    dependencies=[Depends(verify_master_key)]  # All endpoints require master key
)


@router.post("", response_model=CreateAPIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_new_api_key(request: CreateAPIKeyRequest):
    """
    Create a new API key.

    **Requires Master API Key in Authorization header.**

    The full API key is returned ONLY ONCE in this response - save it securely!
    After this, only the key prefix will be shown.

    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/keys" \\
      -H "Authorization: Bearer master_tdvx_xxx" \\
      -H "Content-Type: application/json" \\
      -d '{
        "name": "Production API",
        "description": "Main production key",
        "rate_limit_per_hour": 1000
      }'
    ```
    """
    try:
        key_id, full_key, key_prefix = create_api_key(
            name=request.name,
            rate_limit_per_hour=request.rate_limit_per_hour,
            description=request.description
        )

        # Get details for response
        details = get_api_key_details(key_id)

        if not details:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created key details"
            )

        logger.info(f"Created API key: {key_prefix} ({request.name})")

        return CreateAPIKeyResponse(
            id=key_id,
            api_key=full_key,  # ONLY TIME full key is shown!
            key_prefix=key_prefix,
            name=details["name"],
            description=details["description"],
            rate_limit_per_hour=details["rate_limit_per_hour"],
            created_at=details["created_at"]
        )

    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}"
        )


@router.get("", response_model=List[APIKeyListItem])
async def list_all_api_keys(include_revoked: bool = False):
    """
    List all API keys.

    **Requires Master API Key in Authorization header.**

    By default, only active keys are shown. Set `include_revoked=true`
    to include revoked keys.

    **Example:**
    ```bash
    # List active keys only
    curl "http://localhost:8000/api/keys" \\
      -H "Authorization: Bearer master_tdvx_xxx"

    # List all keys including revoked
    curl "http://localhost:8000/api/keys?include_revoked=true" \\
      -H "Authorization: Bearer master_tdvx_xxx"
    ```
    """
    try:
        keys = list_api_keys(include_revoked=include_revoked)
        return [APIKeyListItem(**key) for key in keys]

    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}"
        )


@router.get("/{key_id}", response_model=APIKeyDetailResponse)
async def get_api_key(key_id: int):
    """
    Get detailed information about a specific API key.

    **Requires Master API Key in Authorization header.**

    Includes usage statistics for the last 24 hours.

    **Example:**
    ```bash
    curl "http://localhost:8000/api/keys/1" \\
      -H "Authorization: Bearer master_tdvx_xxx"
    ```
    """
    try:
        # Get key details
        details = get_api_key_details(key_id)

        if not details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key with ID {key_id} not found"
            )

        # Get usage statistics
        current_usage = get_current_usage(key_id)
        usage_24h = get_usage_stats(key_id, hours=24)

        return APIKeyDetailResponse(
            **details,
            current_hour_usage=current_usage["request_count"],
            usage_stats_24h=usage_24h
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get API key details: {str(e)}"
        )


@router.patch("/{key_id}/rate-limit")
async def update_api_key_rate_limit(
    key_id: int,
    request: UpdateRateLimitRequest
):
    """
    Update the rate limit for an API key.

    **Requires Master API Key in Authorization header.**

    **Example:**
    ```bash
    curl -X PATCH "http://localhost:8000/api/keys/1/rate-limit" \\
      -H "Authorization: Bearer master_tdvx_xxx" \\
      -H "Content-Type: application/json" \\
      -d '{"rate_limit_per_hour": 2000}'
    ```
    """
    try:
        success = update_rate_limit(key_id, request.rate_limit_per_hour)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key with ID {key_id} not found or is revoked"
            )

        logger.info(f"Updated rate limit for key ID {key_id} to {request.rate_limit_per_hour}/hour")

        return {
            "message": "Rate limit updated successfully",
            "key_id": key_id,
            "new_rate_limit": request.rate_limit_per_hour
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update rate limit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update rate limit: {str(e)}"
        )


@router.delete("/{key_id}")
async def revoke_api_key_endpoint(
    key_id: int,
    request: RevokeAPIKeyRequest = None
):
    """
    Revoke an API key.

    **Requires Master API Key in Authorization header.**

    Once revoked, the key can no longer be used. Revoked keys cannot be
    reactivated - create a new key if needed.

    **Example:**
    ```bash
    curl -X DELETE "http://localhost:8000/api/keys/1" \\
      -H "Authorization: Bearer master_tdvx_xxx" \\
      -H "Content-Type: application/json" \\
      -d '{"reason": "Key compromised"}'
    ```
    """
    try:
        reason = request.reason if request else None
        success = revoke_api_key(key_id, reason=reason)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key with ID {key_id} not found or already revoked"
            )

        logger.info(f"Revoked API key ID {key_id}: {reason or 'No reason provided'}")

        return {
            "message": "API key revoked successfully",
            "key_id": key_id,
            "reason": reason
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke API key: {str(e)}"
        )
