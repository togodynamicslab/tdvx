"""Pydantic models for API key management."""

from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from datetime import datetime


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Descriptive name for the API key"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional description of the key's purpose"
    )
    rate_limit_per_hour: int = Field(
        ...,
        gt=0,
        le=10000,
        description="Maximum number of requests per hour"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Production API",
                    "description": "Main production access",
                    "rate_limit_per_hour": 1000
                }
            ]
        }
    }


class CreateAPIKeyResponse(BaseModel):
    """Response model for API key creation (includes full key ONCE)."""

    id: int = Field(..., description="Database ID of the API key")
    api_key: str = Field(
        ...,
        description="SAVE THIS! The full API key will not be shown again"
    )
    key_prefix: str = Field(..., description="Key prefix for identification")
    name: str = Field(..., description="Name of the API key")
    description: Optional[str] = Field(None, description="Description")
    rate_limit_per_hour: int = Field(..., description="Rate limit per hour")
    created_at: str = Field(..., description="Creation timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "api_key": "sk_tdvx_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                    "key_prefix": "sk_tdvx_a1b2",
                    "name": "Production API",
                    "description": "Main production access",
                    "rate_limit_per_hour": 1000,
                    "created_at": "2025-12-02T15:30:00"
                }
            ]
        }
    }


class APIKeyListItem(BaseModel):
    """Model for API key in list view (without full key)."""

    id: int = Field(..., description="Database ID")
    key_prefix: str = Field(..., description="Key prefix (first 12 chars)")
    name: str = Field(..., description="Name of the API key")
    description: Optional[str] = Field(None, description="Description")
    rate_limit_per_hour: int = Field(..., description="Rate limit per hour")
    created_at: str = Field(..., description="Creation timestamp")
    last_used_at: Optional[str] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Whether the key is active")
    revoked_at: Optional[str] = Field(None, description="Revocation timestamp")
    revoked_reason: Optional[str] = Field(None, description="Reason for revocation")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "key_prefix": "sk_tdvx_a1b2",
                    "name": "Production API",
                    "description": "Main production access",
                    "rate_limit_per_hour": 1000,
                    "created_at": "2025-12-02T15:30:00",
                    "last_used_at": "2025-12-02T16:45:00",
                    "is_active": True,
                    "revoked_at": None,
                    "revoked_reason": None
                }
            ]
        }
    }


class APIKeyDetailResponse(BaseModel):
    """Detailed API key information with usage statistics."""

    id: int = Field(..., description="Database ID")
    key_prefix: str = Field(..., description="Key prefix (first 12 chars)")
    name: str = Field(..., description="Name of the API key")
    description: Optional[str] = Field(None, description="Description")
    rate_limit_per_hour: int = Field(..., description="Rate limit per hour")
    created_at: str = Field(..., description="Creation timestamp")
    last_used_at: Optional[str] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Whether the key is active")
    revoked_at: Optional[str] = Field(None, description="Revocation timestamp")
    revoked_reason: Optional[str] = Field(None, description="Reason for revocation")
    current_hour_usage: int = Field(..., description="Requests used this hour")
    usage_stats_24h: dict = Field(..., description="Usage statistics for last 24 hours")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "key_prefix": "sk_tdvx_a1b2",
                    "name": "Production API",
                    "description": "Main production access",
                    "rate_limit_per_hour": 1000,
                    "created_at": "2025-12-02T15:30:00",
                    "last_used_at": "2025-12-02T16:45:00",
                    "is_active": True,
                    "revoked_at": None,
                    "revoked_reason": None,
                    "current_hour_usage": 45,
                    "usage_stats_24h": {
                        "total_requests": 523,
                        "peak_hour": "2025-12-02 14:00",
                        "peak_count": 87
                    }
                }
            ]
        }
    }


class UpdateRateLimitRequest(BaseModel):
    """Request model for updating rate limit."""

    rate_limit_per_hour: int = Field(
        ...,
        gt=0,
        le=10000,
        description="New rate limit (requests per hour)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rate_limit_per_hour": 2000
                }
            ]
        }
    }


class RevokeAPIKeyRequest(BaseModel):
    """Request model for revoking an API key."""

    reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional reason for revocation"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reason": "Key compromised in security incident"
                }
            ]
        }
    }


class ModelInfo(BaseModel):
    """Information about available transcription models."""

    name: str = Field(..., description="Model display name")
    whisper_model: str = Field(..., description="Underlying Whisper model")
    uses_faster_whisper: bool = Field(..., description="Whether it uses faster-whisper")
    description: str = Field(..., description="Model description")
    estimated_speed: str = Field(..., description="Processing speed estimate")


class ModelsResponse(BaseModel):
    """Response model for /models endpoint."""

    default_model: str = Field(..., description="Default model identifier")
    available_models: dict = Field(..., description="Dictionary of available models")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "default_model": "tdv1-fast",
                    "available_models": {
                        "tdv1": {
                            "name": "TDv1",
                            "whisper_model": "large-v3",
                            "uses_faster_whisper": False,
                            "description": "High quality pipeline",
                            "estimated_speed": "~15-20s per 10s of audio"
                        },
                        "tdv1-fast": {
                            "name": "TDv1-Fast",
                            "whisper_model": "medium",
                            "uses_faster_whisper": True,
                            "description": "Real-time pipeline",
                            "estimated_speed": "~4-6s per 10s of audio"
                        }
                    }
                }
            ]
        }
    }
