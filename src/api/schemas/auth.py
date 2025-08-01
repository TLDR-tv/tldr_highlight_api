"""Authentication and API key management schemas.

This module defines Pydantic models for API key management endpoints,
including request/response schemas with validation and documentation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class APIKeyCreate(BaseModel):
    """Request schema for creating a new API key."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for the API key",
        example="Production API Key",
    )

    scopes: List[str] = Field(
        default_factory=list,
        description="List of permission scopes for the API key",
        example=["streams:read", "streams:write", "highlights:read"],
    )

    expires_at: Optional[datetime] = Field(
        None,
        description="Optional expiration timestamp for the API key",
        example="2024-12-31T23:59:59Z",
    )

    @field_validator("scopes")
    def validate_scopes(cls, v):
        """Validate that scopes are from the allowed list."""
        allowed_scopes = {
            "streams:read",
            "streams:write",
            "streams:delete",
            "highlights:read",
            "highlights:write",
            "highlights:delete",
            "webhooks:read",
            "webhooks:write",
            "webhooks:delete",
            "organizations:read",
            "organizations:write",
            "users:read",
            "users:write",
            "admin",
        }

        for scope in v:
            if scope not in allowed_scopes:
                raise ValueError(f"Invalid scope: {scope}")

        return v


class APIKeyUpdate(BaseModel):
    """Request schema for updating an API key."""

    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Human-readable name for the API key",
        example="Updated Production API Key",
    )

    scopes: Optional[List[str]] = Field(
        None,
        description="List of permission scopes for the API key",
        example=["streams:read", "streams:write", "highlights:read"],
    )

    active: Optional[bool] = Field(
        None, description="Whether the API key is active", example=True
    )

    @field_validator("scopes")
    def validate_scopes(cls, v):
        """Validate that scopes are from the allowed list."""
        if v is None:
            return v

        allowed_scopes = {
            "streams:read",
            "streams:write",
            "streams:delete",
            "highlights:read",
            "highlights:write",
            "highlights:delete",
            "webhooks:read",
            "webhooks:write",
            "webhooks:delete",
            "organizations:read",
            "organizations:write",
            "users:read",
            "users:write",
            "admin",
        }

        for scope in v:
            if scope not in allowed_scopes:
                raise ValueError(f"Invalid scope: {scope}")

        return v


class APIKeyResponse(BaseModel):
    """Response schema for API key information (without the actual key)."""

    id: int = Field(..., description="Unique identifier for the API key", example=123)

    name: str = Field(
        ...,
        description="Human-readable name for the API key",
        example="Production API Key",
    )

    scopes: List[str] = Field(
        ...,
        description="List of permission scopes for the API key",
        example=["streams:read", "streams:write", "highlights:read"],
    )

    active: bool = Field(..., description="Whether the API key is active", example=True)

    created_at: datetime = Field(
        ...,
        description="Timestamp when the API key was created",
        example="2024-01-15T10:30:00Z",
    )

    expires_at: Optional[datetime] = Field(
        None,
        description="Optional expiration timestamp for the API key",
        example="2024-12-31T23:59:59Z",
    )

    last_used_at: Optional[datetime] = Field(
        None,
        description="Timestamp of last API call using this key",
        example="2024-01-20T14:45:30Z",
    )

    is_expired: bool = Field(
        ..., description="Whether the API key has expired", example=False
    )

    class Config:
        from_attributes = True


class APIKeyCreateResponse(APIKeyResponse):
    """Response schema for API key creation (includes the actual key once)."""

    key: str = Field(
        ...,
        description="The actual API key string (shown only once)",
        example="tldr_sk_1234567890abcdef",
    )


class APIKeyRotateResponse(BaseModel):
    """Response schema for API key rotation."""

    id: int = Field(..., description="Unique identifier for the API key", example=123)

    key: str = Field(
        ...,
        description="The new API key string (shown only once)",
        example="tldr_sk_abcdef1234567890",
    )

    rotated_at: datetime = Field(
        ...,
        description="Timestamp when the key was rotated",
        example="2024-01-20T15:00:00Z",
    )


class APIKeyListResponse(BaseModel):
    """Response schema for listing API keys."""

    total: int = Field(..., description="Total number of API keys", example=5)

    keys: List[APIKeyResponse] = Field(..., description="List of API keys")


class LoginRequest(BaseModel):
    """Request schema for user login."""

    email: str = Field(
        ..., description="User's email address", example="user@company.com"
    )

    password: str = Field(
        ..., min_length=8, description="User's password", example="securepassword123"
    )


class LoginResponse(BaseModel):
    """Response schema for successful login."""

    access_token: str = Field(
        ...,
        description="JWT access token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    )

    token_type: str = Field(
        default="bearer", description="Token type", example="bearer"
    )

    expires_in: int = Field(
        ..., description="Token expiration time in seconds", example=3600
    )


class TokenData(BaseModel):
    """Data contained in JWT tokens."""

    user_id: int
    email: str
    scopes: List[str] = Field(default_factory=list)
