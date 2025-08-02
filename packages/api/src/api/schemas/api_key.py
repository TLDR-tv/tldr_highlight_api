"""API key request and response schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class APIKeyResponse(BaseModel):
    """API key response model."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    prefix: str
    scopes: list[str]
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool


class APIKeyCreateRequest(BaseModel):
    """Create API key request."""

    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(..., min_items=1)


class APIKeyCreateResponse(BaseModel):
    """Response when creating a new API key."""

    api_key: APIKeyResponse
    raw_key: str
    message: str = "Store this key securely. It will not be shown again."


class APIKeyListResponse(BaseModel):
    """API key list response."""

    api_keys: list[APIKeyResponse]
    total: int
