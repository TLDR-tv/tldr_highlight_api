"""API key request and response models."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class APIKeyResponse(BaseModel):
    """API key response model."""
    
    id: UUID
    name: str
    prefix: str
    scopes: list[str]
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool
    
    class Config:
        from_attributes = True


class APIKeyCreateRequest(BaseModel):
    """Create API key request."""
    
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(..., min_items=1)