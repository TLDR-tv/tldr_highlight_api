"""Organization request and response schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict


class OrganizationResponse(BaseModel):
    """Organization response model."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    slug: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    webhook_url: Optional[str] = None
    has_webhook_configured: bool
    wake_words: list[str]
    
    @field_validator('wake_words', mode='before')
    @classmethod
    def convert_wake_words(cls, v):
        """Convert set to list for JSON serialization."""
        if isinstance(v, set):
            return sorted(list(v))
        return v


class OrganizationUpdateRequest(BaseModel):
    """Update organization request."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    webhook_url: Optional[str] = None
    
    @field_validator('webhook_url')
    @classmethod
    def validate_webhook_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate webhook URL if provided."""
        if v and not (v.startswith('https://') or v.startswith('http://')):
            raise ValueError('Webhook URL must start with https:// or http://')
        return v


class WebhookSecretResponse(BaseModel):
    """Webhook secret response."""
    
    webhook_secret: str
    message: str = "Store this secret securely. It will not be shown again."


class WakeWordRequest(BaseModel):
    """Wake word request."""
    
    wake_word: str = Field(..., min_length=1, max_length=50)


class OrganizationUsageResponse(BaseModel):
    """Organization usage statistics response."""
    
    organization_id: UUID
    name: str
    total_streams_processed: int
    total_highlights_generated: int
    total_processing_seconds: float
    total_processing_hours: float
    avg_highlights_per_stream: float
    avg_processing_seconds_per_stream: float
    created_at: str
    is_active: bool


class MessageResponse(BaseModel):
    """Generic message response."""
    
    message: str