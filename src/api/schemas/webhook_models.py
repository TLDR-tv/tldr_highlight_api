"""Webhook request and response models."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class WebhookEventType(str, Enum):
    """Supported webhook event types."""

    STREAM_STARTED = "stream.started"
    STREAM_ENDED = "stream.ended"
    STREAM_ERROR = "stream.error"
    RECORDING_STARTED = "recording.started"
    RECORDING_COMPLETED = "recording.completed"
    HLS_STARTED = "hls.started"
    RTMP_STARTED = "rtmp.started"


class BaseWebhookEvent(BaseModel):
    """Base webhook event model."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="When the event occurred")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class StreamMetadata(BaseModel):
    """Stream-specific metadata."""

    title: Optional[str] = None
    description: Optional[str] = None
    external_stream_id: Optional[str] = Field(
        None, description="External system's stream ID"
    )
    external_user_id: Optional[str] = Field(
        None, description="External system's user ID"
    )
    external_username: Optional[str] = Field(
        None, description="External system's username"
    )
    tags: List[str] = Field(default_factory=list)
    custom_data: Dict[str, Any] = Field(default_factory=dict)


class StreamStartedWebhookEvent(BaseWebhookEvent):
    """Stream started webhook event."""

    event_type: str = Field(default=WebhookEventType.STREAM_STARTED.value)
    stream_url: str = Field(..., description="URL of the stream")
    user_id: Optional[int] = Field(None, description="Internal user ID if mapped")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    metadata: StreamMetadata = Field(default_factory=StreamMetadata)

    @validator("stream_url")
    def validate_stream_url(cls, v):
        """Validate stream URL format."""
        if not v or not v.startswith(("http://", "https://", "rtmp://", "rtmps://")):
            raise ValueError("Invalid stream URL format")
        return v


class WebhookResponse(BaseModel):
    """Standard webhook response."""

    success: bool = Field(
        ..., description="Whether the webhook was processed successfully"
    )
    message: str = Field(..., description="Response message")
    event_id: Optional[str] = Field(None, description="Processed event ID")
    stream_id: Optional[int] = Field(
        None, description="Created stream ID if applicable"
    )


class WebhookVerificationHeaders(BaseModel):
    """Headers used for webhook verification."""

    signature: Optional[str] = Field(None, description="HMAC signature")
    timestamp: Optional[str] = Field(None, description="Request timestamp")
    event_id: Optional[str] = Field(None, description="Event ID header")
