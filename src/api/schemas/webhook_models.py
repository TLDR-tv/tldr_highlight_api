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


class WebhookPlatform(str, Enum):
    """Supported webhook platforms."""
    HUNDREDMS = "100ms"
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    CUSTOM = "custom"


class BaseWebhookEvent(BaseModel):
    """Base webhook event model."""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="When the event occurred")
    platform: Optional[str] = Field(None, description="Platform that sent the webhook")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StreamMetadata(BaseModel):
    """Stream-specific metadata."""
    title: Optional[str] = None
    description: Optional[str] = None
    platform_stream_id: Optional[str] = None
    platform_user_id: Optional[str] = None
    platform_username: Optional[str] = None
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


class HundredMSWebhookPayload(BaseModel):
    """100ms-specific webhook payload."""
    version: str = Field(..., description="Event version")
    id: str = Field(..., description="Event ID")
    account_id: str = Field(..., description="100ms account ID")
    timestamp: datetime = Field(..., description="Event timestamp")
    type: str = Field(..., description="100ms event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    
    def to_stream_started_event(self) -> Optional[StreamStartedWebhookEvent]:
        """Convert 100ms payload to stream started event."""
        # Map 100ms events to our event types
        event_mapping = {
            "beam.started.success": WebhookEventType.STREAM_STARTED,
            "hls.started.success": WebhookEventType.STREAM_STARTED,
            "rtmp.started.success": WebhookEventType.STREAM_STARTED,
        }
        
        if self.type not in event_mapping:
            return None
        
        # Extract stream URL from data
        stream_url = None
        if "rtmp_urls" in self.data and self.data["rtmp_urls"]:
            stream_url = self.data["rtmp_urls"][0]
        elif "url" in self.data:
            stream_url = self.data["url"]
        elif "recording_url" in self.data:
            stream_url = self.data["recording_url"]
        
        if not stream_url:
            return None
        
        return StreamStartedWebhookEvent(
            event_id=self.id,
            event_type=event_mapping[self.type].value,
            timestamp=self.timestamp,
            platform=WebhookPlatform.HUNDREDMS.value,
            stream_url=stream_url,
            metadata=StreamMetadata(
                platform_stream_id=self.data.get("room_id"),
                custom_data={
                    "account_id": self.account_id,
                    "session_id": self.data.get("session_id"),
                    "room_name": self.data.get("room_name"),
                }
            )
        )


class TwitchWebhookPayload(BaseModel):
    """Twitch EventSub webhook payload."""
    subscription: Dict[str, Any] = Field(..., description="Subscription details")
    event: Dict[str, Any] = Field(..., description="Event data")
    
    def to_stream_started_event(self) -> Optional[StreamStartedWebhookEvent]:
        """Convert Twitch payload to stream started event."""
        if self.subscription.get("type") != "stream.online":
            return None
        
        # Build Twitch stream URL
        broadcaster_login = self.event.get("broadcaster_user_login")
        if not broadcaster_login:
            return None
        
        stream_url = f"https://www.twitch.tv/{broadcaster_login}"
        
        return StreamStartedWebhookEvent(
            event_id=self.event.get("id", f"twitch_{datetime.utcnow().timestamp()}"),
            event_type=WebhookEventType.STREAM_STARTED.value,
            timestamp=datetime.fromisoformat(self.event.get("started_at", datetime.utcnow().isoformat())),
            platform=WebhookPlatform.TWITCH.value,
            stream_url=stream_url,
            metadata=StreamMetadata(
                platform_stream_id=self.event.get("id"),
                platform_user_id=self.event.get("broadcaster_user_id"),
                platform_username=self.event.get("broadcaster_user_name"),
                custom_data={
                    "broadcaster_login": broadcaster_login,
                    "type": self.event.get("type", "live"),
                }
            )
        )


class WebhookResponse(BaseModel):
    """Standard webhook response."""
    success: bool = Field(..., description="Whether the webhook was processed successfully")
    message: str = Field(..., description="Response message")
    event_id: Optional[str] = Field(None, description="Processed event ID")
    stream_id: Optional[int] = Field(None, description="Created stream ID if applicable")


class WebhookVerificationHeaders(BaseModel):
    """Headers used for webhook verification."""
    signature: Optional[str] = Field(None, description="HMAC signature")
    timestamp: Optional[str] = Field(None, description="Request timestamp")
    event_id: Optional[str] = Field(None, description="Event ID header")
    platform: Optional[str] = Field(None, description="Platform identifier")