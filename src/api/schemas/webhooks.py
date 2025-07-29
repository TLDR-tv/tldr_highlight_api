"""Webhook management request/response schemas."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .streams import PaginatedResponse


class WebhookCreate(BaseModel):
    """Request schema for creating a new webhook."""

    url: HttpUrl = Field(description="Webhook endpoint URL")
    events: List[str] = Field(
        min_length=1, description="List of event types to subscribe to"
    )
    secret: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=255,
        description="Optional secret for signature verification (auto-generated if not provided)",
    )
    active: bool = Field(
        default=True, description="Whether the webhook should be active"
    )

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: List[str]) -> List[str]:
        """Validate webhook event types."""
        valid_events = {
            "stream.started",
            "stream.completed",
            "stream.failed",
            "batch.started",
            "batch.completed",
            "batch.failed",
            "highlight.created",
            "highlight.batch_ready",
            "*",  # Subscribe to all events
        }

        for event in v:
            if event not in valid_events:
                raise ValueError(
                    f"Invalid event type: {event}. Valid events: {valid_events}"
                )

        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> HttpUrl:
        """Validate webhook URL."""
        if v.scheme not in ["http", "https"]:
            raise ValueError("Webhook URL must use HTTP or HTTPS")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://api.example.com/webhooks/tldr-highlights",
                "events": ["stream.completed", "highlight.created"],
                "secret": "your-webhook-secret-key",
                "active": True,
            }
        }


class WebhookUpdate(BaseModel):
    """Request schema for updating webhook configuration."""

    url: Optional[HttpUrl] = Field(
        default=None, description="Updated webhook endpoint URL"
    )
    events: Optional[List[str]] = Field(
        default=None, min_length=1, description="Updated list of event types"
    )
    secret: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=255,
        description="Updated secret for signature verification",
    )
    active: Optional[bool] = Field(default=None, description="Updated active status")

    @field_validator("events")
    @classmethod
    def validate_events(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate webhook event types."""
        if v is None:
            return v

        valid_events = {
            "stream.started",
            "stream.completed",
            "stream.failed",
            "batch.started",
            "batch.completed",
            "batch.failed",
            "highlight.created",
            "highlight.batch_ready",
            "*",
        }

        for event in v:
            if event not in valid_events:
                raise ValueError(
                    f"Invalid event type: {event}. Valid events: {valid_events}"
                )

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "events": ["stream.completed", "batch.completed", "highlight.created"],
                "active": True,
            }
        }


class WebhookTest(BaseModel):
    """Request schema for testing webhook delivery."""

    event_type: str = Field(description="Event type to test")
    test_payload: bool = Field(
        default=True, description="Whether to send a test payload or real event data"
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type for testing."""
        valid_events = {
            "stream.started",
            "stream.completed",
            "stream.failed",
            "batch.started",
            "batch.completed",
            "batch.failed",
            "highlight.created",
            "highlight.batch_ready",
        }

        if v not in valid_events:
            raise ValueError(f"Invalid event type: {v}. Valid events: {valid_events}")

        return v

    class Config:
        json_schema_extra = {
            "example": {"event_type": "highlight.created", "test_payload": True}
        }


class WebhookResponse(BaseModel):
    """Response schema for webhook details."""

    id: int = Field(description="Unique webhook identifier")
    user_id: int = Field(description="Owner user ID")
    url: str = Field(description="Webhook endpoint URL")
    events: List[str] = Field(description="Subscribed event types")
    secret: str = Field(description="Secret for signature verification")
    active: bool = Field(description="Whether webhook is active")
    created_at: datetime = Field(description="Creation timestamp")

    # Stats
    total_deliveries: int = Field(
        default=0, description="Total number of delivery attempts"
    )
    successful_deliveries: int = Field(
        default=0, description="Number of successful deliveries"
    )
    last_delivery_at: Optional[datetime] = Field(
        default=None, description="Timestamp of last delivery attempt"
    )
    last_successful_delivery_at: Optional[datetime] = Field(
        default=None, description="Timestamp of last successful delivery"
    )

    # Health
    is_healthy: bool = Field(
        default=True, description="Whether webhook is responding successfully"
    )
    failure_count: int = Field(default=0, description="Consecutive failure count")
    next_retry_at: Optional[datetime] = Field(
        default=None, description="Next retry timestamp (if failing)"
    )

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 987,
                "user_id": 456,
                "url": "https://api.example.com/webhooks/tldr-highlights",
                "events": ["stream.completed", "highlight.created"],
                "secret": "wh_secret_abcd1234...",
                "active": True,
                "created_at": "2024-01-15T10:30:00Z",
                "total_deliveries": 247,
                "successful_deliveries": 243,
                "last_delivery_at": "2024-01-15T14:22:30Z",
                "last_successful_delivery_at": "2024-01-15T14:22:30Z",
                "is_healthy": True,
                "failure_count": 0,
                "next_retry_at": None,
            }
        }


class WebhookTestResponse(BaseModel):
    """Response schema for webhook test results."""

    success: bool = Field(description="Whether test delivery succeeded")
    status_code: Optional[int] = Field(description="HTTP response status code")
    response_time_ms: Optional[int] = Field(description="Response time in milliseconds")
    error_message: Optional[str] = Field(description="Error message if failed")
    headers_sent: dict = Field(description="Headers sent with the request")
    payload_sent: dict = Field(description="Payload sent with the request")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "status_code": 200,
                "response_time_ms": 156,
                "error_message": None,
                "headers_sent": {
                    "Content-Type": "application/json",
                    "X-TLDR-Signature": "sha256=abc123...",
                    "X-TLDR-Event": "highlight.created",
                },
                "payload_sent": {
                    "event": "highlight.created",
                    "timestamp": "2024-01-15T14:30:00Z",
                    "data": {"test": True, "highlight_id": 12345},
                },
            }
        }


class WebhookListResponse(PaginatedResponse):
    """Response schema for paginated webhook list."""

    items: List[WebhookResponse] = Field(description="Webhook items")

    class Config:
        json_schema_extra = {
            "example": {
                "page": 1,
                "per_page": 20,
                "total": 3,
                "pages": 1,
                "has_next": False,
                "has_prev": False,
                "items": [
                    {
                        "id": 987,
                        "url": "https://api.example.com/webhooks/tldr-highlights",
                        "events": ["stream.completed", "highlight.created"],
                        "active": True,
                        "is_healthy": True,
                        "total_deliveries": 247,
                        "successful_deliveries": 243,
                    }
                ],
            }
        }
