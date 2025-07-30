"""Webhook event domain entity."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp


class WebhookEventStatus(Enum):
    """Status of webhook event processing."""

    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class WebhookEventType(Enum):
    """Types of webhook events."""

    STREAM_STARTED = "stream.started"
    STREAM_ENDED = "stream.ended"
    STREAM_ERROR = "stream.error"
    RECORDING_STARTED = "recording.started"
    RECORDING_COMPLETED = "recording.completed"
    CUSTOM = "custom"


@dataclass
class WebhookEvent(Entity[int]):
    """Domain entity representing a received webhook event."""

    # Required fields
    event_id: str
    event_type: WebhookEventType
    platform: str
    received_at: Timestamp

    # Optional fields with defaults
    status: WebhookEventStatus = WebhookEventStatus.RECEIVED
    payload: Dict[str, Any] = field(default_factory=dict)

    # Processing information
    stream_id: Optional[int] = None
    user_id: Optional[int] = None
    processed_at: Optional[Timestamp] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def mark_processing(self) -> "WebhookEvent":
        """Mark event as being processed."""
        self.status = WebhookEventStatus.PROCESSING
        self.updated_at = Timestamp.now()
        return self

    def mark_processed(self, stream_id: int) -> "WebhookEvent":
        """Mark event as successfully processed."""
        self.status = WebhookEventStatus.PROCESSED
        self.stream_id = stream_id
        self.processed_at = Timestamp.now()
        self.updated_at = Timestamp.now()
        self.error_message = None
        return self

    def mark_failed(self, error: str) -> "WebhookEvent":
        """Mark event as failed."""
        self.status = WebhookEventStatus.FAILED
        self.error_message = error
        self.retry_count += 1
        self.updated_at = Timestamp.now()
        return self

    def mark_duplicate(self) -> "WebhookEvent":
        """Mark event as duplicate."""
        self.status = WebhookEventStatus.DUPLICATE
        self.processed_at = Timestamp.now()
        self.updated_at = Timestamp.now()
        return self

    @property
    def is_retryable(self) -> bool:
        """Check if event can be retried."""
        return (
            self.status == WebhookEventStatus.FAILED
            and self.retry_count < 3  # Max 3 retries
        )

    @property
    def processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.processed_at and self.received_at:
            return (self.processed_at.value - self.received_at.value).total_seconds()
        return None
