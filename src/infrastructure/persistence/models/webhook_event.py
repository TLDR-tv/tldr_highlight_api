"""SQLAlchemy model for webhook events."""

from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum as PyEnum

from sqlalchemy import (
    Integer,
    String,
    Text,
    DateTime,
    Enum,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.stream import Stream
    from src.infrastructure.persistence.models.user import User


# Define enums locally to avoid domain dependency
class WebhookEventStatus(str, PyEnum):
    """Status of webhook event processing."""

    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRY = "retry"


class WebhookEventType(str, PyEnum):
    """Type of webhook event."""

    STREAM_STARTED = "stream.started"
    STREAM_ENDED = "stream.ended"
    STREAM_UPDATE = "stream.update"
    HIGHLIGHT_CREATED = "highlight.created"
    VIDEO_UPLOADED = "video.uploaded"


class WebhookEvent(Base):
    """SQLAlchemy model for webhook events."""

    __tablename__ = "webhook_events"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Event identification
    event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[WebhookEventType] = mapped_column(
        Enum(WebhookEventType), nullable=False
    )
    platform: Mapped[str] = mapped_column(String(50), nullable=False)

    # Processing status
    status: Mapped[WebhookEventStatus] = mapped_column(
        Enum(WebhookEventStatus), nullable=False, default=WebhookEventStatus.RECEIVED
    )

    # Event payload
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Processing information
    stream_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("streams.id"), nullable=True
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamps
    received_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    stream: Mapped[Optional["Stream"]] = relationship(
        "Stream", back_populates="webhook_events"
    )
    user: Mapped[Optional["User"]] = relationship(
        "User", back_populates="webhook_events"
    )

    # Indexes
    __table_args__ = (
        # Unique constraint on external event ID per platform
        Index("idx_webhook_event_external_id", "event_id", "platform", unique=True),
        # Index for finding events by status
        Index("idx_webhook_event_status", "status"),
        # Index for finding events by user
        Index("idx_webhook_event_user", "user_id"),
        # Index for finding events by stream
        Index("idx_webhook_event_stream", "stream_id"),
        # Index for finding failed events for retry
        Index("idx_webhook_event_retry", "status", "retry_count"),
    )
