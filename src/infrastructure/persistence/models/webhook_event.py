"""SQLAlchemy model for webhook events."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Enum,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship
from datetime import datetime

from src.infrastructure.persistence.models.base import Base
from src.domain.entities.webhook_event import WebhookEventStatus, WebhookEventType


class WebhookEvent(Base):
    """SQLAlchemy model for webhook events."""

    __tablename__ = "webhook_events"

    # Primary key
    id = Column(Integer, primary_key=True)

    # Event identification
    event_id = Column(String(255), nullable=False)
    event_type = Column(Enum(WebhookEventType), nullable=False)
    platform = Column(String(50), nullable=False)

    # Processing status
    status = Column(
        Enum(WebhookEventStatus), nullable=False, default=WebhookEventStatus.RECEIVED
    )

    # Event payload
    payload = Column(JSON, nullable=False, default=dict)

    # Processing information
    stream_id = Column(Integer, ForeignKey("streams.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    processed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    received_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    stream = relationship("Stream", back_populates="webhook_events")
    user = relationship("User", back_populates="webhook_events")

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
