"""Webhook model for event notifications.

This module defines the Webhook model which manages
webhook endpoints for real-time event notifications.
"""

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.user import User


class WebhookEvent(str):
    """Supported webhook event types."""

    STREAM_STARTED = "stream.started"
    STREAM_COMPLETED = "stream.completed"
    STREAM_FAILED = "stream.failed"
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"
    HIGHLIGHT_CREATED = "highlight.created"
    HIGHLIGHT_BATCH_READY = "highlight.batch_ready"


class Webhook(Base, TimestampMixin):
    """Webhook model for managing event notifications.

    Represents webhook endpoints that receive real-time
    notifications about processing events.

    Attributes:
        id: Unique identifier for the webhook
        user_id: Foreign key to the user who owns this webhook
        url: Webhook endpoint URL
        events: JSON array of subscribed event types
        secret: Secret key for webhook signature verification
        active: Whether the webhook is currently active
        created_at: Timestamp when the webhook was created
    """

    __tablename__ = "webhooks"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the webhook"
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the user who owns this webhook",
    )

    url: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Webhook endpoint URL"
    )

    events: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="JSON array of subscribed event types",
    )

    secret: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Secret key for webhook signature verification",
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Whether the webhook is currently active",
    )

    # New fields for domain model compatibility
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="active",
        comment="Webhook status (active, inactive, failed)",
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Optional description of the webhook",
    )

    consecutive_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of consecutive delivery failures",
    )

    total_deliveries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of delivery attempts",
    )

    successful_deliveries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of successful deliveries",
    )

    custom_headers: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Custom headers to send with webhook requests",
    )

    last_delivery_data: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON data about last delivery attempt",
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User", back_populates="webhooks", lazy="joined"
    )

    def is_subscribed_to(self, event: str) -> bool:
        """Check if the webhook is subscribed to a specific event.

        Args:
            event: The event type to check

        Returns:
            bool: True if subscribed to the event, False otherwise
        """
        return event in self.events or "*" in self.events

    def __repr__(self) -> str:
        """String representation of the Webhook."""
        return f"<Webhook(id={self.id}, url='{self.url}', events={len(self.events)}, active={self.active})>"
