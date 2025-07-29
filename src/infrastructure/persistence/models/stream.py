"""Stream model for livestream processing.

This module defines the Stream model which represents
livestreams being processed for highlight extraction.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.highlight import Highlight
    from src.infrastructure.persistence.models.user import User
    from src.infrastructure.persistence.models.webhook_event import WebhookEvent


class StreamStatus(str, Enum):
    """Status values for stream processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamPlatform(str, Enum):
    """Supported streaming platforms."""

    TWITCH = "twitch"
    YOUTUBE = "youtube"
    RTMP = "rtmp"
    CUSTOM = "custom"


class Stream(Base, TimestampMixin):
    """Stream model for tracking livestream processing.

    Represents a livestream being processed for highlight extraction
    with configurable options and processing status tracking.

    Attributes:
        id: Unique identifier for the stream
        source_url: URL of the livestream
        platform: Streaming platform type
        status: Current processing status
        options: JSON object with processing options
        user_id: Foreign key to the user who created this stream
        created_at: Timestamp when the stream was created
        updated_at: Timestamp when the stream was last updated
        completed_at: Timestamp when processing completed
    """

    __tablename__ = "streams"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the stream"
    )

    source_url: Mapped[str] = mapped_column(
        Text, nullable=False, comment="URL of the livestream"
    )

    platform: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Streaming platform type"
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=StreamStatus.PENDING.value,
        index=True,
        comment="Current processing status",
    )

    options: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="JSON object with processing options",
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the user who created this stream",
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Timestamp when processing completed",
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="streams", lazy="joined")

    highlights: Mapped[List["Highlight"]] = relationship(
        "Highlight",
        back_populates="stream",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    
    webhook_events: Mapped[List["WebhookEvent"]] = relationship(
        "WebhookEvent",
        back_populates="stream",
        lazy="selectin",
    )

    @property
    def is_active(self) -> bool:
        """Check if the stream is currently being processed.

        Returns:
            bool: True if the stream is active, False otherwise
        """
        return self.status in [StreamStatus.PENDING, StreamStatus.PROCESSING]

    @property
    def processing_duration(self) -> Optional[float]:
        """Calculate the processing duration in seconds.

        Returns:
            Optional[float]: Duration in seconds if completed, None otherwise
        """
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None

    def __repr__(self) -> str:
        """String representation of the Stream."""
        return f"<Stream(id={self.id}, platform='{self.platform}', status='{self.status}')>"
