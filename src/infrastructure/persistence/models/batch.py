"""Batch model for video batch processing.

This module defines the Batch model which represents
batch processing jobs for multiple videos.
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.highlight import Highlight
    from src.infrastructure.persistence.models.user import User


class BatchStatus(str, Enum):
    """Status values for batch processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Batch(Base, TimestampMixin):
    """Batch model for processing multiple videos.

    Represents a batch job for processing multiple videos
    for highlight extraction with configurable options.

    Attributes:
        id: Unique identifier for the batch
        status: Current processing status
        options: JSON object with processing options
        user_id: Foreign key to the user who created this batch
        video_count: Number of videos in the batch
        created_at: Timestamp when the batch was created
        updated_at: Timestamp when the batch was last updated
    """

    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the batch"
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=BatchStatus.PENDING.value,
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
        comment="Foreign key to the user who created this batch",
    )

    video_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of videos in the batch"
    )
    
    # Additional fields for domain model compatibility
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Name of the batch job",
    )
    
    processing_options: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON serialized processing options",
    )
    
    items_data: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON serialized batch items data",
    )
    
    dimension_set_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("dimension_sets.id"),
        nullable=True,
        comment="Optional custom dimension set for flexible highlight detection",
    )
    
    type_registry_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("highlight_type_registries.id"),
        nullable=True,
        comment="Optional custom type registry for flexible highlight types",
    )
    
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing started",
    )
    
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing completed",
    )
    
    total_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of items in batch",
    )
    
    processed_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of processed items",
    )
    
    successful_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of successfully processed items",
    )
    
    failed_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of failed items",
    )
    
    webhook_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Webhook URL for completion notification",
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="batches", lazy="joined")

    highlights: Mapped[List["Highlight"]] = relationship(
        "Highlight",
        back_populates="batch",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @property
    def is_active(self) -> bool:
        """Check if the batch is currently being processed.

        Returns:
            bool: True if the batch is active, False otherwise
        """
        return self.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]

    @property
    def processed_count(self) -> int:
        """Get the number of processed videos.

        Returns:
            int: Number of videos that have been processed
        """
        # Group highlights by their source video
        processed_videos = set()
        for highlight in self.highlights:
            if (
                highlight.extra_metadata
                and "source_video_id" in highlight.extra_metadata
            ):
                processed_videos.add(highlight.extra_metadata["source_video_id"])
        return len(processed_videos)

    @property
    def progress_percentage(self) -> float:
        """Calculate the processing progress as a percentage.

        Returns:
            float: Progress percentage (0-100)
        """
        if self.video_count == 0:
            return 0.0
        return (self.processed_count / self.video_count) * 100

    def __repr__(self) -> str:
        """String representation of the Batch."""
        return (
            f"<Batch(id={self.id}, status='{self.status}', videos={self.video_count})>"
        )
