"""Highlight model for extracted video highlights.

This module defines the Highlight model which represents
extracted highlights from streams or batch videos.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.batch import Batch
    from src.infrastructure.persistence.models.stream import Stream


class Highlight(Base):
    """Highlight model for extracted video highlights.

    Represents an extracted highlight from a stream or video
    with associated metadata and confidence scoring.

    Attributes:
        id: Unique identifier for the highlight
        stream_id: Foreign key to the stream (if from stream)
        batch_id: Foreign key to the batch (if from batch)
        title: Title of the highlight
        description: Description of the highlight
        video_url: URL to the highlight video clip
        thumbnail_url: URL to the highlight thumbnail
        duration: Duration of the highlight in seconds
        timestamp: Original timestamp in source content
        confidence_score: AI confidence score (0-1)
        tags: JSON array of tags
        extra_metadata: JSON object with additional metadata
        created_at: Timestamp when the highlight was created
    """

    __tablename__ = "highlights"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the highlight"
    )

    stream_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("streams.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Foreign key to the stream (if from stream)",
    )

    batch_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Foreign key to the batch (if from batch)",
    )

    title: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Title of the highlight"
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Description of the highlight"
    )

    video_url: Mapped[str] = mapped_column(
        Text, nullable=False, comment="URL to the highlight video clip"
    )

    thumbnail_url: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="URL to the highlight thumbnail"
    )

    duration: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Duration of the highlight in seconds"
    )

    timestamp: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        comment="Original timestamp in source content (seconds)",
    )

    confidence_score: Mapped[float] = mapped_column(
        Float, nullable=False, index=True, comment="AI confidence score (0-1)"
    )

    start_time_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Start time in seconds from beginning"
    )

    end_time_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, comment="End time in seconds from beginning"
    )

    highlight_types: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list, comment="Multiple highlight types"
    )

    tags: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list, comment="JSON array of tags"
    )

    sentiment_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Sentiment score (-1.0 to 1.0)"
    )

    viewer_engagement: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Viewer engagement score (0.0 to 1.0)"
    )

    video_analysis: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="JSON serialized video analysis data"
    )

    audio_analysis: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="JSON serialized audio analysis data"
    )

    chat_analysis: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="JSON serialized chat analysis data"
    )

    processed_by: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, comment="AI model/version that processed this"
    )

    clip_url: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="URL to the highlight clip"
    )

    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="JSON object with additional metadata",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="CURRENT_TIMESTAMP",
        comment="Timestamp when the highlight was created",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="CURRENT_TIMESTAMP",
        onupdate=datetime.utcnow,
        comment="Timestamp when the highlight was last updated",
    )

    # Relationships
    stream: Mapped[Optional["Stream"]] = relationship(
        "Stream", back_populates="highlights", lazy="joined"
    )

    batch: Mapped[Optional["Batch"]] = relationship(
        "Batch", back_populates="highlights", lazy="joined"
    )

    @property
    def source_type(self) -> str:
        """Get the source type of the highlight.

        Returns:
            str: Either 'stream' or 'batch'
        """
        if self.stream_id:
            return "stream"
        elif self.batch_id:
            return "batch"
        return "unknown"

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if this is a high confidence highlight.

        Args:
            threshold: Confidence threshold (default: 0.8)

        Returns:
            bool: True if confidence score exceeds threshold
        """
        return self.confidence_score >= threshold

    def __repr__(self) -> str:
        """String representation of the Highlight."""
        return f"<Highlight(id={self.id}, title='{self.title}', confidence={self.confidence_score:.2f})>"
