"""Usage Record model for tracking API usage and billing.

This module defines the UsageRecord model which tracks
API usage for billing and analytics purposes.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.user import User


class UsageRecordType(str):
    """Types of usage records for billing."""

    STREAM_PROCESSED = "stream_processed"
    BATCH_VIDEO_PROCESSED = "batch_video_processed"
    HIGHLIGHT_GENERATED = "highlight_generated"
    API_CALL = "api_call"
    STORAGE_USED = "storage_used"
    BANDWIDTH_USED = "bandwidth_used"


class UsageRecord(Base):
    """Usage Record model for tracking API usage.

    Records various types of usage for billing, analytics,
    and quota management for enterprise customers.

    Attributes:
        id: Unique identifier for the usage record
        user_id: Foreign key to the user
        record_type: Type of usage being recorded
        quantity: Quantity of usage (unit depends on type)
        extra_metadata: JSON object with additional details
        created_at: Timestamp when the record was created
    """

    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the usage record"
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the user",
    )

    record_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Type of usage being recorded"
    )

    quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Quantity of usage (unit depends on type)",
    )

    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="JSON object with additional details",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="CURRENT_TIMESTAMP",
        index=True,
        comment="Timestamp when the record was created",
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User", back_populates="usage_records", lazy="joined"
    )

    def __repr__(self) -> str:
        """String representation of the UsageRecord."""
        return f"<UsageRecord(id={self.id}, type='{self.record_type}', quantity={self.quantity})>"
