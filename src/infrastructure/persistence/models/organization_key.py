"""SQLAlchemy model for organization signing keys.

This module defines the database schema for storing organization-specific
signing keys used for generating signed URLs.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Index,
    Text,
)

from src.infrastructure.persistence.models.base import Base


class OrganizationKey(Base):
    """Organization signing key model."""

    __tablename__ = "organization_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=False, index=True)

    # Key information
    key_id = Column(String(64), unique=True, nullable=False, index=True)
    key_value = Column(Text, nullable=False)  # Encrypted at rest
    algorithm = Column(String(10), nullable=False, default="HS256")

    # Key metadata
    key_version = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)
    is_primary = Column(Boolean, nullable=False, default=False)

    # Lifecycle
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    rotated_at = Column(DateTime, nullable=True)
    deactivated_at = Column(DateTime, nullable=True)

    # Usage tracking
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)

    # Rotation information
    previous_key_id = Column(String(64), nullable=True)
    rotation_reason = Column(String(255), nullable=True)

    # Metadata
    created_by = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)

    __table_args__ = (
        # Ensure only one primary key per organization
        Index(
            "idx_one_primary_per_org",
            organization_id,
            is_primary,
            unique=True,
            postgresql_where="is_primary = true",
        ),
        # Ensure unique active versions per organization
        Index(
            "idx_unique_active_version",
            organization_id,
            key_version,
            unique=True,
            postgresql_where="is_active = true",
        ),
    )

    def __repr__(self):
        return (
            f"<OrganizationKey("
            f"id={self.id}, "
            f"org_id={self.organization_id}, "
            f"key_id={self.key_id}, "
            f"version={self.key_version}, "
            f"active={self.is_active}"
            f")>"
        )
