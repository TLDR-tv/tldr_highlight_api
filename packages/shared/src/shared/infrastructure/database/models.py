"""SQLAlchemy models for database tables."""

from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    Float,
    DateTime,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from .database import Base
from ...domain.models.stream import StreamStatus, StreamSource
from ...domain.models.user import UserRole


class OrganizationModel(Base):
    """Organization database model."""

    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Usage tracking
    total_streams_processed = Column(Integer, default=0, nullable=False)
    total_highlights_generated = Column(Integer, default=0, nullable=False)
    total_processing_seconds = Column(Float, default=0.0, nullable=False)

    # Custom configuration
    webhook_url = Column(String(500), nullable=True)
    webhook_secret = Column(String(255), nullable=True)
    rubric_name = Column(String(50), nullable=False, default="general")

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    users = relationship(
        "UserModel", back_populates="organization", cascade="all, delete-orphan"
    )
    streams = relationship(
        "StreamModel", back_populates="organization", cascade="all, delete-orphan"
    )
    api_keys = relationship(
        "APIKeyModel", back_populates="organization", cascade="all, delete-orphan"
    )
    wake_word_configs = relationship(
        "WakeWordModel", back_populates="organization", cascade="all, delete-orphan"
    )


class UserModel(Base):
    """User database model."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.MEMBER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Security
    hashed_password = Column(String(255), nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    organization = relationship("OrganizationModel", back_populates="users")

    __table_args__ = (Index("idx_user_org_email", "organization_id", "email"),)


class StreamModel(Base):
    """Stream database model."""

    __tablename__ = "streams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )
    stream_url = Column(Text, nullable=False)
    stream_fingerprint = Column(String(255), nullable=False, index=True)
    source_type = Column(
        SQLEnum(StreamSource), default=StreamSource.DIRECT_URL, nullable=False
    )
    status = Column(
        SQLEnum(StreamStatus), default=StreamStatus.PENDING, nullable=False, index=True
    )

    # Metadata
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    platform_user_id = Column(String(255), nullable=True)

    # Processing info
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, default=0.0, nullable=False)
    segments_processed = Column(Integer, default=0, nullable=False)
    highlights_generated = Column(Integer, default=0, nullable=False)

    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    organization = relationship("OrganizationModel", back_populates="streams")
    highlights = relationship(
        "HighlightModel", back_populates="stream", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_stream_org_fingerprint", "organization_id", "stream_fingerprint"),
        Index("idx_stream_status", "status"),
    )


class HighlightModel(Base):
    """Highlight database model."""

    __tablename__ = "highlights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stream_id = Column(UUID(as_uuid=True), ForeignKey("streams.id"), nullable=False)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)

    # Content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    tags = Column(
        JSON, default=list, nullable=False
    )  # Use JSON for SQLite compatibility

    # Scoring (stored as JSON for flexibility)
    dimension_scores = Column(JSON, default=dict, nullable=False)
    overall_score = Column(Float, nullable=False, index=True)

    # Media files
    clip_path = Column(String(500), nullable=True)
    thumbnail_path = Column(String(500), nullable=True)

    # Metadata
    transcript = Column(Text, nullable=True)
    wake_word_triggered = Column(Boolean, default=False, nullable=False)
    wake_word_detected = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    stream = relationship("StreamModel", back_populates="highlights")

    __table_args__ = (
        Index("idx_highlight_org", "organization_id"),
        Index("idx_highlight_score", "overall_score"),
        Index("idx_highlight_wake_word", "wake_word_triggered", "wake_word_detected"),
    )


class APIKeyModel(Base):
    """API key database model."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False)
    prefix = Column(String(50), unique=True, nullable=False, index=True)

    # Permissions
    scopes = Column(
        JSON, default=list, nullable=False
    )  # Use JSON for SQLite compatibility

    # Usage tracking
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)

    # Lifecycle
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    created_by_user_id = Column(UUID(as_uuid=True), nullable=True)
    description = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    organization = relationship("OrganizationModel", back_populates="api_keys")

    __table_args__ = (
        Index("idx_api_key_active", "is_active"),
        Index("idx_api_key_org", "organization_id"),
    )


class WakeWordModel(Base):
    """Wake word configuration database model."""

    __tablename__ = "wake_words"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )
    phrase = Column(String(500), nullable=False)  # Increased size for multi-word phrases
    is_active = Column(Boolean, default=True, nullable=False)

    # Configuration
    case_sensitive = Column(Boolean, default=False, nullable=False)
    exact_match = Column(Boolean, default=True, nullable=False)
    cooldown_seconds = Column(Integer, default=30, nullable=False)
    
    # Fuzzy matching configuration
    max_edit_distance = Column(Integer, default=2, nullable=False)
    similarity_threshold = Column(Float, default=0.8, nullable=False)
    
    # Clip configuration
    pre_roll_seconds = Column(Integer, default=10, nullable=False)
    post_roll_seconds = Column(Integer, default=30, nullable=False)

    # Usage tracking
    trigger_count = Column(Integer, default=0, nullable=False)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    organization = relationship("OrganizationModel", back_populates="wake_word_configs")

    __table_args__ = (
        UniqueConstraint("organization_id", "phrase", name="uq_org_wake_phrase"),
        Index("idx_wake_word_active", "organization_id", "is_active"),
    )


class UsageRecordModel(Base):
    """Usage tracking database model."""

    __tablename__ = "usage_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False
    )

    # What was used
    resource_type = Column(
        String(50), nullable=False
    )  # "stream", "highlight", "api_call"
    resource_id = Column(UUID(as_uuid=True), nullable=True)

    # Usage metrics
    quantity = Column(Float, default=1.0, nullable=False)
    unit = Column(String(50), nullable=False)  # "count", "seconds", "bytes"

    # Metadata
    usage_metadata = Column("metadata", JSON, default=dict, nullable=False)

    # Timestamp
    recorded_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        Index("idx_usage_org_time", "organization_id", "recorded_at"),
        Index("idx_usage_type", "resource_type"),
    )
