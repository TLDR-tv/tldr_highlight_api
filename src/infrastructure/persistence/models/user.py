"""User model for authentication and account management.

This module defines the User model which represents enterprise
customers using the TL;DR Highlight API service.
"""

from typing import TYPE_CHECKING, List

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.api_key import APIKey
    from src.infrastructure.persistence.models.batch import Batch
    from src.infrastructure.persistence.models.organization import Organization
    from src.infrastructure.persistence.models.stream import Stream
    from src.infrastructure.persistence.models.usage_record import UsageRecord
    from src.infrastructure.persistence.models.webhook import Webhook
    from src.infrastructure.persistence.models.webhook_event import WebhookEvent


class User(Base, TimestampMixin):
    """User model representing enterprise customers.

    Stores authentication credentials and company information
    for B2B customers using the highlight extraction API.

    Attributes:
        id: Unique identifier for the user
        email: User's email address (unique)
        password_hash: Bcrypt hashed password
        company_name: Name of the user's company
        created_at: Timestamp when the user was created
        updated_at: Timestamp when the user was last updated
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the user"
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User's email address",
    )

    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Bcrypt hashed password"
    )

    company_name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Name of the user's company"
    )

    # Relationships
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    owned_organizations: Mapped[List["Organization"]] = relationship(
        "Organization",
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    streams: Mapped[List["Stream"]] = relationship(
        "Stream", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    batches: Mapped[List["Batch"]] = relationship(
        "Batch", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    webhooks: Mapped[List["Webhook"]] = relationship(
        "Webhook", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    usage_records: Mapped[List["UsageRecord"]] = relationship(
        "UsageRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    
    webhook_events: Mapped[List["WebhookEvent"]] = relationship(
        "WebhookEvent",
        back_populates="user",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """String representation of the User."""
        return (
            f"<User(id={self.id}, email='{self.email}', company='{self.company_name}')>"
        )
