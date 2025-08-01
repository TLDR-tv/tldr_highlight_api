"""API Key model for authentication and authorization.

This module defines the APIKey model which manages API keys
for enterprise customers with scoped permissions.
"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.user import User


class APIKey(Base):
    """API Key model for authenticating API requests.

    Manages API keys with scoped permissions, expiration,
    and usage tracking for enterprise customers.

    Attributes:
        id: Unique identifier for the API key
        key: The actual API key string (hashed)
        name: Human-readable name for the key
        user_id: Foreign key to the user who owns this key
        scopes: JSON array of permission scopes
        active: Whether the key is currently active
        created_at: Timestamp when the key was created
        expires_at: Optional expiration timestamp
        last_used_at: Timestamp of last API call using this key
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the API key"
    )

    key: Mapped[str] = mapped_column(
        Text,
        unique=True,
        nullable=False,
        index=True,
        comment="The API key string (hashed)",
    )

    name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Human-readable name for the key"
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the user who owns this key",
    )

    scopes: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list, comment="JSON array of permission scopes"
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Whether the key is currently active",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="CURRENT_TIMESTAMP",
        comment="Timestamp when the key was created",
    )

    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Optional expiration timestamp",
    )

    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of last API call using this key",
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User", back_populates="api_keys", lazy="joined"
    )

    def __repr__(self) -> str:
        """String representation of the APIKey."""
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id}, active={self.active})>"
