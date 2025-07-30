"""Organization model for multi-tenant support.

This module defines the Organization model which represents
enterprise organizations with different subscription plans.
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.user import User


class PlanType(str, Enum):
    """Subscription plan types for organizations."""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class Organization(Base):
    """Organization model for multi-tenant architecture.

    Represents enterprise organizations with subscription plans
    and usage limits for the highlight extraction service.

    Attributes:
        id: Unique identifier for the organization
        name: Organization name
        owner_id: Foreign key to the user who owns this organization
        plan_type: Subscription plan type
        created_at: Timestamp when the organization was created
    """

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(
        primary_key=True, comment="Unique identifier for the organization"
    )

    name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, comment="Organization name"
    )

    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the user who owns this organization",
    )

    plan_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=PlanType.STARTER.value,
        comment="Subscription plan type",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="CURRENT_TIMESTAMP",
        comment="Timestamp when the organization was created",
    )

    # Relationships
    owner: Mapped["User"] = relationship(
        "User", back_populates="owned_organizations", lazy="joined"
    )
    dimension_sets = relationship(
        "DimensionSet", back_populates="organization", cascade="all, delete-orphan"
    )
    highlight_type_registry = relationship(
        "HighlightTypeRegistry",
        back_populates="organization",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Plan limits removed - all organizations have unlimited access for first client

    def __repr__(self) -> str:
        """String representation of the Organization."""
        return (
            f"<Organization(id={self.id}, name='{self.name}', plan='{self.plan_type}')>"
        )
