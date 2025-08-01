"""SQLAlchemy model for HighlightTypeRegistry."""

from typing import Dict, Any, TYPE_CHECKING

from sqlalchemy import (
    Integer,
    JSON,
    Boolean,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.persistence.models.base import Base
from src.infrastructure.persistence.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.infrastructure.persistence.models.organization import Organization


class HighlightTypeRegistry(Base, TimestampMixin):
    """Persistence model for highlight type registries.

    Stores custom highlight type definitions for organizations.
    """

    __tablename__ = "highlight_type_registries"
    __table_args__ = (UniqueConstraint("organization_id", name="uq_type_registry_org"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id"), nullable=False
    )

    # JSON field for flexible type definitions
    types: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )  # Dict of type_id -> HighlightTypeDefinition

    # Configuration
    allow_multiple_types: Mapped[bool] = mapped_column(Boolean, default=True)
    max_types_per_highlight: Mapped[int] = mapped_column(Integer, default=3)
    include_built_in_types: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="highlight_type_registry"
    )

    def __repr__(self):
        return f"<HighlightTypeRegistry(id={self.id}, org_id={self.organization_id})>"
