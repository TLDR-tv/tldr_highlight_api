"""SQLAlchemy model for DimensionSet."""

from typing import Dict, Any, TYPE_CHECKING

from sqlalchemy import (
    Integer,
    String,
    Text,
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


class DimensionSet(Base, TimestampMixin):
    """Persistence model for dimension sets.

    Stores configurable dimension definitions for flexible highlight detection.
    """

    __tablename__ = "dimension_sets"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_dimension_set_org_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id"), nullable=False
    )

    # JSON fields for flexible dimension storage
    dimensions: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )  # Dict of dimension_id -> definition
    dimension_weights: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )  # Dict of dimension_id -> weight

    # Configuration
    allow_partial_scoring: Mapped[bool] = mapped_column(Boolean, default=True)
    minimum_dimensions_required: Mapped[int] = mapped_column(Integer, default=3)
    weight_normalization: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="dimension_sets"
    )

    def __repr__(self):
        return f"<DimensionSet(id={self.id}, name='{self.name}', org_id={self.organization_id})>"
