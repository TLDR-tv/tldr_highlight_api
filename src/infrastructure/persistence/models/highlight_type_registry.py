"""SQLAlchemy model for HighlightTypeRegistry."""

from sqlalchemy import (
    Column,
    Integer,
    JSON,
    Boolean,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from src.infrastructure.persistence.models.base import Base
from src.infrastructure.persistence.models.mixins import TimestampMixin


class HighlightTypeRegistry(Base, TimestampMixin):
    """Persistence model for highlight type registries.

    Stores custom highlight type definitions for organizations.
    """

    __tablename__ = "highlight_type_registries"
    __table_args__ = (UniqueConstraint("organization_id", name="uq_type_registry_org"),)

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)

    # JSON field for flexible type definitions
    types = Column(
        JSON, nullable=False, default=dict
    )  # Dict of type_id -> HighlightTypeDefinition

    # Configuration
    allow_multiple_types = Column(Boolean, default=True)
    max_types_per_highlight = Column(Integer, default=3)
    include_built_in_types = Column(Boolean, default=True)

    # Relationships
    organization = relationship(
        "Organization", back_populates="highlight_type_registry"
    )

    def __repr__(self):
        return f"<HighlightTypeRegistry(id={self.id}, org_id={self.organization_id})>"
