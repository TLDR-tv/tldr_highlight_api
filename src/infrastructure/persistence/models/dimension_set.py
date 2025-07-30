"""SQLAlchemy model for DimensionSet."""

from sqlalchemy import Column, Integer, String, Text, JSON, Boolean, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from src.infrastructure.persistence.models.base import Base
from src.infrastructure.persistence.models.mixins import TimestampMixin


class DimensionSet(Base, TimestampMixin):
    """Persistence model for dimension sets.
    
    Stores configurable dimension definitions for flexible highlight detection.
    """
    
    __tablename__ = "dimension_sets"
    __table_args__ = (
        UniqueConstraint('organization_id', 'name', name='uq_dimension_set_org_name'),
    )
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    # JSON fields for flexible dimension storage
    dimensions = Column(JSON, nullable=False, default=dict)  # Dict of dimension_id -> definition
    dimension_weights = Column(JSON, nullable=False, default=dict)  # Dict of dimension_id -> weight
    
    # Configuration
    allow_partial_scoring = Column(Boolean, default=True)
    minimum_dimensions_required = Column(Integer, default=3)
    weight_normalization = Column(Boolean, default=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="dimension_sets")
    
    def __repr__(self):
        return f"<DimensionSet(id={self.id}, name='{self.name}', org_id={self.organization_id})>"