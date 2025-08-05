"""Scoring rubric domain model."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from enum import Enum


class RubricVisibility(Enum):
    """Visibility settings for rubrics."""
    PRIVATE = "private"  # Only visible to the organization
    PUBLIC = "public"    # Can be used as template by other orgs
    SYSTEM = "system"    # Built-in system rubrics


@dataclass
class Rubric:
    """Scoring rubric configuration for highlight detection."""
    
    id: UUID = field(default_factory=uuid4)
    organization_id: Optional[UUID] = None  # None for system rubrics
    name: str = ""
    description: str = ""
    
    # Rubric configuration stored as JSON
    # This includes dimensions, weights, thresholds, etc.
    config: dict = field(default_factory=dict)
    
    # Metadata
    visibility: RubricVisibility = RubricVisibility.PRIVATE
    is_active: bool = True
    version: int = 1
    
    # Usage tracking
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by_user_id: Optional[UUID] = None
    
    def __post_init__(self) -> None:
        """Validate rubric configuration."""
        if not self.name:
            raise ValueError("Rubric name is required")
            
        # System rubrics must not have organization_id
        if self.visibility == RubricVisibility.SYSTEM and self.organization_id:
            raise ValueError("System rubrics cannot belong to an organization")
            
        # Private rubrics must have organization_id
        if self.visibility == RubricVisibility.PRIVATE and not self.organization_id:
            raise ValueError("Private rubrics must belong to an organization")
    
    @property
    def is_system_rubric(self) -> bool:
        """Check if this is a system-provided rubric."""
        return self.visibility == RubricVisibility.SYSTEM
    
    @property
    def is_template(self) -> bool:
        """Check if this rubric can be used as a template."""
        return self.visibility in (RubricVisibility.PUBLIC, RubricVisibility.SYSTEM)
    
    def can_be_edited_by(self, organization_id: UUID) -> bool:
        """Check if rubric can be edited by organization."""
        if self.is_system_rubric:
            return False
        return self.organization_id == organization_id
    
    def can_be_used_by(self, organization_id: UUID) -> bool:
        """Check if rubric can be used by organization."""
        if self.is_system_rubric or self.visibility == RubricVisibility.PUBLIC:
            return True
        return self.organization_id == organization_id
    
    def increment_usage(self) -> None:
        """Record rubric usage."""
        self.usage_count += 1
        self.last_used_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def clone_for_organization(self, organization_id: UUID, name: Optional[str] = None) -> "Rubric":
        """Create a copy of this rubric for an organization."""
        return Rubric(
            organization_id=organization_id,
            name=name or f"{self.name} (Copy)",
            description=self.description,
            config=self.config.copy(),
            visibility=RubricVisibility.PRIVATE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )