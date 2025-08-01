"""Repository interface for Dimension Set Aggregate.

This defines the contract for persisting and retrieving dimension sets,
following DDD repository patterns.
"""

from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime
from dataclasses import dataclass, field

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate


class DimensionSetRepository(Protocol):
    """Repository protocol for Dimension Set aggregates.

    This protocol defines persistence operations for dimension sets
    while keeping the domain model free from infrastructure concerns.
    """

    async def get(self, id: int) -> Optional[DimensionSetAggregate]:
        """Retrieve a dimension set by its ID.

        Args:
            id: The ID of the dimension set

        Returns:
            The dimension set aggregate or None if not found
        """
        ...

    async def list_for_organization(
        self,
        organization_id: int,
        *,
        active_only: bool = True,
        include_public: bool = True,
    ) -> List[DimensionSetAggregate]:
        """Get all dimension sets for an organization.

        Args:
            organization_id: The organization ID
            active_only: Whether to include only active sets
            include_public: Whether to include public sets from other orgs

        Returns:
            List of dimension set aggregates
        """
        ...

    async def find(
        self,
        *,
        organization_id: Optional[int] = None,
        industry: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        min_usage_count: Optional[int] = None,
        limit: int = 100,
    ) -> List[DimensionSetAggregate]:
        """Find dimension sets matching criteria.

        Args:
            organization_id: Filter by organization
            industry: Filter by industry
            content_type: Filter by content type
            tags: Filter by tags (any match)
            is_public: Filter by public status
            min_usage_count: Minimum usage count
            limit: Maximum results to return

        Returns:
            List of matching dimension sets
        """
        ...

    async def save(self, dimension_set: DimensionSetAggregate) -> DimensionSetAggregate:
        """Save a dimension set aggregate.

        This method handles both creation and updates, preserving
        the aggregate's domain events.

        Args:
            dimension_set: The dimension set to save

        Returns:
            The saved dimension set with updated ID if created
        """
        ...

    async def delete(self, id: int) -> bool:
        """Delete a dimension set.

        Args:
            id: The ID of the dimension set to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    async def exists(self, organization_id: int, name: str) -> bool:
        """Check if a dimension set with the given name exists.

        Args:
            organization_id: The organization ID
            name: The dimension set name

        Returns:
            True if exists, False otherwise
        """
        ...

    async def get_usage_statistics(
        self,
        id: int,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get usage statistics for a dimension set.

        Args:
            id: The dimension set ID
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with usage statistics
        """
        ...

    async def list_popular(
        self, *, limit: int = 10, time_window_days: int = 30
    ) -> List[DimensionSetAggregate]:
        """Get most popular dimension sets.

        Args:
            limit: Maximum number of sets to return
            time_window_days: Time window for popularity calculation

        Returns:
            List of popular dimension sets
        """
        ...

    async def clone(
        self,
        source_id: int,
        target_organization_id: int,
        user_id: int,
        *,
        new_name: Optional[str] = None,
    ) -> DimensionSetAggregate:
        """Clone a dimension set for another organization.

        Args:
            source_id: Source dimension set ID
            target_organization_id: Target organization ID
            user_id: User creating the clone
            new_name: Optional new name for the clone

        Returns:
            The cloned dimension set
        """
        ...


@dataclass(frozen=True)
class DimensionSetQuery:
    """Query parameters for finding dimension sets.

    This replaces the builder pattern with a simple immutable dataclass,
    which is more Pythonic and explicit about available query options.
    """

    organization_id: Optional[int] = None
    industry: Optional[str] = None
    content_type: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None
    min_usage_count: Optional[int] = None
    limit: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for repository query, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def for_organization(cls, org_id: int, **kwargs) -> "DimensionSetQuery":
        """Create a query for a specific organization."""
        return cls(organization_id=org_id, **kwargs)

    @classmethod
    def public_only(cls, **kwargs) -> "DimensionSetQuery":
        """Create a query for public dimension sets only."""
        return cls(is_public=True, **kwargs)

    @classmethod
    def active_only(cls, **kwargs) -> "DimensionSetQuery":
        """Create a query for active dimension sets only."""
        return cls(is_active=True, **kwargs)
