"""Repository interface for Dimension Set Aggregate.

This defines the contract for persisting and retrieving dimension sets,
following DDD repository patterns.
"""

from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate


class DimensionSetRepository(Protocol):
    """Repository protocol for Dimension Set aggregates.

    This protocol defines persistence operations for dimension sets
    while keeping the domain model free from infrastructure concerns.
    """

    async def get_by_id(self, dimension_set_id: int) -> Optional[DimensionSetAggregate]:
        """Retrieve a dimension set by its ID.

        Args:
            dimension_set_id: The ID of the dimension set

        Returns:
            The dimension set aggregate or None if not found
        """
        ...

    async def get_by_organization(
        self,
        organization_id: int,
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

    async def find_by_criteria(
        self,
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

    async def delete(self, dimension_set_id: int) -> bool:
        """Delete a dimension set.

        Args:
            dimension_set_id: The ID of the dimension set to delete

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
        dimension_set_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get usage statistics for a dimension set.

        Args:
            dimension_set_id: The dimension set ID
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with usage statistics
        """
        ...

    async def get_popular_sets(
        self, limit: int = 10, time_window_days: int = 30
    ) -> List[DimensionSetAggregate]:
        """Get most popular dimension sets.

        Args:
            limit: Maximum number of sets to return
            time_window_days: Time window for popularity calculation

        Returns:
            List of popular dimension sets
        """
        ...

    async def clone_for_organization(
        self,
        source_dimension_set_id: int,
        target_organization_id: int,
        user_id: int,
        new_name: Optional[str] = None,
    ) -> DimensionSetAggregate:
        """Clone a dimension set for another organization.

        Args:
            source_dimension_set_id: Source dimension set ID
            target_organization_id: Target organization ID
            user_id: User creating the clone
            new_name: Optional new name for the clone

        Returns:
            The cloned dimension set
        """
        ...


class DimensionSetSpecification:
    """Specification pattern for complex dimension set queries.

    This allows building complex queries while keeping them
    in the domain layer.
    """

    def __init__(self) -> None:
        self.criteria: Dict[str, Any] = {}

    def for_organization(self, org_id: int) -> "DimensionSetSpecification":
        """Filter by organization."""
        self.criteria["organization_id"] = org_id
        return self

    def with_industry(self, industry: str) -> "DimensionSetSpecification":
        """Filter by industry."""
        self.criteria["industry"] = industry
        return self

    def with_content_type(self, content_type: str) -> "DimensionSetSpecification":
        """Filter by content type."""
        self.criteria["content_type"] = content_type
        return self

    def with_tags(self, tags: List[str]) -> "DimensionSetSpecification":
        """Filter by tags."""
        self.criteria["tags"] = tags
        return self

    def only_public(self) -> "DimensionSetSpecification":
        """Only public dimension sets."""
        self.criteria["is_public"] = True
        return self

    def only_active(self) -> "DimensionSetSpecification":
        """Only active dimension sets."""
        self.criteria["is_active"] = True
        return self

    def with_minimum_usage(self, count: int) -> "DimensionSetSpecification":
        """Minimum usage count."""
        self.criteria["min_usage_count"] = count
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for repository query."""
        return self.criteria
