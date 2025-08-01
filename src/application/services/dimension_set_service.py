"""Application service for dimension set management.

This service provides high-level operations for dimension sets,
coordinating between domain and infrastructure layers.
"""

from typing import Optional
from src.domain.repositories.dimension_set_repository_interface import DimensionSetRepository
from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.domain.exceptions import EntityNotFoundError


class DimensionSetService:
    """Application service for dimension set operations."""
    
    def __init__(self, dimension_set_repository: DimensionSetRepository):
        """Initialize service with repository.
        
        Args:
            dimension_set_repository: Repository for dimension sets
        """
        self.dimension_set_repository = dimension_set_repository
    
    async def get_dimension_set_for_stream(
        self,
        dimension_set_id: Optional[int],
        organization_id: int,
        user_id: int,
    ) -> DimensionSetAggregate:
        """Get dimension set for stream processing.
        
        If dimension_set_id is provided, retrieves that set.
        Otherwise, returns a default set for the organization.
        
        Args:
            dimension_set_id: Optional specific dimension set ID
            organization_id: Organization ID
            user_id: User ID for creation if needed
            
        Returns:
            DimensionSetAggregate for stream processing
            
        Raises:
            EntityNotFoundError: If specified dimension set not found
        """
        if dimension_set_id:
            # Get specific dimension set
            dimension_set = await self.dimension_set_repository.get_by_id(dimension_set_id)
            if not dimension_set:
                raise EntityNotFoundError(
                    entity_type="DimensionSet",
                    entity_id=dimension_set_id
                )
            return dimension_set
        
        # Get organization's default dimension sets
        org_sets = await self.dimension_set_repository.get_by_organization(
            organization_id=organization_id,
            active_only=True
        )
        
        if org_sets:
            # Return the first active set
            return org_sets[0]
        
        # Get popular public sets as fallback
        popular_sets = await self.dimension_set_repository.get_popular_sets(limit=1)
        if popular_sets:
            return popular_sets[0]
        
        # As last resort, create a default gaming set
        # In production, this would be pre-seeded in the database
        default_set = DimensionSetAggregate.create_gaming_set(
            organization_id=organization_id,
            user_id=user_id
        )
        
        # Save and return
        return await self.dimension_set_repository.save(default_set)
    
    async def get_default_dimension_set(
        self,
        organization_id: int,
        user_id: int,
        content_type: Optional[str] = None,
    ) -> DimensionSetAggregate:
        """Get default dimension set for organization.
        
        Args:
            organization_id: Organization ID
            user_id: User ID for creation if needed
            content_type: Optional content type (gaming, education, etc.)
            
        Returns:
            Default dimension set for the organization
        """
        # Try to find by content type if specified
        if content_type:
            sets = await self.dimension_set_repository.find_by_criteria(
                organization_id=organization_id,
                content_type=content_type,
                is_public=None,
                limit=1
            )
            if sets:
                return sets[0]
        
        # Fall back to general lookup
        return await self.get_dimension_set_for_stream(
            dimension_set_id=None,
            organization_id=organization_id,
            user_id=user_id
        )