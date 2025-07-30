"""Repository interface for DimensionSet entities."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.dimension_set import DimensionSet


class DimensionSetRepository(ABC):
    """Abstract repository for DimensionSet persistence."""
    
    @abstractmethod
    async def get(self, dimension_set_id: int) -> Optional[DimensionSet]:
        """Get a dimension set by ID.
        
        Args:
            dimension_set_id: DimensionSet ID
            
        Returns:
            DimensionSet if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_organization(self, organization_id: int) -> List[DimensionSet]:
        """Get all dimension sets for an organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            List of dimension sets
        """
        pass
    
    @abstractmethod
    async def get_by_name(self, organization_id: int, name: str) -> Optional[DimensionSet]:
        """Get a dimension set by name within an organization.
        
        Args:
            organization_id: Organization ID
            name: DimensionSet name
            
        Returns:
            DimensionSet if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def save(self, dimension_set: DimensionSet) -> DimensionSet:
        """Save a dimension set.
        
        Args:
            dimension_set: DimensionSet to save
            
        Returns:
            Saved DimensionSet with ID
        """
        pass
    
    @abstractmethod
    async def delete(self, dimension_set_id: int) -> bool:
        """Delete a dimension set.
        
        Args:
            dimension_set_id: DimensionSet ID
            
        Returns:
            True if deleted, False otherwise
        """
        pass