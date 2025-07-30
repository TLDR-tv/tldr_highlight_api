"""Repository interface for HighlightTypeRegistry entities."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.highlight_type_registry import HighlightTypeRegistry


class HighlightTypeRegistryRepository(ABC):
    """Abstract repository for HighlightTypeRegistry persistence."""
    
    @abstractmethod
    async def get(self, registry_id: int) -> Optional[HighlightTypeRegistry]:
        """Get a type registry by ID.
        
        Args:
            registry_id: Registry ID
            
        Returns:
            HighlightTypeRegistry if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_organization(self, organization_id: int) -> Optional[HighlightTypeRegistry]:
        """Get the type registry for an organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            HighlightTypeRegistry if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def save(self, registry: HighlightTypeRegistry) -> HighlightTypeRegistry:
        """Save a type registry.
        
        Args:
            registry: HighlightTypeRegistry to save
            
        Returns:
            Saved HighlightTypeRegistry with ID
        """
        pass
    
    @abstractmethod
    async def delete(self, registry_id: int) -> bool:
        """Delete a type registry.
        
        Args:
            registry_id: Registry ID
            
        Returns:
            True if deleted, False otherwise
        """
        pass