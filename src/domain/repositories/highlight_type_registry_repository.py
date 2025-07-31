"""Repository interface for HighlightTypeRegistry entities."""

from typing import Protocol, Optional

from src.domain.entities.highlight_type_registry import HighlightTypeRegistry


class HighlightTypeRegistryRepository(Protocol):
    """Repository protocol for HighlightTypeRegistry persistence."""

    async def get(self, registry_id: int) -> Optional[HighlightTypeRegistry]:
        """Get a type registry by ID."""
        ...

    async def get_by_organization(
        self, organization_id: int
    ) -> Optional[HighlightTypeRegistry]:
        """Get the type registry for an organization."""
        ...

    async def save(self, registry: HighlightTypeRegistry) -> HighlightTypeRegistry:
        """Save a type registry."""
        ...

    async def delete(self, registry_id: int) -> bool:
        """Delete a type registry."""
        ...