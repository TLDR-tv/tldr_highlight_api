"""Repository interface for highlight agent configurations."""

from typing import List, Optional, Protocol
from ..entities.highlight_agent_config import HighlightAgentConfig


class HighlightAgentConfigRepository(Protocol):
    """Repository interface for highlight agent configurations."""

    async def save(self, config: HighlightAgentConfig) -> HighlightAgentConfig:
        """Save a highlight agent configuration."""
        ...

    async def get(self, config_id: int) -> Optional[HighlightAgentConfig]:
        """Get a configuration by ID."""
        ...

    async def get_by_organization(
        self, organization_id: int
    ) -> List[HighlightAgentConfig]:
        """Get all configurations for an organization."""
        ...

    async def get_by_user(self, user_id: int) -> List[HighlightAgentConfig]:
        """Get all configurations created by a user."""
        ...

    async def get_active_for_organization(
        self, organization_id: int
    ) -> List[HighlightAgentConfig]:
        """Get all active configurations for an organization."""
        ...

    async def get_default_for_content_type(
        self, content_type: str
    ) -> Optional[HighlightAgentConfig]:
        """Get default configuration for a content type."""
        ...

    async def delete(self, config_id: int) -> bool:
        """Delete a configuration."""
        ...

    async def exists(self, config_id: int) -> bool:
        """Check if a configuration exists."""
        ...
