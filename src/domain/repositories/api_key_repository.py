"""API key repository protocol."""

from typing import Protocol, Optional, List

from src.domain.repositories.base import Repository
from src.domain.entities.api_key import APIKey
from src.domain.value_objects.timestamp import Timestamp


class APIKeyRepository(Repository[APIKey, int], Protocol):
    """Repository protocol for APIKey entities.

    Extends the base repository with API key-specific operations.
    """
    
    async def get_by_key_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by its hash."""
        ...
    
    async def get_by_user(self, user_id: int, active_only: bool = True) -> List[APIKey]:
        """Get all API keys for a user."""
        ...
        
    async def get_active_keys(self) -> List[APIKey]:
        """Get all active API keys."""
        ...
        
    async def get_expiring_soon(self, days: int = 7) -> List[APIKey]:
        """Get API keys expiring within specified days."""
        ...
        
    async def get_by_scope(self, scope: str, active_only: bool = True) -> List[APIKey]:
        """Get all API keys with specific scope."""
        ...
        
    async def count_by_user(self, user_id: int, active_only: bool = True) -> int:
        """Count API keys for a user."""
        ...
        
    async def cleanup_expired(self) -> int:
        """Remove expired API keys."""
        ...
        
    async def get_usage_stats(
        self, api_key_id: int, since: Optional[Timestamp] = None
    ) -> dict:
        """Get usage statistics for an API key."""
        ...
        
    async def revoke_all_for_user(self, user_id: int) -> int:
        """Revoke all API keys for a user."""
        ...
