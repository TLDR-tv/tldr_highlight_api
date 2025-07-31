"""Base repository protocol."""

from typing import Protocol, TypeVar, Generic, Optional, List

from src.domain.entities.base import Entity


T = TypeVar("T", bound=Entity)
ID = TypeVar("ID")


class Repository(Protocol, Generic[T, ID]):
    """Base repository protocol for entity persistence.

    This protocol defines the basic CRUD operations that all
    repositories must implement.
    """

    async def get(self, id: ID) -> Optional[T]:
        """Get entity by ID."""
        ...

    async def get_many(self, ids: List[ID]) -> List[T]:
        """Get multiple entities by IDs."""
        ...

    async def save(self, entity: T) -> T:
        """Save entity (create or update)."""
        ...

    async def delete(self, id: ID) -> None:
        """Delete entity by ID."""
        ...

    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        ...
