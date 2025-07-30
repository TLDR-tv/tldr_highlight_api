"""Base repository protocol."""

from typing import Protocol, TypeVar, Generic, Optional, List
from abc import abstractmethod

from src.domain.entities.base import Entity


T = TypeVar("T", bound=Entity)
ID = TypeVar("ID")


class Repository(Protocol, Generic[T, ID]):
    """Base repository protocol for entity persistence.

    This protocol defines the basic CRUD operations that all
    repositories must implement.
    """

    @abstractmethod
    async def get(self, id: ID) -> Optional[T]:
        """Get entity by ID."""
        ...

    @abstractmethod
    async def get_many(self, ids: List[ID]) -> List[T]:
        """Get multiple entities by IDs."""
        ...

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity (create or update)."""
        ...

    @abstractmethod
    async def delete(self, id: ID) -> None:
        """Delete entity by ID."""
        ...

    @abstractmethod
    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        ...
