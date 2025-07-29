"""Base entity class for domain entities."""

from dataclasses import dataclass
from typing import TypeVar, Generic, Optional
from abc import ABC

from src.domain.value_objects.timestamp import Timestamp


T = TypeVar('T')


@dataclass
class Entity(ABC, Generic[T]):
    """Base class for all domain entities.
    
    Entities have identity and are distinguishable by their ID,
    not by their attributes.
    """
    
    id: Optional[T]
    created_at: Timestamp
    updated_at: Timestamp
    
    def __eq__(self, other: object) -> bool:
        """Entities are equal if they have the same type and ID."""
        if not isinstance(other, self.__class__):
            return False
        if self.id is None or other.id is None:
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on entity type and ID."""
        if self.id is None:
            raise ValueError("Cannot hash entity without ID")
        return hash((self.__class__.__name__, self.id))