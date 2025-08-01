"""Pythonic mapping utilities for domain entity to persistence model conversion."""

from typing import TypeVar, Protocol, runtime_checkable, Callable, List

from src.domain.entities.base import Entity
from src.infrastructure.persistence.models.base import Base


DomainEntity = TypeVar("DomainEntity", bound=Entity)
PersistenceModel = TypeVar("PersistenceModel", bound=Base)


@runtime_checkable
class DomainMappable(Protocol):
    """Protocol for domain entities that can be mapped to persistence models."""

    @classmethod
    def from_model(cls, model: Base) -> "DomainMappable":
        """Create domain entity from persistence model."""
        ...

    def to_model(self) -> Base:
        """Convert domain entity to persistence model."""
        ...


@runtime_checkable
class PersistenceMappable(Protocol):
    """Protocol for persistence models that can be mapped from domain entities."""

    @classmethod
    def from_entity(cls, entity: Entity) -> "PersistenceMappable":
        """Create persistence model from domain entity."""
        ...

    def to_entity(self) -> Entity:
        """Convert persistence model to domain entity."""
        ...


# Utility functions for mapping lists
def map_to_domain_list(
    models: List[PersistenceModel],
    mapper_func: Callable[[PersistenceModel], DomainEntity],
) -> List[DomainEntity]:
    """Convert list of persistence models to domain entities."""
    return [mapper_func(model) for model in models]


def map_to_persistence_list(
    entities: List[DomainEntity],
    mapper_func: Callable[[DomainEntity], PersistenceModel],
) -> List[PersistenceModel]:
    """Convert list of domain entities to persistence models."""
    return [mapper_func(entity) for entity in entities]


# Convenience functions for protocol-based mapping
def to_domain_list(models: List[PersistenceMappable]) -> List[Entity]:
    """Convert list of mappable models to domain entities."""
    return [model.to_entity() for model in models]


def from_domain_list(entities: List[DomainMappable]) -> List[Base]:
    """Convert list of mappable entities to persistence models."""
    return [entity.to_model() for entity in entities]
