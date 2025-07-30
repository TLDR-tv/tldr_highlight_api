"""Base mapper for domain entity to persistence model conversion."""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.domain.entities.base import Entity
from src.infrastructure.persistence.models.base import Base


DomainEntity = TypeVar("DomainEntity", bound=Entity)
PersistenceModel = TypeVar("PersistenceModel", bound=Base)


class Mapper(ABC, Generic[DomainEntity, PersistenceModel]):
    """Base mapper for converting between domain entities and persistence models.

    This abstract class defines the interface for bidirectional mapping
    between the domain layer and persistence layer.
    """

    @abstractmethod
    def to_domain(self, model: PersistenceModel) -> DomainEntity:
        """Convert persistence model to domain entity."""
        ...

    @abstractmethod
    def to_persistence(self, entity: DomainEntity) -> PersistenceModel:
        """Convert domain entity to persistence model."""
        ...

    def to_domain_list(self, models: list[PersistenceModel]) -> list[DomainEntity]:
        """Convert list of persistence models to domain entities."""
        return [self.to_domain(model) for model in models]

    def to_persistence_list(
        self, entities: list[DomainEntity]
    ) -> list[PersistenceModel]:
        """Convert list of domain entities to persistence models."""
        return [self.to_persistence(entity) for entity in entities]
