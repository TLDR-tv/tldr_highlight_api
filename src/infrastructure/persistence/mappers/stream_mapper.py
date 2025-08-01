"""Pythonic stream mapping functions.

Replaces mapper classes with simple functions and entity class methods.
This follows Python's preference for functions over classes when possible.
"""

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.stream import Stream as DomainStream
from src.infrastructure.persistence.models.stream import Stream as PersistenceStream


# Pythonic mapping functions - use entity class methods instead of mapper classes


def stream_to_domain(model: PersistenceStream) -> DomainStream:
    """Convert persistence model to domain entity using entity class method."""
    return DomainStream.from_model(model)


def stream_to_persistence(entity: DomainStream) -> PersistenceStream:
    """Convert domain entity to persistence model using entity method."""
    return entity.to_model()


# Legacy mapper class for backward compatibility
class StreamMapper(Mapper):
    """Legacy mapper - use stream_to_domain/stream_to_persistence functions instead."""

    def __init__(self):
        super().__init__(stream_to_domain, stream_to_persistence)

    def to_domain(self, model: PersistenceStream) -> DomainStream:
        """Convert persistence model to domain entity."""
        return stream_to_domain(model)

    def to_persistence(self, entity: DomainStream) -> PersistenceStream:
        """Convert domain entity to persistence model."""
        return stream_to_persistence(entity)
