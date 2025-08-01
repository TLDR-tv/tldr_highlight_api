"""Base entity class for domain entities."""

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List

from src.domain.events import DomainEvent
from src.domain.value_objects.timestamp import Timestamp


T = TypeVar("T")


@dataclass(kw_only=True)
class Entity(Generic[T]):
    """Base class for all domain entities.

    Entities have identity and are distinguishable by their ID,
    not by their attributes.
    """

    id: Optional[T] = None

    # Audit timestamps
    created_at: Timestamp = field(default_factory=Timestamp.now)
    updated_at: Timestamp = field(default_factory=Timestamp.now)

    def __post_init__(self):
        """Initialize entity after dataclass initialization."""
        pass

    def _touch_updated_at(self) -> None:
        """Update the updated_at timestamp to current time."""
        object.__setattr__(self, "updated_at", Timestamp.now())

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


@dataclass(kw_only=True)
class AggregateRoot(Entity[T]):
    """Base class for aggregate roots.

    Aggregate roots are the entry point to an aggregate and
    maintain consistency across the aggregate boundary.
    They can raise domain events.
    """

    _domain_events: List[DomainEvent] = field(default_factory=list, init=False)

    def add_domain_event(self, event: DomainEvent) -> None:
        """Add a domain event to be raised."""
        if event.aggregate_id is None:
            event.aggregate_id = self.id
        if event.aggregate_type is None:
            event.aggregate_type = self.__class__.__name__
        self._domain_events.append(event)

    def clear_domain_events(self) -> List[DomainEvent]:
        """Clear and return all domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events

    @property
    def domain_events(self) -> List[DomainEvent]:
        """Get copy of domain events."""
        return self._domain_events.copy()
