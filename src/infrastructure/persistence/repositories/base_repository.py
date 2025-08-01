"""Base repository implementation with mixin-based architecture."""

from typing import (
    TypeVar,
    Generic,
    Optional,
    List,
    Type,
    Protocol,
    runtime_checkable,
    Callable,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists, func
from sqlalchemy.exc import IntegrityError

from src.domain.entities.base import Entity
from src.domain.exceptions import DuplicateEntityError
from src.infrastructure.persistence.models.base import Base


DomainEntity = TypeVar("DomainEntity", bound=Entity)
PersistenceModel = TypeVar("PersistenceModel", bound=Base)
ID = TypeVar("ID")


@runtime_checkable
class RepositoryComponents(Protocol):
    """Protocol defining required repository components."""

    session: AsyncSession
    model_class: Type[PersistenceModel]
    to_domain: Callable[[PersistenceModel], DomainEntity]
    to_persistence: Callable[[DomainEntity], PersistenceModel]


class CRUDMixin:
    """Mixin providing basic CRUD operations."""

    session: AsyncSession
    model_class: Type[PersistenceModel]
    to_domain: Callable[[PersistenceModel], DomainEntity]
    to_persistence: Callable[[DomainEntity], PersistenceModel]

    async def get(self, id: ID) -> Optional[DomainEntity]:
        """Get entity by ID."""
        result = await self.session.get(self.model_class, id)
        return self.to_domain(result) if result else None

    async def get_many(self, ids: List[ID]) -> List[DomainEntity]:
        """Get multiple entities by IDs."""
        if not ids:
            return []

        stmt = select(self.model_class).where(self.model_class.id.in_(ids))
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self.to_domain(model) for model in models]

    async def save(self, entity: DomainEntity) -> DomainEntity:
        """Save entity (create or update)."""
        try:
            model = self.to_persistence(entity)

            if entity.id is None:
                self.session.add(model)
            else:
                await self.session.merge(model)

            await self.session.flush()
            await self.session.refresh(model)
            return self.to_domain(model)

        except IntegrityError as e:
            await self.session.rollback()
            raise DuplicateEntityError(
                entity_type=entity.__class__.__name__,
                duplicate_field="unknown",
                duplicate_value=str(e.orig),
            )

    async def delete(self, id: ID) -> bool:
        """Delete entity by ID."""
        result = await self.session.get(self.model_class, id)
        if result:
            await self.session.delete(result)
            await self.session.flush()
            return True
        return False


class QueryMixin:
    """Mixin providing query capabilities."""

    session: AsyncSession
    model_class: Type[PersistenceModel]
    to_domain: Callable[[PersistenceModel], DomainEntity]

    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        stmt = select(exists().where(self.model_class.id == id))
        result = await self.session.execute(stmt)
        return result.scalar() or False

    async def count(self) -> int:
        """Count total entities."""
        stmt = select(func.count()).select_from(self.model_class)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def find_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[DomainEntity]:
        """Find all entities with optional pagination."""
        stmt = select(self.model_class)

        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self.to_domain(model) for model in models]


class TransactionMixin:
    """Mixin providing transaction management."""

    session: AsyncSession

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()

    async def flush(self) -> None:
        """Flush pending changes without committing."""
        await self.session.flush()


class BaseRepository(
    CRUDMixin, QueryMixin, TransactionMixin, Generic[DomainEntity, PersistenceModel, ID]
):
    """Base repository with mixin-based architecture."""

    def __init__(
        self,
        session: AsyncSession,
        model_class: Type[PersistenceModel],
        to_domain: Optional[Callable[[PersistenceModel], DomainEntity]] = None,
        to_persistence: Optional[Callable[[DomainEntity], PersistenceModel]] = None,
    ):
        """Initialize repository with session, model class, and mapping functions."""
        self.session = session
        self.model_class = model_class

        # Use provided mapping functions or default to entity methods
        if to_domain is None:
            # Assume entity has from_model class method
            def default_to_domain(model):
                entity_class = self._get_entity_class()
                return entity_class.from_model(model)

            self.to_domain = default_to_domain
        else:
            self.to_domain = to_domain

        if to_persistence is None:
            # Assume entity has to_model method
            def default_to_persistence(entity):
                return entity.to_model()

            self.to_persistence = default_to_persistence
        else:
            self.to_persistence = to_persistence

    def _get_entity_class(self):
        """Get the entity class from generic type parameters."""
        # This is a simplified version - in practice you might pass entity_class
        # as a parameter or use more sophisticated type introspection
        raise NotImplementedError("Subclasses must implement entity class detection")

    @classmethod
    def with_entity_mapping(
        cls,
        session: AsyncSession,
        model_class: Type[PersistenceModel],
        entity_class: Type[DomainEntity],
    ):
        """Create repository using entity's from_model/to_model methods.

        This is the preferred Pythonic approach that eliminates mapper classes.
        """

        def to_domain_func(model):
            return entity_class.from_model(model)

        def to_persistence_func(entity):
            return entity.to_model()

        return cls(session, model_class, to_domain_func, to_persistence_func)
