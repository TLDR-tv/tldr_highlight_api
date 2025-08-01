"""Base repository implementation with mixin-based architecture."""

from typing import TypeVar, Generic, Optional, List, Type, Protocol, runtime_checkable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists, func
from sqlalchemy.exc import IntegrityError

from src.domain.entities.base import Entity
from src.domain.exceptions import EntityNotFoundError, DuplicateEntityError
from src.infrastructure.persistence.models.base import Base
from src.infrastructure.persistence.mappers.base import Mapper


DomainEntity = TypeVar("DomainEntity", bound=Entity)
PersistenceModel = TypeVar("PersistenceModel", bound=Base)
ID = TypeVar("ID")


@runtime_checkable
class RepositoryComponents(Protocol):
    """Protocol defining required repository components."""

    session: AsyncSession
    model_class: Type[PersistenceModel]
    mapper: Mapper[DomainEntity, PersistenceModel]


class CRUDMixin:
    """Mixin providing basic CRUD operations."""

    session: AsyncSession
    model_class: Type[PersistenceModel]
    mapper: Mapper[DomainEntity, PersistenceModel]

    async def get(self, id: ID) -> Optional[DomainEntity]:
        """Get entity by ID."""
        result = await self.session.get(self.model_class, id)
        return self.mapper.to_domain(result) if result else None

    async def get_many(self, ids: List[ID]) -> List[DomainEntity]:
        """Get multiple entities by IDs."""
        if not ids:
            return []

        stmt = select(self.model_class).where(self.model_class.id.in_(ids))
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return self.mapper.to_domain_list(list(models))

    async def save(self, entity: DomainEntity) -> DomainEntity:
        """Save entity (create or update)."""
        try:
            model = self.mapper.to_persistence(entity)

            if entity.id is None:
                self.session.add(model)
            else:
                model = await self.session.merge(model)

            await self.session.flush()
            await self.session.refresh(model)
            return self.mapper.to_domain(model)

        except IntegrityError as e:
            await self.session.rollback()
            raise DuplicateEntityError(
                f"Entity violates uniqueness constraint: {str(e)}"
            )

    async def delete(self, id: ID) -> None:
        """Delete entity by ID."""
        model = await self.session.get(self.model_class, id)
        if model is None:
            raise EntityNotFoundError(
                f"{self.model_class.__name__} with id {id} not found"
            )

        await self.session.delete(model)
        await self.session.flush()


class QueryMixin:
    """Mixin providing query operations."""

    session: AsyncSession
    model_class: Type[PersistenceModel]

    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        stmt = select(exists().where(self.model_class.id == id))
        result = await self.session.execute(stmt)
        return result.scalar()

    async def count(self) -> int:
        """Count total entities."""
        stmt = select(func.count()).select_from(self.model_class)
        result = await self.session.execute(stmt)
        return result.scalar() or 0


class TransactionMixin:
    """Mixin providing transaction management."""

    session: AsyncSession

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()


class BaseRepository(
    CRUDMixin, QueryMixin, TransactionMixin, Generic[DomainEntity, PersistenceModel, ID]
):
    """Base repository with mixin-based architecture.

    Combines CRUD, query, and transaction functionality through mixins
    for better separation of concerns and reusability.
    """

    def __init__(
        self,
        session: AsyncSession,
        model_class: Type[PersistenceModel],
        mapper: Mapper[DomainEntity, PersistenceModel],
    ):
        """Initialize repository with session, model class, and mapper."""
        self.session = session
        self.model_class = model_class
        self.mapper = mapper
