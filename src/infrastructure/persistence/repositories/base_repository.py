"""Base repository implementation with common CRUD operations."""

from typing import TypeVar, Generic, Optional, List, Type
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists
from sqlalchemy.exc import IntegrityError

from src.domain.entities.base import Entity
from src.domain.exceptions import EntityNotFoundError, DuplicateEntityError
from src.infrastructure.persistence.models.base import Base
from src.infrastructure.persistence.mappers.base import Mapper


DomainEntity = TypeVar("DomainEntity", bound=Entity)
PersistenceModel = TypeVar("PersistenceModel", bound=Base)
ID = TypeVar("ID")


class BaseRepository(Generic[DomainEntity, PersistenceModel, ID]):
    """Base repository with common CRUD operations.

    This class provides the foundation for all repository implementations,
    handling the basic CRUD operations and entity-model conversions.
    """

    def __init__(
        self,
        session: AsyncSession,
        model_class: Type[PersistenceModel],
        mapper: Mapper[DomainEntity, PersistenceModel],
    ):
        """Initialize repository with session, model class, and mapper.

        Args:
            session: SQLAlchemy async session
            model_class: The SQLAlchemy model class
            mapper: Mapper for converting between domain and persistence
        """
        self.session = session
        self.model_class = model_class
        self.mapper = mapper

    async def get(self, id: ID) -> Optional[DomainEntity]:
        """Get entity by ID.

        Args:
            id: The entity ID

        Returns:
            The domain entity if found, None otherwise
        """
        result = await self.session.get(self.model_class, id)
        if result is None:
            return None
        return self.mapper.to_domain(result)

    async def get_many(self, ids: List[ID]) -> List[DomainEntity]:
        """Get multiple entities by IDs.

        Args:
            ids: List of entity IDs

        Returns:
            List of found domain entities
        """
        if not ids:
            return []

        stmt = select(self.model_class).where(self.model_class.id.in_(ids))
        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return self.mapper.to_domain_list(list(models))

    async def save(self, entity: DomainEntity) -> DomainEntity:
        """Save entity (create or update).

        Args:
            entity: The domain entity to save

        Returns:
            The saved domain entity with updated ID and timestamps

        Raises:
            DuplicateEntityError: If entity violates uniqueness constraints
        """
        try:
            model = self.mapper.to_persistence(entity)

            if entity.id is None:
                # New entity - add to session
                self.session.add(model)
            else:
                # Existing entity - merge with session
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
        """Delete entity by ID.

        Args:
            id: The entity ID to delete

        Raises:
            EntityNotFoundError: If entity doesn't exist
        """
        model = await self.session.get(self.model_class, id)
        if model is None:
            raise EntityNotFoundError(
                f"{self.model_class.__name__} with id {id} not found"
            )

        await self.session.delete(model)
        await self.session.flush()

    async def exists(self, id: ID) -> bool:
        """Check if entity exists.

        Args:
            id: The entity ID to check

        Returns:
            True if entity exists, False otherwise
        """
        stmt = select(exists().where(self.model_class.id == id))
        result = await self.session.execute(stmt)
        return result.scalar()

    async def count(self) -> int:
        """Count total entities.

        Returns:
            Total count of entities
        """
        from sqlalchemy import func

        stmt = select(func.count()).select_from(self.model_class)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def commit(self) -> None:
        """Commit the current transaction.

        This should typically be called at the service/use case level,
        not within individual repository methods.
        """
        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction.

        This should typically be called at the service/use case level
        when an error occurs.
        """
        await self.session.rollback()
