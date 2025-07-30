"""Concrete implementation of HighlightTypeRegistryRepository using SQLAlchemy."""

from typing import Optional
from sqlalchemy import select

from src.domain.repositories.highlight_type_registry_repository import (
    HighlightTypeRegistryRepository as IHighlightTypeRegistryRepository,
)
from src.domain.entities.highlight_type_registry import HighlightTypeRegistry
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.highlight_type_registry import (
    HighlightTypeRegistry as HighlightTypeRegistryModel,
)
from src.infrastructure.persistence.mappers.highlight_type_registry_mapper import (
    HighlightTypeRegistryMapper,
)


class HighlightTypeRegistryRepository(
    BaseRepository[HighlightTypeRegistry, HighlightTypeRegistryModel, int],
    IHighlightTypeRegistryRepository,
):
    """Concrete implementation of HighlightTypeRegistryRepository using SQLAlchemy."""

    def __init__(self, session):
        """Initialize HighlightTypeRegistryRepository with session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(
            session=session,
            model_class=HighlightTypeRegistryModel,
            mapper=HighlightTypeRegistryMapper(),
        )

    async def get_by_organization(
        self, organization_id: int
    ) -> Optional[HighlightTypeRegistry]:
        """Get the type registry for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            HighlightTypeRegistry if found, None otherwise
        """
        stmt = select(HighlightTypeRegistryModel).where(
            HighlightTypeRegistryModel.organization_id == organization_id
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        return self.mapper.to_domain(model) if model else None

    async def delete(self, registry_id: int) -> bool:
        """Delete a type registry.

        Args:
            registry_id: Registry ID

        Returns:
            True if deleted, False otherwise
        """
        stmt = select(HighlightTypeRegistryModel).where(
            HighlightTypeRegistryModel.id == registry_id
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model:
            await self.session.delete(model)
            await self.session.commit()
            return True

        return False
