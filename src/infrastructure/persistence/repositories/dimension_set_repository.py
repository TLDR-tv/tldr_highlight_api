"""Concrete implementation of DimensionSetRepository using SQLAlchemy."""

from typing import List, Optional
from sqlalchemy import select, and_

from src.domain.repositories.dimension_set_repository_interface import (
    DimensionSetRepository as IDimensionSetRepository,
)
from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.dimension_set import (
    DimensionSet as DimensionSetModel,
)
from src.infrastructure.persistence.mappers.dimension_set_mapper import (
    DimensionSetMapper,
)


class DimensionSetRepository(
    BaseRepository[DimensionSetAggregate, DimensionSetModel, int], IDimensionSetRepository
):
    """Concrete implementation of DimensionSetRepository using SQLAlchemy."""

    def __init__(self, session):
        """Initialize DimensionSetRepository with session.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(
            session=session, model_class=DimensionSetModel, mapper=DimensionSetMapper()
        )

    async def get_by_organization(self, organization_id: int) -> List[DimensionSetAggregate]:
        """Get all dimension sets for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            List of dimension sets
        """
        stmt = (
            select(DimensionSetModel)
            .where(DimensionSetModel.organization_id == organization_id)
            .order_by(DimensionSetModel.name)
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_name(
        self, organization_id: int, name: str
    ) -> Optional[DimensionSetAggregate]:
        """Get a dimension set by name within an organization.

        Args:
            organization_id: Organization ID
            name: DimensionSet name

        Returns:
            DimensionSet if found, None otherwise
        """
        stmt = select(DimensionSetModel).where(
            and_(
                DimensionSetModel.organization_id == organization_id,
                DimensionSetModel.name == name,
            )
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        return self.mapper.to_domain(model) if model else None

    async def delete(self, dimension_set_id: int) -> bool:
        """Delete a dimension set.

        Args:
            dimension_set_id: DimensionSet ID

        Returns:
            True if deleted, False otherwise
        """
        stmt = select(DimensionSetModel).where(DimensionSetModel.id == dimension_set_id)

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model:
            await self.session.delete(model)
            await self.session.commit()
            return True

        return False
