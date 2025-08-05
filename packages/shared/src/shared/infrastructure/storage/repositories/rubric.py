"""Repository for rubric persistence."""

from typing import Optional, List
from uuid import UUID
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.domain.models.rubric import Rubric, RubricVisibility
from shared.infrastructure.database.models import RubricModel
from .base import BaseRepository


class RubricRepository(BaseRepository[Rubric, RubricModel]):
    """Repository for managing rubrics."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with session."""
        super().__init__(session, Rubric, RubricModel)

    async def get_by_organization(
        self, organization_id: UUID, include_system: bool = True
    ) -> List[Rubric]:
        """Get all rubrics available to an organization.
        
        Args:
            organization_id: Organization ID
            include_system: Whether to include system rubrics
            
        Returns:
            List of available rubrics
        """
        # Build query conditions
        conditions = []
        
        # Organization's own rubrics
        conditions.append(RubricModel.organization_id == organization_id)
        
        # System rubrics
        if include_system:
            conditions.append(RubricModel.visibility == RubricVisibility.SYSTEM.value)
        
        # Public rubrics from other orgs
        conditions.append(RubricModel.visibility == RubricVisibility.PUBLIC.value)
        
        # Execute query
        query = select(RubricModel).where(
            and_(
                RubricModel.is_active == True,
                or_(*conditions)
            )
        ).order_by(RubricModel.name)
        
        result = await self.session.execute(query)
        models = result.scalars().all()
        
        return [self._to_domain(model) for model in models]
    
    async def get_default_for_organization(
        self, organization_id: UUID
    ) -> Optional[Rubric]:
        """Get the default rubric for an organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            Default rubric if set, otherwise None
        """
        # First check if organization has a default rubric set
        from shared.infrastructure.database.models import OrganizationModel
        
        org_query = select(OrganizationModel).where(
            OrganizationModel.id == organization_id
        )
        org_result = await self.session.execute(org_query)
        org = org_result.scalar_one_or_none()
        
        if org and org.default_rubric_id:
            return await self.get(org.default_rubric_id)
        
        # Otherwise return the general system rubric
        query = select(RubricModel).where(
            and_(
                RubricModel.visibility == RubricVisibility.SYSTEM.value,
                RubricModel.name == "General Highlights",
                RubricModel.is_active == True
            )
        )
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()
        
        return self._to_domain(model) if model else None
    
    async def get_by_name_and_org(
        self, name: str, organization_id: Optional[UUID] = None
    ) -> Optional[Rubric]:
        """Get rubric by name and organization.
        
        Args:
            name: Rubric name
            organization_id: Organization ID (None for system rubrics)
            
        Returns:
            Rubric if found
        """
        conditions = [
            RubricModel.name == name,
            RubricModel.is_active == True
        ]
        
        if organization_id:
            conditions.append(RubricModel.organization_id == organization_id)
        else:
            conditions.append(RubricModel.organization_id.is_(None))
        
        query = select(RubricModel).where(and_(*conditions))
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()
        
        return self._to_domain(model) if model else None
    
    async def increment_usage(self, rubric_id: UUID) -> None:
        """Increment usage count for a rubric.
        
        Args:
            rubric_id: Rubric ID
        """
        rubric = await self.get(rubric_id)
        if rubric:
            rubric.increment_usage()
            await self.update(rubric)
    
    def _to_domain(self, model: RubricModel) -> Rubric:
        """Convert database model to domain model."""
        return Rubric(
            id=model.id,
            organization_id=model.organization_id,
            name=model.name,
            description=model.description,
            config=model.config,
            visibility=RubricVisibility(model.visibility),
            is_active=model.is_active,
            version=model.version,
            usage_count=model.usage_count,
            last_used_at=model.last_used_at,
            created_at=model.created_at,
            updated_at=model.updated_at,
            created_by_user_id=model.created_by_user_id,
        )
    
    def _to_model(self, entity: Rubric) -> RubricModel:
        """Convert domain model to database model."""
        return RubricModel(
            id=entity.id,
            organization_id=entity.organization_id,
            name=entity.name,
            description=entity.description,
            config=entity.config,
            visibility=entity.visibility.value,
            is_active=entity.is_active,
            version=entity.version,
            usage_count=entity.usage_count,
            last_used_at=entity.last_used_at,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            created_by_user_id=entity.created_by_user_id,
        )