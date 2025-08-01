"""Organization repository implementation."""

from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ....domain.models.organization import Organization
from ..models import OrganizationModel


class OrganizationRepository:
    """SQLAlchemy implementation of organization repository."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def add(self, entity: Organization) -> Organization:
        """Add organization to repository."""
        model = OrganizationModel(
            id=entity.id,
            name=entity.name,
            slug=entity.slug,
            is_active=entity.is_active,
            total_streams_processed=entity.total_streams_processed,
            total_highlights_generated=entity.total_highlights_generated,
            total_processing_seconds=entity.total_processing_seconds,
            wake_words=list(entity.wake_words),
            webhook_url=entity.webhook_url,
            webhook_secret=entity.webhook_secret,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)

        return self._to_entity(model)

    async def get(self, id: UUID) -> Optional[Organization]:
        """Get organization by ID."""
        model = await self.session.get(OrganizationModel, id)
        return self._to_entity(model) if model else None

    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        result = await self.session.execute(
            select(OrganizationModel).where(OrganizationModel.slug == slug)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_api_key(self, api_key: str) -> Optional[Organization]:
        """Get organization associated with API key."""
        # This would join with API key table
        # For now, returning None as API key lookup isn't implemented
        return None

    async def update(self, entity: Organization) -> Organization:
        """Update existing organization."""
        model = await self.session.get(OrganizationModel, entity.id)
        if not model:
            raise ValueError(f"Organization {entity.id} not found")

        # Update fields
        model.name = entity.name
        model.slug = entity.slug
        model.is_active = entity.is_active
        model.total_streams_processed = entity.total_streams_processed
        model.total_highlights_generated = entity.total_highlights_generated
        model.total_processing_seconds = entity.total_processing_seconds
        model.wake_words = list(entity.wake_words)
        model.webhook_url = entity.webhook_url
        model.webhook_secret = entity.webhook_secret
        model.updated_at = entity.updated_at

        await self.session.commit()
        await self.session.refresh(model)

        return self._to_entity(model)

    async def delete(self, id: UUID) -> None:
        """Delete organization by ID."""
        model = await self.session.get(OrganizationModel, id)
        if model:
            await self.session.delete(model)
            await self.session.commit()

    async def list(self, **filters) -> list[Organization]:
        """List organizations with optional filters."""
        query = select(OrganizationModel)

        # Apply filters
        if "is_active" in filters:
            query = query.where(OrganizationModel.is_active == filters["is_active"])

        result = await self.session.execute(query)
        models = result.scalars().all()

        return [self._to_entity(model) for model in models]

    def _to_entity(self, model: OrganizationModel) -> Organization:
        """Convert model to entity."""
        return Organization(
            id=model.id,
            name=model.name,
            slug=model.slug,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
            total_streams_processed=model.total_streams_processed,
            total_highlights_generated=model.total_highlights_generated,
            total_processing_seconds=model.total_processing_seconds,
            wake_words=set(model.wake_words) if model.wake_words else set(),
            webhook_url=model.webhook_url,
            webhook_secret=model.webhook_secret,
        )
