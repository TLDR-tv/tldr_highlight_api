"""Organization repository implementation."""

from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ....domain.models.organization import Organization
from ...database.models import OrganizationModel


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
            webhook_url=entity.webhook_url,
            webhook_secret=entity.webhook_secret,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

        self.session.add(model)

        # Add wake words as separate entities
        from ...database.models import WakeWordModel

        for word in entity.wake_words:
            wake_word = WakeWordModel(
                organization_id=entity.id,
                word=word,
                is_active=True,
            )
            self.session.add(wake_word)

        await self.session.commit()
        await self.session.refresh(model)

        return self._to_entity(model)

    # Alias for compatibility with tests
    async def create(self, entity: Organization) -> Organization:
        """Create new organization (alias for add)."""
        return await self.add(entity)

    async def get(self, id: UUID) -> Optional[Organization]:
        """Get organization by ID."""
        from sqlalchemy.orm import selectinload

        result = await self.session.execute(
            select(OrganizationModel)
            .where(OrganizationModel.id == id)
            .options(selectinload(OrganizationModel.wake_word_configs))
        )
        model = result.scalar_one_or_none()
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
        from sqlalchemy.orm import selectinload
        from ...database.models import WakeWordModel

        # Load organization with wake words eagerly
        result = await self.session.execute(
            select(OrganizationModel)
            .where(OrganizationModel.id == entity.id)
            .options(selectinload(OrganizationModel.wake_word_configs))
        )
        model = result.scalar_one_or_none()

        if not model:
            raise ValueError(f"Organization {entity.id} not found")

        # Update fields
        model.name = entity.name
        model.slug = entity.slug
        model.is_active = entity.is_active
        model.total_streams_processed = entity.total_streams_processed
        model.total_highlights_generated = entity.total_highlights_generated
        model.total_processing_seconds = entity.total_processing_seconds
        model.webhook_url = entity.webhook_url
        model.webhook_secret = entity.webhook_secret
        model.updated_at = entity.updated_at

        # Update wake words - only add/remove changed ones
        existing_words = {ww.phrase for ww in model.wake_word_configs if ww.is_active}
        new_words = entity.wake_words

        # Remove words that are no longer in the set
        for ww in model.wake_word_configs[
            :
        ]:  # Use slice to avoid modifying during iteration
            if ww.phrase not in new_words:
                model.wake_word_configs.remove(ww)

        # Add new words that don't exist
        for word in new_words:
            if word not in existing_words:
                wake_word = WakeWordModel(
                    organization_id=entity.id,
                    phrase=word,
                    is_active=True,
                )
                model.wake_word_configs.append(wake_word)

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
        # Extract wake words from relationship
        wake_words = set()
        # Only access wake_word_configs if it's already loaded
        if "wake_word_configs" in model.__dict__:
            wake_words = {ww.phrase for ww in model.wake_word_configs if ww.is_active}

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
            wake_words=wake_words,
            webhook_url=model.webhook_url,
            webhook_secret=model.webhook_secret,
        )
