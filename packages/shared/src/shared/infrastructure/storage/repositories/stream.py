"""Stream repository implementation."""

from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ....domain.models.stream import Stream, StreamStatus, StreamSource, StreamType
from ...database.models import StreamModel


class StreamRepository:
    """SQLAlchemy implementation of stream repository."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def add(self, entity: Stream) -> Stream:
        """Add stream to repository."""
        model = StreamModel(
            id=entity.id,
            organization_id=entity.organization_id,
            stream_url=entity.url,
            stream_fingerprint=entity.stream_fingerprint,
            source_type=entity.source_type,
            status=entity.status,
            title=entity.name or entity.url[:100],
            description=entity.metadata.get("description", "") if entity.metadata else "",
            started_at=entity.started_at,
            completed_at=entity.completed_at,
            duration_seconds=entity.duration_seconds,
            segments_processed=entity.segments_processed,
            highlights_generated=entity.highlights_generated,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            error_message=entity.error_message,
            retry_count=entity.retry_count,
        )

        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)

        return self._to_entity(model)

    # Alias for compatibility with tests
    async def create(self, entity: Stream) -> Stream:
        """Create new stream (alias for add)."""
        return await self.add(entity)

    async def get(self, id: UUID) -> Optional[Stream]:
        """Get stream by ID."""
        model = await self.session.get(StreamModel, id)
        return self._to_entity(model) if model else None

    async def get_by_fingerprint(
        self, fingerprint: str, org_id: UUID
    ) -> Optional[Stream]:
        """Get stream by fingerprint within an organization."""
        return None

    async def list_active(self, org_id: UUID) -> list[Stream]:
        """List active streams for an organization."""
        return []

    async def list_by_organization(
        self, org_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Stream]:
        """List streams for an organization with pagination."""
        result = await self.session.execute(
            select(StreamModel)
            .where(StreamModel.organization_id == org_id)
            .order_by(StreamModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        models = result.scalars().all()
        return [self._to_entity(model) for model in models]
    
    async def count_by_organization(self, org_id: UUID) -> int:
        """Count total streams for an organization."""
        from sqlalchemy import func
        
        result = await self.session.execute(
            select(func.count(StreamModel.id))
            .where(StreamModel.organization_id == org_id)
        )
        return result.scalar() or 0

    async def update(self, entity: Stream) -> Stream:
        """Update existing stream."""
        return entity

    async def delete(self, id: UUID) -> None:
        """Delete stream by ID."""
        pass

    async def list(self, **filters) -> list[Stream]:
        """List streams with optional filters."""
        return []

    def _to_entity(self, model: StreamModel) -> Stream:
        """Convert model to entity."""
        return Stream(
            id=model.id,
            organization_id=model.organization_id,
            url=model.stream_url,
            name=model.title,
            type=StreamType.LIVESTREAM,  # Default or determine from metadata
            status=StreamStatus(model.status),
            celery_task_id=None,  # Would need to be stored in model
            metadata={},  # Would need to be stored in model
            stream_fingerprint=model.stream_fingerprint,
            source_type=StreamSource(model.source_type),
            started_at=model.started_at,
            completed_at=model.completed_at,
            duration_seconds=model.duration_seconds,
            segments_processed=model.segments_processed,
            highlights_generated=model.highlights_generated,
            stats=None,  # Would need to be stored in model
            created_at=model.created_at,
            updated_at=model.updated_at,
            error_message=model.error_message,
            retry_count=model.retry_count,
        )
