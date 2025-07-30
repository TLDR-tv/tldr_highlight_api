"""SQLAlchemy implementation of webhook event repository."""

from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.webhook_event import (
    WebhookEvent as DomainWebhookEvent,
    WebhookEventStatus,
    WebhookEventType,
)
from src.domain.repositories.webhook_event_repository import WebhookEventRepository
from src.infrastructure.persistence.models.webhook_event import (
    WebhookEvent as PersistenceWebhookEvent,
)
from src.infrastructure.persistence.mappers.webhook_event_mapper import (
    WebhookEventMapper,
)
from src.infrastructure.persistence.repositories.base_repository import BaseRepository


class WebhookEventRepositoryImpl(
    BaseRepository[DomainWebhookEvent, PersistenceWebhookEvent, int],
    WebhookEventRepository,
):
    """SQLAlchemy implementation of webhook event repository."""

    def __init__(self, session: AsyncSession):
        """Initialize webhook event repository.

        Args:
            session: SQLAlchemy async session
        """
        super().__init__(
            session=session,
            model_class=PersistenceWebhookEvent,
            mapper=WebhookEventMapper(),
        )

    async def get_by_external_id(
        self, external_event_id: str, platform: str
    ) -> Optional[DomainWebhookEvent]:
        """Get a webhook event by external event ID and platform."""
        stmt = select(PersistenceWebhookEvent).where(
            and_(
                PersistenceWebhookEvent.event_id == external_event_id,
                PersistenceWebhookEvent.platform == platform,
            )
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        return self.mapper.to_domain(model) if model else None

    async def get_recent_events(
        self,
        limit: int = 100,
        status: Optional[WebhookEventStatus] = None,
        event_type: Optional[WebhookEventType] = None,
        platform: Optional[str] = None,
    ) -> List[DomainWebhookEvent]:
        """Get recent webhook events with optional filters."""
        stmt = select(PersistenceWebhookEvent)

        # Apply filters
        conditions = []
        if status:
            conditions.append(PersistenceWebhookEvent.status == status)
        if event_type:
            conditions.append(PersistenceWebhookEvent.event_type == event_type)
        if platform:
            conditions.append(PersistenceWebhookEvent.platform == platform)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Order by received_at descending and limit
        stmt = stmt.order_by(PersistenceWebhookEvent.received_at.desc()).limit(limit)

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self.mapper.to_domain(model) for model in models]

    async def get_failed_events_for_retry(
        self, max_retry_count: int = 3, limit: int = 100
    ) -> List[DomainWebhookEvent]:
        """Get failed events that can be retried."""
        stmt = (
            select(PersistenceWebhookEvent)
            .where(
                and_(
                    PersistenceWebhookEvent.status == WebhookEventStatus.FAILED,
                    PersistenceWebhookEvent.retry_count < max_retry_count,
                )
            )
            .order_by(PersistenceWebhookEvent.updated_at.asc())
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self.mapper.to_domain(model) for model in models]

    async def exists_by_external_id(
        self, external_event_id: str, platform: str
    ) -> bool:
        """Check if a webhook event exists by external ID and platform."""
        stmt = (
            select(PersistenceWebhookEvent.id)
            .where(
                and_(
                    PersistenceWebhookEvent.event_id == external_event_id,
                    PersistenceWebhookEvent.platform == platform,
                )
            )
            .limit(1)
        )

        result = await self.session.execute(stmt)
        return result.scalar() is not None

    async def get_events_by_stream(self, stream_id: int) -> List[DomainWebhookEvent]:
        """Get all webhook events associated with a stream."""
        stmt = (
            select(PersistenceWebhookEvent)
            .where(PersistenceWebhookEvent.stream_id == stream_id)
            .order_by(PersistenceWebhookEvent.received_at.asc())
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self.mapper.to_domain(model) for model in models]

    async def cleanup_old_events(self, days: int = 30) -> int:
        """Clean up old processed webhook events."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Delete processed events older than cutoff
        stmt = select(PersistenceWebhookEvent).where(
            and_(
                PersistenceWebhookEvent.status.in_(
                    [WebhookEventStatus.PROCESSED, WebhookEventStatus.DUPLICATE]
                ),
                PersistenceWebhookEvent.created_at < cutoff_date,
            )
        )

        result = await self.session.execute(stmt)
        models_to_delete = result.scalars().all()

        for model in models_to_delete:
            await self.session.delete(model)

        await self.session.commit()

        return len(models_to_delete)
