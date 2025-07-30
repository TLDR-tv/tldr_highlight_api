"""Stream repository implementation."""

from typing import Optional, List
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload

from src.domain.repositories.stream_repository import (
    StreamRepository as IStreamRepository,
)
from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.stream import Stream as StreamModel
from src.infrastructure.persistence.models.user import User as UserModel
from src.infrastructure.persistence.mappers.stream_mapper import StreamMapper


class StreamRepository(BaseRepository[Stream, StreamModel, int], IStreamRepository):
    """Concrete implementation of StreamRepository using SQLAlchemy."""

    def __init__(self, session):
        """Initialize StreamRepository with session."""
        super().__init__(
            session=session, model_class=StreamModel, mapper=StreamMapper()
        )

    async def get_by_user(
        self,
        user_id: int,
        status: Optional[StreamStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Stream]:
        """Get streams for a user, optionally filtered by status.

        Args:
            user_id: User ID
            status: Optional status filter
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of streams for the user
        """
        stmt = select(StreamModel).where(StreamModel.user_id == user_id)

        if status:
            stmt = stmt.where(StreamModel.status == status.value)

        stmt = stmt.order_by(StreamModel.created_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_organization(
        self,
        organization_id: int,
        status: Optional[StreamStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Stream]:
        """Get streams for an organization.

        Args:
            organization_id: Organization ID
            status: Optional status filter
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of streams for the organization
        """
        # Join with users to find streams by organization members
        stmt = (
            select(StreamModel)
            .join(UserModel, StreamModel.user_id == UserModel.id)
            .join(UserModel.owned_organizations)
            .where(UserModel.owned_organizations.any(id=organization_id))
        )

        if status:
            stmt = stmt.where(StreamModel.status == status.value)

        stmt = stmt.order_by(StreamModel.created_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_active_streams(self) -> List[Stream]:
        """Get all streams currently being processed.

        Returns:
            List of active streams
        """
        stmt = (
            select(StreamModel)
            .where(
                StreamModel.status.in_(
                    [StreamStatus.PENDING.value, StreamStatus.PROCESSING.value]
                )
            )
            .order_by(StreamModel.created_at)
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_platform(
        self, platform: StreamPlatform, limit: int = 100, offset: int = 0
    ) -> List[Stream]:
        """Get streams by platform.

        Args:
            platform: Streaming platform
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of streams from the platform
        """
        stmt = (
            select(StreamModel)
            .where(StreamModel.platform == platform.value)
            .order_by(StreamModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_date_range(
        self, start: Timestamp, end: Timestamp, user_id: Optional[int] = None
    ) -> List[Stream]:
        """Get streams within a date range.

        Args:
            start: Start timestamp
            end: End timestamp
            user_id: Optional user ID filter

        Returns:
            List of streams in the date range
        """
        stmt = select(StreamModel).where(
            and_(
                StreamModel.created_at >= start.value,
                StreamModel.created_at <= end.value,
            )
        )

        if user_id:
            stmt = stmt.where(StreamModel.user_id == user_id)

        stmt = stmt.order_by(StreamModel.created_at.desc())

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def count_by_status(
        self, status: StreamStatus, user_id: Optional[int] = None
    ) -> int:
        """Count streams by status.

        Args:
            status: Stream status
            user_id: Optional user ID filter

        Returns:
            Count of streams with the status
        """
        stmt = (
            select(func.count())
            .select_from(StreamModel)
            .where(StreamModel.status == status.value)
        )

        if user_id:
            stmt = stmt.where(StreamModel.user_id == user_id)

        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_with_highlights(self, stream_id: int) -> Optional[Stream]:
        """Get stream with its highlights loaded.

        Args:
            stream_id: Stream ID

        Returns:
            Stream with highlights if found, None otherwise
        """
        stmt = (
            select(StreamModel)
            .where(StreamModel.id == stream_id)
            .options(selectinload(StreamModel.highlights))
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self.mapper.to_domain(model)

    async def get_processing_stats(self, user_id: int) -> dict:
        """Get processing statistics for a user.

        Args:
            user_id: User ID

        Returns:
            Dictionary with processing statistics
        """
        # Count by status
        status_counts = {}
        for status in StreamStatus:
            count = await self.count_by_status(status, user_id)
            status_counts[status.value] = count

        # Get average processing time for completed streams
        avg_processing_stmt = select(
            func.avg(
                func.extract("epoch", StreamModel.completed_at - StreamModel.started_at)
            )
        ).where(
            and_(
                StreamModel.user_id == user_id,
                StreamModel.status == StreamStatus.COMPLETED.value,
                StreamModel.started_at.isnot(None),
                StreamModel.completed_at.isnot(None),
            )
        )

        result = await self.session.execute(avg_processing_stmt)
        avg_processing_seconds = result.scalar() or 0

        # Get total processing minutes
        total_processing_stmt = select(func.sum(StreamModel.duration_seconds)).where(
            and_(
                StreamModel.user_id == user_id,
                StreamModel.status == StreamStatus.COMPLETED.value,
                StreamModel.duration_seconds.isnot(None),
            )
        )

        result = await self.session.execute(total_processing_stmt)
        total_duration_seconds = result.scalar() or 0

        return {
            "by_status": status_counts,
            "total_streams": sum(status_counts.values()),
            "avg_processing_time_seconds": float(avg_processing_seconds),
            "total_processed_minutes": float(total_duration_seconds / 60),
        }

    async def cleanup_old_streams(self, older_than: Timestamp) -> int:
        """Clean up old completed/failed streams.

        Args:
            older_than: Timestamp before which to clean up

        Returns:
            Number of streams cleaned up
        """
        # Find old completed/failed streams
        stmt = select(StreamModel).where(
            and_(
                StreamModel.created_at < older_than.value,
                StreamModel.status.in_(
                    [
                        StreamStatus.COMPLETED.value,
                        StreamStatus.FAILED.value,
                        StreamStatus.CANCELLED.value,
                    ]
                ),
            )
        )

        result = await self.session.execute(stmt)
        old_streams = list(result.scalars().unique())

        # Delete them (cascading will handle highlights)
        for stream in old_streams:
            await self.session.delete(stream)

        await self.session.flush()
        return len(old_streams)
