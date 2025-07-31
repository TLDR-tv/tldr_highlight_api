"""Highlight repository implementation."""

import json
from typing import Optional, List, Dict, Any
from sqlalchemy import select, func, and_, or_, text

from src.domain.repositories.highlight_repository import (
    HighlightRepository as IHighlightRepository,
)
from src.domain.entities.highlight import Highlight
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.highlight import Highlight as HighlightModel
from src.infrastructure.persistence.models.stream import Stream as StreamModel
from src.infrastructure.persistence.mappers.highlight_mapper import HighlightMapper


class HighlightRepository(
    BaseRepository[Highlight, HighlightModel, int], IHighlightRepository
):
    """Concrete implementation of HighlightRepository using SQLAlchemy."""

    def __init__(self, session):
        """Initialize HighlightRepository with session."""
        super().__init__(
            session=session, model_class=HighlightModel, mapper=HighlightMapper()
        )

    async def get_by_stream(
        self,
        stream_id: int,
        min_confidence: Optional[float] = None,
        types: Optional[List[str]] = None,
    ) -> List[Highlight]:
        """Get highlights for a stream with optional filters.

        Args:
            stream_id: Stream ID
            min_confidence: Minimum confidence score filter
            types: List of highlight types to filter by

        Returns:
            List of highlights for the stream
        """
        stmt = select(HighlightModel).where(HighlightModel.stream_id == stream_id)

        if min_confidence is not None:
            stmt = stmt.where(HighlightModel.confidence_score >= min_confidence)

        if types:
            type_values = types  # Already strings
            stmt = stmt.where(HighlightModel.highlight_type.in_(type_values))

        # Order by confidence score and start time
        stmt = stmt.order_by(
            HighlightModel.confidence_score.desc(),
            HighlightModel.start_time_seconds.asc(),
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_user(
        self, user_id: int, limit: int = 100, offset: int = 0
    ) -> List[Highlight]:
        """Get all highlights for a user across all streams.

        Args:
            user_id: User ID
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of highlights for the user
        """
        stmt = (
            select(HighlightModel)
            .join(StreamModel, HighlightModel.stream_id == StreamModel.id)
            .where(StreamModel.user_id == user_id)
            .order_by(HighlightModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_confidence_range(
        self, min_score: float, max_score: float, user_id: Optional[int] = None
    ) -> List[Highlight]:
        """Get highlights within a confidence score range.

        Args:
            min_score: Minimum confidence score
            max_score: Maximum confidence score
            user_id: Optional user ID filter

        Returns:
            List of highlights in the confidence range
        """
        stmt = select(HighlightModel).where(
            and_(
                HighlightModel.confidence_score >= min_score,
                HighlightModel.confidence_score <= max_score,
            )
        )

        if user_id:
            stmt = stmt.join(
                StreamModel, HighlightModel.stream_id == StreamModel.id
            ).where(StreamModel.user_id == user_id)

        stmt = stmt.order_by(HighlightModel.confidence_score.desc())

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_type(
        self, highlight_type: str, user_id: Optional[int] = None, limit: int = 100
    ) -> List[Highlight]:
        """Get highlights by type.

        Args:
            highlight_type: Type of highlight
            user_id: Optional user ID filter
            limit: Maximum number of results

        Returns:
            List of highlights of the specified type
        """
        stmt = select(HighlightModel).where(
            HighlightModel.highlight_type == highlight_type
        )

        if user_id:
            stmt = stmt.join(
                StreamModel, HighlightModel.stream_id == StreamModel.id
            ).where(StreamModel.user_id == user_id)

        stmt = stmt.order_by(HighlightModel.confidence_score.desc()).limit(limit)

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_tags(
        self, tags: List[str], match_all: bool = False, user_id: Optional[int] = None
    ) -> List[Highlight]:
        """Get highlights by tags (match any or all).

        Args:
            tags: List of tags to search for
            match_all: If True, match all tags; if False, match any tag
            user_id: Optional user ID filter

        Returns:
            List of highlights matching the tag criteria
        """
        if not tags:
            return []

        stmt = select(HighlightModel)

        if match_all:
            # All tags must be present - use JSON containment
            for tag in tags:
                stmt = stmt.where(
                    func.json_extract(HighlightModel.tags, "$").contains(f'"{tag}"')
                )
        else:
            # Any tag can be present
            tag_conditions = []
            for tag in tags:
                tag_conditions.append(
                    func.json_extract(HighlightModel.tags, "$").contains(f'"{tag}"')
                )
            stmt = stmt.where(or_(*tag_conditions))

        if user_id:
            stmt = stmt.join(
                StreamModel, HighlightModel.stream_id == StreamModel.id
            ).where(StreamModel.user_id == user_id)

        stmt = stmt.order_by(HighlightModel.confidence_score.desc())

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_trending(
        self, time_window: Timestamp, limit: int = 10
    ) -> List[Highlight]:
        """Get trending highlights based on engagement.

        Args:
            time_window: Timestamp defining the start of the time window
            limit: Maximum number of results

        Returns:
            List of trending highlights
        """
        # Calculate trending score: confidence * engagement * recency_factor
        recency_weight = 0.3
        confidence_weight = 0.4
        engagement_weight = 0.3

        stmt = (
            select(
                HighlightModel,
                (
                    confidence_weight * HighlightModel.confidence_score
                    + engagement_weight
                    * func.coalesce(HighlightModel.viewer_engagement, 0.0)
                    + recency_weight
                    * func.exp(
                        -func.extract("epoch", func.now() - HighlightModel.created_at)
                        / 86400.0
                    )
                ).label("trending_score"),
            )
            .where(HighlightModel.created_at >= time_window.value)
            .order_by(text("trending_score DESC"))
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        models = [row[0] for row in result.unique()]

        return self.mapper.to_domain_list(models)

    async def get_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get highlight statistics for a user.

        Args:
            user_id: User ID

        Returns:
            Dictionary with user's highlight statistics
        """
        # Total count
        total_stmt = (
            select(func.count())
            .select_from(HighlightModel)
            .join(StreamModel, HighlightModel.stream_id == StreamModel.id)
            .where(StreamModel.user_id == user_id)
        )

        total_result = await self.session.execute(total_stmt)
        total_count = total_result.scalar() or 0

        # Count by type
        type_stmt = (
            select(HighlightModel.highlight_type, func.count())
            .select_from(HighlightModel)
            .join(StreamModel, HighlightModel.stream_id == StreamModel.id)
            .where(StreamModel.user_id == user_id)
            .group_by(HighlightModel.highlight_type)
        )

        type_result = await self.session.execute(type_stmt)
        type_counts = {row[0]: row[1] for row in type_result}

        # Average confidence score
        avg_confidence_stmt = (
            select(func.avg(HighlightModel.confidence_score))
            .select_from(HighlightModel)
            .join(StreamModel, HighlightModel.stream_id == StreamModel.id)
            .where(StreamModel.user_id == user_id)
        )

        avg_result = await self.session.execute(avg_confidence_stmt)
        avg_confidence = avg_result.scalar() or 0.0

        # High confidence count (>= 0.8)
        high_conf_stmt = (
            select(func.count())
            .select_from(HighlightModel)
            .join(StreamModel, HighlightModel.stream_id == StreamModel.id)
            .where(
                and_(
                    StreamModel.user_id == user_id,
                    HighlightModel.confidence_score >= 0.8,
                )
            )
        )

        high_conf_result = await self.session.execute(high_conf_stmt)
        high_confidence_count = high_conf_result.scalar() or 0

        # Most common tags
        tags_stmt = (
            select(HighlightModel.tags)
            .select_from(HighlightModel)
            .join(StreamModel, HighlightModel.stream_id == StreamModel.id)
            .where(StreamModel.user_id == user_id)
        )

        tags_result = await self.session.execute(tags_stmt)
        all_tags = []
        for row in tags_result:
            if row[0]:  # tags field is not null
                tags_list = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                all_tags.extend(tags_list)

        # Count tag frequencies
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Get top 10 tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_highlights": total_count,
            "by_type": type_counts,
            "average_confidence": float(avg_confidence),
            "high_confidence_count": high_confidence_count,
            "high_confidence_percentage": (high_confidence_count / total_count * 100)
            if total_count > 0
            else 0,
            "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
        }

    async def search(
        self, query: str, user_id: Optional[int] = None, limit: int = 100
    ) -> List[Highlight]:
        """Search highlights by title or description.

        Args:
            query: Search query
            user_id: Optional user ID filter
            limit: Maximum number of results

        Returns:
            List of highlights matching the search query
        """
        search_term = f"%{query.lower()}%"

        stmt = select(HighlightModel).where(
            or_(
                func.lower(HighlightModel.title).like(search_term),
                func.lower(HighlightModel.description).like(search_term),
            )
        )

        if user_id:
            stmt = stmt.join(
                StreamModel, HighlightModel.stream_id == StreamModel.id
            ).where(StreamModel.user_id == user_id)

        # Order by relevance (exact title matches first, then confidence)
        stmt = stmt.order_by(
            func.lower(HighlightModel.title).like(f"%{query.lower()}%").desc(),
            HighlightModel.confidence_score.desc(),
        ).limit(limit)

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def bulk_create(self, highlights: List[Highlight]) -> List[Highlight]:
        """Create multiple highlights at once.

        Args:
            highlights: List of highlight entities to create

        Returns:
            List of created highlight entities with IDs
        """
        if not highlights:
            return []

        try:
            # Convert to persistence models
            models = [self.mapper.to_persistence(highlight) for highlight in highlights]

            # Add all models to session
            self.session.add_all(models)
            await self.session.flush()  # Flush to get IDs

            # Refresh all models to get complete data
            for model in models:
                await self.session.refresh(model)

            # Convert back to domain entities
            return self.mapper.to_domain_list(models)

        except Exception:
            await self.session.rollback()
            raise

    async def count_by_stream(self, stream_id: int) -> int:
        """Count the total number of highlights for a stream.

        Args:
            stream_id: Stream ID

        Returns:
            Total count of highlights for the stream
        """
        # Build count query
        count_stmt = (
            select(func.count())
            .select_from(HighlightModel)
            .where(HighlightModel.stream_id == stream_id)
        )
        count_result = await self.session.execute(count_stmt)
        return count_result.scalar() or 0

    async def get_by_stream_with_pagination(
        self,
        stream_id: int,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "confidence",
    ) -> Dict[str, Any]:
        """Get highlights for a stream with pagination and metadata.

        Args:
            stream_id: Stream ID
            limit: Maximum number of results per page
            offset: Number of results to skip
            order_by: Ordering criteria ("confidence", "time", "created")

        Returns:
            Dictionary containing highlights and pagination metadata
        """
        # Build base query
        stmt = select(HighlightModel).where(HighlightModel.stream_id == stream_id)

        # Apply ordering
        if order_by == "confidence":
            stmt = stmt.order_by(HighlightModel.confidence_score.desc())
        elif order_by == "time":
            stmt = stmt.order_by(HighlightModel.start_time_seconds.asc())
        elif order_by == "created":
            stmt = stmt.order_by(HighlightModel.created_at.desc())
        else:
            stmt = stmt.order_by(HighlightModel.confidence_score.desc())

        # Get total count
        count_stmt = (
            select(func.count())
            .select_from(HighlightModel)
            .where(HighlightModel.stream_id == stream_id)
        )
        count_result = await self.session.execute(count_stmt)
        total_count = count_result.scalar() or 0

        # Apply pagination
        stmt = stmt.limit(limit).offset(offset)

        # Execute query
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        # Convert to domain entities
        highlights = self.mapper.to_domain_list(models)

        # Calculate pagination metadata
        total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
        current_page = (offset // limit) + 1 if limit > 0 else 1
        has_next = offset + limit < total_count
        has_prev = offset > 0

        return {
            "highlights": highlights,
            "pagination": {
                "total_count": total_count,
                "total_pages": total_pages,
                "current_page": current_page,
                "page_size": limit,
                "has_next": has_next,
                "has_prev": has_prev,
            },
        }
