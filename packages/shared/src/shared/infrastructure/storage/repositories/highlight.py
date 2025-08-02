"""Highlight repository implementation."""

from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import json

from ....domain.models.highlight import Highlight, DimensionScore
from ...database.models import HighlightModel


class HighlightRepository:
    """SQLAlchemy implementation of highlight repository."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def add(self, entity: Highlight) -> Highlight:
        """Add highlight to repository."""
        # Convert dimension scores to JSON format
        dimension_scores_json = {
            ds.name: {"score": ds.score, "confidence": ds.confidence}
            for ds in entity.dimension_scores
        }

        model = HighlightModel(
            id=entity.id,
            stream_id=entity.stream_id,
            organization_id=entity.organization_id,
            start_time=entity.start_time,
            end_time=entity.end_time,
            duration=entity.duration,
            title=entity.title,
            description=entity.description,
            tags=entity.tags,
            dimension_scores=dimension_scores_json,
            overall_score=entity.overall_score,
            clip_path=entity.clip_path,
            thumbnail_path=entity.thumbnail_path,
            transcript=entity.transcript,
            wake_word_triggered=entity.wake_word_triggered,
            wake_word_detected=entity.wake_word_detected,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)

        return self._to_entity(model)

    # Alias for compatibility with tests
    async def create(self, entity: Highlight) -> Highlight:
        """Create new highlight (alias for add)."""
        return await self.add(entity)

    async def get(self, id: UUID) -> Optional[Highlight]:
        """Get highlight by ID."""
        model = await self.session.get(HighlightModel, id)
        return self._to_entity(model) if model else None

    async def list_by_stream(self, stream_id: UUID) -> list[Highlight]:
        """List highlights for a stream."""
        result = await self.session.execute(
            select(HighlightModel)
            .where(HighlightModel.stream_id == stream_id)
            .order_by(HighlightModel.start_time)
        )
        models = result.scalars().all()
        return [self._to_entity(model) for model in models]

    async def list_by_organization(
        self, org_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Highlight]:
        """List highlights for an organization."""
        result = await self.session.execute(
            select(HighlightModel)
            .where(HighlightModel.organization_id == org_id)
            .order_by(HighlightModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        models = result.scalars().all()
        return [self._to_entity(model) for model in models]

    async def list_by_wake_word(self, org_id: UUID, wake_word: str) -> list[Highlight]:
        """List highlights triggered by a specific wake word."""
        return []

    async def update(self, entity: Highlight) -> Highlight:
        """Update existing highlight."""
        return entity

    async def delete(self, id: UUID) -> None:
        """Delete highlight by ID."""
        pass

    async def list(self, **filters) -> list[Highlight]:
        """List highlights with optional filters."""
        query = select(HighlightModel)

        # Apply filters
        conditions = []
        if "organization_id" in filters:
            conditions.append(
                HighlightModel.organization_id == filters["organization_id"]
            )
        if "stream_id" in filters:
            conditions.append(HighlightModel.stream_id == filters["stream_id"])
        if "wake_word_triggered" in filters:
            conditions.append(
                HighlightModel.wake_word_triggered == filters["wake_word_triggered"]
            )
        if "min_score" in filters:
            conditions.append(HighlightModel.overall_score >= filters["min_score"])

        if conditions:
            query = query.where(and_(*conditions))

        # Apply ordering
        order_by = filters.get("order_by", "created_at")
        if order_by == "score":
            query = query.order_by(HighlightModel.overall_score.desc())
        else:
            query = query.order_by(HighlightModel.created_at.desc())

        # Apply pagination
        limit = filters.get("limit", 100)
        offset = filters.get("offset", 0)
        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._to_entity(model) for model in models]

    def _to_entity(self, model: HighlightModel) -> Highlight:
        """Convert model to entity."""
        # Convert dimension scores from JSON
        dimension_scores = []
        if model.dimension_scores:
            for name, data in model.dimension_scores.items():
                dimension_scores.append(
                    DimensionScore(
                        name=name, score=data["score"], confidence=data["confidence"]
                    )
                )

        return Highlight(
            id=model.id,
            stream_id=model.stream_id,
            organization_id=model.organization_id,
            start_time=model.start_time,
            end_time=model.end_time,
            duration=model.duration,
            title=model.title,
            description=model.description,
            tags=model.tags or [],
            dimension_scores=dimension_scores,
            overall_score=model.overall_score,
            clip_path=model.clip_path,
            thumbnail_path=model.thumbnail_path,
            transcript=model.transcript,
            wake_word_triggered=model.wake_word_triggered,
            wake_word_detected=model.wake_word_detected,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
