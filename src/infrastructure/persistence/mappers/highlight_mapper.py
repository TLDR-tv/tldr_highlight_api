"""Highlight mapper for domain entity to persistence model conversion."""

import json
from typing import Optional

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.highlight import Highlight as DomainHighlight, HighlightType
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.url import Url
from src.infrastructure.persistence.models.highlight import Highlight as PersistenceHighlight


class HighlightMapper(Mapper[DomainHighlight, PersistenceHighlight]):
    """Maps between Highlight domain entity and persistence model."""
    
    def to_domain(self, model: PersistenceHighlight) -> DomainHighlight:
        """Convert Highlight persistence model to domain entity."""
        # Parse analysis data
        video_analysis = json.loads(model.video_analysis) if model.video_analysis else {}
        audio_analysis = json.loads(model.audio_analysis) if model.audio_analysis else {}
        chat_analysis = json.loads(model.chat_analysis) if model.chat_analysis else {}
        
        # Parse tags
        tags = json.loads(model.tags) if model.tags else []
        
        return DomainHighlight(
            id=model.id,
            stream_id=model.stream_id,
            start_time=Duration(model.start_time_seconds),
            end_time=Duration(model.end_time_seconds),
            confidence_score=ConfidenceScore(model.confidence_score),
            highlight_type=HighlightType(model.highlight_type),
            title=model.title,
            description=model.description,
            thumbnail_url=Url(model.thumbnail_url) if model.thumbnail_url else None,
            clip_url=Url(model.clip_url) if model.clip_url else None,
            tags=tags,
            sentiment_score=model.sentiment_score,
            viewer_engagement=model.viewer_engagement,
            video_analysis=video_analysis,
            audio_analysis=audio_analysis,
            chat_analysis=chat_analysis,
            processed_by=model.processed_by,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at)
        )
    
    def to_persistence(self, entity: DomainHighlight) -> PersistenceHighlight:
        """Convert Highlight domain entity to persistence model."""
        model = PersistenceHighlight()
        
        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id
        
        model.stream_id = entity.stream_id
        model.start_time_seconds = float(entity.start_time)
        model.end_time_seconds = float(entity.end_time)
        model.confidence_score = float(entity.confidence_score)
        model.highlight_type = entity.highlight_type.value
        
        # Set content attributes
        model.title = entity.title
        model.description = entity.description
        model.thumbnail_url = entity.thumbnail_url.value if entity.thumbnail_url else None
        model.clip_url = entity.clip_url.value if entity.clip_url else None
        
        # Serialize tags
        model.tags = json.dumps(entity.tags)
        
        # Set analysis scores
        model.sentiment_score = entity.sentiment_score
        model.viewer_engagement = entity.viewer_engagement
        
        # Serialize analysis data
        model.video_analysis = json.dumps(entity.video_analysis) if entity.video_analysis else None
        model.audio_analysis = json.dumps(entity.audio_analysis) if entity.audio_analysis else None
        model.chat_analysis = json.dumps(entity.chat_analysis) if entity.chat_analysis else None
        
        model.processed_by = entity.processed_by
        
        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value
        
        return model