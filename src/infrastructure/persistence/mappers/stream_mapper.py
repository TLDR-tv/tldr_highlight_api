"""Stream mapper for domain entity to persistence model conversion."""

import json
from typing import Optional

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.stream import Stream as DomainStream, StreamStatus, StreamPlatform
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.processing_options import ProcessingOptions
from src.infrastructure.persistence.models.stream import Stream as PersistenceStream


class StreamMapper(Mapper[DomainStream, PersistenceStream]):
    """Maps between Stream domain entity and persistence model."""
    
    def to_domain(self, model: PersistenceStream) -> DomainStream:
        """Convert Stream persistence model to domain entity."""
        # Parse processing options from JSON
        processing_opts_data = json.loads(model.processing_options) if model.processing_options else {}
        processing_options = ProcessingOptions(**processing_opts_data)
        
        # Parse platform data
        platform_data = json.loads(model.platform_data) if model.platform_data else {}
        
        return DomainStream(
            id=model.id,
            url=Url(model.url),
            platform=StreamPlatform(model.platform),
            status=StreamStatus(model.status),
            user_id=model.user_id,
            processing_options=processing_options,
            title=model.title,
            channel_name=model.channel_name,
            game_category=model.game_category,
            language=model.language,
            viewer_count=model.viewer_count,
            duration=Duration(model.duration_seconds) if model.duration_seconds else None,
            started_at=Timestamp(model.started_at) if model.started_at else None,
            completed_at=Timestamp(model.completed_at) if model.completed_at else None,
            error_message=model.error_message,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at),
            highlight_ids=[h.id for h in model.highlights] if model.highlights else [],
            platform_data=platform_data
        )
    
    def to_persistence(self, entity: DomainStream) -> PersistenceStream:
        """Convert Stream domain entity to persistence model."""
        model = PersistenceStream()
        
        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id
        
        model.url = entity.url.value
        model.platform = entity.platform.value
        model.status = entity.status.value
        model.user_id = entity.user_id
        
        # Serialize processing options
        model.processing_options = json.dumps(entity.processing_options.to_dict())
        
        # Set optional attributes
        model.title = entity.title
        model.channel_name = entity.channel_name
        model.game_category = entity.game_category
        model.language = entity.language
        model.viewer_count = entity.viewer_count
        model.duration_seconds = float(entity.duration) if entity.duration else None
        
        # Set timestamps
        model.started_at = entity.started_at.value if entity.started_at else None
        model.completed_at = entity.completed_at.value if entity.completed_at else None
        model.error_message = entity.error_message
        
        # Serialize platform data
        model.platform_data = json.dumps(entity.platform_data) if entity.platform_data else None
        
        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value
        
        return model