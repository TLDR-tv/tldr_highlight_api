"""Batch mapper for domain entity to persistence model conversion."""

import json
from typing import Optional, List

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.batch import Batch as DomainBatch, BatchStatus, BatchItem
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.processing_options import ProcessingOptions
from src.infrastructure.persistence.models.batch import Batch as PersistenceBatch


class BatchMapper(Mapper[DomainBatch, PersistenceBatch]):
    """Maps between Batch domain entity and persistence model."""
    
    def to_domain(self, model: PersistenceBatch) -> DomainBatch:
        """Convert Batch persistence model to domain entity."""
        # Parse processing options
        processing_opts_data = json.loads(model.processing_options) if model.processing_options else {}
        processing_options = ProcessingOptions(**processing_opts_data) if processing_opts_data else ProcessingOptions()
        
        # Parse batch items
        items = []
        if model.items_data:
            items_data = json.loads(model.items_data)
            for item_data in items_data:
                item = BatchItem(
                    url=Url(item_data['url']),
                    item_id=item_data['item_id'],
                    status=BatchStatus(item_data.get('status', 'pending')),
                    stream_id=item_data.get('stream_id'),
                    error_message=item_data.get('error_message'),
                    metadata=item_data.get('metadata', {})
                )
                items.append(item)
        
        return DomainBatch(
            id=model.id,
            name=model.name,
            user_id=model.user_id,
            status=BatchStatus(model.status),
            processing_options=processing_options,
            items=items,
            started_at=Timestamp(model.started_at) if model.started_at else None,
            completed_at=Timestamp(model.completed_at) if model.completed_at else None,
            total_items=model.total_items,
            processed_items=model.processed_items,
            successful_items=model.successful_items,
            failed_items=model.failed_items,
            webhook_url=Url(model.webhook_url) if model.webhook_url else None,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at)
        )
    
    def to_persistence(self, entity: DomainBatch) -> PersistenceBatch:
        """Convert Batch domain entity to persistence model."""
        model = PersistenceBatch()
        
        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id
        
        model.name = entity.name
        model.user_id = entity.user_id
        model.status = entity.status.value
        
        # Serialize processing options
        model.processing_options = json.dumps(entity.processing_options.to_dict())
        
        # Serialize batch items
        items_data = []
        for item in entity.items:
            item_data = {
                'url': item.url.value,
                'item_id': item.item_id,
                'status': item.status.value,
                'stream_id': item.stream_id,
                'error_message': item.error_message,
                'metadata': item.metadata
            }
            items_data.append(item_data)
        model.items_data = json.dumps(items_data)
        
        # Set processing timestamps
        model.started_at = entity.started_at.value if entity.started_at else None
        model.completed_at = entity.completed_at.value if entity.completed_at else None
        
        # Set counters
        model.total_items = entity.total_items
        model.processed_items = entity.processed_items
        model.successful_items = entity.successful_items
        model.failed_items = entity.failed_items
        
        # Set webhook URL
        model.webhook_url = entity.webhook_url.value if entity.webhook_url else None
        
        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value
        
        return model