"""Mapper for webhook event entities."""

from typing import Optional

from src.domain.entities.webhook_event import (
    WebhookEvent as DomainWebhookEvent,
    WebhookEventStatus,
    WebhookEventType
)
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.webhook_event import WebhookEvent as PersistenceWebhookEvent
from src.infrastructure.persistence.mappers.base import Mapper


class WebhookEventMapper(Mapper[DomainWebhookEvent, PersistenceWebhookEvent]):
    """Maps between domain and persistence webhook event models."""
    
    def to_domain(self, model: PersistenceWebhookEvent) -> DomainWebhookEvent:
        """Convert persistence model to domain entity.
        
        Args:
            model: Persistence webhook event model
            
        Returns:
            Domain webhook event entity
        """
        return DomainWebhookEvent(
            id=model.id,
            event_id=model.event_id,
            event_type=model.event_type,
            platform=model.platform,
            status=model.status,
            payload=model.payload or {},
            stream_id=model.stream_id,
            user_id=model.user_id,
            processed_at=Timestamp(model.processed_at) if model.processed_at else None,
            error_message=model.error_message,
            retry_count=model.retry_count,
            received_at=Timestamp(model.received_at),
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at)
        )
    
    def to_persistence(self, entity: DomainWebhookEvent) -> PersistenceWebhookEvent:
        """Convert domain entity to persistence model.
        
        Args:
            entity: Domain webhook event entity
            
        Returns:
            Persistence webhook event model
        """
        return PersistenceWebhookEvent(
            id=entity.id,
            event_id=entity.event_id,
            event_type=entity.event_type,
            platform=entity.platform,
            status=entity.status,
            payload=entity.payload,
            stream_id=entity.stream_id,
            user_id=entity.user_id,
            processed_at=entity.processed_at.value if entity.processed_at else None,
            error_message=entity.error_message,
            retry_count=entity.retry_count,
            received_at=entity.received_at.value,
            created_at=entity.created_at.value,
            updated_at=entity.updated_at.value
        )