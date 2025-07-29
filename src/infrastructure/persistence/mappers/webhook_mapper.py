"""Webhook mapper for domain entity to persistence model conversion."""

import json
from typing import Optional

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.webhook import (
    Webhook as DomainWebhook, 
    WebhookEvent,
    WebhookStatus,
    WebhookDelivery
)
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.webhook import Webhook as PersistenceWebhook


class WebhookMapper(Mapper[DomainWebhook, PersistenceWebhook]):
    """Maps between Webhook domain entity and persistence model."""
    
    def to_domain(self, model: PersistenceWebhook) -> DomainWebhook:
        """Convert Webhook persistence model to domain entity."""
        # Parse events and custom headers
        events = [WebhookEvent(e) for e in json.loads(model.events)] if model.events else []
        custom_headers = json.loads(model.custom_headers) if model.custom_headers else {}
        
        # Parse last delivery if present
        last_delivery = None
        if model.last_delivery_data:
            delivery_data = json.loads(model.last_delivery_data)
            last_delivery = WebhookDelivery(
                delivered_at=Timestamp.from_iso_string(delivery_data['delivered_at']),
                status_code=delivery_data['status_code'],
                response_time_ms=delivery_data['response_time_ms'],
                error_message=delivery_data.get('error_message')
            )
        
        return DomainWebhook(
            id=model.id,
            url=Url(model.url),
            user_id=model.user_id,
            events=events,
            secret=model.secret,
            description=model.description,
            status=WebhookStatus(model.status),
            last_delivery=last_delivery,
            consecutive_failures=model.consecutive_failures,
            total_deliveries=model.total_deliveries,
            successful_deliveries=model.successful_deliveries,
            custom_headers=custom_headers,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at)
        )
    
    def to_persistence(self, entity: DomainWebhook) -> PersistenceWebhook:
        """Convert Webhook domain entity to persistence model."""
        model = PersistenceWebhook()
        
        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id
        
        model.url = entity.url.value
        model.user_id = entity.user_id
        model.secret = entity.secret
        model.description = entity.description
        model.status = entity.status.value
        
        # Serialize events
        model.events = json.dumps([event.value for event in entity.events])
        
        # Serialize custom headers
        model.custom_headers = json.dumps(entity.custom_headers) if entity.custom_headers else '{}'
        
        # Set delivery statistics
        model.consecutive_failures = entity.consecutive_failures
        model.total_deliveries = entity.total_deliveries
        model.successful_deliveries = entity.successful_deliveries
        
        # Serialize last delivery if present
        if entity.last_delivery:
            delivery_data = {
                'delivered_at': entity.last_delivery.delivered_at.iso_string,
                'status_code': entity.last_delivery.status_code,
                'response_time_ms': entity.last_delivery.response_time_ms,
                'error_message': entity.last_delivery.error_message
            }
            model.last_delivery_data = json.dumps(delivery_data)
        
        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value
        
        return model