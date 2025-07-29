"""Domain entities.

Entities are domain objects with identity. Unlike value objects,
entities are defined by their ID rather than their attributes.
All entities in this module use dataclasses for immutability
and clear structure.
"""

from src.domain.entities.base import Entity
from src.domain.entities.user import User
from src.domain.entities.organization import Organization, PlanType, PlanLimits
from src.domain.entities.api_key import APIKey
from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
from src.domain.entities.highlight import Highlight, HighlightType
from src.domain.entities.webhook import Webhook, WebhookEvent, WebhookStatus, WebhookDelivery
from src.domain.entities.batch import Batch, BatchStatus, BatchItem
from src.domain.entities.usage_record import UsageRecord, UsageType

__all__ = [
    # Base
    "Entity",
    
    # User and Organization
    "User",
    "Organization",
    "PlanType",
    "PlanLimits",
    
    # API Key
    "APIKey",
    
    # Stream and Highlight
    "Stream",
    "StreamStatus", 
    "StreamPlatform",
    "Highlight",
    "HighlightType",
    
    # Webhook
    "Webhook",
    "WebhookEvent",
    "WebhookStatus",
    "WebhookDelivery",
    
    # Batch
    "Batch",
    "BatchStatus",
    "BatchItem",
    
    # Usage
    "UsageRecord",
    "UsageType",
]