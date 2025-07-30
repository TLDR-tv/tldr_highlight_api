"""Database models for the TL;DR Highlight API.

This package contains all SQLAlchemy models for the enterprise
highlight extraction API service.
"""

from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.base import Base, TimestampMixin
from src.infrastructure.persistence.models.batch import Batch, BatchStatus
from src.infrastructure.persistence.models.dimension_set import DimensionSet
from src.infrastructure.persistence.models.highlight import Highlight
from src.infrastructure.persistence.models.highlight_type_registry import (
    HighlightTypeRegistry,
)
from src.infrastructure.persistence.models.organization import Organization, PlanType
from src.infrastructure.persistence.models.stream import (
    Stream,
    StreamPlatform,
    StreamStatus,
)
from src.infrastructure.persistence.models.usage_record import (
    UsageRecord,
    UsageRecordType,
)
from src.infrastructure.persistence.models.user import User
from src.infrastructure.persistence.models.webhook import (
    Webhook,
    WebhookEvent as WebhookEventType,
)
from src.infrastructure.persistence.models.webhook_event import WebhookEvent

__all__ = [
    # Base classes
    "Base",
    "TimestampMixin",
    # Models
    "User",
    "APIKey",
    "Organization",
    "Stream",
    "Batch",
    "DimensionSet",
    "Highlight",
    "HighlightTypeRegistry",
    "Webhook",
    "UsageRecord",
    "WebhookEvent",
    # Enums
    "PlanType",
    "StreamStatus",
    "StreamPlatform",
    "BatchStatus",
    "WebhookEventType",
    "UsageRecordType",
]
