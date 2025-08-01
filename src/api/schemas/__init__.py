"""API schemas for request/response validation."""

from .common import PaginatedResponse, PaginationParams
from .highlights import (
    HighlightFilters,
    HighlightListResponse,
    HighlightResponse,
    HighlightSearch,
    HighlightUpdate,
)
from .streams import (
    StreamCreate,
    StreamListResponse,
    StreamOptions,
    StreamResponse,
    StreamUpdate,
)
from .webhooks import (
    WebhookCreate,
    WebhookListResponse,
    WebhookResponse,
    WebhookTest,
    WebhookUpdate,
)
from .webhook_models import (
    WebhookEventType,
    BaseWebhookEvent,
    StreamMetadata,
    StreamStartedWebhookEvent,
    WebhookResponse as WebhookProcessingResponse,
    WebhookVerificationHeaders,
)

__all__ = [
    # Stream schemas
    "StreamCreate",
    "StreamUpdate",
    "StreamResponse",
    "StreamListResponse",
    "StreamOptions",
    # Highlight schemas
    "HighlightResponse",
    "HighlightUpdate",
    "HighlightListResponse",
    "HighlightSearch",
    "HighlightFilters",
    # Webhook schemas
    "WebhookCreate",
    "WebhookUpdate",
    "WebhookResponse",
    "WebhookListResponse",
    "WebhookTest",
    # Webhook processing schemas
    "WebhookEventType",
    "BaseWebhookEvent",
    "StreamMetadata",
    "StreamStartedWebhookEvent",
    "WebhookProcessingResponse",
    "WebhookVerificationHeaders",
    # Common schemas
    "PaginationParams",
    "PaginatedResponse",
]
