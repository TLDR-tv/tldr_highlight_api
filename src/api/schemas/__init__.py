"""API schemas for request/response validation."""

from .batches import (
    BatchCreate,
    BatchListResponse,
    BatchOptions,
    BatchResponse,
    BatchUpdate,
    VideoInput,
)
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
    WebhookPlatform,
    BaseWebhookEvent,
    StreamMetadata,
    StreamStartedWebhookEvent,
    HundredMSWebhookPayload,
    TwitchWebhookPayload,
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
    # Batch schemas
    "BatchCreate",
    "BatchUpdate",
    "BatchResponse",
    "BatchListResponse",
    "BatchOptions",
    "VideoInput",
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
    "WebhookPlatform",
    "BaseWebhookEvent",
    "StreamMetadata",
    "StreamStartedWebhookEvent",
    "HundredMSWebhookPayload",
    "TwitchWebhookPayload",
    "WebhookProcessingResponse",
    "WebhookVerificationHeaders",
    # Common schemas
    "PaginationParams",
    "PaginatedResponse",
]