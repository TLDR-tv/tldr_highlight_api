"""API routers for the TL;DR Highlight API.

This package contains all API route handlers organized by functionality.
Each router handles a specific domain of the API (authentication, streams, etc.).
"""

from src.api.routers.auth import router as auth_router
from src.api.routers.content import router as content_router
from src.api.routers.health import router as health_router
from src.api.routers.highlights import router as highlights_router
from src.api.routers.streams import router as streams_router
from src.api.routers.webhooks import router as webhooks_router
from src.api.routers.webhook_receiver import router as webhook_receiver_router

__all__ = [
    "auth_router",
    "content_router",
    "health_router",
    "highlights_router",
    "streams_router",
    "webhooks_router",
    "webhook_receiver_router",
]
