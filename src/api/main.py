"""Main FastAPI application for TL;DR Highlight API.

This module contains the FastAPI application setup with middleware,
exception handlers, and router configuration for the enterprise
highlight extraction service.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from src.api.exceptions import setup_exception_handlers
from src.api.middleware import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.routers import (
    auth_router,
    batches_router,
    health_router,
    highlights_router,
    streams_router,
    webhooks_router,
    webhook_receiver_router,
)
from src.infrastructure.cache import get_redis_cache
from src.infrastructure.config import settings
from src.infrastructure.database import close_db, init_db
from src.infrastructure.observability import (
    configure_logfire,
    LogfireMiddleware,
    metrics,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events.

    Handles database initialization, Redis connection setup, and cleanup.
    """
    # Startup
    try:
        logger.info("Starting TL;DR Highlight API...")
        
        # Initialize Logfire observability
        configure_logfire(app)
        logger.info("Logfire observability configured successfully")
        
        # Start metrics background collection
        await metrics.start_background_collection()
        logger.info("Metrics collection started")

        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")

        # Connect to Redis
        cache = await get_redis_cache()
        logger.info("Redis connected successfully")

        logger.info("Application startup completed")

    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise

    yield

    # Shutdown
    try:
        logger.info("Shutting down TL;DR Highlight API...")
        
        # Stop metrics background collection
        await metrics.stop_background_collection()
        logger.info("Metrics collection stopped")

        # Close Redis connection
        cache = await get_redis_cache()
        await cache.disconnect()
        logger.info("Redis disconnected successfully")

        # Close database connections
        await close_db()
        logger.info("Database connections closed successfully")

        logger.info("Application shutdown completed")

    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise B2B API for AI-powered highlight extraction from livestreams and video content",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "health",
                "description": "Health check and system status endpoints",
            },
            {"name": "auth", "description": "Authentication and API key management"},
            {"name": "streams", "description": "Livestream processing and management"},
            {"name": "batches", "description": "Batch video processing operations"},
            {"name": "highlights", "description": "Highlight access and management"},
            {"name": "webhooks", "description": "Webhook configuration and management"},
        ],
    )

    # Configure OpenAPI security schemes
    if app.openapi_schema is None:
        from fastapi.openapi.utils import get_openapi

        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema

            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
            )

            # Add API key security scheme
            openapi_schema["components"]["securitySchemes"] = {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": settings.api_key_header,
                    "description": "API key for authentication",
                }
            }

            # Apply security to all endpoints except health
            for path, path_item in openapi_schema["paths"].items():
                if not path.startswith("/health"):
                    for method, operation in path_item.items():
                        if method.lower() in ["get", "post", "put", "delete", "patch"]:
                            operation.setdefault("security", []).append(
                                {"ApiKeyAuth": []}
                            )

            app.openapi_schema = openapi_schema
            return app.openapi_schema

        app.openapi = custom_openapi

    # Add middleware (order matters - first added is outermost)

    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Logfire observability middleware
    app.add_middleware(
        LogfireMiddleware,
        capture_request_headers=settings.logfire_capture_headers,
        capture_request_body=settings.logfire_capture_body,
    )

    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # Error handling middleware
    app.add_middleware(ErrorHandlingMiddleware)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Trusted host middleware
    if settings.allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Include routers
    app.include_router(health_router, prefix="/health", tags=["health"])

    app.include_router(
        auth_router, prefix=f"{settings.api_v1_prefix}/auth", tags=["auth"]
    )

    app.include_router(
        streams_router, prefix=f"{settings.api_v1_prefix}/streams", tags=["streams"]
    )

    app.include_router(
        batches_router, prefix=f"{settings.api_v1_prefix}/batches", tags=["batches"]
    )

    app.include_router(
        highlights_router,
        prefix=f"{settings.api_v1_prefix}/highlights",
        tags=["highlights"],
    )

    app.include_router(
        webhooks_router, prefix=f"{settings.api_v1_prefix}/webhooks", tags=["webhooks"]
    )
    
    app.include_router(
        webhook_receiver_router, prefix=f"{settings.api_v1_prefix}/webhooks", tags=["webhooks"]
    )

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with basic API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "operational",
            "docs_url": "/docs" if not settings.is_production else None,
            "api_version": "v1",
        }

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
        workers=settings.worker_count,
    )
