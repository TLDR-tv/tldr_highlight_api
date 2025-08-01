"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from .routes import auth, streams, highlights, organizations, webhooks, health
from .middleware.logging import LoggingMiddleware
from ..infrastructure.config import get_settings

logger = get_logger()
settings = get_settings()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="TLDR Highlight API",
        description="Enterprise B2B API for AI-powered highlight extraction",
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(LoggingMiddleware)
    
    # Routes
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(organizations.router, prefix="/api/v1/organizations", tags=["organizations"])
    app.include_router(streams.router, prefix="/api/v1/streams", tags=["streams"])
    app.include_router(highlights.router, prefix="/api/v1/highlights", tags=["highlights"])
    app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["webhooks"])
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "TLDR Highlight API",
            "version": "1.0.0",
            "docs": "/docs" if settings.environment != "production" else None
        }
    
    return app


app = create_app()