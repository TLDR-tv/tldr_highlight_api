"""Main FastAPI application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from structlog import get_logger

from .routes import auth, users, streams, highlights, organizations, webhooks, health, tokens
from .middleware.logging import LoggingMiddleware
from .middleware.rate_limit import RateLimiter, RateLimitHeaderMiddleware, rate_limit_error_handler
from shared.infrastructure.config.config import get_settings

logger = get_logger()
settings = get_settings()

# Global rate limiter instance
rate_limiter: RateLimiter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global rate_limiter
    rate_limiter = RateLimiter(settings)
    app.state.limiter = rate_limiter.limiter  # Make limiter available to app
    logger.info("Rate limiter initialized", enabled=settings.rate_limit_enabled)
    
    yield
    
    # Shutdown
    if rate_limiter:
        await rate_limiter.close()
        logger.info("Rate limiter closed")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="TLDR Highlight API",
        description="Enterprise B2B API for AI-powered highlight extraction",
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
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
    
    # Add rate limiting if enabled
    if settings.rate_limit_enabled:
        # Add rate limit exceeded handler
        app.add_exception_handler(RateLimitExceeded, rate_limit_error_handler)

    # Routes
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
    app.include_router(
        organizations.router, prefix="/api/v1/organizations", tags=["organizations"]
    )
    app.include_router(streams.router, prefix="/api/v1/streams", tags=["streams"])
    app.include_router(
        highlights.router, prefix="/api/v1/highlights", tags=["highlights"]
    )
    app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["webhooks"])
    app.include_router(tokens.router, tags=["tokens"])

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "TLDR Highlight API",
            "version": "1.0.0",
            "docs": "/docs" if settings.environment != "production" else None,
        }

    return app


app = create_app()
