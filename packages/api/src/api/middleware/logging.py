"""Logging middleware."""

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from structlog import get_logger

logger = get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next):
        """Log requests and responses."""
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Start timer
        start_time = time.time()

        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None,
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=round(duration, 3),
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
