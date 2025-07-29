"""Error handling middleware for the TL;DR Highlight API.

This middleware provides centralized error handling and logging
for unhandled exceptions that occur during request processing.
"""

import logging
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.exceptions import create_error_response
from src.core.config import settings

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle unhandled exceptions and provide consistent error responses."""

    def __init__(self, app):
        """Initialize the error handling middleware.

        Args:
            app: FastAPI application instance
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle request and catch any unhandled exceptions.

        Args:
            request: HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            Response: HTTP response or error response
        """
        try:
            response = await call_next(request)
            return response

        except Exception as exc:
            return await self._handle_exception(request, exc)

    async def _handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle unhandled exceptions and create error response.

        Args:
            request: HTTP request
            exc: Exception that occurred

        Returns:
            JSONResponse: Standardized error response
        """
        request_id = getattr(request.state, "request_id", None)

        # Log the exception with context
        logger.error(
            f"Unhandled exception in middleware: {type(exc).__name__}",
            extra={
                "request_id": request_id,
                "path": str(request.url),
                "method": request.method,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
            },
            exc_info=True,
        )

        # Determine appropriate status code and message
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_code = "INTERNAL_SERVER_ERROR"

        # Don't expose internal error details in production
        if settings.is_production:
            message = "An internal server error occurred"
            details = None
        else:
            message = f"{type(exc).__name__}: {str(exc)}"
            details = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }

        # Create error response
        return JSONResponse(
            status_code=status_code,
            content=create_error_response(
                status_code=status_code,
                message=message,
                error_code=error_code,
                details=details,
                request_id=request_id,
            ),
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request.

        Args:
            request: HTTP request

        Returns:
            str: Client IP address
        """
        # Check forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"
