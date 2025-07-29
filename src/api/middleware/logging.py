"""Request logging middleware for the TL;DR Highlight API.

This middleware logs detailed information about HTTP requests and responses
for monitoring, debugging, and analytics purposes.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses with detailed information."""

    def __init__(
        self, app, log_request_body: bool = False, log_response_body: bool = False
    ):
        """Initialize the request logging middleware.

        Args:
            app: FastAPI application instance
            log_request_body: Whether to log request body (be careful with sensitive data)
            log_response_body: Whether to log response body (be careful with large responses)
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details.

        Args:
            request: HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            Response: HTTP response
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        api_key = request.headers.get("x-api-key", "")

        # Log request start
        start_time = time.time()

        # Prepare request logging data
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "headers": dict(request.headers),
            "api_key_prefix": api_key[:8] + "..." if api_key else None,
        }

        # Log request body if enabled (be careful with sensitive data)
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Only log if body is not too large and doesn't contain sensitive data
                    if len(body) < 10000:  # 10KB limit
                        request_data["body_size"] = len(body)
                        # Don't log actual body content for security - just indicate presence
                        request_data["has_body"] = True
                    else:
                        request_data["body_size"] = len(body)
                        request_data["body_truncated"] = True
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")

        logger.info(
            f"Request started: {request.method} {request.url.path}", extra=request_data
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Prepare response logging data
            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "response_headers": dict(response.headers),
            }

            # Log response body size if available
            if hasattr(response, "body") and response.body:
                response_data["response_size"] = len(response.body)

            # Add custom headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))

            # Log response
            if response.status_code >= 400:
                logger.error(
                    f"Request failed: {request.method} {request.url.path} - {response.status_code}",
                    extra={**request_data, **response_data},
                )
            else:
                logger.info(
                    f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                    extra={**request_data, **response_data},
                )

            return response

        except Exception as e:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time

            # Log exception
            error_data = {
                "request_id": request_id,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "process_time_ms": round(process_time * 1000, 2),
            }

            logger.error(
                f"Request exception: {request.method} {request.url.path}",
                extra={**request_data, **error_data},
                exc_info=True,
            )

            # Re-raise the exception
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers.

        Args:
            request: HTTP request

        Returns:
            str: Client IP address
        """
        # Check for forwarded headers (when behind proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    def _should_skip_logging(self, request: Request) -> bool:
        """Determine if request should be skipped from logging.

        Args:
            request: HTTP request

        Returns:
            bool: True if logging should be skipped
        """
        # Skip health check endpoints to reduce noise
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return True

        # Skip static files
        if request.url.path.startswith("/static/"):
            return True

        # Skip metrics endpoints
        if request.url.path.startswith("/metrics"):
            return True

        return False
