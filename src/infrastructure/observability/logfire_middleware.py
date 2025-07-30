"""Logfire middleware for FastAPI request tracking and observability.

This middleware provides comprehensive request/response logging, performance
tracking, and error monitoring through Pydantic Logfire.
"""

import time
import uuid
from typing import Callable, Optional, Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import logfire
from src.infrastructure.config import settings


class LogfireMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request tracking with Logfire."""

    def __init__(
        self,
        app: ASGIApp,
        service_name: Optional[str] = None,
        capture_request_headers: bool = True,
        capture_response_headers: bool = True,
        capture_request_body: bool = False,
        capture_response_body: bool = False,
        excluded_paths: Optional[set[str]] = None,
    ):
        """Initialize the Logfire middleware.

        Args:
            app: FastAPI application
            service_name: Optional service name override
            capture_request_headers: Whether to capture request headers
            capture_response_headers: Whether to capture response headers
            capture_request_body: Whether to capture request body (careful with sensitive data)
            capture_response_body: Whether to capture response body (careful with large responses)
            excluded_paths: Set of paths to exclude from tracking
        """
        super().__init__(app)
        self.service_name = service_name or settings.logfire_service_name
        self.capture_request_headers = capture_request_headers
        self.capture_response_headers = capture_response_headers
        self.capture_request_body = capture_request_body
        self.capture_response_body = capture_response_body
        self.excluded_paths = excluded_paths or {
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and track with Logfire.

        Args:
            request: The incoming request
            call_next: Next middleware/handler

        Returns:
            Response: The response
        """
        # Skip excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Generate request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract request attributes
        attributes = self._extract_request_attributes(request)

        # Start timing
        start_time = time.time()

        # Create span for the request
        with logfire.span(
            f"http.request {request.method} {request.url.path}",
            _span_type="span",
            **attributes,
        ) as span:
            try:
                # Add request ID to span
                span.set_attribute("http.request_id", request_id)

                # Extract authentication context
                self._add_auth_context(request, span)

                # Process request
                response = await call_next(request)

                # Calculate duration
                duration = time.time() - start_time

                # Add response attributes
                self._add_response_attributes(response, duration, span)

                # Add custom response headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = f"{duration * 1000:.2f}"

                # Log based on status code
                if response.status_code >= 500:
                    logfire.error(
                        f"Request failed with server error: {response.status_code}",
                        status_code=response.status_code,
                        duration_ms=duration * 1000,
                    )
                elif response.status_code >= 400:
                    logfire.warning(
                        f"Request failed with client error: {response.status_code}",
                        status_code=response.status_code,
                        duration_ms=duration * 1000,
                    )
                else:
                    logfire.info(
                        f"Request completed successfully: {response.status_code}",
                        status_code=response.status_code,
                        duration_ms=duration * 1000,
                    )

                # Track metrics
                self._track_metrics(request, response, duration)

                return response

            except Exception as e:
                duration = time.time() - start_time

                # Log exception
                logfire.error(
                    f"Request failed with exception: {type(e).__name__}",
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    duration_ms=duration * 1000,
                    exc_info=True,
                )

                # Track error metrics
                self._track_error_metrics(request, e, duration)

                # Re-raise the exception
                raise

    def _extract_request_attributes(self, request: Request) -> Dict[str, Any]:
        """Extract attributes from the request.

        Args:
            request: The request object

        Returns:
            Dict of attributes
        """
        attributes = {
            "http.method": request.method,
            "http.url": str(request.url),
            "http.path": request.url.path,
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname,
            "http.user_agent": request.headers.get("user-agent", ""),
            "http.client_ip": self._get_client_ip(request),
            "service.name": self.service_name,
        }

        # Add query parameters
        if request.query_params:
            attributes["http.query_params"] = dict(request.query_params)

        # Add headers if enabled
        if self.capture_request_headers:
            # Filter sensitive headers
            headers = self._filter_sensitive_headers(dict(request.headers))
            attributes["http.request_headers"] = headers

        # Add path parameters
        if hasattr(request, "path_params") and request.path_params:
            attributes["http.path_params"] = request.path_params

        return attributes

    def _add_auth_context(self, request: Request, span: Any) -> None:
        """Add authentication context to the span.

        Args:
            request: The request object
            span: The current span
        """
        # Check for API key
        api_key = request.headers.get("x-api-key")
        if api_key:
            # Only log prefix for security
            span.set_attribute("auth.api_key_prefix", api_key[:8] + "...")
            span.set_attribute("auth.method", "api_key")

        # Check for JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            span.set_attribute("auth.method", "jwt")

        # Add user context if available in request state
        if hasattr(request.state, "user"):
            user = request.state.user
            if hasattr(user, "organization_id"):
                span.set_attribute("user.organization_id", str(user.organization_id))
            if hasattr(user, "id"):
                span.set_attribute("user.id", str(user.id))

    def _add_response_attributes(
        self, response: Response, duration: float, span: Any
    ) -> None:
        """Add response attributes to the span.

        Args:
            response: The response object
            duration: Request duration in seconds
            span: The current span
        """
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("http.duration_ms", duration * 1000)

        # Add response headers if enabled
        if self.capture_response_headers:
            headers = self._filter_sensitive_headers(dict(response.headers))
            span.set_attribute("http.response_headers", headers)

        # Add response size if available
        if hasattr(response, "body") and response.body:
            span.set_attribute("http.response_size_bytes", len(response.body))

    def _track_metrics(
        self, request: Request, response: Response, duration: float
    ) -> None:
        """Track request metrics.

        Args:
            request: The request object
            response: The response object
            duration: Request duration in seconds
        """
        # Track request count
        logfire.info(
            "metric.http_requests_total",
            value=1,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            metric_type="counter",
        )

        # Track request duration
        logfire.info(
            "metric.http_request_duration_seconds",
            value=duration,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            metric_type="histogram",
        )

        # Track response size if available
        if hasattr(response, "body") and response.body:
            logfire.info(
                "metric.http_response_size_bytes",
                value=len(response.body),
                method=request.method,
                path=request.url.path,
                metric_type="histogram",
            )

    def _track_error_metrics(
        self, request: Request, exception: Exception, duration: float
    ) -> None:
        """Track error metrics.

        Args:
            request: The request object
            exception: The exception that occurred
            duration: Request duration in seconds
        """
        logfire.info(
            "metric.http_errors_total",
            value=1,
            method=request.method,
            path=request.url.path,
            exception_type=type(exception).__name__,
            metric_type="counter",
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request.

        Args:
            request: The request object

        Returns:
            Client IP address
        """
        # Check forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client
        return request.client.host if request.client else "unknown"

    def _filter_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter out sensitive headers.

        Args:
            headers: Original headers

        Returns:
            Filtered headers
        """
        sensitive_headers = {
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie",
            "x-csrf-token",
        }

        filtered = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                # Mask sensitive values
                if key.lower() == "x-api-key" and len(value) > 8:
                    filtered[key] = value[:8] + "..."
                else:
                    filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value

        return filtered
