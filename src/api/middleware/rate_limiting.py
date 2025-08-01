"""Rate limiting middleware for the TL;DR Highlight API.

This middleware implements rate limiting using Redis-based sliding window
algorithm to prevent API abuse and ensure fair usage.
"""

import logging
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.exceptions import create_error_response
from src.infrastructure.cache import get_rate_limiter
from src.infrastructure.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits based on API keys and IP addresses."""

    def __init__(self, app):
        """Initialize the rate limiting middleware.

        Args:
            app: FastAPI application instance
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and enforce rate limits.

        Args:
            request: HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            Response: HTTP response or rate limit error
        """
        # Skip rate limiting for certain endpoints
        if self._should_skip_rate_limiting(request):
            return await call_next(request)

        # Get rate limit key (API key or IP-based)
        rate_limit_key = self._get_rate_limit_key(request)

        # Get rate limits (may vary by API key tier)
        per_minute_limit, per_hour_limit = await self._get_rate_limits(request)

        # Check rate limits
        rate_limit_exceeded, remaining_requests = await self._check_rate_limit(
            rate_limit_key, per_minute_limit, per_hour_limit
        )

        if rate_limit_exceeded:
            return await self._create_rate_limit_response(request, remaining_requests)

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        self._add_rate_limit_headers(
            response, remaining_requests, per_minute_limit, per_hour_limit
        )

        return response

    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Determine if rate limiting should be skipped for this request.

        Args:
            request: HTTP request

        Returns:
            bool: True if rate limiting should be skipped
        """
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return True

        # Skip for root endpoint
        if request.url.path == "/":
            return True

        # Skip for OpenAPI docs (if enabled)
        if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            return True

        return False

    def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key for the request.

        Args:
            request: HTTP request

        Returns:
            str: Rate limit key
        """
        # Prefer API key for authenticated requests
        api_key = request.headers.get(settings.api_key_header.lower())
        if api_key:
            return f"api_key:{api_key[:16]}"  # Use prefix for security

        # Fallback to IP-based rate limiting
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"

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

    async def _get_rate_limits(self, request: Request) -> tuple[int, int]:
        """Get rate limits for the request (may vary by API key tier).

        Args:
            request: HTTP request

        Returns:
            tuple: (per_minute_limit, per_hour_limit)
        """
        # TODO: Implement tier-based rate limiting by looking up API key
        # For now, use default limits from settings
        api_key = request.headers.get(settings.api_key_header.lower())

        if api_key:
            # TODO: Query database for API key tier and custom limits
            # This would involve looking up the API key in the database
            # and returning tier-specific limits
            pass

        # Return default limits
        return settings.rate_limit_per_minute, settings.rate_limit_per_hour

    async def _check_rate_limit(
        self, key: str, per_minute_limit: int, per_hour_limit: int
    ) -> tuple[bool, dict]:
        """Check if rate limit is exceeded.

        Args:
            key: Rate limit key
            per_minute_limit: Requests per minute limit
            per_hour_limit: Requests per hour limit

        Returns:
            tuple: (is_exceeded, remaining_info)
        """
        try:
            # Get rate limiter instance
            limiter = await get_rate_limiter()

            # Check per-minute limit
            minute_allowed, minute_remaining = await limiter.is_allowed(
                f"{key}:minute", per_minute_limit, 60
            )

            # Check per-hour limit
            hour_allowed, hour_remaining = await limiter.is_allowed(
                f"{key}:hour", per_hour_limit, 3600
            )

            # Rate limit exceeded if either limit is hit
            is_exceeded = not minute_allowed or not hour_allowed

            remaining_info = {
                "minute": minute_remaining,
                "hour": hour_remaining,
                "per_minute_limit": per_minute_limit,
                "per_hour_limit": per_hour_limit,
            }

            if is_exceeded:
                logger.warning(
                    f"Rate limit exceeded for key: {key}",
                    extra={
                        "rate_limit_key": key,
                        "minute_remaining": minute_remaining,
                        "hour_remaining": hour_remaining,
                        "per_minute_limit": per_minute_limit,
                        "per_hour_limit": per_hour_limit,
                    },
                )

            return is_exceeded, remaining_info

        except Exception as e:
            logger.error(f"Error checking rate limit for key {key}: {e}")
            # Fail open - allow request on error
            return False, {
                "minute": per_minute_limit,
                "hour": per_hour_limit,
                "per_minute_limit": per_minute_limit,
                "per_hour_limit": per_hour_limit,
            }

    async def _create_rate_limit_response(
        self, request: Request, remaining_info: dict
    ) -> JSONResponse:
        """Create rate limit exceeded response.

        Args:
            request: HTTP request
            remaining_info: Remaining requests information

        Returns:
            JSONResponse: Rate limit error response
        """
        request_id = getattr(request.state, "request_id", None)

        # Determine which limit was hit
        if remaining_info["minute"] <= 0:
            message = f"Rate limit exceeded: {remaining_info['per_minute_limit']} requests per minute"
            retry_after = "60"
        else:
            message = f"Rate limit exceeded: {remaining_info['per_hour_limit']} requests per hour"
            retry_after = "3600"

        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=create_error_response(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                message=message,
                error_code="RATE_LIMIT_EXCEEDED",
                details={
                    "remaining_minute": remaining_info["minute"],
                    "remaining_hour": remaining_info["hour"],
                    "limit_minute": remaining_info["per_minute_limit"],
                    "limit_hour": remaining_info["per_hour_limit"],
                },
                request_id=request_id,
            ),
        )

        # Add rate limit headers
        self._add_rate_limit_headers(
            response,
            remaining_info,
            remaining_info["per_minute_limit"],
            remaining_info["per_hour_limit"],
        )

        # Add retry-after header
        response.headers["Retry-After"] = retry_after

        return response

    def _add_rate_limit_headers(
        self,
        response: Response,
        remaining_info: dict,
        per_minute_limit: int,
        per_hour_limit: int,
    ) -> None:
        """Add rate limit headers to response.

        Args:
            response: HTTP response
            remaining_info: Remaining requests information
            per_minute_limit: Per minute rate limit
            per_hour_limit: Per hour rate limit
        """
        # Standard rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(per_minute_limit)
        response.headers["X-RateLimit-Limit-Hour"] = str(per_hour_limit)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            remaining_info.get("minute", 0)
        )
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            remaining_info.get("hour", 0)
        )

        # Alternative header format (some clients prefer this)
        response.headers["X-RateLimit-Limit"] = str(per_minute_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining_info.get("minute", 0))
