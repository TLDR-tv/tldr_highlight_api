"""Rate limiting middleware and utilities."""

import redis.asyncio as redis
from fastapi import Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from typing import Optional, Callable
import time
from uuid import UUID

from shared.infrastructure.config.config import Settings
from shared.domain.models.organization import Organization
from shared.domain.models.api_key import APIKey


class RateLimiter:
    """Custom rate limiter with organization-based limits."""
    
    def __init__(self, settings: Settings):
        """Initialize rate limiter with settings."""
        self.settings = settings
        self.enabled = settings.rate_limit_enabled
        
        # Initialize Redis connection for rate limit storage
        if self.enabled:
            self.redis_client = redis.from_url(
                settings.rate_limit_storage_url,
                encoding="utf-8",
                decode_responses=True
            )
        else:
            self.redis_client = None
        
        # Create limiter with custom key function
        self.limiter = Limiter(
            key_func=self._get_rate_limit_key,
            default_limits=[settings.rate_limit_global] if self.enabled else [],
            enabled=self.enabled,
            storage_uri=settings.rate_limit_storage_url if self.enabled else None,
            headers_enabled=True,
            swallow_errors=False,
        )
        
        # Organization tier limits
        self.tier_limits = {
            "free": settings.rate_limit_tier_free,
            "pro": settings.rate_limit_tier_pro,
            "enterprise": settings.rate_limit_tier_enterprise,
        }
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key based on organization or IP."""
        # Check if we have an authenticated organization
        if hasattr(request.state, "api_key") and request.state.api_key:
            # Use organization ID for authenticated requests
            api_key: APIKey = request.state.api_key
            return f"org:{api_key.organization_id}"
        elif hasattr(request.state, "user") and request.state.user:
            # Use user's organization for user-authenticated requests
            user = request.state.user
            return f"org:{user.organization_id}"
        else:
            # Fall back to IP address for unauthenticated requests
            return get_remote_address(request)
    
    def get_organization_limit(self, organization: Organization) -> str:
        """Get rate limit for organization based on tier."""
        tier = organization.billing_tier or "free"
        return self.tier_limits.get(tier, self.tier_limits["free"])
    
    async def check_rate_limit(
        self,
        request: Request,
        limit_override: Optional[str] = None
    ) -> Optional[Response]:
        """Check if request exceeds rate limit."""
        if not self.enabled:
            return None
        
        key = self._get_rate_limit_key(request)
        limit = limit_override or self.settings.rate_limit_global
        
        # Check organization-specific limit
        if hasattr(request.state, "organization") and request.state.organization:
            org: Organization = request.state.organization
            limit = self.get_organization_limit(org)
        
        # Implement token bucket algorithm
        bucket_key = f"rate_limit:{key}:{limit}"
        tokens_key = f"{bucket_key}:tokens"
        timestamp_key = f"{bucket_key}:timestamp"
        
        # Parse limit (e.g., "100/minute" -> 100 requests per 60 seconds)
        try:
            rate_parts = limit.split("/")
            max_requests = int(rate_parts[0])
            
            # Convert time unit to seconds
            time_unit = rate_parts[1].lower()
            if time_unit == "second":
                window_seconds = 1
            elif time_unit == "minute":
                window_seconds = 60
            elif time_unit == "hour":
                window_seconds = 3600
            elif time_unit == "day":
                window_seconds = 86400
            else:
                window_seconds = 60  # Default to minute
        except (ValueError, IndexError):
            max_requests = 100
            window_seconds = 60
        
        current_time = time.time()
        
        # Get current token count and last refill time
        pipe = self.redis_client.pipeline()
        pipe.get(tokens_key)
        pipe.get(timestamp_key)
        results = await pipe.execute()
        
        current_tokens = float(results[0]) if results[0] else float(max_requests)
        last_refill = float(results[1]) if results[1] else current_time
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = (time_elapsed / window_seconds) * max_requests
        
        # Update token count (capped at max_requests)
        new_tokens = min(current_tokens + tokens_to_add, max_requests)
        
        # Check if request can proceed
        if new_tokens >= 1:
            # Consume a token
            new_tokens -= 1
            
            # Update Redis
            pipe = self.redis_client.pipeline()
            pipe.set(tokens_key, new_tokens, ex=window_seconds * 2)
            pipe.set(timestamp_key, current_time, ex=window_seconds * 2)
            await pipe.execute()
            
            # Add rate limit headers
            remaining = int(new_tokens)
            reset_time = int(current_time + (window_seconds - time_elapsed))
            
            return None  # Request allowed
        else:
            # Calculate retry after
            tokens_needed = 1 - new_tokens
            seconds_until_token = (tokens_needed * window_seconds) / max_requests
            retry_after = int(seconds_until_token)
            
            # Create rate limit exceeded response
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + seconds_until_token))
                }
            )
            
            return response
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


def create_endpoint_limiter(limit: str):
    """Create a simple rate limit dependency for endpoints."""
    from ..dependencies import get_rate_limiter
    
    async def rate_limit_check(
        request: Request,
        rate_limiter = Depends(get_rate_limiter)
    ):
        """Check rate limit for this endpoint."""
        if not rate_limiter or not rate_limiter.enabled:
            return None
            
        # Perform rate limit check
        response = await rate_limiter.check_rate_limit(request, limit_override=limit)
        if response:
            # Rate limit exceeded
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": response.headers.get("Retry-After", "60"),
                    "X-RateLimit-Limit": response.headers.get("X-RateLimit-Limit", ""),
                    "X-RateLimit-Remaining": response.headers.get("X-RateLimit-Remaining", "0"),
                    "X-RateLimit-Reset": response.headers.get("X-RateLimit-Reset", "")
                }
            )
        return None
    
    return Depends(rate_limit_check)


# Middleware for adding rate limit headers to all responses
class RateLimitHeaderMiddleware:
    """Add rate limit headers to responses."""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        self.app = app
        self.rate_limiter = rate_limiter
    
    async def __call__(self, scope, receive, send):
        """ASGI middleware for rate limit headers."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        async def send_wrapper(message):
            if message["type"] == "http.response.start" and self.rate_limiter.enabled:
                # Add rate limit headers if available
                headers = dict(message.get("headers", []))
                
                # These would be set by SlowAPI
                if b"x-ratelimit-limit" not in headers:
                    # Default headers can be added here if needed
                    pass
                    
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


# Custom error handler for rate limit exceeded
async def rate_limit_error_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom error handler for rate limit exceeded."""
    retry_after = exc.headers.get("Retry-After", "60")
    
    response = JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down your requests.",
            "retry_after": int(retry_after),
            "error_code": "RATE_LIMIT_EXCEEDED"
        }
    )
    
    # Copy headers from exception
    response.headers.update(exc.headers)
    
    return response