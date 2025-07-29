"""Rate limiting service for the TL;DR Highlight API.

This module provides Redis-based rate limiting with sliding window implementation
for different permission levels and API key-based tracking.
"""

import asyncio
import logging
from typing import Tuple

from src.core.cache import RedisCache
from src.infrastructure.persistence.models.api_key import APIKey

logger = logging.getLogger(__name__)


class RateLimitService:
    """Service for handling rate limiting using Redis sliding window algorithm."""

    def __init__(self, cache: RedisCache):
        """Initialize rate limiting service.

        Args:
            cache: Redis cache instance for rate limiting data
        """
        self.cache = cache

    async def check_rate_limit(
        self, api_key: APIKey, window_seconds: int = 60
    ) -> Tuple[bool, int]:
        """Check if a request is allowed under the rate limit.

        Uses a sliding window algorithm implemented with Redis sorted sets.

        Args:
            api_key: The API key making the request
            window_seconds: Time window in seconds (default 60 for per-minute)

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        try:
            # Get rate limit for this API key
            max_requests = await self.get_rate_limit(api_key)

            # Generate rate limit key
            rate_limit_key = self._get_rate_limit_key(api_key)

            # Get current time for sliding window
            current_time = int(asyncio.get_event_loop().time())
            window_start = current_time - window_seconds

            # Use Lua script for atomic sliding window operation
            lua_script = """
            local key = KEYS[1]
            local window_start = tonumber(ARGV[1])
            local current_time = tonumber(ARGV[2])
            local max_requests = tonumber(ARGV[3])
            local window_seconds = tonumber(ARGV[4])
            
            -- Remove old entries outside the window
            redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
            
            -- Count current requests in window
            local current_requests = redis.call('ZCARD', key)
            
            if current_requests < max_requests then
                -- Add new request timestamp
                redis.call('ZADD', key, current_time, current_time)
                -- Set expiration for cleanup
                redis.call('EXPIRE', key, window_seconds + 1)
                return {1, max_requests - current_requests - 1}
            else
                return {0, 0}
            end
            """

            async with self.cache.get_client() as client:
                result = await client.eval(
                    lua_script,
                    1,  # Number of keys
                    rate_limit_key,
                    window_start,
                    current_time,
                    max_requests,
                    window_seconds,
                )

                is_allowed = bool(result[0])
                remaining = result[1]

                if not is_allowed:
                    logger.warning(
                        f"Rate limit exceeded for API key {api_key.id}. "
                        f"Limit: {max_requests}/{window_seconds}s"
                    )

                return is_allowed, remaining

        except Exception as e:
            logger.error(f"Rate limiter error for API key {api_key.id}: {e}")
            # Fail open - allow request on error
            max_requests = await self.get_rate_limit(api_key)
            return True, max_requests

    async def get_rate_limit(self, api_key: APIKey) -> int:
        """Get the rate limit for an API key based on the user's plan.

        Args:
            api_key: The API key object

        Returns:
            int: Rate limit per minute
        """
        try:
            # Get user's organization plan
            if api_key.user.owned_organizations:
                org = api_key.user.owned_organizations[0]
                limits = org.plan_limits
                return limits["api_rate_limit_per_minute"]

            # Default rate limit if no organization
            return 60

        except Exception as e:
            logger.error(f"Error getting rate limit for API key {api_key.id}: {e}")
            return 60  # Default fallback

    async def get_current_usage(self, api_key: APIKey, window_seconds: int = 60) -> int:
        """Get the current usage count for an API key within the time window.

        Args:
            api_key: The API key object
            window_seconds: Time window in seconds

        Returns:
            int: Current number of requests in the window
        """
        try:
            rate_limit_key = self._get_rate_limit_key(api_key)
            current_time = int(asyncio.get_event_loop().time())
            window_start = current_time - window_seconds

            async with self.cache.get_client() as client:
                # Remove expired entries
                await client.zremrangebyscore(rate_limit_key, 0, window_start)
                # Count remaining entries
                return await client.zcard(rate_limit_key)

        except Exception as e:
            logger.error(f"Error getting current usage for API key {api_key.id}: {e}")
            return 0

    async def reset_rate_limit(self, api_key: APIKey) -> bool:
        """Reset the rate limit for an API key (admin function).

        Args:
            api_key: The API key object

        Returns:
            bool: True if successfully reset, False otherwise
        """
        try:
            rate_limit_key = self._get_rate_limit_key(api_key)
            return await self.cache.delete(rate_limit_key)

        except Exception as e:
            logger.error(f"Error resetting rate limit for API key {api_key.id}: {e}")
            return False

    def _get_rate_limit_key(self, api_key: APIKey) -> str:
        """Generate Redis key for rate limiting an API key.

        Args:
            api_key: The API key object

        Returns:
            str: Redis key for rate limiting
        """
        return f"rate_limit:api_key:{api_key.id}"

    async def get_rate_limit_info(self, api_key: APIKey) -> dict:
        """Get comprehensive rate limit information for an API key.

        Args:
            api_key: The API key object

        Returns:
            dict: Rate limit information
        """
        try:
            max_requests = await self.get_rate_limit(api_key)
            current_usage = await self.get_current_usage(api_key)
            remaining = max(0, max_requests - current_usage)

            return {
                "max_requests_per_minute": max_requests,
                "current_usage": current_usage,
                "remaining": remaining,
                "reset_time": int(asyncio.get_event_loop().time()) + 60,  # Next minute
                "percentage_used": (current_usage / max_requests) * 100
                if max_requests > 0
                else 0,
            }

        except Exception as e:
            logger.error(f"Error getting rate limit info for API key {api_key.id}: {e}")
            return {
                "max_requests_per_minute": 60,
                "current_usage": 0,
                "remaining": 60,
                "reset_time": int(asyncio.get_event_loop().time()) + 60,
                "percentage_used": 0,
            }

    async def check_burst_limit(
        self,
        api_key: APIKey,
        burst_window_seconds: int = 10,
        burst_max_requests: int = 20,
    ) -> Tuple[bool, int]:
        """Check burst rate limit for short-term protection.

        Args:
            api_key: The API key object
            burst_window_seconds: Burst window in seconds (default 10s)
            burst_max_requests: Maximum requests in burst window

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        try:
            burst_key = f"burst:{self._get_rate_limit_key(api_key)}"
            current_time = int(asyncio.get_event_loop().time())
            window_start = current_time - burst_window_seconds

            lua_script = """
            local key = KEYS[1]
            local window_start = tonumber(ARGV[1])
            local current_time = tonumber(ARGV[2])
            local max_requests = tonumber(ARGV[3])
            local window_seconds = tonumber(ARGV[4])
            
            redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
            local current_requests = redis.call('ZCARD', key)
            
            if current_requests < max_requests then
                redis.call('ZADD', key, current_time, current_time)
                redis.call('EXPIRE', key, window_seconds + 1)
                return {1, max_requests - current_requests - 1}
            else
                return {0, 0}
            end
            """

            async with self.cache.get_client() as client:
                result = await client.eval(
                    lua_script,
                    1,
                    burst_key,
                    window_start,
                    current_time,
                    burst_max_requests,
                    burst_window_seconds,
                )

                is_allowed = bool(result[0])
                remaining = result[1]

                if not is_allowed:
                    logger.warning(
                        f"Burst limit exceeded for API key {api_key.id}. "
                        f"Limit: {burst_max_requests}/{burst_window_seconds}s"
                    )

                return is_allowed, remaining

        except Exception as e:
            logger.error(f"Burst rate limiter error for API key {api_key.id}: {e}")
            return True, burst_max_requests

    async def check_daily_limit(self, api_key: APIKey) -> Tuple[bool, int]:
        """Check daily usage limits based on organization plan.

        Args:
            api_key: The API key object

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        try:
            # Get daily limits from organization plan
            if api_key.user.owned_organizations:
                org = api_key.user.owned_organizations[0]
                limits = org.plan_limits

                # Calculate daily API limit (rate limit * 24 * 60)
                _daily_limit = limits["api_rate_limit_per_minute"] * 24 * 60

                # Use 24-hour sliding window
                return await self.check_rate_limit(
                    api_key,
                    window_seconds=24 * 60 * 60,  # 24 hours
                )

            # Default daily limit
            return await self.check_rate_limit(api_key, window_seconds=24 * 60 * 60)

        except Exception as e:
            logger.error(f"Daily limit check error for API key {api_key.id}: {e}")
            return True, 86400  # 24 hours worth of default rate
