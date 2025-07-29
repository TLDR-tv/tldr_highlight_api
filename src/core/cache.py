"""Redis connection and caching utilities for the TL;DR Highlight API."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Optional

import redis
import redis.asyncio as redis_async
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from src.core.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis cache manager with connection pooling and retry logic."""

    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis_async.Redis] = None
        self._lock = asyncio.Lock()

    async def _create_pool(self) -> ConnectionPool:
        """Create Redis connection pool with retry logic."""
        return redis_async.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            db=settings.redis_db,
            max_connections=settings.redis_max_connections,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

    async def connect(self) -> None:
        """Initialize Redis connection with retry logic."""
        async with self._lock:
            if self._client:
                return

            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    self._pool = await self._create_pool()
                    self._client = redis_async.Redis(connection_pool=self._pool)

                    # Test connection
                    await self._client.ping()
                    logger.info("Successfully connected to Redis")
                    return
                except (ConnectionError, TimeoutError) as e:
                    logger.error(f"Redis connection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    else:
                        raise

    async def disconnect(self) -> None:
        """Close Redis connection and cleanup resources."""
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None
            if self._pool:
                await self._pool.disconnect()
                self._pool = None
            logger.info("Disconnected from Redis")

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[redis_async.Redis, None]:
        """Get Redis client with automatic connection management."""
        if not self._client:
            await self.connect()
        yield self._client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic JSON deserialization."""
        try:
            async with self.get_client() as client:
                value = await client.get(key)
                if value:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return (
                            value.decode("utf-8") if isinstance(value, bytes) else value
                        )
                return None
        except Exception as e:
            logger.error(f"Error getting key {key} from cache: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with automatic JSON serialization."""
        try:
            async with self.get_client() as client:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                elif not isinstance(value, (str, bytes)):
                    value = str(value)

                if ttl:
                    await client.setex(key, ttl, value)
                else:
                    await client.set(key, value)
                return True
        except Exception as e:
            logger.error(f"Error setting key {key} in cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            async with self.get_client() as client:
                result = await client.delete(key)
                return bool(result)
        except Exception as e:
            logger.error(f"Error deleting key {key} from cache: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            async with self.get_client() as client:
                return bool(await client.exists(key))
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        try:
            async with self.get_client() as client:
                return bool(await client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Error setting TTL for key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache."""
        try:
            async with self.get_client() as client:
                return await client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing key {key}: {e}")
            return None

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        try:
            async with self.get_client() as client:
                values = await client.mget(keys)
                result = {}
                for key, value in zip(keys, values):
                    if value:
                        try:
                            result[key] = json.loads(value)
                        except json.JSONDecodeError:
                            result[key] = (
                                value.decode("utf-8")
                                if isinstance(value, bytes)
                                else value
                            )
                return result
        except Exception as e:
            logger.error(f"Error getting multiple keys from cache: {e}")
            return {}

    async def set_many(
        self, mapping: dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        try:
            async with self.get_client() as client:
                # Serialize values
                serialized = {}
                for key, value in mapping.items():
                    if isinstance(value, (dict, list)):
                        serialized[key] = json.dumps(value)
                    elif not isinstance(value, (str, bytes)):
                        serialized[key] = str(value)
                    else:
                        serialized[key] = value

                # Use pipeline for atomic operation
                pipe = client.pipeline()
                for key, value in serialized.items():
                    if ttl:
                        pipe.setex(key, ttl, value)
                    else:
                        pipe.set(key, value)
                await pipe.execute()
                return True
        except Exception as e:
            logger.error(f"Error setting multiple keys in cache: {e}")
            return False


# Global cache instance
cache = RedisCache()


def cached(key_prefix: str, ttl: int = 300, key_func: Optional[Callable] = None):
    """
    Decorator for caching async function results.

    Args:
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key from arguments
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = f"{key_prefix}:{key_func(*args, **kwargs)}"
            else:
                # Simple key generation from args
                key_parts = [str(arg) for arg in args]
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{':'.join(key_parts)}"

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_value

            # Execute function and cache result
            logger.debug(f"Cache miss for key: {cache_key}")
            result = await func(*args, **kwargs)

            if result is not None:
                await cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


class RateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""

    def __init__(self, cache_instance: RedisCache):
        self.cache = cache_instance

    async def is_allowed(
        self, key: str, max_requests: int, window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if request is allowed under rate limit.

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        current_time = int(asyncio.get_event_loop().time())
        window_start = current_time - window_seconds
        rate_limit_key = f"rate_limit:{key}"

        try:
            async with self.cache.get_client() as client:
                # Use Lua script for atomic operation
                lua_script = """
                local key = KEYS[1]
                local window_start = tonumber(ARGV[1])
                local current_time = tonumber(ARGV[2])
                local max_requests = tonumber(ARGV[3])
                local window_seconds = tonumber(ARGV[4])
                
                -- Remove old entries
                redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
                
                -- Count current requests
                local current_requests = redis.call('ZCARD', key)
                
                if current_requests < max_requests then
                    -- Add new request
                    redis.call('ZADD', key, current_time, current_time)
                    redis.call('EXPIRE', key, window_seconds)
                    return {1, max_requests - current_requests - 1}
                else
                    return {0, 0}
                end
                """

                result = await client.eval(
                    lua_script,
                    1,
                    rate_limit_key,
                    window_start,
                    current_time,
                    max_requests,
                    window_seconds,
                )

                is_allowed = bool(result[0])
                remaining = result[1]

                return is_allowed, remaining
        except Exception as e:
            logger.error(f"Rate limiter error for key {key}: {e}")
            # Fail open - allow request on error
            return True, max_requests


# Global rate limiter instance
rate_limiter = RateLimiter(cache)


def get_redis_client():
    """Get a synchronous Redis client for use in Celery tasks and other sync contexts.
    
    Returns:
        redis.Redis: Synchronous Redis client instance
    """
    
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        db=settings.redis_db,
        decode_responses=settings.redis_decode_responses,
        socket_timeout=settings.redis_socket_timeout,
        socket_connect_timeout=settings.redis_socket_connect_timeout,
        retry_on_timeout=settings.redis_retry_on_timeout,
        health_check_interval=30,
    )
