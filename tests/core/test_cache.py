"""Comprehensive tests for cache module."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from redis.exceptions import ConnectionError, TimeoutError
import redis.asyncio as redis_async

from src.core.cache import (
    RedisCache,
    cache,
    cached,
    RateLimiter,
    rate_limiter,
    get_redis_client,
)


class TestRedisCache:
    """Test cases for RedisCache class."""

    @pytest.mark.asyncio
    async def test_connection_success(self):
        """Test successful Redis connection."""
        redis_cache = RedisCache()
        
        with patch('src.core.cache.redis_async.ConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool_class.return_value = mock_pool
            
            with patch('src.core.cache.redis_async.Redis') as mock_redis_class:
                mock_client = AsyncMock()
                mock_client.ping = AsyncMock(return_value=True)
                mock_redis_class.return_value = mock_client
                
                await redis_cache.connect()
                
                assert redis_cache._client is not None
                assert redis_cache._pool is not None
                mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_retry(self):
        """Test Redis connection with retries."""
        redis_cache = RedisCache()
        
        with patch('src.core.cache.redis_async.ConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool_class.return_value = mock_pool
            
            with patch('src.core.cache.redis_async.Redis') as mock_redis_class:
                mock_client = AsyncMock()
                # Fail twice, then succeed
                mock_client.ping = AsyncMock(side_effect=[
                    ConnectionError("Connection failed"),
                    TimeoutError("Timeout"),
                    True
                ])
                mock_redis_class.return_value = mock_client
                
                with patch('asyncio.sleep') as mock_sleep:
                    await redis_cache.connect()
                    
                    assert mock_client.ping.call_count == 3
                    assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test Redis connection failure after all retries."""
        redis_cache = RedisCache()
        
        with patch('src.core.cache.redis_async.ConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool_class.return_value = mock_pool
            
            with patch('src.core.cache.redis_async.Redis') as mock_redis_class:
                mock_client = AsyncMock()
                mock_client.ping = AsyncMock(side_effect=ConnectionError("Connection failed"))
                mock_redis_class.return_value = mock_client
                
                with patch('asyncio.sleep'):
                    with pytest.raises(ConnectionError):
                        await redis_cache.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test Redis disconnection."""
        redis_cache = RedisCache()
        
        # Set up mock client and pool
        redis_cache._client = AsyncMock()
        redis_cache._pool = AsyncMock()
        
        await redis_cache.disconnect()
        
        redis_cache._client.close.assert_called_once()
        redis_cache._pool.disconnect.assert_called_once()
        assert redis_cache._client is None
        assert redis_cache._pool is None

    @pytest.mark.asyncio
    async def test_get_client_context(self):
        """Test get_client context manager."""
        redis_cache = RedisCache()
        
        # Mock existing client
        mock_client = AsyncMock()
        redis_cache._client = mock_client
        
        async with redis_cache.get_client() as client:
            assert client is mock_client

    @pytest.mark.asyncio
    async def test_get_client_connects_if_needed(self):
        """Test get_client connects if no client exists."""
        redis_cache = RedisCache()
        
        with patch.object(redis_cache, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_client = AsyncMock()
            redis_cache._client = mock_client  # Set after connect
            
            async with redis_cache.get_client() as client:
                assert client is mock_client
            
            # Should not call connect if client exists
            mock_connect.assert_not_called()
            
        # Test when client is None
        redis_cache._client = None
        with patch.object(redis_cache, 'connect', new_callable=AsyncMock) as mock_connect:
            redis_cache._client = mock_client  # Set after connect
            
            async with redis_cache.get_client() as client:
                assert client is mock_client
            
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_json_value(self):
        """Test getting JSON values from cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=b'{"key": "value", "number": 42}')
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.get("test_key")
            
            assert result == {"key": "value", "number": 42}
            mock_client.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_string_value(self):
        """Test getting string values from cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=b'plain string')
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.get("test_key")
            
            assert result == "plain string"

    @pytest.mark.asyncio
    async def test_get_none_value(self):
        """Test getting None when key doesn't exist."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.get("nonexistent_key")
            
            assert result is None

    @pytest.mark.asyncio
    async def test_get_error_handling(self):
        """Test error handling in get operation."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Redis error"))
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.get("test_key")
            
            assert result is None  # Should return None on error

    @pytest.mark.asyncio
    async def test_set_json_value(self):
        """Test setting JSON values in cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)
        mock_client.setex = AsyncMock(return_value=True)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            # Without TTL
            result = await redis_cache.set("test_key", {"data": "value"})
            assert result is True
            mock_client.set.assert_called_once_with("test_key", '{"data": "value"}')
            
            # With TTL
            result = await redis_cache.set("test_key2", {"data": "value2"}, ttl=3600)
            assert result is True
            mock_client.setex.assert_called_once_with("test_key2", 3600, '{"data": "value2"}')

    @pytest.mark.asyncio
    async def test_set_string_value(self):
        """Test setting string values in cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.set("test_key", "string value")
            
            assert result is True
            mock_client.set.assert_called_once_with("test_key", "string value")

    @pytest.mark.asyncio
    async def test_set_numeric_value(self):
        """Test setting numeric values in cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.set("test_key", 42)
            
            assert result is True
            mock_client.set.assert_called_once_with("test_key", "42")

    @pytest.mark.asyncio
    async def test_set_error_handling(self):
        """Test error handling in set operation."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.set = AsyncMock(side_effect=Exception("Redis error"))
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.set("test_key", "value")
            
            assert result is False  # Should return False on error

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting keys from cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(return_value=1)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.delete("test_key")
            
            assert result is True
            mock_client.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent key."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(return_value=0)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.delete("nonexistent_key")
            
            assert result is False

    @pytest.mark.asyncio
    async def test_delete_error_handling(self):
        """Test error handling in delete operation."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(side_effect=Exception("Redis error"))
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.delete("test_key")
            
            assert result is False

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test checking key existence."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.exists = AsyncMock(return_value=1)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.exists("test_key")
            
            assert result is True
            mock_client.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists_not_found(self):
        """Test checking non-existent key."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.exists = AsyncMock(return_value=0)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.exists("nonexistent_key")
            
            assert result is False

    @pytest.mark.asyncio
    async def test_expire(self):
        """Test setting TTL for existing key."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.expire = AsyncMock(return_value=True)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.expire("test_key", 3600)
            
            assert result is True
            mock_client.expire.assert_called_once_with("test_key", 3600)

    @pytest.mark.asyncio
    async def test_increment(self):
        """Test incrementing counter in cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.incrby = AsyncMock(return_value=5)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.increment("counter_key")
            
            assert result == 5
            mock_client.incrby.assert_called_once_with("counter_key", 1)
            
            # Test with custom amount
            mock_client.incrby.reset_mock()
            result = await redis_cache.increment("counter_key", 10)
            
            assert result == 5
            mock_client.incrby.assert_called_once_with("counter_key", 10)

    @pytest.mark.asyncio
    async def test_get_many(self):
        """Test getting multiple values from cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_client.mget = AsyncMock(return_value=[
            b'{"data": 1}',
            b'plain string',
            None,
            b'{"data": 2}'
        ])
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            result = await redis_cache.get_many(["key1", "key2", "key3", "key4"])
            
            assert result == {
                "key1": {"data": 1},
                "key2": "plain string",
                "key4": {"data": 2}
            }
            mock_client.mget.assert_called_once_with(["key1", "key2", "key3", "key4"])

    @pytest.mark.asyncio
    async def test_set_many(self):
        """Test setting multiple values in cache."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_client.pipeline = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=True)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            mapping = {
                "key1": {"data": 1},
                "key2": "string value",
                "key3": 42
            }
            
            result = await redis_cache.set_many(mapping)
            
            assert result is True
            mock_client.pipeline.assert_called_once()
            
            # Check pipeline calls
            assert mock_pipeline.set.call_count == 3
            mock_pipeline.set.assert_any_call("key1", '{"data": 1}')
            mock_pipeline.set.assert_any_call("key2", "string value")
            mock_pipeline.set.assert_any_call("key3", "42")

    @pytest.mark.asyncio
    async def test_set_many_with_ttl(self):
        """Test setting multiple values with TTL."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_client.pipeline = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=True)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            mapping = {"key1": "value1", "key2": "value2"}
            
            result = await redis_cache.set_many(mapping, ttl=3600)
            
            assert result is True
            assert mock_pipeline.setex.call_count == 2
            mock_pipeline.setex.assert_any_call("key1", 3600, "value1")
            mock_pipeline.setex.assert_any_call("key2", 3600, "value2")


class TestCachedDecorator:
    """Test cases for cached decorator."""

    @pytest.mark.asyncio
    async def test_cached_hit(self):
        """Test cache hit scenario."""
        call_count = 0
        
        @cached("test_func", ttl=300)
        async def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        with patch.object(cache, 'get', new_callable=AsyncMock) as mock_get:
            with patch.object(cache, 'set', new_callable=AsyncMock) as mock_set:
                mock_get.return_value = 42  # Cache hit
                
                result = await test_function(10, 20)
                
                assert result == 42
                assert call_count == 0  # Function not called
                mock_get.assert_called_once()
                mock_set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_miss(self):
        """Test cache miss scenario."""
        call_count = 0
        
        @cached("test_func", ttl=300)
        async def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        with patch.object(cache, 'get', new_callable=AsyncMock) as mock_get:
            with patch.object(cache, 'set', new_callable=AsyncMock) as mock_set:
                mock_get.return_value = None  # Cache miss
                
                result = await test_function(10, 20)
                
                assert result == 30
                assert call_count == 1  # Function called
                mock_get.assert_called_once()
                mock_set.assert_called_once_with(
                    "test_func:10:20",
                    30,
                    300
                )

    @pytest.mark.asyncio
    async def test_cached_with_kwargs(self):
        """Test caching with keyword arguments."""
        @cached("test_kwargs", ttl=300)
        async def test_function(x, y=10, z=20):
            return x + y + z
        
        with patch.object(cache, 'get', new_callable=AsyncMock) as mock_get:
            with patch.object(cache, 'set', new_callable=AsyncMock) as mock_set:
                mock_get.return_value = None
                
                result = await test_function(5, z=30)
                
                assert result == 45
                # Check cache key includes kwargs
                cache_key = mock_set.call_args[0][0]
                assert "test_kwargs:5:z=30" == cache_key

    @pytest.mark.asyncio
    async def test_cached_with_key_func(self):
        """Test caching with custom key function."""
        def custom_key_func(user_id, *args, **kwargs):
            return f"user_{user_id}"
        
        @cached("test_custom", ttl=300, key_func=custom_key_func)
        async def get_user_data(user_id, include_details=False):
            return {"id": user_id, "details": include_details}
        
        with patch.object(cache, 'get', new_callable=AsyncMock) as mock_get:
            with patch.object(cache, 'set', new_callable=AsyncMock) as mock_set:
                mock_get.return_value = None
                
                result = await get_user_data(123, include_details=True)
                
                assert result["id"] == 123
                # Check custom key was used
                cache_key = mock_set.call_args[0][0]
                assert cache_key == "test_custom:user_123"

    @pytest.mark.asyncio
    async def test_cached_none_result(self):
        """Test caching when function returns None."""
        @cached("test_none", ttl=300)
        async def test_function():
            return None
        
        with patch.object(cache, 'get', new_callable=AsyncMock) as mock_get:
            with patch.object(cache, 'set', new_callable=AsyncMock) as mock_set:
                mock_get.return_value = None  # Cache miss
                
                result = await test_function()
                
                assert result is None
                # Should not cache None results
                mock_set.assert_not_called()


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self):
        """Test rate limiting when request is allowed."""
        limiter = RateLimiter(cache)
        
        mock_client = AsyncMock()
        mock_client.eval = AsyncMock(return_value=[1, 4])  # Allowed, 4 remaining
        
        with patch.object(cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            is_allowed, remaining = await limiter.is_allowed("test_key", 5, 60)
            
            assert is_allowed is True
            assert remaining == 4

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limiting when limit is exceeded."""
        limiter = RateLimiter(cache)
        
        mock_client = AsyncMock()
        mock_client.eval = AsyncMock(return_value=[0, 0])  # Not allowed, 0 remaining
        
        with patch.object(cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            is_allowed, remaining = await limiter.is_allowed("test_key", 5, 60)
            
            assert is_allowed is False
            assert remaining == 0

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self):
        """Test rate limiting error handling (fail open)."""
        limiter = RateLimiter(cache)
        
        mock_client = AsyncMock()
        mock_client.eval = AsyncMock(side_effect=Exception("Redis error"))
        
        with patch.object(cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            is_allowed, remaining = await limiter.is_allowed("test_key", 5, 60)
            
            # Should fail open - allow request on error
            assert is_allowed is True
            assert remaining == 5

    @pytest.mark.asyncio
    async def test_rate_limit_lua_script(self):
        """Test that Lua script is called correctly."""
        limiter = RateLimiter(cache)
        
        mock_client = AsyncMock()
        mock_client.eval = AsyncMock(return_value=[1, 2])
        
        with patch.object(cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.time.return_value = 1000.0
                
                await limiter.is_allowed("user:123", 10, 300)
                
                # Check eval was called with correct parameters
                call_args = mock_client.eval.call_args
                assert call_args[0][1] == 1  # Number of keys
                assert call_args[0][2] == "rate_limit:user:123"  # Key
                assert call_args[0][3] == 700  # Window start (1000 - 300)
                assert call_args[0][4] == 1000  # Current time
                assert call_args[0][5] == 10  # Max requests
                assert call_args[0][6] == 300  # Window seconds


class TestGetRedisClient:
    """Test cases for get_redis_client function."""

    def test_get_redis_client(self):
        """Test getting synchronous Redis client."""
        with patch('src.core.cache.redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            
            client = get_redis_client()
            
            assert client is mock_client
            mock_redis.assert_called_once()
            
            # Check connection parameters
            call_kwargs = mock_redis.call_args[1]
            assert 'host' in call_kwargs
            assert 'port' in call_kwargs
            assert 'db' in call_kwargs
            assert call_kwargs['health_check_interval'] == 30

    def test_get_redis_client_with_password(self):
        """Test getting Redis client with password."""
        with patch('src.core.cache.settings') as mock_settings:
            mock_settings.redis_host = 'localhost'
            mock_settings.redis_port = 6379
            mock_settings.redis_password = 'secret_password'
            mock_settings.redis_db = 1
            mock_settings.redis_decode_responses = True
            mock_settings.redis_socket_timeout = 5
            mock_settings.redis_socket_connect_timeout = 5
            mock_settings.redis_retry_on_timeout = True
            
            with patch('src.core.cache.redis.Redis') as mock_redis:
                get_redis_client()
                
                call_kwargs = mock_redis.call_args[1]
                assert call_kwargs['password'] == 'secret_password'
                assert call_kwargs['decode_responses'] is True


class TestGlobalInstances:
    """Test cases for global instances."""

    def test_cache_global_instance(self):
        """Test global cache instance."""
        from src.core.cache import cache as cache1
        from src.core.cache import cache as cache2
        
        assert cache1 is cache2
        assert isinstance(cache1, RedisCache)

    def test_rate_limiter_global_instance(self):
        """Test global rate limiter instance."""
        from src.core.cache import rate_limiter as limiter1
        from src.core.cache import rate_limiter as limiter2
        
        assert limiter1 is limiter2
        assert isinstance(limiter1, RateLimiter)
        assert limiter1.cache is cache


class TestIntegration:
    """Integration tests for cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_workflow(self):
        """Test complete cache workflow."""
        redis_cache = RedisCache()
        
        mock_client = AsyncMock()
        # Set up responses for workflow
        mock_client.get = AsyncMock(side_effect=[None, b'"cached_value"'])
        mock_client.set = AsyncMock(return_value=True)
        mock_client.exists = AsyncMock(return_value=1)
        mock_client.delete = AsyncMock(return_value=1)
        
        with patch.object(redis_cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            # Initial get - cache miss
            result = await redis_cache.get("workflow_key")
            assert result is None
            
            # Set value
            assert await redis_cache.set("workflow_key", "cached_value") is True
            
            # Check exists
            assert await redis_cache.exists("workflow_key") is True
            
            # Get value - cache hit
            result = await redis_cache.get("workflow_key")
            assert result == "cached_value"
            
            # Delete value
            assert await redis_cache.delete("workflow_key") is True

    @pytest.mark.asyncio
    async def test_rate_limiter_workflow(self):
        """Test rate limiter workflow."""
        limiter = RateLimiter(cache)
        
        mock_client = AsyncMock()
        # Simulate decreasing remaining count
        mock_client.eval = AsyncMock(side_effect=[
            [1, 4],  # First request: allowed, 4 remaining
            [1, 3],  # Second request: allowed, 3 remaining
            [1, 2],  # Third request: allowed, 2 remaining
            [1, 1],  # Fourth request: allowed, 1 remaining
            [1, 0],  # Fifth request: allowed, 0 remaining
            [0, 0],  # Sixth request: not allowed
        ])
        
        with patch.object(cache, 'get_client') as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_client
            
            # Make requests up to limit
            for i in range(5):
                is_allowed, remaining = await limiter.is_allowed("api_key", 5, 60)
                assert is_allowed is True
                assert remaining == 4 - i
            
            # Exceed limit
            is_allowed, remaining = await limiter.is_allowed("api_key", 5, 60)
            assert is_allowed is False
            assert remaining == 0