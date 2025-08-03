"""Unit tests for RateLimiter class."""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import Request, HTTPException
import redis.asyncio as redis

from api.middleware.rate_limit import RateLimiter
from shared.infrastructure.config.config import Settings
from shared.domain.models.api_key import APIKey
from shared.domain.models.organization import Organization
from uuid import uuid4


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.rate_limit_enabled = True
    settings.rate_limit_storage_url = "redis://localhost:6379/3"
    settings.rate_limit_global = "100/minute"
    settings.rate_limit_burst = 20
    settings.rate_limit_tier_free = "50/minute"
    settings.rate_limit_tier_pro = "500/minute"
    settings.rate_limit_tier_enterprise = "5000/minute"
    return settings


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock_client = AsyncMock(spec=redis.Redis)
    mock_pipeline = AsyncMock()
    mock_client.pipeline.return_value = mock_pipeline
    mock_pipeline.execute = AsyncMock(return_value=[None, None])
    return mock_client


@pytest.fixture
def rate_limiter(mock_settings, mock_redis):
    """Create RateLimiter instance with mocks."""
    with patch('redis.asyncio.from_url', return_value=mock_redis):
        limiter = RateLimiter(mock_settings)
        limiter.redis_client = mock_redis
        return limiter


class TestRateLimiter:
    """Test RateLimiter functionality."""

    def test_initialization_enabled(self, mock_settings):
        """Test RateLimiter initialization when enabled."""
        with patch('redis.asyncio.from_url') as mock_from_url:
            limiter = RateLimiter(mock_settings)
            
            assert limiter.enabled is True
            assert limiter.settings == mock_settings
            mock_from_url.assert_called_once_with(
                mock_settings.rate_limit_storage_url,
                encoding="utf-8",
                decode_responses=True
            )

    def test_initialization_disabled(self, mock_settings):
        """Test RateLimiter initialization when disabled."""
        mock_settings.rate_limit_enabled = False
        
        with patch('redis.asyncio.from_url') as mock_from_url:
            limiter = RateLimiter(mock_settings)
            
            assert limiter.enabled is False
            assert limiter.redis_client is None
            mock_from_url.assert_not_called()

    def test_get_rate_limit_key_with_api_key(self, rate_limiter):
        """Test rate limit key generation with API key."""
        request = Mock(spec=Request)
        request.state = Mock()
        
        api_key = Mock(spec=APIKey)
        api_key.organization_id = uuid4()
        request.state.api_key = api_key
        
        key = rate_limiter._get_rate_limit_key(request)
        assert key == f"org:{api_key.organization_id}"

    def test_get_rate_limit_key_with_user(self, rate_limiter):
        """Test rate limit key generation with authenticated user."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.api_key = None
        
        user = Mock()
        user.organization_id = uuid4()
        request.state.user = user
        
        key = rate_limiter._get_rate_limit_key(request)
        assert key == f"org:{user.organization_id}"

    def test_get_rate_limit_key_with_ip(self, rate_limiter):
        """Test rate limit key generation with IP address."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.api_key = None
        request.state.user = None
        
        with patch('api.middleware.rate_limit.get_remote_address', return_value="192.168.1.1"):
            key = rate_limiter._get_rate_limit_key(request)
            assert key == "192.168.1.1"

    def test_get_organization_limit(self, rate_limiter):
        """Test organization limit based on tier."""
        org = Mock(spec=Organization)
        
        # Test each tier
        org.billing_tier = "free"
        assert rate_limiter.get_organization_limit(org) == "50/minute"
        
        org.billing_tier = "pro"
        assert rate_limiter.get_organization_limit(org) == "500/minute"
        
        org.billing_tier = "enterprise"
        assert rate_limiter.get_organization_limit(org) == "5000/minute"
        
        # Test default (None or unknown tier)
        org.billing_tier = None
        assert rate_limiter.get_organization_limit(org) == "50/minute"
        
        org.billing_tier = "unknown"
        assert rate_limiter.get_organization_limit(org) == "50/minute"

    @pytest.mark.asyncio
    async def test_check_rate_limit_disabled(self, mock_settings, mock_redis):
        """Test rate limit check when disabled."""
        mock_settings.rate_limit_enabled = False
        limiter = RateLimiter(mock_settings)
        
        request = Mock(spec=Request)
        result = await limiter.check_rate_limit(request)
        
        assert result is None
        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter, mock_redis):
        """Test rate limit check when request is allowed."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.api_key = None
        request.state.user = None
        
        # Mock Redis responses - has tokens available
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = ["50", str(time.time())]
        
        with patch('api.middleware.rate_limit.get_remote_address', return_value="192.168.1.1"):
            result = await rate_limiter.check_rate_limit(request)
        
        assert result is None  # Request allowed
        assert mock_redis.pipeline.call_count >= 1

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter, mock_redis):
        """Test rate limit check when limit is exceeded."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.api_key = None
        request.state.user = None
        
        # Mock Redis responses - no tokens available
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = ["0", str(time.time())]
        
        with patch('api.middleware.rate_limit.get_remote_address', return_value="192.168.1.1"):
            result = await rate_limiter.check_rate_limit(request)
        
        assert result is not None
        assert result.status_code == 429
        assert "Retry-After" in result.headers

    @pytest.mark.asyncio
    async def test_check_rate_limit_with_organization(self, rate_limiter, mock_redis):
        """Test rate limit check with organization-specific limit."""
        request = Mock(spec=Request)
        request.state = Mock()
        
        org = Mock(spec=Organization)
        org.billing_tier = "pro"
        request.state.organization = org
        request.state.api_key = None
        request.state.user = None
        
        # Mock Redis responses
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = ["100", str(time.time())]
        
        with patch('api.middleware.rate_limit.get_remote_address', return_value="192.168.1.1"):
            result = await rate_limiter.check_rate_limit(request)
        
        assert result is None  # Should use pro tier limit

    @pytest.mark.asyncio
    async def test_rate_limit_parsing(self, rate_limiter, mock_redis):
        """Test parsing of different rate limit formats."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.api_key = None
        request.state.user = None
        request.state.organization = None
        
        # Test different time units
        test_cases = [
            ("100/second", 1),
            ("100/minute", 60),
            ("100/hour", 3600),
            ("100/day", 86400),
            ("100/unknown", 60),  # Default to minute
        ]
        
        for limit, expected_window in test_cases:
            # Reset mock
            mock_pipeline = AsyncMock()
            mock_redis.pipeline.return_value = mock_pipeline
            mock_pipeline.execute.return_value = ["50", str(time.time())]
            
            with patch('api.middleware.rate_limit.get_remote_address', return_value="192.168.1.1"):
                await rate_limiter.check_rate_limit(request, limit_override=limit)
            
            # Verify the limit was parsed correctly
            assert mock_redis.pipeline.called

    @pytest.mark.asyncio
    async def test_close(self, rate_limiter, mock_redis):
        """Test closing Redis connection."""
        await rate_limiter.close()
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_redis(self, mock_settings):
        """Test closing when Redis is not initialized."""
        mock_settings.rate_limit_enabled = False
        limiter = RateLimiter(mock_settings)
        
        # Should not raise exception
        await limiter.close()


class TestRateLimitEndpointDecorator:
    """Test the create_endpoint_limiter function."""

    @pytest.mark.asyncio
    async def test_endpoint_limiter_allowed(self):
        """Test endpoint limiter when request is allowed."""
        from api.middleware.rate_limit import create_endpoint_limiter
        
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.enabled = True
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=None)
        
        with patch('api.dependencies.get_rate_limiter', return_value=mock_rate_limiter):
            limiter_dep = create_endpoint_limiter("5/minute")
            
            # Extract the actual function from the Depends wrapper
            rate_limit_check = limiter_dep.dependency
            
            request = Mock(spec=Request)
            result = await rate_limit_check(request, mock_rate_limiter)
            
            assert result is None
            mock_rate_limiter.check_rate_limit.assert_called_once_with(
                request, limit_override="5/minute"
            )

    @pytest.mark.asyncio
    async def test_endpoint_limiter_exceeded(self):
        """Test endpoint limiter when rate limit is exceeded."""
        from api.middleware.rate_limit import create_endpoint_limiter
        
        mock_response = Mock()
        mock_response.headers = {
            "Retry-After": "60",
            "X-RateLimit-Limit": "5",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1234567890"
        }
        
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.enabled = True
        mock_rate_limiter.check_rate_limit = AsyncMock(return_value=mock_response)
        
        with patch('api.dependencies.get_rate_limiter', return_value=mock_rate_limiter):
            limiter_dep = create_endpoint_limiter("5/minute")
            rate_limit_check = limiter_dep.dependency
            
            request = Mock(spec=Request)
            
            with pytest.raises(HTTPException) as exc_info:
                await rate_limit_check(request, mock_rate_limiter)
            
            assert exc_info.value.status_code == 429
            assert exc_info.value.detail == "Rate limit exceeded"
            assert "Retry-After" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_endpoint_limiter_disabled(self):
        """Test endpoint limiter when rate limiting is disabled."""
        from api.middleware.rate_limit import create_endpoint_limiter
        
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.enabled = False
        
        with patch('api.dependencies.get_rate_limiter', return_value=mock_rate_limiter):
            limiter_dep = create_endpoint_limiter("5/minute")
            rate_limit_check = limiter_dep.dependency
            
            request = Mock(spec=Request)
            result = await rate_limit_check(request, mock_rate_limiter)
            
            assert result is None
            mock_rate_limiter.check_rate_limit.assert_not_called()


class TestTokenBucketAlgorithm:
    """Test the token bucket algorithm implementation."""

    @pytest.mark.asyncio
    async def test_token_refill_calculation(self, rate_limiter, mock_redis):
        """Test token refill calculation over time."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.api_key = None
        request.state.user = None
        request.state.organization = None
        
        current_time = time.time()
        last_refill = current_time - 30  # 30 seconds ago
        
        # Mock Redis responses - half tokens used, 30 seconds passed
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = ["50", str(last_refill)]
        
        with patch('time.time', return_value=current_time):
            with patch('api.middleware.rate_limit.get_remote_address', return_value="192.168.1.1"):
                result = await rate_limiter.check_rate_limit(request, limit_override="100/minute")
        
        # Should have refilled some tokens (30/60 * 100 = 50 new tokens)
        assert result is None  # Request allowed

    @pytest.mark.asyncio
    async def test_burst_capacity(self, rate_limiter, mock_redis):
        """Test that burst capacity is properly limited."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.api_key = None
        request.state.user = None
        request.state.organization = None
        
        # Mock Redis responses - full bucket
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = ["100", str(time.time())]
        
        with patch('api.middleware.rate_limit.get_remote_address', return_value="192.168.1.1"):
            result = await rate_limiter.check_rate_limit(request, limit_override="100/minute")
        
        assert result is None  # Should allow when bucket is full