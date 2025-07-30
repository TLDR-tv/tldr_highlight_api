"""Tests for rate limiting service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.services.rate_limit import RateLimitService
from src.models.api_key import APIKey
from src.models.user import User
from src.models.organization import Organization, PlanType


@pytest.fixture
def cache_mock():
    """Create mock cache."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=None)
    cache.delete = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def rate_limit_service(cache_mock):
    """Create rate limit service instance."""
    return RateLimitService(cache_mock)


@pytest.fixture
def sample_api_key():
    """Create sample API key."""
    user = User(
        id=1,
        email="test@example.com",
        company_name="Test Company",
        password_hash="hashed",
    )

    org = Organization(
        id=1, name="Test Org", owner_id=1, plan_type=PlanType.PROFESSIONAL.value
    )
    user.owned_organizations = [org]

    api_key = APIKey(
        id=1,
        key="hashed_key",
        name="Test Key",
        user_id=1,
        scopes=["read", "write"],
        active=True,
    )
    api_key.user = user

    return api_key


def create_redis_mock(eval_return_value=None):
    """Create a properly mocked Redis client with async context manager."""
    redis_client = MagicMock()
    redis_client.eval = AsyncMock(return_value=eval_return_value)
    redis_client.zremrangebyscore = AsyncMock(return_value=None)
    redis_client.zcard = AsyncMock(return_value=0)

    # Create async context manager
    async def async_context_manager():
        return redis_client

    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=redis_client)
    context_manager.__aexit__ = AsyncMock(return_value=None)

    return context_manager


class TestRateLimitService:
    """Test rate limiting service."""

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test rate limit check when request is allowed."""
        # Mock Redis client
        context_manager = create_redis_mock(eval_return_value=[1, 299])
        cache_mock.get_client = MagicMock(return_value=context_manager)

        is_allowed, remaining = await rate_limit_service.check_rate_limit(
            sample_api_key
        )

        assert is_allowed is True
        assert remaining == 299

    @pytest.mark.asyncio
    async def test_check_rate_limit_blocked(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test rate limit check when request is blocked."""
        # Mock Redis client
        context_manager = create_redis_mock(eval_return_value=[0, 0])
        cache_mock.get_client = MagicMock(return_value=context_manager)

        is_allowed, remaining = await rate_limit_service.check_rate_limit(
            sample_api_key
        )

        assert is_allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_check_rate_limit_window_seconds(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test rate limit with custom window."""
        # Mock Redis client
        context_manager = create_redis_mock(eval_return_value=[1, 299])
        cache_mock.get_client = MagicMock(return_value=context_manager)

        is_allowed, remaining = await rate_limit_service.check_rate_limit(
            sample_api_key,
            window_seconds=3600,  # 1 hour
        )

        assert is_allowed is True
        assert remaining == 299

    @pytest.mark.asyncio
    async def test_check_rate_limit_redis_error(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test rate limit when Redis has an error."""
        # Mock Redis error
        cache_mock.get_client.side_effect = Exception("Redis error")

        is_allowed, remaining = await rate_limit_service.check_rate_limit(
            sample_api_key
        )

        # Should fail open (allow request) on error
        assert is_allowed is True
        assert remaining == 300  # Professional plan limit

    @pytest.mark.asyncio
    async def test_get_rate_limit_starter_plan(
        self, rate_limit_service, sample_api_key
    ):
        """Test getting rate limit for starter plan."""
        # Change to starter plan
        sample_api_key.user.owned_organizations[0].plan_type = PlanType.STARTER.value

        limit = await rate_limit_service.get_rate_limit(sample_api_key)

        assert limit == 60

    @pytest.mark.asyncio
    async def test_get_rate_limit_professional_plan(
        self, rate_limit_service, sample_api_key
    ):
        """Test getting rate limit for professional plan."""
        limit = await rate_limit_service.get_rate_limit(sample_api_key)

        assert limit == 300

    @pytest.mark.asyncio
    async def test_get_rate_limit_enterprise_plan(
        self, rate_limit_service, sample_api_key
    ):
        """Test getting rate limit for enterprise plan."""
        # Change to enterprise plan
        sample_api_key.user.owned_organizations[0].plan_type = PlanType.ENTERPRISE.value

        limit = await rate_limit_service.get_rate_limit(sample_api_key)

        assert limit == 1000

    @pytest.mark.asyncio
    async def test_get_rate_limit_no_organization(
        self, rate_limit_service, sample_api_key
    ):
        """Test getting rate limit when user has no organization."""
        sample_api_key.user.owned_organizations = []

        limit = await rate_limit_service.get_rate_limit(sample_api_key)

        assert limit == 60  # Default

    @pytest.mark.asyncio
    async def test_get_current_usage(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test getting current usage count."""
        # Mock Redis client
        redis_client = MagicMock()
        redis_client.zremrangebyscore = AsyncMock(return_value=None)
        redis_client.zcard = AsyncMock(return_value=25)

        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=redis_client)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        cache_mock.get_client = MagicMock(return_value=context_manager)

        usage = await rate_limit_service.get_current_usage(sample_api_key)

        assert usage == 25

    @pytest.mark.asyncio
    async def test_get_current_usage_redis_error(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test getting current usage when Redis has an error."""
        # Mock Redis error
        cache_mock.get_client.side_effect = Exception("Redis error")

        usage = await rate_limit_service.get_current_usage(sample_api_key)

        assert usage == 0  # Default on error

    @pytest.mark.asyncio
    async def test_reset_rate_limit(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test resetting rate limit for an API key."""
        result = await rate_limit_service.reset_rate_limit(sample_api_key)

        assert result is True
        cache_mock.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_rate_limit_key(self, rate_limit_service, sample_api_key):
        """Test generating rate limit key."""
        key = rate_limit_service._get_rate_limit_key(sample_api_key)

        assert key == f"rate_limit:api_key:{sample_api_key.id}"

    @pytest.mark.asyncio
    async def test_check_multiple_requests_burst(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test multiple rapid requests hitting rate limit."""
        # Simulate multiple requests
        responses = [
            [1, 299],  # First request allowed
            [1, 298],  # Second request allowed
            [1, 297],  # Third request allowed
            [0, 0],  # Fourth request blocked
        ]

        # Create a mock that returns different values on each call
        redis_client = MagicMock()
        redis_client.eval = AsyncMock(side_effect=responses)

        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=redis_client)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        cache_mock.get_client = MagicMock(return_value=context_manager)

        results = []
        for _ in range(4):
            result = await rate_limit_service.check_rate_limit(sample_api_key)
            results.append(result)

        # First 3 should be allowed, 4th blocked
        assert results[0] == (True, 299)
        assert results[1] == (True, 298)
        assert results[2] == (True, 297)
        assert results[3] == (False, 0)

    @pytest.mark.asyncio
    async def test_sliding_window_behavior(
        self, rate_limit_service, cache_mock, sample_api_key
    ):
        """Test that the sliding window works correctly."""
        # Mock Redis client
        redis_client = MagicMock()
        redis_client.eval = AsyncMock(return_value=[1, 299])

        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=redis_client)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        cache_mock.get_client = MagicMock(return_value=context_manager)

        is_allowed, remaining = await rate_limit_service.check_rate_limit(
            sample_api_key
        )

        # Verify the eval call was made with correct parameters
        redis_client.eval.assert_called_once()

        # Check that the call included the expected parameters
        args = redis_client.eval.call_args[0]  # positional args
        assert len(args) >= 2  # Script and key count

        assert is_allowed is True
        assert remaining == 299
