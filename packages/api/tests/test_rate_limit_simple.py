"""Simple test to verify rate limiting setup."""

import pytest
from unittest.mock import Mock, patch

from shared.infrastructure.config.config import Settings


def test_rate_limiter_import():
    """Test that rate limiter can be imported."""
    from api.middleware.rate_limit import RateLimiter
    assert RateLimiter is not None


def test_rate_limiter_initialization():
    """Test RateLimiter initialization."""
    from api.middleware.rate_limit import RateLimiter
    
    mock_settings = Mock(spec=Settings)
    mock_settings.rate_limit_enabled = False
    mock_settings.rate_limit_storage_url = "redis://localhost:6379/3"
    mock_settings.rate_limit_global = "100/minute"
    mock_settings.rate_limit_burst = 20
    mock_settings.rate_limit_tier_free = "50/minute"
    mock_settings.rate_limit_tier_pro = "500/minute"
    mock_settings.rate_limit_tier_enterprise = "5000/minute"
    
    # Should initialize without Redis when disabled
    limiter = RateLimiter(mock_settings)
    assert limiter.enabled is False
    assert limiter.redis_client is None


def test_endpoint_limiter_creation():
    """Test creating endpoint limiter."""
    from api.middleware.rate_limit import create_endpoint_limiter
    
    limiter = create_endpoint_limiter("5/minute")
    assert limiter is not None
    assert hasattr(limiter, 'dependency')


@pytest.mark.asyncio
async def test_rate_limit_check_disabled():
    """Test rate limit check when disabled."""
    from api.middleware.rate_limit import RateLimiter
    from fastapi import Request
    
    mock_settings = Mock(spec=Settings)
    mock_settings.rate_limit_enabled = False
    mock_settings.rate_limit_storage_url = "redis://localhost:6379/3"
    mock_settings.rate_limit_global = "100/minute"
    mock_settings.rate_limit_burst = 20
    mock_settings.rate_limit_tier_free = "50/minute"
    mock_settings.rate_limit_tier_pro = "500/minute"
    mock_settings.rate_limit_tier_enterprise = "5000/minute"
    
    limiter = RateLimiter(mock_settings)
    
    mock_request = Mock(spec=Request)
    result = await limiter.check_rate_limit(mock_request)
    
    assert result is None  # Should allow when disabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])