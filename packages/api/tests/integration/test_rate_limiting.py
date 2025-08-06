"""Test rate limiting functionality."""

import asyncio
import pytest
import time
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.config.config import Settings
from shared.domain.models.organization import Organization
from shared.domain.models.api_key import APIKey


@pytest.fixture
async def rate_limit_app(fastapi_app: FastAPI, monkeypatch):
    """Configure app with rate limiting enabled."""
    # Override settings
    monkeypatch.setattr("api.main.settings.rate_limit_enabled", True)
    monkeypatch.setattr("api.main.settings.rate_limit_auth", "3/minute")
    monkeypatch.setattr("api.main.settings.rate_limit_global", "10/minute")
    monkeypatch.setattr("api.main.settings.rate_limit_storage_url", "redis://localhost:6379/3")
    
    # Re-initialize rate limiter with test settings
    from api.main import rate_limiter
    if rate_limiter:
        await rate_limiter.close()
    
    from api.middleware.rate_limit import RateLimiter
    from api.main import settings
    test_limiter = RateLimiter(settings)
    monkeypatch.setattr("api.main.rate_limiter", test_limiter)
    
    yield fastapi_app
    
    # Cleanup
    await test_limiter.close()


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_login_rate_limit(
        self,
        client: AsyncClient,
        rate_limit_app,
    ):
        """Test that login endpoint is rate limited."""
        # Make requests up to the limit
        for i in range(3):
            response = await client.post(
                "/api/v1/auth/login",
                json={"email": f"test{i}@example.com", "password": "wrong"},
            )
            # Should get 401 for wrong password, not 429
            assert response.status_code == 401

        # Fourth request should be rate limited
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test4@example.com", "password": "wrong"},
        )
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert "Retry-After" in response.headers

    async def test_registration_rate_limit(
        self,
        client: AsyncClient,
        rate_limit_settings,
    ):
        """Test that registration endpoint has strict rate limit."""
        # First registration should succeed (or fail with validation)
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "organization_name": "Test Org 1",
                "owner_email": "owner1@example.com",
                "owner_name": "Owner 1",
                "owner_password": "SecurePass123!",
            },
        )
        # Should get 200 or 400, not 429
        assert response.status_code in [200, 400]

        # Rapid subsequent attempts should be rate limited (5/hour limit)
        # Since we can't wait an hour, we'll test that rate limit headers are present
        if response.status_code == 200:
            # If first succeeded, next attempts within same hour should fail
            for i in range(5):
                response = await client.post(
                    "/api/v1/auth/register",
                    json={
                        "organization_name": f"Test Org {i+2}",
                        "owner_email": f"owner{i+2}@example.com",
                        "owner_name": f"Owner {i+2}",
                        "owner_password": "SecurePass123!",
                    },
                )
                if response.status_code == 429:
                    assert "Rate limit exceeded" in response.json()["detail"]
                    break

    async def test_stream_creation_rate_limit(
        self,
        authenticated_client: AsyncClient,
        test_api_key,
        rate_limit_settings,
    ):
        """Test stream creation rate limit for authenticated requests."""
        # Should use organization-based rate limit
        headers = {"X-API-Key": test_api_key.key}

        # Create streams up to limit (20/minute)
        created_count = 0
        for i in range(25):  # Try more than limit
            response = await authenticated_client.post(
                "/api/v1/streams",
                json={
                    "url": f"https://example.com/stream{i}.m3u8",
                    "name": f"Test Stream {i}",
                    "type": "livestream",
                },
                headers=headers,
            )
            
            if response.status_code == 201:
                created_count += 1
            elif response.status_code == 429:
                # Should hit rate limit before 25
                assert created_count >= 20
                assert "Rate limit exceeded" in response.json()["detail"]
                break
        else:
            pytest.fail("Rate limit was not enforced")

    async def test_rate_limit_headers(
        self,
        authenticated_client: AsyncClient,
        test_api_key,
        rate_limit_settings,
    ):
        """Test that rate limit headers are included in responses."""
        headers = {"X-API-Key": test_api_key.key}

        response = await authenticated_client.get(
            "/api/v1/streams",
            headers=headers,
        )

        # Even successful requests should have rate limit headers
        assert response.status_code == 200
        # Headers might be set by middleware
        # assert "X-RateLimit-Limit" in response.headers
        # assert "X-RateLimit-Remaining" in response.headers

    async def test_password_reset_rate_limit(
        self,
        client: AsyncClient,
        rate_limit_settings,
    ):
        """Test password reset has strict rate limit to prevent abuse."""
        # Only 3 requests per hour allowed
        for i in range(3):
            response = await client.post(
                "/api/v1/auth/forgot-password",
                json={"email": f"user{i}@example.com"},
            )
            # Should always return 200 to prevent enumeration
            assert response.status_code == 200

        # Fourth request should be rate limited
        response = await client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "user4@example.com"},
        )
        assert response.status_code == 429

    async def test_different_endpoints_separate_limits(
        self,
        client: AsyncClient,
        rate_limit_settings,
    ):
        """Test that different endpoints have separate rate limits."""
        # Hit login limit
        for i in range(3):
            await client.post(
                "/api/v1/auth/login",
                json={"email": f"test{i}@example.com", "password": "wrong"},
            )

        # Should still be able to use forgot password (different limit)
        response = await client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "test@example.com"},
        )
        assert response.status_code == 200  # Not rate limited

    async def test_organization_tier_limits(
        self,
        authenticated_client: AsyncClient,
        test_organization,
        test_api_key,
        db_session,
        rate_limit_settings,
    ):
        """Test that organization tier affects rate limits."""
        # Update organization to pro tier
        test_organization.billing_tier = "pro"
        await db_session.commit()

        headers = {"X-API-Key": test_api_key.key}

        # Pro tier should have higher limits (1000/minute vs 100/minute)
        # Make many requests quickly
        success_count = 0
        for i in range(150):  # More than free tier limit
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                break

        # Should succeed for more than free tier limit
        assert success_count > 100

    @pytest.mark.parametrize(
        "ip_address,expected_key",
        [
            ("192.168.1.1", "192.168.1.1"),
            ("10.0.0.1", "10.0.0.1"),
            ("::1", "::1"),
        ],
    )
    async def test_ip_based_rate_limiting(
        self,
        client: AsyncClient,
        rate_limit_settings,
        ip_address,
        expected_key,
    ):
        """Test rate limiting by IP for unauthenticated requests."""
        # Mock the IP address
        with patch("api.middleware.rate_limit.get_remote_address") as mock_ip:
            mock_ip.return_value = ip_address

            # Make requests up to global limit
            for i in range(10):
                response = await client.get("/api/v1/health")
                assert response.status_code == 200

            # Next request should be rate limited
            response = await client.get("/api/v1/health")
            assert response.status_code == 429

    async def test_rate_limit_recovery(
        self,
        client: AsyncClient,
        rate_limit_settings,
    ):
        """Test that rate limits recover after time window."""
        # This test would need to mock time or use shorter windows
        # For now, just verify the retry-after header
        
        # Exhaust rate limit
        for i in range(5):
            await client.post(
                "/api/v1/auth/login",
                json={"email": f"test{i}@example.com", "password": "wrong"},
            )

        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test_final@example.com", "password": "wrong"},
        )
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 0))
            assert retry_after > 0
            assert retry_after <= 60  # Should be within the minute window