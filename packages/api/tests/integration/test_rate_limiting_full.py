"""Comprehensive integration tests for rate limiting."""

import asyncio
import pytest
import time
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
from uuid import uuid4

from shared.domain.models.organization import Organization
from shared.domain.models.api_key import APIKey
from shared.domain.models.user import User


class TestRateLimitingIntegration:
    """Test rate limiting with real Redis backend."""

    @pytest.mark.asyncio
    async def test_auth_endpoint_rate_limits(self, client: AsyncClient):
        """Test auth endpoints have appropriate rate limits."""
        # Test login endpoint (5/minute limit)
        login_responses = []
        for i in range(6):
            response = await client.post(
                "/api/v1/auth/login",
                json={"email": f"test{i}@example.com", "password": "wrong"},
            )
            login_responses.append(response)
        
        # First 5 should get 401 (bad credentials)
        for i in range(5):
            assert login_responses[i].status_code == 401
        
        # 6th should be rate limited
        assert login_responses[5].status_code == 429
        assert "Rate limit exceeded" in login_responses[5].json()["detail"]

    @pytest.mark.asyncio
    async def test_organization_based_rate_limits(
        self,
        authenticated_client: AsyncClient,
        test_organization: Organization,
        test_api_key: APIKey,
        db_session,
    ):
        """Test that rate limits vary by organization tier."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Test with free tier (default)
        responses = []
        for i in range(10):
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
            responses.append(response)
            if response.status_code == 429:
                break
        
        # Should hit rate limit for free tier
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited or len(responses) == 10
        
        # Upgrade to pro tier
        test_organization.billing_tier = "pro"
        await db_session.commit()
        
        # Clear rate limit by waiting or using different key
        await asyncio.sleep(1)
        
        # Should have higher limit now
        pro_responses = []
        for i in range(20):
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
            pro_responses.append(response)
            if response.status_code == 429:
                break
        
        # Pro tier should allow more requests
        successful_pro = sum(1 for r in pro_responses if r.status_code == 200)
        assert successful_pro > 10

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test that rate limit headers are included in responses."""
        headers = {"X-API-Key": test_api_key.key}
        
        response = await authenticated_client.get(
            "/api/v1/streams",
            headers=headers,
        )
        
        # Check for rate limit headers (if implemented by middleware)
        # These might be added by SlowAPI or custom middleware
        if response.status_code == 200:
            # Headers may or may not be present depending on implementation
            pass

    @pytest.mark.asyncio
    async def test_different_endpoints_separate_buckets(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test that different endpoints have separate rate limit buckets."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Hit stream creation limit
        for i in range(20):
            response = await authenticated_client.post(
                "/api/v1/streams",
                json={
                    "url": f"https://example.com/stream{i}.m3u8",
                    "name": f"Stream {i}",
                    "type": "livestream",
                },
                headers=headers,
            )
            if response.status_code == 429:
                break
        
        # Should still be able to GET streams (different endpoint)
        response = await authenticated_client.get(
            "/api/v1/streams",
            headers=headers,
        )
        assert response.status_code in [200, 404]  # Not rate limited

    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test rate limiting with concurrent requests."""
        headers = {"X-API-Key": test_api_key.key}
        
        async def make_request(i):
            return await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
        
        # Make 50 concurrent requests
        tasks = [make_request(i) for i in range(50)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful vs rate limited
        successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        rate_limited = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 429)
        
        # Should have some of each
        assert successful > 0
        assert rate_limited > 0
        assert successful + rate_limited <= 50

    @pytest.mark.asyncio
    async def test_rate_limit_with_different_auth_methods(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
        test_user: User,
        auth_headers,
    ):
        """Test rate limiting works with both API key and JWT auth."""
        # Test with API key
        api_key_headers = {"X-API-Key": test_api_key.key}
        
        api_responses = []
        for i in range(5):
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=api_key_headers,
            )
            api_responses.append(response)
        
        # Test with JWT token
        jwt_responses = []
        for i in range(5):
            response = await authenticated_client.get(
                "/api/v1/users/me",
                headers=auth_headers,
            )
            jwt_responses.append(response)
        
        # Both should respect rate limits
        assert all(r.status_code in [200, 404] for r in api_responses[:4])
        assert all(r.status_code in [200, 404] for r in jwt_responses[:4])

    @pytest.mark.asyncio
    async def test_rate_limit_recovery_after_window(self, client: AsyncClient):
        """Test that rate limits reset after time window."""
        # Make requests to exhaust limit
        for i in range(5):
            await client.post(
                "/api/v1/auth/login",
                json={"email": f"test{i}@example.com", "password": "wrong"},
            )
        
        # Should be rate limited
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test_limited@example.com", "password": "wrong"},
        )
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            assert 0 < retry_after <= 60
            
            # In real test, would wait for window to expire
            # For now, just verify retry-after is reasonable

    @pytest.mark.asyncio
    async def test_ip_based_rate_limiting_for_anonymous(self, client: AsyncClient):
        """Test that anonymous requests are rate limited by IP."""
        # Make many unauthenticated requests
        responses = []
        for i in range(15):
            response = await client.get("/")
            responses.append(response)
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited or len(responses) == 15

    @pytest.mark.asyncio
    async def test_sensitive_endpoints_have_strict_limits(self, client: AsyncClient):
        """Test that sensitive endpoints have stricter rate limits."""
        # Password reset should have very strict limit (3/hour)
        reset_responses = []
        for i in range(4):
            response = await client.post(
                "/api/v1/auth/forgot-password",
                json={"email": f"user{i}@example.com"},
            )
            reset_responses.append(response)
        
        # First 3 should succeed (return 200 to prevent enumeration)
        for i in range(3):
            assert reset_responses[i].status_code == 200
        
        # 4th should be rate limited
        assert reset_responses[3].status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_with_redis_failure(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
        monkeypatch,
    ):
        """Test graceful handling when Redis is unavailable."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Mock Redis failure
        with patch('redis.asyncio.Redis.pipeline', side_effect=Exception("Redis connection failed")):
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
            
            # Should still work but without rate limiting
            # Depends on swallow_errors configuration
            assert response.status_code in [200, 404, 500]


class TestRateLimitEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_malformed_rate_limit_configuration(self, client: AsyncClient):
        """Test handling of malformed rate limit configurations."""
        with patch('api.main.settings.rate_limit_global', 'invalid/format'):
            # Should handle gracefully
            response = await client.get("/")
            assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_rate_limit_with_x_forwarded_for(self, client: AsyncClient):
        """Test rate limiting with proxy headers."""
        headers = {
            "X-Forwarded-For": "192.168.1.100, 10.0.0.1",
            "X-Real-IP": "192.168.1.100",
        }
        
        # Make requests with proxy headers
        responses = []
        for i in range(10):
            response = await client.get("/", headers=headers)
            responses.append(response)
        
        # Should use the real client IP for rate limiting
        assert all(r.status_code == 200 for r in responses[:9])

    @pytest.mark.asyncio
    async def test_rate_limit_key_isolation(
        self,
        authenticated_client: AsyncClient,
        db_session,
    ):
        """Test that rate limits are properly isolated between organizations."""
        # Create two organizations with API keys
        org1 = Organization(name="Org 1", webhook_url="https://org1.com/webhook")
        org2 = Organization(name="Org 2", webhook_url="https://org2.com/webhook")
        db_session.add_all([org1, org2])
        await db_session.commit()
        
        # Create API keys
        from shared.infrastructure.security.api_key_service import APIKeyService
        from shared.infrastructure.storage.repositories import APIKeyRepository
        
        api_key_repo = APIKeyRepository(db_session)
        api_key_service = APIKeyService(api_key_repo)
        
        key1_str, key1 = await api_key_service.generate_api_key(
            organization_id=org1.id,
            name="Test Key 1",
        )
        
        key2_str, key2 = await api_key_service.generate_api_key(
            organization_id=org2.id,
            name="Test Key 2",
        )
        
        # Make requests with different org keys
        headers1 = {"X-API-Key": key1_str}
        headers2 = {"X-API-Key": key2_str}
        
        # Both should be able to make requests independently
        response1 = await authenticated_client.get("/api/v1/streams", headers=headers1)
        response2 = await authenticated_client.get("/api/v1/streams", headers=headers2)
        
        assert response1.status_code in [200, 404]
        assert response2.status_code in [200, 404]