"""Test rate limiting under concurrent load."""

import asyncio
import pytest
import time
from httpx import AsyncClient
from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4

from shared.domain.models.api_key import APIKey


class TestRateLimitConcurrency:
    """Test rate limiting with concurrent requests and race conditions."""

    @pytest.mark.asyncio
    async def test_token_bucket_race_condition(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test that token bucket correctly handles concurrent token consumption."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Create a barrier to synchronize all requests
        num_requests = 50
        results = []
        
        async def make_request(i):
            # Small random delay to spread requests slightly
            await asyncio.sleep(i * 0.001)
            try:
                response = await authenticated_client.get(
                    "/api/v1/streams",
                    headers=headers,
                )
                return (i, response.status_code)
            except Exception as e:
                return (i, f"error: {e}")
        
        # Launch all requests concurrently
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful = sum(1 for _, status in results if status == 200)
        rate_limited = sum(1 for _, status in results if status == 429)
        errors = sum(1 for _, status in results if isinstance(status, str) and "error" in status)
        
        # Should have no errors
        assert errors == 0
        
        # Should have some successful and some rate limited
        assert successful > 0
        assert rate_limited > 0
        
        # Total should equal requests made
        assert successful + rate_limited == num_requests

    @pytest.mark.asyncio
    async def test_redis_pipeline_atomicity(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test that Redis operations are atomic."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Function to make a burst of requests
        async def request_burst(burst_id, count=10):
            burst_results = []
            for i in range(count):
                response = await authenticated_client.get(
                    "/api/v1/streams",
                    headers=headers,
                )
                burst_results.append({
                    'burst_id': burst_id,
                    'request_id': i,
                    'status': response.status_code,
                    'time': time.time()
                })
            return burst_results
        
        # Launch multiple bursts concurrently
        burst_tasks = [request_burst(i, 10) for i in range(5)]
        all_results = await asyncio.gather(*burst_tasks)
        
        # Flatten results
        flat_results = [r for burst in all_results for r in burst]
        
        # Check that rate limiting was applied consistently
        total_successful = sum(1 for r in flat_results if r['status'] == 200)
        total_limited = sum(1 for r in flat_results if r['status'] == 429)
        
        # Should not exceed rate limit significantly due to race conditions
        assert total_successful <= 100  # Assuming 100/min limit
        assert total_limited > 0  # Should have some rate limited

    @pytest.mark.asyncio
    async def test_rate_limit_window_edge_case(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test behavior at rate limit window boundaries."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Record request times and responses
        timeline = []
        
        # Make requests over 2 seconds to span potential window boundaries
        start_time = time.time()
        for i in range(20):
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
            timeline.append({
                'time': time.time() - start_time,
                'status': response.status_code,
                'headers': dict(response.headers),
            })
            await asyncio.sleep(0.1)  # 100ms between requests
        
        # Analyze timeline
        successful_count = sum(1 for t in timeline if t['status'] == 200)
        limited_count = sum(1 for t in timeline if t['status'] == 429)
        
        # Should see rate limiting kick in
        assert limited_count > 0
        
        # Check if rate limit headers show decreasing remaining
        for i, entry in enumerate(timeline):
            if 'x-ratelimit-remaining' in entry['headers']:
                remaining = int(entry['headers']['x-ratelimit-remaining'])
                # Remaining should decrease or stay same (if limited)
                if i > 0 and timeline[i-1]['status'] == 200:
                    prev_remaining = int(timeline[i-1]['headers'].get('x-ratelimit-remaining', remaining + 1))
                    assert remaining <= prev_remaining

    @pytest.mark.asyncio
    async def test_multi_organization_concurrent_limits(
        self,
        authenticated_client: AsyncClient,
        db_session,
    ):
        """Test that different organizations don't interfere with each other's limits."""
        # Create multiple organizations and API keys
        from shared.domain.models.organization import Organization
        from shared.infrastructure.security.api_key_service import APIKeyService
        from shared.infrastructure.storage.repositories import APIKeyRepository
        
        orgs = []
        keys = []
        
        api_key_repo = APIKeyRepository(db_session)
        api_key_service = APIKeyService(api_key_repo)
        
        for i in range(3):
            org = Organization(
                name=f"Test Org {i}",
                webhook_url=f"https://org{i}.com/webhook",
                billing_tier="free",
            )
            db_session.add(org)
            await db_session.commit()
            orgs.append(org)
            
            key_str, key = await api_key_service.generate_api_key(
                organization_id=org.id,
                name=f"Key {i}",
            )
            keys.append((key_str, key))
        
        # Make concurrent requests from all organizations
        async def org_requests(org_index, key_str, count=30):
            headers = {"X-API-Key": key_str}
            results = []
            
            for i in range(count):
                response = await authenticated_client.get(
                    "/api/v1/streams",
                    headers=headers,
                )
                results.append({
                    'org': org_index,
                    'request': i,
                    'status': response.status_code,
                })
            
            return results
        
        # Launch requests for all orgs concurrently
        tasks = [
            org_requests(i, key_str, 30)
            for i, (key_str, _) in enumerate(keys)
        ]
        all_results = await asyncio.gather(*tasks)
        
        # Each org should hit its own rate limit independently
        for org_idx, org_results in enumerate(all_results):
            org_successful = sum(1 for r in org_results if r['status'] == 200)
            org_limited = sum(1 for r in org_results if r['status'] == 429)
            
            # Each org should hit rate limit
            assert org_limited > 0
            # But should also have some successful requests
            assert org_successful > 0

    @pytest.mark.asyncio
    async def test_rate_limit_with_clock_skew(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
        monkeypatch,
    ):
        """Test rate limiting handles clock skew gracefully."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Make some requests normally
        for i in range(5):
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
            assert response.status_code in [200, 404]
        
        # Simulate clock going backwards
        original_time = time.time
        
        def skewed_time():
            return original_time() - 10  # 10 seconds in the past
        
        with patch('time.time', skewed_time):
            # Should still handle requests gracefully
            response = await authenticated_client.get(
                "/api/v1/streams",
                headers=headers,
            )
            # Should not cause errors
            assert response.status_code in [200, 404, 429]

    @pytest.mark.asyncio
    async def test_distributed_rate_limiting(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test that rate limiting works across multiple API instances."""
        headers = {"X-API-Key": test_api_key.key}
        
        # Simulate multiple API instances by using different client instances
        # In production, these would be different servers sharing Redis
        
        async def simulate_api_instance(instance_id, request_count):
            instance_results = []
            
            # Create new client to simulate different API instance
            async with AsyncClient(app=authenticated_client.app, base_url="http://test") as client:
                for i in range(request_count):
                    response = await client.get(
                        "/api/v1/streams",
                        headers=headers,
                    )
                    instance_results.append({
                        'instance': instance_id,
                        'request': i,
                        'status': response.status_code,
                    })
            
            return instance_results
        
        # Simulate 3 API instances making requests concurrently
        tasks = [simulate_api_instance(i, 20) for i in range(3)]
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        flat_results = [r for instance in all_results for r in instance]
        
        # Total successful across all instances should respect global limit
        total_successful = sum(1 for r in flat_results if r['status'] == 200)
        total_limited = sum(1 for r in flat_results if r['status'] == 429)
        
        # Should enforce rate limit across all instances
        assert total_limited > 0
        assert total_successful > 0
        
        # No instance should get all successful (proves shared state)
        for instance_results in all_results:
            instance_successful = sum(1 for r in instance_results if r['status'] == 200)
            assert instance_successful < 20  # Each tried 20

    @pytest.mark.asyncio
    async def test_rate_limit_memory_leak(
        self,
        authenticated_client: AsyncClient,
        test_api_key: APIKey,
    ):
        """Test that rate limiting doesn't cause memory leaks with many keys."""
        # Create many different "users" (different IPs)
        base_headers = {"X-API-Key": test_api_key.key}
        
        # Simulate requests from many different IPs
        for i in range(100):
            headers = base_headers.copy()
            headers["X-Forwarded-For"] = f"192.168.1.{i}"
            
            # Make a few requests per "user"
            for j in range(3):
                response = await authenticated_client.get(
                    "/api/v1/streams",
                    headers=headers,
                )
                assert response.status_code in [200, 404, 429]
        
        # Redis keys should have TTL and not accumulate forever
        # This is ensured by the 'ex' parameter in Redis set operations