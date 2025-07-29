"""Tests for health check router."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from src.api.main import app
from src.core.config import settings


@pytest.mark.asyncio
class TestHealthRouter:
    """Test cases for health check router endpoints."""

    async def test_comprehensive_health_check_healthy(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test comprehensive health check when all services are healthy."""
        with patch("src.api.routers.health._check_database_health") as mock_db, \
             patch("src.api.routers.health._check_redis_health") as mock_redis, \
             patch("src.api.routers.health._check_storage_health") as mock_storage:
            
            # Mock healthy responses
            mock_db.return_value = {
                "status": "healthy",
                "response_time_ms": 5.5,
                "database": "tldr_test",
                "user": "testuser",
                "version": "PostgreSQL 15.0",
            }
            mock_redis.return_value = {
                "status": "healthy",
                "response_time_ms": 2.1,
                "version": "7.0.0",
                "mode": "standalone",
            }
            mock_storage.return_value = {
                "status": "healthy",
                "response_time_ms": 15.3,
                "region": "us-east-1",
                "buckets_total": 3,
            }
            
            response = await async_client.get("/api/v1/health/")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Check overall response structure
            assert data["status"] == "healthy"
            assert data["version"] == settings.app_version
            assert "timestamp" in data
            assert "services" in data
            
            # Check individual services
            assert data["services"]["api"]["status"] == "healthy"
            assert data["services"]["database"]["status"] == "healthy"
            assert data["services"]["redis"]["status"] == "healthy"
            assert data["services"]["storage"]["status"] == "healthy"

    async def test_comprehensive_health_check_degraded(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test comprehensive health check when some services are unhealthy."""
        with patch("src.api.routers.health._check_database_health") as mock_db, \
             patch("src.api.routers.health._check_redis_health") as mock_redis, \
             patch("src.api.routers.health._check_storage_health") as mock_storage:
            
            # Mock mixed health responses
            mock_db.return_value = {"status": "healthy", "response_time_ms": 5.5}
            mock_redis.return_value = {
                "status": "unhealthy",
                "response_time_ms": 100.0,
                "error": "Connection timeout",
            }
            mock_storage.return_value = {"status": "healthy", "response_time_ms": 15.3}
            
            response = await async_client.get("/api/v1/health/")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Should be degraded when one service is unhealthy
            assert data["status"] == "degraded"
            assert data["services"]["redis"]["status"] == "unhealthy"

    async def test_comprehensive_health_check_unhealthy(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test comprehensive health check when services fail with exceptions."""
        with patch("src.api.routers.health._check_database_health") as mock_db, \
             patch("src.api.routers.health._check_redis_health") as mock_redis, \
             patch("src.api.routers.health._check_storage_health") as mock_storage:
            
            # Mock exception responses
            mock_db.side_effect = Exception("Database connection failed")
            mock_redis.return_value = {"status": "healthy", "response_time_ms": 2.1}
            mock_storage.return_value = {"status": "healthy", "response_time_ms": 15.3}
            
            response = await async_client.get("/api/v1/health/")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Should be unhealthy when exceptions occur
            assert data["status"] == "unhealthy"
            assert data["services"]["database"]["status"] == "unhealthy"
            assert "Database connection failed" in data["services"]["database"]["error"]

    async def test_liveness_check(self, async_client: AsyncClient):
        """Test liveness probe endpoint."""
        response = await async_client.get("/api/v1/health/live")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    async def test_readiness_check_ready(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test readiness probe when services are ready."""
        with patch("src.core.cache.cache.get_client") as mock_cache:
            # Mock Redis client
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock(return_value=True)
            mock_cache.return_value.__aenter__.return_value = mock_redis_client
            
            response = await async_client.get("/api/v1/health/ready")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "ready"
            assert "timestamp" in data

    async def test_readiness_check_not_ready_db_failure(
        self, async_client: AsyncClient
    ):
        """Test readiness probe when database is not ready."""
        with patch("src.api.routers.health.get_db") as mock_get_db:
            # Mock database failure
            mock_db = AsyncMock()
            mock_db.execute.side_effect = SQLAlchemyError("Database unavailable")
            mock_get_db.return_value = mock_db
            
            response = await async_client.get("/api/v1/health/ready")
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert data["detail"] == "Service not ready"

    async def test_readiness_check_not_ready_redis_failure(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test readiness probe when Redis is not ready."""
        with patch("src.core.cache.cache.get_client") as mock_cache:
            # Mock Redis failure
            mock_redis_client = AsyncMock()
            mock_redis_client.ping.side_effect = Exception("Redis connection failed")
            mock_cache.return_value.__aenter__.return_value = mock_redis_client
            
            response = await async_client.get("/api/v1/health/ready")
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert data["detail"] == "Service not ready"

    async def test_database_health_endpoint(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test dedicated database health endpoint."""
        response = await async_client.get("/api/v1/health/database")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "response_time_ms" in data
        assert "database" in data
        assert "user" in data
        assert "version" in data
        assert "connection_info" in data

    async def test_database_health_endpoint_failure(
        self, async_client: AsyncClient
    ):
        """Test database health endpoint when database fails."""
        with patch("src.api.routers.health.get_db") as mock_get_db:
            # Mock database failure
            mock_db = AsyncMock()
            mock_db.execute.side_effect = SQLAlchemyError("Connection timeout")
            mock_get_db.return_value = mock_db
            
            response = await async_client.get("/api/v1/health/database")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "error_type" in data
            assert "response_time_ms" in data

    async def test_redis_health_endpoint(self, async_client: AsyncClient):
        """Test dedicated Redis health endpoint."""
        with patch("src.core.cache.cache.get_client") as mock_cache:
            # Mock Redis client
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock(return_value=True)
            mock_redis_client.info = AsyncMock(return_value={
                "redis_version": "7.0.0",
                "redis_mode": "standalone",
                "connected_clients": 5,
                "used_memory_human": "10MB",
            })
            mock_redis_client.set = AsyncMock(return_value=True)
            mock_redis_client.get = AsyncMock(return_value=b"test_value")
            mock_redis_client.delete = AsyncMock(return_value=1)
            mock_cache.return_value.__aenter__.return_value = mock_redis_client
            
            response = await async_client.get("/api/v1/health/redis")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["version"] == "7.0.0"
            assert data["mode"] == "standalone"
            assert data["connected_clients"] == 5
            assert data["operations_test"] == "passed"

    async def test_redis_health_endpoint_failure(self, async_client: AsyncClient):
        """Test Redis health endpoint when Redis fails."""
        with patch("src.core.cache.cache.get_client") as mock_cache:
            # Mock Redis failure
            mock_cache.side_effect = Exception("Redis connection refused")
            
            response = await async_client.get("/api/v1/health/redis")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "error_type" in data

    async def test_storage_health_endpoint(self, async_client: AsyncClient):
        """Test dedicated storage health endpoint."""
        with patch("src.services.storage.storage_service") as mock_storage:
            # Mock storage service
            mock_storage.list_buckets = AsyncMock(return_value=["bucket1", "bucket2", "bucket3"])
            mock_storage.list_objects = AsyncMock(return_value=[])
            
            response = await async_client.get("/api/v1/health/storage")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"
            assert "response_time_ms" in data
            assert data["buckets_total"] == 3
            assert "key_buckets" in data

    async def test_storage_health_endpoint_failure(self, async_client: AsyncClient):
        """Test storage health endpoint when storage fails."""
        with patch("src.services.storage.storage_service") as mock_storage:
            # Mock storage failure
            mock_storage.list_buckets.side_effect = Exception("S3 access denied")
            
            response = await async_client.get("/api/v1/health/storage")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "S3 access denied" in data["error"]

    async def test_health_check_response_times(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test that health checks include accurate response times."""
        response = await async_client.get("/api/v1/health/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check that all services have response times
        for service_name, service_data in data["services"].items():
            if service_name != "api":  # API has 0 response time
                assert "response_time_ms" in service_data
                if service_data["status"] == "healthy":
                    assert isinstance(service_data["response_time_ms"], (int, float))
                    assert service_data["response_time_ms"] >= 0

    async def test_concurrent_health_checks(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test that health checks run concurrently."""
        with patch("src.api.routers.health._check_database_health") as mock_db, \
             patch("src.api.routers.health._check_redis_health") as mock_redis, \
             patch("src.api.routers.health._check_storage_health") as mock_storage, \
             patch("asyncio.gather") as mock_gather:
            
            # Mock healthy responses
            mock_db.return_value = {"status": "healthy", "response_time_ms": 5.0}
            mock_redis.return_value = {"status": "healthy", "response_time_ms": 2.0}
            mock_storage.return_value = {"status": "healthy", "response_time_ms": 10.0}
            
            # Mock gather to return the results
            mock_gather.return_value = [
                {"status": "healthy", "response_time_ms": 5.0},
                {"status": "healthy", "response_time_ms": 2.0},
                {"status": "healthy", "response_time_ms": 10.0}
            ]
            
            response = await async_client.get("/api/v1/health/")
            
            assert response.status_code == status.HTTP_200_OK
            # Verify asyncio.gather was called (concurrent execution)
            mock_gather.assert_called_once()

    async def test_health_check_with_partial_failures(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test health check when some services fail."""
        with patch("src.api.routers.health._check_database_health") as mock_db, \
             patch("src.api.routers.health._check_redis_health") as mock_redis, \
             patch("src.api.routers.health._check_storage_health") as mock_storage:
            
            # Mix of success and failure
            mock_db.return_value = {"status": "healthy", "response_time_ms": 5.0}
            mock_redis.side_effect = Exception("Redis connection failed")
            mock_storage.return_value = {"status": "degraded", "response_time_ms": 50.0}
            
            response = await async_client.get("/api/v1/health/")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Overall status should be unhealthy due to Redis exception
            assert data["status"] == "unhealthy"
            assert data["services"]["database"]["status"] == "healthy"
            assert data["services"]["redis"]["status"] == "unhealthy"
            assert "Redis connection failed" in data["services"]["redis"]["error"]
            assert data["services"]["storage"]["status"] == "degraded"

    async def test_database_health_detailed_info(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test detailed database health information."""
        with patch.object(db_session, "execute") as mock_execute:
            # Mock database version info
            mock_result = AsyncMock()
            mock_result.fetchone.return_value = (
                "PostgreSQL 15.0 on x86_64-pc-linux-gnu",
                "tldr_test",
                "testuser"
            )
            mock_execute.return_value = mock_result
            
            response = await async_client.get("/api/v1/health/database")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["database"] == "tldr_test"
            assert data["user"] == "testuser"
            assert data["version"] == "PostgreSQL"
            assert "connection_info" in data
            assert "pool_size" in data["connection_info"]
            assert "max_overflow" in data["connection_info"]

    async def test_redis_health_detailed_info(self, async_client: AsyncClient):
        """Test detailed Redis health information."""
        with patch("src.core.cache.cache.get_client") as mock_cache:
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock(return_value=True)
            mock_redis_client.info = AsyncMock(return_value={
                "redis_version": "7.0.0",
                "redis_mode": "standalone",
                "connected_clients": 10,
                "used_memory_human": "50MB",
            })
            mock_redis_client.set = AsyncMock(return_value=True)
            mock_redis_client.get = AsyncMock(return_value=b"test_value")
            mock_redis_client.delete = AsyncMock(return_value=1)
            mock_cache.return_value.__aenter__.return_value = mock_redis_client
            
            response = await async_client.get("/api/v1/health/redis")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["version"] == "7.0.0"
            assert data["mode"] == "standalone"
            assert data["connected_clients"] == 10
            assert data["used_memory_human"] == "50MB"
            assert data["operations_test"] == "passed"
            
            # Verify operations were tested
            mock_redis_client.set.assert_called_once()
            mock_redis_client.get.assert_called_once()
            mock_redis_client.delete.assert_called_once()

    async def test_redis_health_operations_test_failed(self, async_client: AsyncClient):
        """Test Redis health when operations test fails."""
        with patch("src.core.cache.cache.get_client") as mock_cache:
            mock_redis_client = AsyncMock()
            mock_redis_client.ping = AsyncMock(return_value=True)
            mock_redis_client.info = AsyncMock(return_value={"redis_version": "7.0.0"})
            mock_redis_client.set = AsyncMock(return_value=True)
            mock_redis_client.get = AsyncMock(return_value=b"wrong_value")  # Wrong value
            mock_redis_client.delete = AsyncMock(return_value=1)
            mock_cache.return_value.__aenter__.return_value = mock_redis_client
            
            response = await async_client.get("/api/v1/health/redis")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"  # Still healthy (ping succeeded)
            assert data["operations_test"] == "failed"  # But operations test failed

    async def test_storage_health_detailed_info(self, async_client: AsyncClient):
        """Test detailed storage health information."""
        with patch("src.services.storage.storage_service") as mock_storage, \
             patch("src.core.config.settings") as mock_settings:
            
            # Mock settings
            mock_settings.s3_highlights_bucket = "highlights-bucket"
            mock_settings.s3_thumbnails_bucket = "thumbnails-bucket"
            mock_settings.s3_temp_bucket = "temp-bucket"
            mock_settings.s3_region = "us-east-1"
            mock_settings.s3_endpoint_url = None
            
            # Mock storage responses
            mock_storage.list_buckets = AsyncMock(return_value=[
                "highlights-bucket", "thumbnails-bucket", "temp-bucket", "other-bucket"
            ])
            mock_storage.list_objects = AsyncMock(return_value=[])  # Successful access
            
            response = await async_client.get("/api/v1/health/storage")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["region"] == "us-east-1"
            assert data["endpoint"] == "default"
            assert data["buckets_total"] == 4
            assert "key_buckets" in data
            assert data["key_buckets"]["highlights-bucket"] == "accessible"
            assert data["key_buckets"]["thumbnails-bucket"] == "accessible"
            assert data["key_buckets"]["temp-bucket"] == "accessible"

    async def test_storage_health_bucket_access_errors(self, async_client: AsyncClient):
        """Test storage health when some buckets are inaccessible."""
        with patch("src.services.storage.storage_service") as mock_storage, \
             patch("src.core.config.settings") as mock_settings:
            
            # Mock settings
            mock_settings.s3_highlights_bucket = "highlights-bucket"
            mock_settings.s3_thumbnails_bucket = "thumbnails-bucket"
            mock_settings.s3_temp_bucket = "temp-bucket"
            mock_settings.s3_region = "us-east-1"
            mock_settings.s3_endpoint_url = "https://custom.s3.endpoint"
            
            # Mock storage responses
            mock_storage.list_buckets = AsyncMock(return_value=["highlights-bucket", "thumbnails-bucket"])
            
            # Mock bucket access - some succeed, some fail
            async def mock_list_objects(bucket_name, limit=None):
                if bucket_name == "highlights-bucket":
                    return []  # Success
                elif bucket_name == "thumbnails-bucket":
                    raise Exception("Access denied")
                else:  # temp-bucket
                    raise Exception("Bucket not found")
            
            mock_storage.list_objects = mock_list_objects
            
            response = await async_client.get("/api/v1/health/storage")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "healthy"  # Overall still healthy
            assert data["endpoint"] == "https://custom.s3.endpoint"
            assert data["buckets_total"] == 2
            assert data["key_buckets"]["highlights-bucket"] == "accessible"
            assert "Access denied" in data["key_buckets"]["thumbnails-bucket"]
            assert "Bucket not found" in data["key_buckets"]["temp-bucket"]

    async def test_health_endpoints_error_format_consistency(self, async_client: AsyncClient):
        """Test that all health endpoints return consistent error formats."""
        # Test database error format
        with patch("src.api.routers.health.get_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_db.execute.side_effect = SQLAlchemyError("Database connection timeout")
            mock_get_db.return_value = mock_db
            
            response = await async_client.get("/api/v1/health/database")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "error_type" in data
            assert "response_time_ms" in data
            assert data["error_type"] == "SQLAlchemyError"

        # Test Redis error format
        with patch("src.core.cache.cache.get_client") as mock_cache:
            mock_cache.side_effect = Exception("Redis connection refused")
            
            response = await async_client.get("/api/v1/health/redis")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "error_type" in data
            assert "response_time_ms" in data
            assert data["error_type"] == "Exception"

        # Test storage error format
        with patch("src.services.storage.storage_service") as mock_storage:
            mock_storage.list_buckets.side_effect = Exception("S3 service unavailable")
            
            response = await async_client.get("/api/v1/health/storage")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "error_type" in data
            assert "response_time_ms" in data

    async def test_api_service_info_in_health_check(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test that API service info is included in health check."""
        with patch("src.api.routers.health._check_database_health") as mock_db, \
             patch("src.api.routers.health._check_redis_health") as mock_redis, \
             patch("src.api.routers.health._check_storage_health") as mock_storage:
            
            # Mock all services as healthy
            mock_db.return_value = {"status": "healthy", "response_time_ms": 5.0}
            mock_redis.return_value = {"status": "healthy", "response_time_ms": 2.0}
            mock_storage.return_value = {"status": "healthy", "response_time_ms": 10.0}
            
            response = await async_client.get("/api/v1/health/")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Check API service info
            assert "api" in data["services"]
            api_info = data["services"]["api"]
            assert api_info["status"] == "healthy"
            assert "version" in api_info
            assert "environment" in api_info
            assert api_info["response_time_ms"] == 0  # Immediate response

    async def test_health_check_timestamps(self, async_client: AsyncClient):
        """Test that health checks include valid timestamps."""
        response = await async_client.get("/api/v1/health/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify timestamp format
        timestamp = data["timestamp"]
        assert isinstance(timestamp, str)
        # Should be able to parse as ISO format
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        
        # Test other endpoints
        for endpoint in ["/api/v1/health/live", "/api/v1/health/ready"]:
            response = await async_client.get(endpoint)
            if response.status_code == 200:  # Skip if fails due to dependencies
                data = response.json()
                assert "timestamp" in data
                datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    async def test_health_check_version_consistency(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test that version is consistent across health endpoints."""
        with patch("src.core.config.settings") as mock_settings:
            mock_settings.app_version = "1.2.3"
            mock_settings.environment = "test"
            
            response = await async_client.get("/api/v1/health/")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Check main version
            assert data["version"] == "1.2.3"
            # Check API service version
            assert data["services"]["api"]["version"] == "1.2.3"
            assert data["services"]["api"]["environment"] == "test"