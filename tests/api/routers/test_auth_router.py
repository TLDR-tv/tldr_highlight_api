"""Tests for authentication router."""

import pytest
from datetime import datetime
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession



@pytest.mark.asyncio
class TestAuthRouter:
    """Test cases for authentication router endpoints."""

    async def test_auth_status(self, async_client: AsyncClient):
        """Test authentication service status endpoint."""
        response = await async_client.get("/api/v1/auth/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "Authentication service operational"
        assert "timestamp" in data
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    async def test_create_api_key_not_implemented(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test create API key endpoint returns not implemented."""
        response = await async_client.post("/api/v1/auth/api-keys")

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        data = response.json()
        assert data["detail"] == "API key creation will be implemented in Phase 2.3"

    async def test_list_api_keys_not_implemented(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test list API keys endpoint returns not implemented."""
        response = await async_client.get("/api/v1/auth/api-keys")

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        data = response.json()
        assert data["detail"] == "API key listing will be implemented in Phase 2.3"

    async def test_delete_api_key_not_implemented(
        self, async_client: AsyncClient, db_session: AsyncSession
    ):
        """Test delete API key endpoint returns not implemented."""
        response = await async_client.delete("/api/v1/auth/api-keys/test-key-id")

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        data = response.json()
        assert data["detail"] == "API key deletion will be implemented in Phase 2.3"

    async def test_auth_status_response_format(self, async_client: AsyncClient):
        """Test auth status response matches expected schema."""
        response = await async_client.get("/api/v1/auth/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all required fields are present
        assert "status" in data
        assert "timestamp" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)

    async def test_auth_endpoints_without_db(self, async_client: AsyncClient):
        """Test auth endpoints handle database dependency properly."""
        # Status endpoint should work without DB
        response = await async_client.get("/api/v1/auth/status")
        assert response.status_code == status.HTTP_200_OK

        # Other endpoints should fail gracefully
        response = await async_client.post("/api/v1/auth/api-keys")
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

    @pytest.mark.parametrize(
        "endpoint,method",
        [
            ("/api/v1/auth/api-keys", "post"),
            ("/api/v1/auth/api-keys", "get"),
            ("/api/v1/auth/api-keys/test-id", "delete"),
        ],
    )
    async def test_placeholder_endpoints(
        self, async_client: AsyncClient, endpoint: str, method: str
    ):
        """Test all placeholder endpoints return 501 Not Implemented."""
        response = await async_client.request(method, endpoint)
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "Phase 2.3" in response.json()["detail"]
