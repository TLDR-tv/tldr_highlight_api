"""Integration tests for stream management API endpoints."""

import pytest
from uuid import uuid4
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from shared.domain.models.api_key import APIScopes
from shared.domain.models.stream import StreamStatus, StreamType
from shared.infrastructure.storage.repositories import (
    OrganizationRepository,
    StreamRepository,
    APIKeyRepository,
)
from tests.factories import (
    create_test_organization,
    create_test_api_key,
    create_test_stream,
)


class TestStreamEndpoints:
    """Test stream management endpoints."""

    @pytest.mark.asyncio
    async def test_list_streams_empty(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test listing streams when none exist."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # List streams
        response = await client.get(
            "/api/v1/streams/",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["streams"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 20

    @pytest.mark.asyncio
    async def test_list_streams_with_data(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test listing streams with data."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ}
        )

        # Create multiple streams
        stream1 = create_test_stream(
            organization_id=org.id,
            url="https://example.com/stream1.m3u8",
            name="Stream 1",
        )
        stream2 = create_test_stream(
            organization_id=org.id,
            url="https://example.com/stream2.m3u8",
            name="Stream 2",
        )
        stream3 = create_test_stream(
            organization_id=org.id,
            url="https://example.com/stream3.m3u8",
            name="Stream 3",
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream1)
        await stream_repo.create(stream2)
        await stream_repo.create(stream3)
        await test_session.commit()

        # List streams
        response = await client.get(
            "/api/v1/streams/",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["streams"]) == 3
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["per_page"] == 20

        # Check stream order (newest first)
        stream_names = [s["name"] for s in data["streams"]]
        assert stream_names == ["Stream 3", "Stream 2", "Stream 1"]

    @pytest.mark.asyncio
    async def test_list_streams_pagination(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test stream list pagination."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ}
        )

        # Create 5 streams
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        
        for i in range(5):
            stream = create_test_stream(
                organization_id=org.id,
                url=f"https://example.com/stream{i}.m3u8",
                name=f"Stream {i}",
            )
            await stream_repo.create(stream)
        
        await test_session.commit()

        # Get page 1 with page_size 2
        response = await client.get(
            "/api/v1/streams/?page=1&page_size=2",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["streams"]) == 2
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["per_page"] == 2

        # Get page 2
        response = await client.get(
            "/api/v1/streams/?page=2&page_size=2",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["streams"]) == 2
        assert data["page"] == 2

        # Get page 3
        response = await client.get(
            "/api/v1/streams/?page=3&page_size=2",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["streams"]) == 1
        assert data["page"] == 3

    @pytest.mark.asyncio
    async def test_list_streams_wrong_organization(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test that streams from other organizations are not visible."""
        # Create two organizations
        org1 = create_test_organization(name="Org 1")
        org2 = create_test_organization(name="Org 2")
        
        api_key1, raw_key1 = create_test_api_key(
            organization_id=org1.id,
            scopes={APIScopes.STREAMS_READ}
        )
        
        # Create streams for org2
        stream_org2 = create_test_stream(
            organization_id=org2.id,
            url="https://example.com/org2-stream.m3u8",
            name="Org2 Stream",
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        
        await org_repo.create(org1)
        await org_repo.create(org2)
        await api_key_repo.create(api_key1)
        await stream_repo.create(stream_org2)
        await test_session.commit()

        # Try to list streams as org1
        response = await client.get(
            "/api/v1/streams/",
            headers=api_key_headers(raw_key1),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["streams"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_streams_insufficient_scope(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test listing streams with insufficient API key scope."""
        # Create organization and API key without STREAMS_READ scope
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.HIGHLIGHTS_READ}  # No STREAMS_READ
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Try to list streams
        response = await client.get(
            "/api/v1/streams/",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "scope" in response.json()["detail"].lower()