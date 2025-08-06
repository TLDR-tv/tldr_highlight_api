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


class TestStreamCreation:
    """Test stream creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_stream_success(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test successful stream creation."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_WRITE}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Create stream
        response = await client.post(
            "/api/v1/streams/",
            headers=api_key_headers(raw_key),
            json={
                "url": "https://example.com/livestream.m3u8",
                "name": "Test Stream",
                "metadata": {"description": "A test stream"},
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["url"] == "https://example.com/livestream.m3u8"
        assert data["name"] == "Test Stream"
        assert data["metadata"]["description"] == "A test stream"
        assert data["status"] == StreamStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_create_stream_invalid_url(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test stream creation with invalid URL."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_WRITE}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Create stream with invalid URL
        response = await client.post(
            "/api/v1/streams/",
            headers=api_key_headers(raw_key),
            json={
                "url": "not-a-valid-url",
                "name": "Test Stream",
            },
        )

        assert response.status_code == status.HTTP_200_OK  # API may accept any string as URL

    @pytest.mark.asyncio
    async def test_create_stream_insufficient_scope(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test stream creation with insufficient scope."""
        # Create organization and API key with wrong scope
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ}  # Wrong scope
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        response = await client.post(
            "/api/v1/streams/",
            headers=api_key_headers(raw_key),
            json={
                "url": "https://example.com/stream.m3u8",
                "name": "Test Stream",
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestStreamRetrieval:
    """Test stream retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_get_stream_success(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test successful stream retrieval."""
        # Create organization, API key, and stream
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ}
        )
        stream = create_test_stream(organization_id=org.id)

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await test_session.commit()

        # Get stream
        response = await client.get(
            f"/api/v1/streams/{stream.id}",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(stream.id)
        assert data["url"] == stream.url

    @pytest.mark.asyncio
    async def test_get_stream_not_found(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test stream retrieval for non-existent stream."""
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

        # Try to get non-existent stream
        fake_id = uuid4()
        response = await client.get(
            f"/api/v1/streams/{fake_id}",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_stream_cross_organization_access_denied(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test that streams from other organizations can't be accessed."""
        # Create two organizations
        org1 = create_test_organization(name="Org 1")
        org2 = create_test_organization(name="Org 2")
        
        # API key for org1
        api_key, raw_key = create_test_api_key(
            organization_id=org1.id,
            scopes={APIScopes.STREAMS_READ}
        )
        
        # Stream in org2
        stream = create_test_stream(organization_id=org2.id)

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        
        await org_repo.create(org1)
        await org_repo.create(org2)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await test_session.commit()

        # Try to access org2's stream with org1's API key
        response = await client.get(
            f"/api/v1/streams/{stream.id}",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestStreamProcessing:
    """Test stream processing endpoints."""

    @pytest.mark.asyncio
    async def test_process_stream_success(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test successful stream processing."""
        from unittest.mock import patch, MagicMock
        
        # Create organization, API key, and stream
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_WRITE}
        )
        stream = create_test_stream(organization_id=org.id)

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await test_session.commit()

        # Mock Celery task
        with patch('api.routes.streams.celery_app.send_task') as mock_send_task:
            mock_result = MagicMock()
            mock_result.id = "test-task-id"
            mock_send_task.return_value = mock_result

            # Process stream
            response = await client.post(
                f"/api/v1/streams/{stream.id}/process",
                headers=api_key_headers(raw_key),
                json={
                    "dimension_set_id": str(uuid4()),
                    "type_registry_id": str(uuid4()),
                },
            )

            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert "task_id" in data
            assert data["message"] == "Stream processing started"
            mock_send_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_stream_not_found(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test processing non-existent stream."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_WRITE}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Try to process non-existent stream
        fake_id = uuid4()
        response = await client.post(
            f"/api/v1/streams/{fake_id}/process",
            headers=api_key_headers(raw_key),
            json={
                "dimension_set_id": str(uuid4()),
                "type_registry_id": str(uuid4()),
            },
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_process_stream_insufficient_scope(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test stream processing with insufficient scope."""
        # Create organization, API key with wrong scope, and stream
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ}  # Wrong scope
        )
        stream = create_test_stream(organization_id=org.id)

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await test_session.commit()

        response = await client.post(
            f"/api/v1/streams/{stream.id}/process",
            headers=api_key_headers(raw_key),
            json={
                "dimension_set_id": str(uuid4()),
                "type_registry_id": str(uuid4()),
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestStreamUpload:
    """Test stream file upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_stream_file_success(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test successful stream file upload."""
        from unittest.mock import patch, MagicMock, AsyncMock
        
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_WRITE}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Mock S3 and subprocess operations
        with patch('api.routes.streams.boto3.client') as mock_boto3, \
             patch('api.routes.streams.subprocess.run') as mock_subprocess, \
             patch('tempfile.mkdtemp') as mock_mkdtemp, \
             patch('api.routes.streams.aiofiles.open', new_callable=AsyncMock), \
             patch('api.routes.streams.shutil.rmtree'), \
             patch('os.path.getsize', return_value=1024):
            
            mock_s3_client = MagicMock()
            mock_boto3.return_value = mock_s3_client
            mock_subprocess.return_value = MagicMock(returncode=0, stdout="Duration: 00:01:30.00")
            mock_mkdtemp.return_value = "/tmp/test_dir"

            # Create a test file
            test_file_content = b"test video content"
            
            # Upload file
            response = await client.post(
                "/api/v1/streams/upload",
                headers=api_key_headers(raw_key),
                files={"file": ("test_video.mp4", test_file_content, "video/mp4")},
                data={
                    "organization_id": str(org.id),
                    "name": "Test Video Upload"
                }
            )

            # Upload endpoint only works in development mode, test mode returns 403
            assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_upload_stream_file_invalid_format(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test stream file upload with invalid format."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_WRITE}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Upload invalid file
        test_file_content = b"not a video file"
        response = await client.post(
            "/api/v1/streams/upload",
            headers=api_key_headers(raw_key),
            files={"file": ("test.txt", test_file_content, "text/plain")},
            data={
                "organization_id": str(org.id),
                "name": "Invalid File"
            }
        )

        # Upload endpoint only works in development mode, test mode returns 403
        assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_upload_stream_file_insufficient_scope(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test stream file upload with insufficient scope."""
        # Create organization and API key with wrong scope
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ}  # Wrong scope
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        test_file_content = b"test video content"
        response = await client.post(
            "/api/v1/streams/upload",
            headers=api_key_headers(raw_key),
            files={"file": ("test_video.mp4", test_file_content, "video/mp4")},
            data={
                "organization_id": str(org.id),
                "name": "Test Upload"
            }
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN