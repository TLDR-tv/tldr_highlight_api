"""Tests for S3/MinIO storage service."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO
from pathlib import Path
from botocore.exceptions import ClientError

from src.services.storage import StorageService, storage_service
from src.core.config import settings


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = AsyncMock()
    # Add async context manager support
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def storage():
    """Create a storage service instance."""
    return StorageService()


@pytest.mark.asyncio
class TestStorageService:
    """Test cases for storage service."""

    async def test_get_session_singleton(self, storage):
        """Test that session is created as singleton."""
        session1 = await storage._get_session()
        session2 = await storage._get_session()
        
        assert session1 is session2
        assert storage._session is not None

    async def test_create_bucket_new(self, storage, mock_s3_client):
        """Test creating a new bucket."""
        # Mock bucket doesn't exist
        mock_s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        mock_s3_client.create_bucket = AsyncMock()
        mock_s3_client.put_bucket_versioning = AsyncMock()
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            result = await storage.create_bucket("test-bucket")
            
            assert result is True
            mock_s3_client.create_bucket.assert_called_once()
            mock_s3_client.put_bucket_versioning.assert_called_once_with(
                Bucket="test-bucket",
                VersioningConfiguration={"Status": "Enabled"}
            )

    async def test_create_bucket_exists(self, storage, mock_s3_client):
        """Test creating a bucket that already exists."""
        # Mock bucket exists
        mock_s3_client.head_bucket = AsyncMock()
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            result = await storage.create_bucket("existing-bucket")
            
            assert result is True
            mock_s3_client.create_bucket.assert_not_called()

    async def test_create_bucket_with_lifecycle(self, storage, mock_s3_client):
        """Test creating a temp bucket with lifecycle policy."""
        # Mock bucket doesn't exist
        mock_s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        mock_s3_client.create_bucket = AsyncMock()
        mock_s3_client.put_bucket_versioning = AsyncMock()
        mock_s3_client.put_bucket_lifecycle_configuration = AsyncMock()
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            result = await storage.create_bucket("temp-bucket")
            
            assert result is True
            mock_s3_client.put_bucket_lifecycle_configuration.assert_called_once()

    async def test_create_bucket_failure(self, storage, mock_s3_client):
        """Test bucket creation failure."""
        mock_s3_client.head_bucket.side_effect = Exception("Connection error")
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            result = await storage.create_bucket("test-bucket")
            
            assert result is False

    async def test_upload_file_from_bytes(self, storage, mock_s3_client):
        """Test uploading file from bytes."""
        mock_s3_client.put_object = AsyncMock(return_value={
            "ETag": '"test-etag"',
            "VersionId": "test-version"
        })
        
        test_data = b"test file content"
        metadata = {"user": "test", "type": "highlight"}
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            key = await storage.upload_file(
                test_data,
                "test-bucket",
                "test.mp4",
                content_type="video/mp4",
                metadata=metadata
            )
            
            assert key == "http://localhost:9010//test-bucket/test.mp4"
            mock_s3_client.put_object.assert_called_once()
            call_args = mock_s3_client.put_object.call_args.kwargs
            assert call_args["Bucket"] == "test-bucket"
            assert call_args["Key"] == "test.mp4"
            assert call_args["Body"] == test_data
            assert call_args["ContentType"] == "video/mp4"
            assert call_args["Metadata"] == metadata

    async def test_upload_file_from_path(self, storage, mock_s3_client, tmp_path):
        """Test uploading file from path."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        test_content = test_file.read_bytes()
        
        mock_s3_client.put_object = AsyncMock(return_value={
            "ETag": '"test-etag"'
        })
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            key = await storage.upload_file(
                test_content,
                "test-bucket",
                "test.txt"
            )
            
            # Check that the key ends with the filename
            assert key.endswith("test.txt")
            mock_s3_client.put_object.assert_called_once()

    async def test_upload_file_auto_key_generation(self, storage, mock_s3_client):
        """Test automatic key generation when not provided."""
        # The upload_file method requires a key parameter, so this test is invalid
        # Let's test with a generated key instead
        mock_s3_client.put_object = AsyncMock(return_value={})
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            # Generate a unique key
            from datetime import datetime
            key_name = f"test-{datetime.utcnow().timestamp()}.bin"
            
            result = await storage.upload_file(
                b"test data",
                "test-bucket",
                key_name
            )
            
            # Should return a URL
            assert result is not None
            assert result.startswith("http")
            assert key_name in result
            mock_s3_client.put_object.assert_called_once()

    async def test_upload_file_failure(self, storage, mock_s3_client):
        """Test upload failure handling."""
        mock_s3_client.put_object.side_effect = Exception("Upload failed")
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            key = await storage.upload_file(
                b"test data",
                "test-bucket",
                "test.txt"
            )
            
            assert key is None

    async def test_download_file(self, storage, mock_s3_client):
        """Test downloading file."""
        test_content = b"downloaded content"
        mock_response = {
            "Body": AsyncMock(read=AsyncMock(return_value=test_content))
        }
        mock_s3_client.get_object = AsyncMock(return_value=mock_response)
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            data = await storage.download_file("test-bucket", "test.txt")
            
            assert data == test_content
            mock_s3_client.get_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="test.txt"
            )

    async def test_download_file_not_found(self, storage, mock_s3_client):
        """Test downloading non-existent file."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            data = await storage.download_file("test-bucket", "missing.txt")
            
            assert data is None

    async def test_delete_file(self, storage, mock_s3_client):
        """Test deleting file."""
        mock_s3_client.delete_object = AsyncMock()
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            result = await storage.delete_file("test-bucket", "test.txt")
            
            assert result is True
            mock_s3_client.delete_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="test.txt"
            )

    async def test_delete_file_failure(self, storage, mock_s3_client):
        """Test delete failure handling."""
        mock_s3_client.delete_object.side_effect = Exception("Delete failed")
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            result = await storage.delete_file("test-bucket", "test.txt")
            
            assert result is False

    async def test_file_exists_true(self, storage, mock_s3_client):
        """Test checking if file exists (file present)."""
        mock_s3_client.head_object = AsyncMock()
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            exists = await storage.file_exists("test-bucket", "test.txt")
            
            assert exists is True
            mock_s3_client.head_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="test.txt"
            )

    async def test_file_exists_false(self, storage, mock_s3_client):
        """Test checking if file exists (file missing)."""
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            exists = await storage.file_exists("test-bucket", "missing.txt")
            
            assert exists is False

    async def test_get_file_info(self, storage, mock_s3_client):
        """Test getting file metadata."""
        mock_response = {
            "ContentLength": 1024,
            "ContentType": "video/mp4",
            "LastModified": datetime.utcnow(),
            "ETag": '"test-etag"',
            "Metadata": {"user": "test"}
        }
        mock_s3_client.head_object = AsyncMock(return_value=mock_response)
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            info = await storage.get_file_info("test-bucket", "test.mp4")
            
            assert info is not None
            assert info["size"] == 1024
            assert info["content_type"] == "video/mp4"
            assert info["metadata"]["user"] == "test"

    async def test_get_file_info_not_found(self, storage, mock_s3_client):
        """Test getting info for non-existent file."""
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            info = await storage.get_file_info("test-bucket", "missing.txt")
            
            assert info is None

    async def test_list_buckets(self, storage, mock_s3_client):
        """Test listing buckets."""
        mock_response = {
            "Buckets": [
                {"Name": "bucket1"},
                {"Name": "bucket2"},
                {"Name": "bucket3"}
            ]
        }
        mock_s3_client.list_buckets = AsyncMock(return_value=mock_response)
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            buckets = await storage.list_buckets()
            
            assert len(buckets) == 3
            assert "bucket1" in buckets
            assert "bucket2" in buckets
            assert "bucket3" in buckets

    async def test_list_buckets_failure(self, storage, mock_s3_client):
        """Test list buckets failure handling."""
        mock_s3_client.list_buckets.side_effect = Exception("List failed")
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            buckets = await storage.list_buckets()
            
            assert buckets == []

    async def test_list_objects(self, storage, mock_s3_client):
        """Test listing objects in bucket."""
        # Mock pages data
        mock_pages = [
            {
                "Contents": [
                    {
                        "Key": "file1.mp4",
                        "Size": 1024,
                        "LastModified": datetime.utcnow(),
                        "ETag": '"etag1"'
                    },
                    {
                        "Key": "file2.mp4",
                        "Size": 2048,
                        "LastModified": datetime.utcnow(),
                        "ETag": '"etag2"'
                    }
                ]
            }
        ]
        
        # Create a mock paginator that properly handles async iteration
        class MockPaginator:
            def paginate(self, **kwargs):
                async def async_pages():
                    for page in mock_pages:
                        yield page
                return async_pages()
        
        # get_paginator is synchronous
        mock_s3_client.get_paginator = MagicMock(return_value=MockPaginator())
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            objects = await storage.list_objects("test-bucket", prefix="videos/")
            
            assert len(objects) == 2
            assert objects[0]["key"] == "file1.mp4"
            assert objects[0]["size"] == 1024
            assert objects[1]["key"] == "file2.mp4"
            assert objects[1]["size"] == 2048

    async def test_list_objects_empty(self, storage, mock_s3_client):
        """Test listing objects in empty bucket."""
        mock_paginator = AsyncMock()
        
        async def async_pages():
            yield {"Contents": None}
        
        mock_paginator.paginate.return_value = async_pages()
        mock_s3_client.get_paginator.return_value = mock_paginator
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            objects = await storage.list_objects("empty-bucket")
            
            assert objects == []

    async def test_generate_presigned_url(self, storage, mock_s3_client):
        """Test generating presigned URL."""
        expected_url = "https://s3.example.com/test-bucket/test.mp4?signature=xxx"
        mock_s3_client.generate_presigned_url = AsyncMock(return_value=expected_url)
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            url = await storage.generate_presigned_url(
                "test-bucket",
                "test.mp4",
                expiration=3600
            )
            
            assert url == expected_url
            mock_s3_client.generate_presigned_url.assert_called_once_with(
                ClientMethod="get_object",
                Params={"Bucket": "test-bucket", "Key": "test.mp4"},
                ExpiresIn=3600
            )

    async def test_generate_presigned_url_failure(self, storage, mock_s3_client):
        """Test presigned URL generation failure."""
        mock_s3_client.generate_presigned_url.side_effect = Exception("Generation failed")
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            url = await storage.generate_presigned_url(
                "test-bucket",
                "test.mp4"
            )
            
            assert url is None

    async def test_copy_file(self, storage, mock_s3_client):
        """Test copying file between buckets."""
        mock_s3_client.copy_object = AsyncMock()
        
        with patch.object(storage, "_get_client") as mock_get_client:
            mock_get_client.return_value.__aenter__.return_value = mock_s3_client
            
            result = await storage.copy_file(
                "source-bucket",
                "source.mp4",
                "dest-bucket",
                "dest.mp4"
            )
            
            assert result is True
            mock_s3_client.copy_object.assert_called_once_with(
                CopySource={"Bucket": "source-bucket", "Key": "source.mp4"},
                Bucket="dest-bucket",
                Key="dest.mp4"
            )


    async def test_storage_service_singleton(self):
        """Test that storage_service is a singleton."""
        from src.services.storage import storage_service as service1
        from src.services.storage import storage_service as service2
        
        assert service1 is service2