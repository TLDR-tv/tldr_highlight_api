"""S3/MinIO storage implementation.

This module provides S3-compatible object storage as infrastructure,
using Pythonic patterns and clean separation from business logic.
"""

import asyncio
import hashlib
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, Optional, Union, Dict, List
from dataclasses import dataclass
from datetime import datetime

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class StorageObject:
    """Information about a stored object."""
    key: str
    size: int
    last_modified: datetime
    etag: str
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class PresignedUrl:
    """Presigned URL information."""
    url: str
    expires_at: datetime
    method: str


class S3Storage:
    """S3/MinIO storage service with async operations and retry logic.
    
    This class provides infrastructure-level storage operations,
    keeping storage concerns separate from business logic.
    """

    def __init__(self):
        """Initialize S3 storage with connection configuration."""
        self._session: Optional[aioboto3.Session] = None
        self._config = Config(
            region_name=settings.s3_region,
            signature_version="s3v4",
            retries={
                "max_attempts": 3,
                "mode": "adaptive",
            },
            max_pool_connections=50,
        )
        self._lock = asyncio.Lock()

    async def _get_session(self) -> aioboto3.Session:
        """Get or create aioboto3 session."""
        async with self._lock:
            if not self._session:
                self._session = aioboto3.Session()
            return self._session

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[Any, None]:
        """Get S3 client with connection management.
        
        This context manager ensures proper resource cleanup
        following Pythonic patterns.
        """
        session = await self._get_session()

        async with session.client(
            "s3",
            endpoint_url=str(settings.s3_endpoint_url) if settings.s3_endpoint_url else None,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            config=self._config,
        ) as client:
            yield client

    async def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """Ensure S3 bucket exists, creating if necessary.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            True if bucket exists or was created successfully
        """
        try:
            async with self._get_client() as client:
                # Check if bucket exists
                try:
                    await client.head_bucket(Bucket=bucket_name)
                    logger.info(f"Bucket {bucket_name} already exists")
                    return True
                except ClientError as e:
                    if e.response["Error"]["Code"] != "404":
                        raise

                # Create bucket
                await self._create_bucket(client, bucket_name)
                logger.info(f"Created bucket {bucket_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to ensure bucket {bucket_name}: {e}")
            return False

    async def _create_bucket(self, client: Any, bucket_name: str) -> None:
        """Create S3 bucket with proper configuration."""
        if settings.s3_region == "us-east-1":
            await client.create_bucket(Bucket=bucket_name)
        else:
            await client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    "LocationConstraint": settings.s3_region
                },
            )

        # Enable versioning
        await client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={"Status": "Enabled"}
        )

        # Set lifecycle for temp buckets
        if "temp" in bucket_name:
            await self._set_temp_bucket_lifecycle(client, bucket_name)

    async def _set_temp_bucket_lifecycle(self, client: Any, bucket_name: str) -> None:
        """Set lifecycle policy for temporary buckets."""
        await client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration={
                "Rules": [{
                    "ID": "delete-temp-files",
                    "Status": "Enabled",
                    "Expiration": {"Days": 7},
                    "Filter": {"Prefix": ""},
                }]
            },
        )

    async def upload(
        self,
        data: Union[bytes, BinaryIO],
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Upload data to S3.
        
        Args:
            data: File data as bytes or file-like object
            bucket: Bucket name
            key: Object key
            content_type: MIME type
            metadata: Object metadata
            tags: Object tags
            
        Returns:
            Object URL if successful, None otherwise
        """
        try:
            # Ensure data is bytes
            if hasattr(data, "read"):
                data = data.read()

            # Auto-detect content type
            if not content_type:
                content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"

            # Calculate MD5 for integrity
            import base64
            md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()

            async with self._get_client() as client:
                # Build upload arguments
                upload_args = {
                    "Bucket": bucket,
                    "Key": key,
                    "Body": data,
                    "ContentType": content_type,
                    "ContentMD5": md5_hash,
                }

                if metadata:
                    upload_args["Metadata"] = metadata

                if tags:
                    tag_string = "&".join(f"{k}={v}" for k, v in tags.items())
                    upload_args["Tagging"] = tag_string

                # Upload
                response = await client.put_object(**upload_args)

                # Generate URL
                url = self._generate_object_url(bucket, key)
                
                logger.info(
                    f"Uploaded {len(data)} bytes to {bucket}/{key} "
                    f"(ETag: {response.get('ETag')})"
                )
                return url

        except Exception as e:
            logger.error(f"Failed to upload to {bucket}/{key}: {e}")
            return None

    async def download(self, bucket: str, key: str) -> Optional[bytes]:
        """Download data from S3.
        
        Args:
            bucket: Bucket name
            key: Object key
            
        Returns:
            File data as bytes if successful, None otherwise
        """
        try:
            async with self._get_client() as client:
                response = await client.get_object(Bucket=bucket, Key=key)
                data = await response["Body"].read()
                logger.info(f"Downloaded {len(data)} bytes from {bucket}/{key}")
                return data

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Object not found: {bucket}/{key}")
            else:
                logger.error(f"Failed to download from {bucket}/{key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to download from {bucket}/{key}: {e}")
            return None

    async def delete(self, bucket: str, key: str) -> bool:
        """Delete object from S3.
        
        Args:
            bucket: Bucket name
            key: Object key
            
        Returns:
            True if successful
        """
        try:
            async with self._get_client() as client:
                await client.delete_object(Bucket=bucket, Key=key)
                logger.info(f"Deleted {bucket}/{key}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete {bucket}/{key}: {e}")
            return False

    async def exists(self, bucket: str, key: str) -> bool:
        """Check if object exists in S3.
        
        Args:
            bucket: Bucket name
            key: Object key
            
        Returns:
            True if object exists
        """
        try:
            async with self._get_client() as client:
                await client.head_object(Bucket=bucket, Key=key)
                return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Failed to check existence of {bucket}/{key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to check existence of {bucket}/{key}: {e}")
            return False

    async def get_object_info(self, bucket: str, key: str) -> Optional[StorageObject]:
        """Get object metadata from S3.
        
        Args:
            bucket: Bucket name
            key: Object key
            
        Returns:
            StorageObject with metadata if successful
        """
        try:
            async with self._get_client() as client:
                response = await client.head_object(Bucket=bucket, Key=key)
                
                return StorageObject(
                    key=key,
                    size=response.get("ContentLength", 0),
                    last_modified=response.get("LastModified"),
                    etag=response.get("ETag", "").strip('"'),
                    content_type=response.get("ContentType"),
                    metadata=response.get("Metadata", {}),
                )

        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                logger.error(f"Failed to get info for {bucket}/{key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get info for {bucket}/{key}: {e}")
            return None

    async def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        limit: int = 1000,
    ) -> List[StorageObject]:
        """List objects in S3 bucket.
        
        Args:
            bucket: Bucket name
            prefix: Optional prefix to filter objects
            limit: Maximum number of objects to return
            
        Returns:
            List of StorageObject instances
        """
        objects = []
        try:
            async with self._get_client() as client:
                paginator = client.get_paginator("list_objects_v2")

                config = {"Bucket": bucket, "MaxKeys": min(limit, 1000)}
                if prefix:
                    config["Prefix"] = prefix

                async for page in paginator.paginate(**config):
                    for obj in page.get("Contents", []):
                        objects.append(StorageObject(
                            key=obj["Key"],
                            size=obj["Size"],
                            last_modified=obj["LastModified"],
                            etag=obj["ETag"].strip('"'),
                        ))

                        if len(objects) >= limit:
                            break

                    if len(objects) >= limit:
                        break

                logger.info(f"Listed {len(objects)} objects from {bucket}/{prefix or ''}")

        except Exception as e:
            logger.error(f"Failed to list objects in {bucket}/{prefix or ''}: {e}")

        return objects

    async def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        expiration: int = 3600,
        method: str = "GET",
    ) -> Optional[PresignedUrl]:
        """Generate presigned URL for direct access.
        
        Args:
            bucket: Bucket name
            key: Object key
            expiration: URL expiration in seconds
            method: HTTP method (GET or PUT)
            
        Returns:
            PresignedUrl if successful
        """
        try:
            async with self._get_client() as client:
                client_method = "get_object" if method == "GET" else "put_object"
                
                url = await client.generate_presigned_url(
                    ClientMethod=client_method,
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expiration,
                )
                
                expires_at = datetime.utcnow().timestamp() + expiration
                
                logger.info(
                    f"Generated presigned {method} URL for {bucket}/{key} "
                    f"(expires in {expiration}s)"
                )
                
                return PresignedUrl(
                    url=url,
                    expires_at=datetime.fromtimestamp(expires_at),
                    method=method
                )

        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {bucket}/{key}: {e}")
            return None

    async def copy(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> bool:
        """Copy object within or between buckets.
        
        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key
            
        Returns:
            True if successful
        """
        try:
            async with self._get_client() as client:
                copy_source = {"Bucket": source_bucket, "Key": source_key}
                await client.copy_object(
                    CopySource=copy_source,
                    Bucket=dest_bucket,
                    Key=dest_key,
                )
                logger.info(
                    f"Copied {source_bucket}/{source_key} to {dest_bucket}/{dest_key}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to copy object: {e}")
            return False

    async def move(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> bool:
        """Move object within or between buckets.
        
        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key
            
        Returns:
            True if successful
        """
        if await self.copy(source_bucket, source_key, dest_bucket, dest_key):
            return await self.delete(source_bucket, source_key)
        return False

    def _generate_object_url(self, bucket: str, key: str) -> str:
        """Generate public URL for an object."""
        if settings.s3_endpoint_url:
            return f"{str(settings.s3_endpoint_url)}/{bucket}/{key}"
        else:
            return f"https://{bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"


# Singleton instance management
_storage_instance: Optional[S3Storage] = None


def get_storage_service() -> S3Storage:
    """Get or create the global storage service instance.
    
    This follows the singleton pattern for connection reuse.
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = S3Storage()
    return _storage_instance