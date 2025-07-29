"""S3/MinIO storage service for the TL;DR Highlight API."""

import asyncio
import hashlib
import logging
import mimetypes
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, Optional, Union
from urllib.parse import urlparse

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from src.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """S3/MinIO storage service with async operations and retry logic."""

    def __init__(self):
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
        """Get S3 client with connection management."""
        session = await self._get_session()

        async with session.client(
            "s3",
            endpoint_url=str(settings.s3_endpoint_url)
            if settings.s3_endpoint_url
            else None,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            config=self._config,
        ) as client:
            yield client

    async def create_bucket(self, bucket_name: str) -> bool:
        """Create S3 bucket if it doesn't exist."""
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
                if settings.s3_region == "us-east-1":
                    await client.create_bucket(Bucket=bucket_name)
                else:
                    await client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={
                            "LocationConstraint": settings.s3_region
                        },
                    )

                # Set bucket versioning
                await client.put_bucket_versioning(
                    Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
                )

                # Set bucket lifecycle policy for temp files
                if "temp" in bucket_name:
                    await client.put_bucket_lifecycle_configuration(
                        Bucket=bucket_name,
                        LifecycleConfiguration={
                            "Rules": [
                                {
                                    "ID": "delete-temp-files",
                                    "Status": "Enabled",
                                    "Expiration": {"Days": 7},
                                    "Filter": {"Prefix": ""},
                                }
                            ]
                        },
                    )

                logger.info(f"Created bucket {bucket_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
            return False

    async def upload_file(
        self,
        file_data: Union[bytes, BinaryIO],
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Upload file to S3/MinIO.

        Args:
            file_data: File data as bytes or file-like object
            bucket: Bucket name
            key: Object key
            content_type: MIME type
            metadata: Object metadata
            tags: Object tags

        Returns:
            Object URL if successful, None otherwise
        """
        try:
            # Ensure file_data is bytes
            if hasattr(file_data, "read"):
                file_data = file_data.read()

            # Auto-detect content type
            if not content_type:
                content_type = (
                    mimetypes.guess_type(key)[0] or "application/octet-stream"
                )

            # Calculate MD5 for integrity check (base64 encoded for AWS)
            import base64

            md5_hash = base64.b64encode(hashlib.md5(file_data).digest()).decode()

            async with self._get_client() as client:
                # Prepare upload arguments
                upload_args = {
                    "Bucket": bucket,
                    "Key": key,
                    "Body": file_data,
                    "ContentType": content_type,
                    "ContentMD5": md5_hash,
                }

                if metadata:
                    upload_args["Metadata"] = metadata

                if tags:
                    tag_string = "&".join(f"{k}={v}" for k, v in tags.items())
                    upload_args["Tagging"] = tag_string

                # Upload file
                response = await client.put_object(**upload_args)

                # Generate URL
                if settings.s3_endpoint_url:
                    url = f"{str(settings.s3_endpoint_url)}/{bucket}/{key}"
                else:
                    url = (
                        f"https://{bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"
                    )

                logger.info(
                    f"Uploaded file to {bucket}/{key} (ETag: {response.get('ETag')})"
                )
                return url

        except Exception as e:
            logger.error(f"Failed to upload file to {bucket}/{key}: {e}")
            return None

    async def download_file(
        self,
        bucket: str,
        key: str,
    ) -> Optional[bytes]:
        """Download file from S3/MinIO."""
        try:
            async with self._get_client() as client:
                response = await client.get_object(Bucket=bucket, Key=key)
                data = await response["Body"].read()
                logger.info(f"Downloaded file from {bucket}/{key} ({len(data)} bytes)")
                return data

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"File not found: {bucket}/{key}")
            else:
                logger.error(f"Failed to download file from {bucket}/{key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to download file from {bucket}/{key}: {e}")
            return None

    async def delete_file(
        self,
        bucket: str,
        key: str,
    ) -> bool:
        """Delete file from S3/MinIO."""
        try:
            async with self._get_client() as client:
                await client.delete_object(Bucket=bucket, Key=key)
                logger.info(f"Deleted file {bucket}/{key}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete file {bucket}/{key}: {e}")
            return False

    async def file_exists(
        self,
        bucket: str,
        key: str,
    ) -> bool:
        """Check if file exists in S3/MinIO."""
        try:
            async with self._get_client() as client:
                await client.head_object(Bucket=bucket, Key=key)
                return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Failed to check file existence {bucket}/{key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to check file existence {bucket}/{key}: {e}")
            return False

    async def get_file_info(
        self,
        bucket: str,
        key: str,
    ) -> Optional[dict[str, Any]]:
        """Get file metadata from S3/MinIO."""
        try:
            async with self._get_client() as client:
                response = await client.head_object(Bucket=bucket, Key=key)
                return {
                    "size": response.get("ContentLength"),
                    "content_type": response.get("ContentType"),
                    "last_modified": response.get("LastModified"),
                    "etag": response.get("ETag"),
                    "metadata": response.get("Metadata", {}),
                }

        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                logger.error(f"Failed to get file info {bucket}/{key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get file info {bucket}/{key}: {e}")
            return None

    async def list_buckets(self) -> list[str]:
        """List all buckets."""
        try:
            async with self._get_client() as client:
                response = await client.list_buckets()
                buckets = [bucket["Name"] for bucket in response.get("Buckets", [])]
                logger.info(f"Listed {len(buckets)} buckets")
                return buckets
        except Exception as e:
            logger.error(f"Failed to list buckets: {e}")
            return []

    async def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """List objects in S3/MinIO bucket."""
        files = []
        try:
            async with self._get_client() as client:
                paginator = client.get_paginator("list_objects_v2")

                page_config = {"Bucket": bucket, "MaxKeys": limit}
                if prefix:
                    page_config["Prefix"] = prefix

                async for page in paginator.paginate(**page_config):
                    for obj in page.get("Contents", []):
                        files.append(
                            {
                                "key": obj["Key"],
                                "size": obj["Size"],
                                "last_modified": obj["LastModified"],
                                "etag": obj["ETag"],
                            }
                        )

                        if len(files) >= limit:
                            break

                    if len(files) >= limit:
                        break

                logger.info(f"Listed {len(files)} objects from {bucket}/{prefix or ''}")

        except Exception as e:
            logger.error(f"Failed to list objects in {bucket}/{prefix or ''}: {e}")

        return files

    async def list_files(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """List files in S3/MinIO bucket."""
        files = []
        try:
            async with self._get_client() as client:
                paginator = client.get_paginator("list_objects_v2")

                page_config = {"Bucket": bucket, "MaxKeys": max_keys}
                if prefix:
                    page_config["Prefix"] = prefix

                async for page in paginator.paginate(**page_config):
                    for obj in page.get("Contents", []):
                        files.append(
                            {
                                "key": obj["Key"],
                                "size": obj["Size"],
                                "last_modified": obj["LastModified"],
                                "etag": obj["ETag"],
                            }
                        )

                        if len(files) >= max_keys:
                            break

                    if len(files) >= max_keys:
                        break

                logger.info(f"Listed {len(files)} files from {bucket}/{prefix or ''}")

        except Exception as e:
            logger.error(f"Failed to list files in {bucket}/{prefix or ''}: {e}")

        return files

    async def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> Optional[str]:
        """
        Generate presigned URL for direct access.

        Args:
            bucket: Bucket name
            key: Object key
            expiration: URL expiration in seconds
            method: S3 method (get_object or put_object)

        Returns:
            Presigned URL
        """
        try:
            async with self._get_client() as client:
                url = await client.generate_presigned_url(
                    ClientMethod=method,
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expiration,
                )
                logger.info(
                    f"Generated presigned URL for {bucket}/{key} (expires in {expiration}s)"
                )
                return url

        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {bucket}/{key}: {e}")
            return None

    async def copy_file(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> bool:
        """Copy file within or between buckets."""
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
            logger.error(f"Failed to copy file: {e}")
            return False

    async def move_file(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
    ) -> bool:
        """Move file within or between buckets."""
        if await self.copy_file(source_bucket, source_key, dest_bucket, dest_key):
            return await self.delete_file(source_bucket, source_key)
        return False

    async def create_multipart_upload(
        self,
        bucket: str,
        key: str,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Initiate multipart upload for large files."""
        try:
            async with self._get_client() as client:
                args = {"Bucket": bucket, "Key": key}
                if content_type:
                    args["ContentType"] = content_type

                response = await client.create_multipart_upload(**args)
                upload_id = response["UploadId"]
                logger.info(
                    f"Created multipart upload for {bucket}/{key} (ID: {upload_id})"
                )
                return upload_id

        except Exception as e:
            logger.error(f"Failed to create multipart upload: {e}")
            return None

    async def upload_part(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> Optional[str]:
        """Upload part in multipart upload."""
        try:
            async with self._get_client() as client:
                response = await client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=data,
                )
                etag = response["ETag"]
                logger.info(f"Uploaded part {part_number} for {bucket}/{key}")
                return etag

        except Exception as e:
            logger.error(f"Failed to upload part: {e}")
            return None

    async def complete_multipart_upload(
        self,
        bucket: str,
        key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> bool:
        """Complete multipart upload."""
        try:
            async with self._get_client() as client:
                await client.complete_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
                logger.info(f"Completed multipart upload for {bucket}/{key}")
                return True

        except Exception as e:
            logger.error(f"Failed to complete multipart upload: {e}")
            return False

    async def abort_multipart_upload(
        self,
        bucket: str,
        key: str,
        upload_id: str,
    ) -> bool:
        """Abort multipart upload."""
        try:
            async with self._get_client() as client:
                await client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
                logger.info(f"Aborted multipart upload for {bucket}/{key}")
                return True

        except Exception as e:
            logger.error(f"Failed to abort multipart upload: {e}")
            return False


# Global storage instance
storage_service = StorageService()


class StorageHelper:
    """Helper class for common storage operations."""

    @staticmethod
    def generate_key(
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        filename: str,
    ) -> str:
        """Generate standardized S3 key."""
        # Clean filename
        safe_filename = Path(filename).name.replace(" ", "_")

        # Generate timestamp
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")

        # Build key
        return f"{tenant_id}/{resource_type}/{timestamp}/{resource_id}/{safe_filename}"

    @staticmethod
    def parse_s3_url(url: str) -> tuple[str, str]:
        """Parse S3 URL to extract bucket and key."""
        parsed = urlparse(url)

        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        else:
            # Handle HTTP(S) URLs
            path_parts = parsed.path.lstrip("/").split("/", 1)
            if len(path_parts) == 2:
                bucket, key = path_parts
            else:
                raise ValueError(f"Invalid S3 URL: {url}")

        return bucket, key

    @staticmethod
    async def upload_from_url(
        url: str,
        bucket: str,
        key: str,
        storage_service: StorageService,
    ) -> Optional[str]:
        """Download from URL and upload to S3."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.read()
                        content_type = response.headers.get("Content-Type")

                        return await storage_service.upload_file(
                            data,
                            bucket,
                            key,
                            content_type=content_type,
                        )
            return None

        except Exception as e:
            logger.error(f"Failed to upload from URL {url}: {e}")
            return None
