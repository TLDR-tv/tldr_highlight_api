"""Storage helper utilities.

This module provides utility functions for storage operations,
keeping them separate from the main storage implementation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StorageHelper:
    """Helper class for common storage operations.
    
    This class provides utility methods for storage-related tasks,
    following Pythonic patterns with static methods where appropriate.
    """

    @staticmethod
    def generate_key(
        organization_id: str,
        resource_type: str,
        resource_id: str,
        filename: str,
    ) -> str:
        """Generate standardized S3 key with proper organization.
        
        Args:
            organization_id: Organization/tenant identifier
            resource_type: Type of resource (e.g., 'highlights', 'streams')
            resource_id: Unique resource identifier
            filename: Original filename
            
        Returns:
            Standardized S3 key path
            
        Example:
            >>> StorageHelper.generate_key("org123", "highlights", "hl456", "clip.mp4")
            'org123/highlights/2024/01/15/hl456/clip.mp4'
        """
        # Clean filename
        safe_filename = Path(filename).name.replace(" ", "_")
        
        # Generate timestamp path
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        
        # Build hierarchical key
        return f"{organization_id}/{resource_type}/{timestamp}/{resource_id}/{safe_filename}"

    @staticmethod
    def parse_s3_url(url: str) -> Tuple[str, str]:
        """Parse S3 URL to extract bucket and key.
        
        Args:
            url: S3 URL in various formats
            
        Returns:
            Tuple of (bucket, key)
            
        Raises:
            ValueError: If URL format is invalid
            
        Examples:
            >>> StorageHelper.parse_s3_url("s3://mybucket/path/to/file.mp4")
            ('mybucket', 'path/to/file.mp4')
            
            >>> StorageHelper.parse_s3_url("https://mybucket.s3.amazonaws.com/path/to/file.mp4")
            ('mybucket', 'path/to/file.mp4')
        """
        parsed = urlparse(url)
        
        if parsed.scheme == "s3":
            # s3://bucket/key format
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif parsed.scheme in ("http", "https"):
            # Handle various S3 HTTP URL formats
            if ".s3." in parsed.netloc or ".s3-" in parsed.netloc:
                # Virtual-hosted-style: bucket.s3.region.amazonaws.com
                bucket = parsed.netloc.split(".")[0]
                key = parsed.path.lstrip("/")
            else:
                # Path-style: s3.region.amazonaws.com/bucket/key
                path_parts = parsed.path.lstrip("/").split("/", 1)
                if len(path_parts) == 2:
                    bucket, key = path_parts
                else:
                    raise ValueError(f"Invalid S3 URL format: {url}")
        else:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        
        if not bucket or not key:
            raise ValueError(f"Could not parse bucket and key from URL: {url}")
            
        return bucket, key

    @staticmethod
    def get_content_type(filename: str) -> str:
        """Get content type from filename.
        
        Args:
            filename: File name or path
            
        Returns:
            MIME content type
        """
        import mimetypes
        
        content_type = mimetypes.guess_type(filename)[0]
        
        # Additional mappings for streaming content
        extension_map = {
            ".m3u8": "application/x-mpegURL",
            ".ts": "video/MP2T",
            ".flv": "video/x-flv",
            ".webm": "video/webm",
        }
        
        ext = Path(filename).suffix.lower()
        return extension_map.get(ext, content_type or "application/octet-stream")

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for storage
        """
        import re
        
        # Get the base name without path
        name = Path(filename).name
        
        # Replace unsafe characters
        safe_name = re.sub(r'[^\w\s.-]', '_', name)
        
        # Replace multiple spaces/underscores with single underscore
        safe_name = re.sub(r'[\s_]+', '_', safe_name)
        
        # Remove leading/trailing underscores
        safe_name = safe_name.strip('_')
        
        # Ensure non-empty
        if not safe_name:
            safe_name = "unnamed_file"
            
        return safe_name

    @staticmethod
    def estimate_multipart_parts(file_size: int, part_size: int = 5 * 1024 * 1024) -> int:
        """Estimate number of parts for multipart upload.
        
        Args:
            file_size: Total file size in bytes
            part_size: Size of each part (default 5MB)
            
        Returns:
            Number of parts needed
        """
        if file_size <= 0:
            return 0
            
        parts = (file_size + part_size - 1) // part_size
        
        # S3 limits: max 10,000 parts
        if parts > 10000:
            # Increase part size to stay under limit
            min_part_size = (file_size + 9999) // 10000
            logger.warning(
                f"File size {file_size} requires part size of at least "
                f"{min_part_size} bytes to stay under 10,000 part limit"
            )
            parts = 10000
            
        return parts

    @staticmethod
    async def download_and_upload(
        source_url: str,
        storage_service: 'S3Storage',
        bucket: str,
        key: str,
    ) -> Optional[str]:
        """Download from URL and upload to S3.
        
        Args:
            source_url: URL to download from
            storage_service: S3Storage instance
            bucket: Destination bucket
            key: Destination key
            
        Returns:
            S3 URL if successful, None otherwise
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source_url) as response:
                    if response.status == 200:
                        data = await response.read()
                        content_type = response.headers.get("Content-Type")
                        
                        return await storage_service.upload(
                            data,
                            bucket,
                            key,
                            content_type=content_type,
                        )
                    else:
                        logger.error(
                            f"Failed to download from {source_url}: "
                            f"HTTP {response.status}"
                        )
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to download and upload from {source_url}: {e}")
            return None