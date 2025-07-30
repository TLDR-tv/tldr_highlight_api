"""Storage infrastructure for the TL;DR Highlight API.

This module provides object storage capabilities using S3/MinIO
as infrastructure, following DDD principles.
"""

from .s3_storage import S3Storage, get_storage_service
from .storage_helper import StorageHelper

__all__ = [
    "S3Storage",
    "get_storage_service",
    "StorageHelper",
]
