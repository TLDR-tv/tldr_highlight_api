"""Health check endpoints for the TL;DR Highlight API.

This module provides health check endpoints to monitor the status
of the API and its dependencies (database, Redis, S3).
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.common import HealthCheckResponse, StatusResponse
from src.core.cache import cache
from src.core.config import settings
from src.core.database import get_db
from src.services.storage import storage_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Comprehensive health check",
    description="Check the health of the API and all its dependencies",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthCheckResponse:
    """Perform comprehensive health check of all services.

    Returns:
        HealthCheckResponse: Detailed health status
    """
    timestamp = datetime.utcnow()
    services = {}
    overall_status = "healthy"

    # Check all services concurrently
    health_checks = {
        "database": _check_database_health(db),
        "redis": _check_redis_health(),
        "storage": _check_storage_health(),
    }

    # Wait for all health checks to complete
    results = await asyncio.gather(*health_checks.values(), return_exceptions=True)

    # Process results
    for service_name, result in zip(health_checks.keys(), results):
        if isinstance(result, Exception):
            services[service_name] = {
                "status": "unhealthy",
                "error": str(result),
                "response_time_ms": None,
            }
            overall_status = "unhealthy"
        else:
            services[service_name] = result
            if result["status"] != "healthy":
                overall_status = (
                    "degraded" if overall_status == "healthy" else "unhealthy"
                )

    # Add API service info
    services["api"] = {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
        "response_time_ms": 0,  # Immediate response
    }

    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        timestamp=timestamp,
        services=services,
    )


@router.get(
    "/live",
    response_model=StatusResponse,
    summary="Liveness probe",
    description="Simple liveness check (API is running)",
)
async def liveness_check() -> StatusResponse:
    """Liveness probe for Kubernetes/Docker health checks.

    Returns:
        StatusResponse: Simple status response
    """
    return StatusResponse(status="alive", timestamp=datetime.utcnow())


@router.get(
    "/ready",
    response_model=StatusResponse,
    summary="Readiness probe",
    description="Check if API is ready to serve requests",
)
async def readiness_check(db: AsyncSession = Depends(get_db)) -> StatusResponse:
    """Readiness probe for Kubernetes/Docker health checks.

    Args:
        db: Database session

    Returns:
        StatusResponse: Readiness status

    Raises:
        HTTPException: If not ready to serve requests
    """
    # Check critical dependencies
    try:
        # Quick database check
        await db.execute(text("SELECT 1"))

        # Quick Redis check
        async with cache.get_client() as client:
            await client.ping()

        return StatusResponse(status="ready", timestamp=datetime.utcnow())

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready"
        )


@router.get(
    "/database",
    response_model=Dict[str, Any],
    summary="Database health check",
    description="Detailed database health information",
)
async def database_health(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get detailed database health information.

    Args:
        db: Database session

    Returns:
        Dict: Database health details
    """
    return await _check_database_health(db)


@router.get(
    "/redis",
    response_model=Dict[str, Any],
    summary="Redis health check",
    description="Detailed Redis health information",
)
async def redis_health() -> Dict[str, Any]:
    """Get detailed Redis health information.

    Returns:
        Dict: Redis health details
    """
    return await _check_redis_health()


@router.get(
    "/storage",
    response_model=Dict[str, Any],
    summary="Storage health check",
    description="Detailed storage (S3) health information",
)
async def storage_health() -> Dict[str, Any]:
    """Get detailed storage health information.

    Returns:
        Dict: Storage health details
    """
    return await _check_storage_health()


async def _check_database_health(db: AsyncSession) -> Dict[str, Any]:
    """Check database health and return detailed information.

    Args:
        db: Database session

    Returns:
        Dict: Database health information
    """
    start_time = datetime.utcnow()

    try:
        # Test basic connectivity
        await db.execute(text("SELECT 1"))

        # Test query performance
        result = await db.execute(
            text("SELECT version(), current_database(), current_user")
        )
        db_info = result.fetchone()

        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "database": db_info[1] if db_info else "unknown",
            "user": db_info[2] if db_info else "unknown",
            "version": db_info[0].split()[0] if db_info else "unknown",
            "connection_info": {
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
            },
        }

    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Database health check failed: {e}")

        return {
            "status": "unhealthy",
            "response_time_ms": round(response_time, 2),
            "error": str(e),
            "error_type": type(e).__name__,
        }


async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis health and return detailed information.

    Returns:
        Dict: Redis health information
    """
    start_time = datetime.utcnow()

    try:
        async with cache.get_client() as client:
            # Test basic connectivity
            await client.ping()

            # Get Redis info
            info = await client.info()

            # Test basic operations
            test_key = "health_check_test"
            await client.set(test_key, "test_value", ex=10)
            test_value = await client.get(test_key)
            await client.delete(test_key)

            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "version": info.get("redis_version", "unknown"),
                "mode": info.get("redis_mode", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "operations_test": "passed"
                if test_value == b"test_value"
                else "failed",
            }

    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Redis health check failed: {e}")

        return {
            "status": "unhealthy",
            "response_time_ms": round(response_time, 2),
            "error": str(e),
            "error_type": type(e).__name__,
        }


async def _check_storage_health() -> Dict[str, Any]:
    """Check S3 storage health and return detailed information.

    Returns:
        Dict: Storage health information
    """
    start_time = datetime.utcnow()

    try:
        # Test S3 connectivity by listing buckets
        buckets = await storage_service.list_buckets()

        # Test bucket access for key buckets
        bucket_status = {}
        key_buckets = [
            settings.s3_highlights_bucket,
            settings.s3_thumbnails_bucket,
            settings.s3_temp_bucket,
        ]

        for bucket_name in key_buckets:
            try:
                # Test bucket access (list objects with limit)
                await storage_service.list_objects(bucket_name, limit=1)
                bucket_status[bucket_name] = "accessible"
            except Exception as bucket_error:
                bucket_status[bucket_name] = f"error: {str(bucket_error)}"

        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "region": settings.s3_region,
            "endpoint": str(settings.s3_endpoint_url)
            if settings.s3_endpoint_url
            else "default",
            "buckets_total": len(buckets),
            "key_buckets": bucket_status,
        }

    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Storage health check failed: {e}")

        return {
            "status": "unhealthy",
            "response_time_ms": round(response_time, 2),
            "error": str(e),
            "error_type": type(e).__name__,
        }
