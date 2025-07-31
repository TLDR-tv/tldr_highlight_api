"""Security-related dependencies for FastAPI.

This module provides dependency injection for security components.
"""

from typing import Annotated
from fastapi import Depends
import redis.asyncio as redis

from src.infrastructure.security.url_signer import URLSigner
from src.infrastructure.security.config import SecurityConfig
from src.infrastructure.security.key_service import KeyGenerationService
from src.domain.repositories.organization_key_repository import (
    OrganizationKeyRepository,
)
from src.api.dependencies.repositories import get_organization_key_repository
from src.api.dependencies.cache import get_redis_client
from src.infrastructure.config import settings


async def get_security_config() -> SecurityConfig:
    """Get security configuration.

    Returns:
        SecurityConfig instance
    """
    # In a real implementation, this would load from environment or config file
    return SecurityConfig(
        master_signing_key=settings.jwt_secret_key,
        jwt_default_algorithm="HS256",
        jwt_issuer="tldr-api",
    )


async def get_key_generation_service(
    security_config: Annotated[SecurityConfig, Depends(get_security_config)],
) -> KeyGenerationService:
    """Get key generation service.

    Args:
        security_config: Security configuration

    Returns:
        KeyGenerationService instance
    """
    # Master encryption key should come from secure storage
    # For now, using a derived key from the JWT secret
    master_encryption_key = settings.jwt_secret_key.get_secret_value() + "_encryption"

    return KeyGenerationService(
        security_config=security_config, master_encryption_key=master_encryption_key
    )


async def get_url_signer(
    security_config: Annotated[SecurityConfig, Depends(get_security_config)],
    key_repository: Annotated[
        OrganizationKeyRepository, Depends(get_organization_key_repository)
    ],
    key_service: Annotated[KeyGenerationService, Depends(get_key_generation_service)],
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)],
) -> URLSigner:
    """Get URL signer instance.

    Args:
        security_config: Security configuration
        key_repository: Organization key repository
        key_service: Key generation service
        redis_client: Redis client

    Returns:
        URLSigner instance
    """
    return URLSigner(
        security_config=security_config,
        key_repository=key_repository,
        key_service=key_service,
        redis_client=redis_client,
    )
