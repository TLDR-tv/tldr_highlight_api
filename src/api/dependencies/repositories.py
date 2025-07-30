"""Repository dependencies for FastAPI.

This module provides dependency injection for infrastructure repositories.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db as get_db_session
from src.infrastructure.persistence.repositories.user_repository import UserRepository
from src.infrastructure.persistence.repositories.api_key_repository import (
    APIKeyRepository,
)
from src.infrastructure.persistence.repositories.organization_repository import (
    OrganizationRepository,
)
from src.infrastructure.persistence.repositories.stream_repository import (
    StreamRepository,
)
from src.infrastructure.persistence.repositories.highlight_repository import (
    HighlightRepository,
)
from src.infrastructure.persistence.repositories.webhook_repository import (
    WebhookRepository,
)
from src.infrastructure.persistence.repositories.webhook_event_repository import (
    WebhookEventRepository,
)
from src.infrastructure.persistence.repositories.usage_record_repository import (
    UsageRecordRepository,
)
from src.infrastructure.persistence.repositories.dimension_set_repository import (
    DimensionSetRepository,
)
from src.infrastructure.persistence.repositories.highlight_type_registry_repository import (
    HighlightTypeRegistryRepository,
)


# Repository Dependencies


async def get_user_repository(
    db: AsyncSession = Depends(get_db_session),
) -> UserRepository:
    """Get user repository instance."""
    return UserRepository(db)


async def get_api_key_repository(
    db: AsyncSession = Depends(get_db_session),
) -> APIKeyRepository:
    """Get API key repository instance."""
    return APIKeyRepository(db)


async def get_organization_repository(
    db: AsyncSession = Depends(get_db_session),
) -> OrganizationRepository:
    """Get organization repository instance."""
    return OrganizationRepository(db)


async def get_stream_repository(
    db: AsyncSession = Depends(get_db_session),
) -> StreamRepository:
    """Get stream repository instance."""
    return StreamRepository(db)


async def get_highlight_repository(
    db: AsyncSession = Depends(get_db_session),
) -> HighlightRepository:
    """Get highlight repository instance."""
    return HighlightRepository(db)


async def get_webhook_repository(
    db: AsyncSession = Depends(get_db_session),
) -> WebhookRepository:
    """Get webhook repository instance."""
    return WebhookRepository(db)


async def get_webhook_event_repository(
    db: AsyncSession = Depends(get_db_session),
) -> WebhookEventRepository:
    """Get webhook event repository instance."""
    return WebhookEventRepository(db)


async def get_usage_record_repository(
    db: AsyncSession = Depends(get_db_session),
) -> UsageRecordRepository:
    """Get usage record repository instance."""
    return UsageRecordRepository(db)


async def get_dimension_set_repository(
    db: AsyncSession = Depends(get_db_session),
) -> DimensionSetRepository:
    """Get dimension set repository instance."""
    return DimensionSetRepository(db)


async def get_highlight_type_registry_repository(
    db: AsyncSession = Depends(get_db_session),
) -> HighlightTypeRegistryRepository:
    """Get highlight type registry repository instance."""
    return HighlightTypeRegistryRepository(db)
