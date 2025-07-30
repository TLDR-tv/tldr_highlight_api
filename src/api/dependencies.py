"""Common dependencies for API endpoints."""

from typing import Annotated, Optional, Dict

from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from src.core.database import get_db as get_db_session
from src.core.config import settings
from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.user import User

# Import repositories
from src.infrastructure.persistence.repositories.user_repository import UserRepository
from src.infrastructure.persistence.repositories.api_key_repository import APIKeyRepository
from src.infrastructure.persistence.repositories.organization_repository import OrganizationRepository
from src.infrastructure.persistence.repositories.stream_repository import StreamRepository
from src.infrastructure.persistence.repositories.highlight_repository import HighlightRepository
from src.infrastructure.persistence.repositories.batch_repository import BatchRepository
from src.infrastructure.persistence.repositories.webhook_repository import WebhookRepository
from src.infrastructure.persistence.repositories.webhook_event_repository import WebhookEventRepository
from src.infrastructure.persistence.repositories.usage_record_repository import UsageRecordRepository

# Import domain services
from src.domain.services.organization_management_service import OrganizationManagementService
from src.domain.services.stream_processing_service import StreamProcessingService
from src.domain.services.highlight_detection_service import HighlightDetectionService
from src.domain.services.webhook_delivery_service import WebhookDeliveryService
from src.domain.services.usage_tracking_service import UsageTrackingService

# Import use cases
from src.application.use_cases.authentication import AuthenticationUseCase
from src.application.use_cases.stream_processing import StreamProcessingUseCase
from src.application.use_cases.batch_processing import BatchProcessingUseCase
from src.application.use_cases.webhook_processing import WebhookProcessingUseCase

# Import security
from src.infrastructure.security.webhook_validator import WebhookValidatorFactory


# Security scheme for API key authentication
security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    """Get the current authenticated user from API key.

    Args:
        credentials: HTTP Bearer token credentials
        db: Database session

    Returns:
        User: The authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    # Extract API key from Bearer token
    api_key = credentials.credentials

    # Look up API key in database
    result = await db.execute(
        select(APIKey)
        .where(APIKey.key == api_key, APIKey.is_active)
        .options(selectinload(APIKey.user))
    )
    api_key_obj = result.scalar_one_or_none()

    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if API key is expired
    if api_key_obj.is_expired:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key_obj.user


async def get_optional_current_user(
    request: Request, db: Annotated[AsyncSession, Depends(get_db_session)]
) -> Optional[User]:
    """Get the current user if authenticated, or None.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Optional[User]: The authenticated user or None
    """
    # Check for Authorization header
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        return None

    # Extract API key
    api_key = authorization.replace("Bearer ", "")

    # Look up API key in database
    result = await db.execute(
        select(APIKey)
        .where(APIKey.key == api_key, APIKey.is_active)
        .options(selectinload(APIKey.user))
    )
    api_key_obj = result.scalar_one_or_none()

    if not api_key_obj or api_key_obj.is_expired:
        return None

    return api_key_obj.user


# Ownership verification is now handled inline in the router functions


# Type aliases for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalCurrentUser = Annotated[Optional[User], Depends(get_optional_current_user)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]


# Repository Dependencies

async def get_user_repository(
    db: AsyncSession = Depends(get_db_session)
) -> UserRepository:
    """Get user repository instance."""
    return UserRepository(db)


async def get_api_key_repository(
    db: AsyncSession = Depends(get_db_session)
) -> APIKeyRepository:
    """Get API key repository instance."""
    return APIKeyRepository(db)


async def get_organization_repository(
    db: AsyncSession = Depends(get_db_session)
) -> OrganizationRepository:
    """Get organization repository instance."""
    return OrganizationRepository(db)


async def get_stream_repository(
    db: AsyncSession = Depends(get_db_session)
) -> StreamRepository:
    """Get stream repository instance."""
    return StreamRepository(db)


async def get_highlight_repository(
    db: AsyncSession = Depends(get_db_session)
) -> HighlightRepository:
    """Get highlight repository instance."""
    return HighlightRepository(db)


async def get_batch_repository(
    db: AsyncSession = Depends(get_db_session)
) -> BatchRepository:
    """Get batch repository instance."""
    return BatchRepository(db)


async def get_webhook_repository(
    db: AsyncSession = Depends(get_db_session)
) -> WebhookRepository:
    """Get webhook repository instance."""
    return WebhookRepository(db)


async def get_webhook_event_repository(
    db: AsyncSession = Depends(get_db_session)
) -> WebhookEventRepository:
    """Get webhook event repository instance."""
    return WebhookEventRepository(db)


async def get_usage_record_repository(
    db: AsyncSession = Depends(get_db_session)
) -> UsageRecordRepository:
    """Get usage record repository instance."""
    return UsageRecordRepository(db)


# Domain Service Dependencies

async def get_organization_management_service(
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    usage_repo: UsageRecordRepository = Depends(get_usage_record_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    webhook_repo: WebhookRepository = Depends(get_webhook_repository)
) -> OrganizationManagementService:
    """Get organization management service instance."""
    return OrganizationManagementService(
        org_repo=org_repo,
        user_repo=user_repo,
        usage_repo=usage_repo,
        stream_repo=stream_repo,
        api_key_repo=api_key_repo,
        webhook_repo=webhook_repo
    )


async def get_stream_processing_service(
    stream_repo: StreamRepository = Depends(get_stream_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository)
) -> StreamProcessingService:
    """Get stream processing service instance."""
    return StreamProcessingService(
        stream_repo=stream_repo,
        org_repo=org_repo
    )


async def get_highlight_detection_service(
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository)
) -> HighlightDetectionService:
    """Get highlight detection service instance."""
    return HighlightDetectionService(
        highlight_repo=highlight_repo,
        stream_repo=stream_repo
    )


async def get_webhook_delivery_service(
    webhook_repo: WebhookRepository = Depends(get_webhook_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    batch_repo: BatchRepository = Depends(get_batch_repository)
) -> WebhookDeliveryService:
    """Get webhook delivery service instance."""
    return WebhookDeliveryService(
        webhook_repo=webhook_repo,
        stream_repo=stream_repo,
        highlight_repo=highlight_repo,
        batch_repo=batch_repo
    )


async def get_usage_tracking_service(
    usage_repo: UsageRecordRepository = Depends(get_usage_record_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    user_repo: UserRepository = Depends(get_user_repository)
) -> UsageTrackingService:
    """Get usage tracking service instance."""
    return UsageTrackingService(
        usage_repo=usage_repo,
        org_repo=org_repo,
        user_repo=user_repo
    )


# Use Case Dependencies

async def get_authentication_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    org_service: OrganizationManagementService = Depends(get_organization_management_service)
) -> AuthenticationUseCase:
    """Get authentication use case instance."""
    return AuthenticationUseCase(
        user_repo=user_repo,
        api_key_repo=api_key_repo,
        org_repo=org_repo,
        org_service=org_service
    )


async def get_stream_processing_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    stream_service: StreamProcessingService = Depends(get_stream_processing_service),
    highlight_service: HighlightDetectionService = Depends(get_highlight_detection_service),
    webhook_service: WebhookDeliveryService = Depends(get_webhook_delivery_service),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> StreamProcessingUseCase:
    """Get stream processing use case instance."""
    return StreamProcessingUseCase(
        user_repo=user_repo,
        stream_repo=stream_repo,
        highlight_repo=highlight_repo,
        stream_service=stream_service,
        highlight_service=highlight_service,
        webhook_service=webhook_service,
        usage_service=usage_service
    )


async def get_batch_processing_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    batch_repo: BatchRepository = Depends(get_batch_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    org_service: OrganizationManagementService = Depends(get_organization_management_service),
    webhook_service: WebhookDeliveryService = Depends(get_webhook_delivery_service),
    usage_service: UsageTrackingService = Depends(get_usage_tracking_service)
) -> BatchProcessingUseCase:
    """Get batch processing use case instance."""
    return BatchProcessingUseCase(
        user_repo=user_repo,
        batch_repo=batch_repo,
        org_repo=org_repo,
        org_service=org_service,
        webhook_service=webhook_service,
        usage_service=usage_service
    )


async def get_webhook_processing_use_case(
    webhook_event_repo: WebhookEventRepository = Depends(get_webhook_event_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    stream_processing_use_case: StreamProcessingUseCase = Depends(get_stream_processing_use_case)
) -> WebhookProcessingUseCase:
    """Get webhook processing use case instance."""
    # Get webhook secrets from settings
    webhook_secrets = {
        "100ms": getattr(settings, 'WEBHOOK_SECRET_100MS', ''),
        "twitch": getattr(settings, 'WEBHOOK_SECRET_TWITCH', ''),
        "custom": getattr(settings, 'WEBHOOK_SECRET_CUSTOM', ''),
    }
    
    validator_factory = WebhookValidatorFactory(webhook_secrets)
    
    return WebhookProcessingUseCase(
        webhook_event_repo=webhook_event_repo,
        user_repo=user_repo,
        api_key_repo=api_key_repo,
        stream_processing_use_case=stream_processing_use_case,
        validator_factory=validator_factory
    )
