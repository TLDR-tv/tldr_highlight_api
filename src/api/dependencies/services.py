"""Domain service dependencies for FastAPI.

This module provides dependency injection for domain services.
"""

from fastapi import Depends

from src.domain.services.organization_management_service import OrganizationManagementService
from src.domain.services.stream_processing_service import StreamProcessingService
from src.domain.services.highlight_detection_service import HighlightDetectionService
from src.domain.services.webhook_delivery_service import WebhookDeliveryService
from src.domain.services.usage_tracking_service import UsageTrackingService

from .repositories import (
    get_user_repository,
    get_api_key_repository,
    get_organization_repository,
    get_stream_repository,
    get_highlight_repository,
    get_batch_repository,
    get_webhook_repository,
    get_usage_record_repository,
)

from src.infrastructure.persistence.repositories.user_repository import UserRepository
from src.infrastructure.persistence.repositories.api_key_repository import APIKeyRepository
from src.infrastructure.persistence.repositories.organization_repository import OrganizationRepository
from src.infrastructure.persistence.repositories.stream_repository import StreamRepository
from src.infrastructure.persistence.repositories.highlight_repository import HighlightRepository
from src.infrastructure.persistence.repositories.batch_repository import BatchRepository
from src.infrastructure.persistence.repositories.webhook_repository import WebhookRepository
from src.infrastructure.persistence.repositories.usage_record_repository import UsageRecordRepository


# Domain Service Dependencies

async def get_organization_management_service(
    user_repo: UserRepository = Depends(get_user_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository)
) -> OrganizationManagementService:
    """Get organization management service instance."""
    return OrganizationManagementService(
        user_repo=user_repo,
        org_repo=org_repo,
        api_key_repo=api_key_repo
    )


async def get_stream_processing_service(
    stream_repo: StreamRepository = Depends(get_stream_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    usage_repo: UsageRecordRepository = Depends(get_usage_record_repository),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository)
) -> StreamProcessingService:
    """Get stream processing service instance."""
    return StreamProcessingService(
        stream_repo=stream_repo,
        user_repo=user_repo,
        org_repo=org_repo,
        usage_repo=usage_repo,
        highlight_repo=highlight_repo
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