"""Domain service dependencies for FastAPI.

This module provides clean FastAPI dependency injection for domain services.
"""

from fastapi import Depends

from src.application.workflows import (
    OrganizationManager,
    UsageTracker,
    StreamProcessor,
    DimensionManager,
    WebhookNotifier,
    Authenticator,
)
from src.domain.services.stream_processing_service import StreamProcessingService
from src.domain.services.highlight_detection_service import HighlightDetectionService
from src.domain.services.webhook_delivery_service import WebhookDeliveryService

from .repositories import (
    get_user_repository,
    get_api_key_repository,
    get_organization_repository,
    get_stream_repository,
    get_highlight_repository,
    get_webhook_repository,
    get_usage_record_repository,
    get_dimension_set_repository,
    get_highlight_type_registry_repository,
)

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
from src.infrastructure.persistence.repositories.usage_record_repository import (
    UsageRecordRepository,
)
from src.infrastructure.persistence.repositories.dimension_set_repository import (
    DimensionSetRepository,
)
from src.infrastructure.persistence.repositories.highlight_type_registry_repository import (
    HighlightTypeRegistryRepository,
)


# Domain Service Dependencies - clean FastAPI dependency injection


def get_organization_manager(
    user_repo: UserRepository = Depends(get_user_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
) -> OrganizationManager:
    """Get organization manager instance."""
    return OrganizationManager(
        user_repo=user_repo,
        organization_repo=org_repo,
        api_key_repo=api_key_repo,
    )


def get_stream_processing_service(
    stream_repo: StreamRepository = Depends(get_stream_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    usage_repo: UsageRecordRepository = Depends(get_usage_record_repository),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
) -> StreamProcessingService:
    """Get stream processing service instance."""
    return StreamProcessingService(
        stream_repo=stream_repo,
        user_repo=user_repo,
        org_repo=org_repo,
        usage_repo=usage_repo,
        highlight_repo=highlight_repo,
    )


def get_highlight_detection_service(
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    dimension_set_repo: DimensionSetRepository = Depends(get_dimension_set_repository),
    type_registry_repo: HighlightTypeRegistryRepository = Depends(
        get_highlight_type_registry_repository
    ),
) -> HighlightDetectionService:
    """Get highlight detection service instance."""
    return HighlightDetectionService(
        highlight_repo=highlight_repo,
        stream_repo=stream_repo,
        dimension_set_repo=dimension_set_repo,
        type_registry_repo=type_registry_repo,
    )


def get_webhook_delivery_service(
    webhook_repo: WebhookRepository = Depends(get_webhook_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
) -> WebhookDeliveryService:
    """Get webhook delivery service instance."""
    return WebhookDeliveryService(
        webhook_repo=webhook_repo,
        stream_repo=stream_repo,
        highlight_repo=highlight_repo,
    )


def get_usage_tracker(
    usage_repo: UsageRecordRepository = Depends(get_usage_record_repository),
    user_repo: UserRepository = Depends(get_user_repository),
) -> UsageTracker:
    """Get usage tracker instance."""
    return UsageTracker(
        usage_repo=usage_repo,
        user_repo=user_repo,
    )


def get_stream_processor(
    stream_repo: StreamRepository = Depends(get_stream_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    usage_repo: UsageRecordRepository = Depends(get_usage_record_repository),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
) -> StreamProcessor:
    """Get stream processor instance."""
    # TODO: Add missing dependencies like task_service, platform_detector, agent_config_repo
    return StreamProcessor(
        stream_repo=stream_repo,
        user_repo=user_repo,
        org_repo=org_repo,
        usage_repo=usage_repo,
        highlight_repo=highlight_repo,
        agent_config_repo=None,  # TODO: Implement
        platform_detector=None,  # TODO: Implement
        task_service=None,  # TODO: Implement
    )


def get_dimension_manager(
    dimension_repo: DimensionSetRepository = Depends(get_dimension_set_repository),
    registry_repo: HighlightTypeRegistryRepository = Depends(get_highlight_type_registry_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
) -> DimensionManager:
    """Get dimension manager instance."""
    return DimensionManager(
        dimension_repo=dimension_repo,
        registry_repo=registry_repo,
        organization_repo=org_repo,
    )


def get_webhook_notifier(
    webhook_repo: WebhookRepository = Depends(get_webhook_repository),
    event_repo: WebhookEventRepository = Depends(get_webhook_event_repository),
) -> WebhookNotifier:
    """Get webhook notifier instance."""
    return WebhookNotifier(
        webhook_repo=webhook_repo,
        event_repo=event_repo,
    )


def get_authenticator(
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
) -> Authenticator:
    """Get authenticator instance."""
    return Authenticator(
        user_repo=user_repo,
        api_key_repo=api_key_repo,
    )
