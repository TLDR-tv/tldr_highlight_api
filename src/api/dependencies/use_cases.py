"""Use case dependencies for FastAPI.

This module provides dependency injection for application use cases,
keeping them separate from authentication concerns.
"""

from fastapi import Depends

# Import repositories
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

# Import domain services
from src.application.workflows import OrganizationManager
from src.domain.services.stream_processing_service import StreamProcessingService
from src.domain.services.highlight_detection_service import HighlightDetectionService
from src.domain.services.webhook_delivery_service import WebhookDeliveryService
from src.application.workflows import UsageTracker

# Import use cases
from src.application.use_cases.user_registration import UserRegistrationUseCase
from src.application.use_cases.user_login import UserLoginUseCase
from src.application.use_cases.api_key_validation import APIKeyValidationUseCase
from src.application.use_cases.api_key_management import APIKeyManagementUseCase
from src.application.use_cases.stream_processing import StreamProcessingUseCase
from src.application.use_cases.webhook_processing import WebhookProcessingUseCase
from src.application.use_cases.user_management import UserManagementUseCase
from src.application.use_cases.organization_management import (
    OrganizationManagementUseCase,
)
from src.application.use_cases.highlight_management import HighlightManagementUseCase
from src.application.use_cases.webhook_configuration import WebhookConfigurationUseCase

# Import security dependencies
from src.infrastructure.security.url_signer import URLSigner
from .security import get_url_signer
from .security_services import get_password_hashing_service, get_api_key_hashing_service
from src.domain.services.security_services import (
    PasswordHashingService,
    APIKeyHashingService,
)

from .repositories import (
    get_user_repository,
    get_api_key_repository,
    get_organization_repository,
    get_stream_repository,
    get_highlight_repository,
    get_webhook_repository,
    get_webhook_event_repository,
)

from .services import (
    get_organization_manager,
    get_stream_processing_service,
    get_highlight_detection_service,
    get_webhook_delivery_service,
    get_usage_tracker,
)


# Use Case Dependencies


async def get_user_registration_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    org_manager: OrganizationManager = Depends(get_organization_manager),
    password_service: PasswordHashingService = Depends(get_password_hashing_service),
    api_key_service: APIKeyHashingService = Depends(get_api_key_hashing_service),
) -> UserRegistrationUseCase:
    """Get user registration use case instance."""
    return UserRegistrationUseCase(
        user_repo=user_repo,
        api_key_repo=api_key_repo,
        org_repo=org_repo,
        org_service=org_manager,
        password_service=password_service,
        api_key_service=api_key_service,
    )


async def get_user_login_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    password_service: PasswordHashingService = Depends(get_password_hashing_service),
    api_key_service: APIKeyHashingService = Depends(get_api_key_hashing_service),
) -> UserLoginUseCase:
    """Get user login use case instance."""
    return UserLoginUseCase(
        user_repo=user_repo,
        api_key_repo=api_key_repo,
        org_repo=org_repo,
        password_service=password_service,
        api_key_service=api_key_service,
    )


async def get_api_key_validation_use_case(
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    api_key_service: APIKeyHashingService = Depends(get_api_key_hashing_service),
) -> APIKeyValidationUseCase:
    """Get API key validation use case instance."""
    return APIKeyValidationUseCase(
        api_key_repo=api_key_repo,
        org_repo=org_repo,
        api_key_service=api_key_service,
    )


async def get_api_key_management_use_case(
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    api_key_service: APIKeyHashingService = Depends(get_api_key_hashing_service),
) -> APIKeyManagementUseCase:
    """Get API key management use case instance."""
    return APIKeyManagementUseCase(
        api_key_repo=api_key_repo,
        user_repo=user_repo,
        org_repo=org_repo,
        api_key_service=api_key_service,
    )


async def get_stream_processing_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    stream_service: StreamProcessingService = Depends(get_stream_processing_service),
    highlight_service: HighlightDetectionService = Depends(
        get_highlight_detection_service
    ),
    webhook_service: WebhookDeliveryService = Depends(get_webhook_delivery_service),
    usage_tracker: UsageTracker = Depends(get_usage_tracker),
) -> StreamProcessingUseCase:
    """Get stream processing use case instance."""
    # For now, use a mock agent config repository
    # TODO: Implement proper agent config repository
    agent_config_repo = None

    return StreamProcessingUseCase(
        user_repo=user_repo,
        stream_repo=stream_repo,
        highlight_repo=highlight_repo,
        agent_config_repo=agent_config_repo,
        stream_service=stream_service,
        highlight_service=highlight_service,
        webhook_service=webhook_service,
        usage_service=usage_tracker,
    )


async def get_webhook_processing_use_case(
    webhook_event_repo: WebhookEventRepository = Depends(get_webhook_event_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    stream_processing_use_case: StreamProcessingUseCase = Depends(
        get_stream_processing_use_case
    ),
) -> WebhookProcessingUseCase:
    """Get webhook processing use case instance."""
    # Create validator factory with webhook secrets from config
    # In production, these would come from environment variables or secure config
    from src.infrastructure.security.webhook_validator import WebhookValidatorFactory

    webhook_secrets = {
        "100ms": "your-100ms-webhook-secret",
        "twitch": "your-twitch-webhook-secret",
        "custom": "your-custom-webhook-secret",
    }
    validator_factory = WebhookValidatorFactory(webhook_secrets)

    return WebhookProcessingUseCase(
        webhook_event_repo=webhook_event_repo,
        user_repo=user_repo,
        api_key_repo=api_key_repo,
        stream_processing_use_case=stream_processing_use_case,
        validator_factory=validator_factory,
    )


async def get_user_management_use_case(
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_repo: APIKeyRepository = Depends(get_api_key_repository),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
) -> UserManagementUseCase:
    """Get user management use case instance."""
    return UserManagementUseCase(
        user_repo=user_repo, api_key_repo=api_key_repo, org_repo=org_repo
    )


async def get_organization_management_use_case(
    org_repo: OrganizationRepository = Depends(get_organization_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    org_manager: OrganizationManager = Depends(get_organization_manager),
) -> OrganizationManagementUseCase:
    """Get organization management use case instance."""
    return OrganizationManagementUseCase(
        org_repo=org_repo, user_repo=user_repo, org_service=org_service
    )


async def get_highlight_management_use_case(
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    url_signer: URLSigner = Depends(get_url_signer),
) -> HighlightManagementUseCase:
    """Get highlight management use case instance."""
    return HighlightManagementUseCase(
        highlight_repo=highlight_repo,
        stream_repo=stream_repo,
        user_repo=user_repo,
        url_signer=url_signer,
    )


async def get_webhook_configuration_use_case(
    webhook_repo: WebhookRepository = Depends(get_webhook_repository),
    webhook_event_repo: WebhookEventRepository = Depends(get_webhook_event_repository),
) -> WebhookConfigurationUseCase:
    """Get webhook configuration use case instance."""
    return WebhookConfigurationUseCase(
        webhook_repo=webhook_repo, webhook_event_repo=webhook_event_repo
    )
