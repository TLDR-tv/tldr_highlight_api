"""API dependencies for FastAPI.

This module provides dependency injection for the API layer,
following clean architecture principles.
"""

from .auth import (
    # Core dependencies
    get_current_user,
    get_current_api_key,
    get_current_organization,
    authenticated_user,
    authenticated_organization,
    optional_authentication,
    # Type aliases
    CurrentUser,
    OptionalCurrentUser,
    # Scope requirements
    require_scope,
    require_authenticated_scope,
    admin_required,
    read_required,
    write_required,
    streams_required,
    batches_required,
    webhooks_required,
    analytics_required,
    # Rate limiting
    check_rate_limit,
    check_burst_limit,
)

# Re-export database dependency from infrastructure
from src.infrastructure.database import get_db as get_db_session

# Import repository dependencies
from .repositories import (
    get_user_repository,
    get_api_key_repository,
    get_organization_repository,
    get_stream_repository,
    get_highlight_repository,
    get_batch_repository,
    get_webhook_repository,
    get_webhook_event_repository,
    get_usage_record_repository,
)

# Import service dependencies
from .services import (
    get_organization_management_service,
    get_stream_processing_service,
    get_highlight_detection_service,
    get_webhook_delivery_service,
    get_usage_tracking_service,
)

# Import use case dependencies
from .use_cases import (
    get_authentication_use_case,
    get_stream_processing_use_case,
    get_batch_processing_use_case,
    get_webhook_processing_use_case,
)

__all__ = [
    # Database
    "get_db_session",
    # Authentication
    "get_current_user",
    "get_current_api_key",
    "get_current_organization",
    "authenticated_user",
    "authenticated_organization",
    "optional_authentication",
    # Type aliases
    "CurrentUser",
    "OptionalCurrentUser",
    # Permissions
    "require_scope",
    "require_authenticated_scope",
    "admin_required",
    "read_required",
    "write_required",
    "streams_required",
    "batches_required",
    "webhooks_required",
    "analytics_required",
    # Rate limiting
    "check_rate_limit",
    "check_burst_limit",
    # Repositories
    "get_user_repository",
    "get_api_key_repository",
    "get_organization_repository",
    "get_stream_repository",
    "get_highlight_repository",
    "get_batch_repository",
    "get_webhook_repository",
    "get_webhook_event_repository",
    "get_usage_record_repository",
    # Services
    "get_organization_management_service",
    "get_stream_processing_service",
    "get_highlight_detection_service",
    "get_webhook_delivery_service",
    "get_usage_tracking_service",
    # Use cases
    "get_authentication_use_case",
    "get_stream_processing_use_case",
    "get_batch_processing_use_case",
    "get_webhook_processing_use_case",
]