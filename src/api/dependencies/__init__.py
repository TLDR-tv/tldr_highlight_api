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

# Import existing dependencies to maintain compatibility
from src.api.dependencies import (
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
    # Use cases
    "get_authentication_use_case",
    "get_stream_processing_use_case",
    "get_batch_processing_use_case",
    "get_webhook_processing_use_case",
]