"""API request and response schemas."""

from .highlights import (
    HighlightResponse,
    HighlightListResponse,
    StreamHighlightResponse,
    StreamHighlightListResponse,
    DimensionScoreResponse,
    HighlightListParams,
)
from .auth import (
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    RegisterOrganizationRequest,
    RegisterOrganizationResponse,
    PasswordResetRequest,
    PasswordResetResponse,
    PasswordResetConfirmRequest,
    PasswordChangeRequest,
    MessageResponse,
)
from .organization import (
    OrganizationResponse,
    OrganizationUpdateRequest,
    WebhookSecretResponse,
    WakeWordRequest,
    OrganizationUsageResponse,
)
from .user import (
    UserResponse,
    UserCreateRequest,
    UserUpdateRequest,
    UserRoleUpdateRequest,
    UserListResponse,
)
from .api_key import (
    APIKeyResponse,
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyListResponse,
)

__all__ = [
    # Highlights
    "HighlightResponse",
    "HighlightListResponse", 
    "StreamHighlightResponse",
    "StreamHighlightListResponse",
    "DimensionScoreResponse",
    "HighlightListParams",
    # Auth
    "LoginRequest",
    "TokenResponse",
    "RefreshTokenRequest",
    "RegisterOrganizationRequest",
    "RegisterOrganizationResponse",
    "PasswordResetRequest",
    "PasswordResetResponse",
    "PasswordResetConfirmRequest",
    "PasswordChangeRequest",
    "MessageResponse",
    # Organization
    "OrganizationResponse",
    "OrganizationUpdateRequest",
    "WebhookSecretResponse",
    "WakeWordRequest",
    "OrganizationUsageResponse",
    # User
    "UserResponse",
    "UserCreateRequest",
    "UserUpdateRequest",
    "UserRoleUpdateRequest",
    "UserListResponse",
    # API Key
    "APIKeyResponse",
    "APIKeyCreateRequest",
    "APIKeyCreateResponse",
    "APIKeyListResponse",
]