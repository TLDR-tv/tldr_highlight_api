"""Authentication request and response schemas."""

from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict


class LoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


class RegisterOrganizationRequest(BaseModel):
    """Register new organization request."""

    # Organization details
    organization_name: str = Field(..., min_length=1, max_length=100)
    webhook_url: Optional[str] = None

    # Owner details
    owner_email: EmailStr
    owner_name: str = Field(..., min_length=1, max_length=100)
    owner_password: str = Field(..., min_length=8)

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate webhook URL if provided."""
        if v and not (v.startswith("https://") or v.startswith("http://")):
            raise ValueError("Webhook URL must start with https:// or http://")
        return v


class RegisterOrganizationResponse(BaseModel):
    """Register organization response."""

    message: str
    organization: dict  # Will be OrganizationResponse dict
    user: dict  # Will be UserResponse dict


class PasswordResetRequest(BaseModel):
    """Password reset request."""

    email: EmailStr


class PasswordResetResponse(BaseModel):
    """Password reset response."""

    message: str
    reset_token: Optional[str] = Field(None, exclude=True)  # Only for development


class PasswordResetWithTokenResponse(PasswordResetResponse):
    """Password reset response with token (for development)."""

    reset_token: str = Field(..., description="Reset token for development")


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation."""

    token: str
    new_password: str = Field(..., min_length=8)


class PasswordChangeRequest(BaseModel):
    """Password change request."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8)


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
