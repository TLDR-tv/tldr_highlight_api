"""User management schemas.

This module defines Pydantic models for user management endpoints,
including request/response schemas with validation and documentation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserUpdate(BaseModel):
    """Request schema for updating user information."""

    email: Optional[EmailStr] = Field(
        None, description="User's email address", example="newemail@company.com"
    )

    company_name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Name of the user's company",
        example="New Company Name Inc",
    )

    password: Optional[str] = Field(
        None,
        min_length=8,
        max_length=128,
        description="New password for the user",
        example="newsecurepassword123",
    )

    @field_validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        if v is None:
            return v

        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Check for at least one uppercase, one lowercase, and one digit
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)

        if not (has_upper and has_lower and has_digit):
            raise ValueError(
                "Password must contain at least one uppercase letter, "
                "one lowercase letter, and one digit"
            )

        return v


class UserResponse(BaseModel):
    """Response schema for user information."""

    id: int = Field(..., description="Unique identifier for the user", example=123)

    email: str = Field(
        ..., description="User's email address", example="user@company.com"
    )

    company_name: str = Field(
        ..., description="Name of the user's company", example="Tech Solutions Inc"
    )

    created_at: datetime = Field(
        ...,
        description="Timestamp when the user was created",
        example="2024-01-15T10:30:00Z",
    )

    updated_at: datetime = Field(
        ...,
        description="Timestamp when the user was last updated",
        example="2024-01-20T14:45:30Z",
    )

    class Config:
        from_attributes = True


class UserProfileResponse(UserResponse):
    """Extended response schema for user profile (includes additional details)."""

    api_keys_count: int = Field(..., description="Number of active API keys", example=3)

    organizations_count: int = Field(
        ..., description="Number of organizations owned", example=1
    )

    last_login: Optional[datetime] = Field(
        None, description="Timestamp of last login", example="2024-01-20T09:15:00Z"
    )


class UserRegistrationRequest(BaseModel):
    """Request schema for user registration."""

    email: EmailStr = Field(
        ..., description="User's email address", example="user@company.com"
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User's password",
        example="securepassword123",
    )

    company_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the user's company",
        example="Tech Solutions Inc",
    )

    @field_validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Check for at least one uppercase, one lowercase, and one digit
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)

        if not (has_upper and has_lower and has_digit):
            raise ValueError(
                "Password must contain at least one uppercase letter, "
                "one lowercase letter, and one digit"
            )

        return v


class UserRegistrationResponse(BaseModel):
    """Response schema for user registration."""

    id: int = Field(..., description="Unique identifier for the new user", example=123)

    email: str = Field(
        ..., description="User's email address", example="user@company.com"
    )

    company_name: str = Field(
        ..., description="Name of the user's company", example="Tech Solutions Inc"
    )

    created_at: datetime = Field(
        ...,
        description="Timestamp when the user was created",
        example="2024-01-15T10:30:00Z",
    )

    access_token: str = Field(
        ...,
        description="JWT access token for immediate login",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    )

    token_type: str = Field(
        default="bearer", description="Token type", example="bearer"
    )

    expires_in: int = Field(
        ..., description="Token expiration time in seconds", example=3600
    )


class UserListResponse(BaseModel):
    """Response schema for listing users (admin only)."""

    total: int = Field(..., description="Total number of users", example=150)

    page: int = Field(..., description="Current page number", example=1)

    per_page: int = Field(..., description="Number of users per page", example=20)

    users: List[UserResponse] = Field(..., description="List of users")


class PasswordChangeRequest(BaseModel):
    """Request schema for changing password."""

    current_password: str = Field(
        ..., description="User's current password", example="oldpassword123"
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password",
        example="newsecurepassword123",
    )

    @field_validator("new_password")
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # Check for at least one uppercase, one lowercase, and one digit
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)

        if not (has_upper and has_lower and has_digit):
            raise ValueError(
                "Password must contain at least one uppercase letter, "
                "one lowercase letter, and one digit"
            )

        return v


class PasswordChangeResponse(BaseModel):
    """Response schema for password change."""

    message: str = Field(
        default="Password changed successfully",
        description="Success message",
        example="Password changed successfully",
    )

    changed_at: datetime = Field(
        ...,
        description="Timestamp when the password was changed",
        example="2024-01-20T15:30:00Z",
    )
