"""Organization management schemas.

This module defines Pydantic models for organization management endpoints,
including request/response schemas with validation and documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.infrastructure.persistence.models.organization import PlanType


class OrganizationUpdate(BaseModel):
    """Request schema for updating an organization."""

    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Organization name",
        example="Acme Corp Updated",
    )

    plan_type: Optional[PlanType] = Field(
        None, description="Subscription plan type", example=PlanType.PROFESSIONAL
    )


class OrganizationResponse(BaseModel):
    """Response schema for organization information."""

    id: int = Field(
        ..., description="Unique identifier for the organization", example=123
    )

    name: str = Field(..., description="Organization name", example="Acme Corp")

    owner_id: int = Field(
        ..., description="ID of the user who owns this organization", example=456
    )

    plan_type: str = Field(
        ..., description="Subscription plan type", example="professional"
    )

    created_at: datetime = Field(
        ...,
        description="Timestamp when the organization was created",
        example="2024-01-15T10:30:00Z",
    )

    # Plan limits removed - all organizations have unlimited access for now

    class Config:
        from_attributes = True


class OrganizationUserResponse(BaseModel):
    """Response schema for organization user information."""

    id: int = Field(..., description="User ID", example=789)

    email: str = Field(
        ..., description="User's email address", example="user@company.com"
    )

    company_name: str = Field(
        ..., description="User's company name", example="Tech Solutions Inc"
    )

    created_at: datetime = Field(
        ...,
        description="Timestamp when the user was created",
        example="2024-01-10T09:15:00Z",
    )

    class Config:
        from_attributes = True


class OrganizationUsersListResponse(BaseModel):
    """Response schema for listing organization users."""

    total: int = Field(
        ..., description="Total number of users in the organization", example=3
    )

    users: List[OrganizationUserResponse] = Field(
        ..., description="List of users in the organization"
    )


class AddUserToOrganizationRequest(BaseModel):
    """Request schema for adding a user to an organization."""

    user_id: int = Field(
        ..., description="ID of the user to add to the organization", example=789
    )

    role: Optional[str] = Field(
        "member", description="Role of the user in the organization", example="member"
    )

    @field_validator("role")
    def validate_role(cls, v):
        """Validate that role is from the allowed list."""
        allowed_roles = {"owner", "admin", "member", "viewer"}
        if v not in allowed_roles:
            raise ValueError(f"Invalid role: {v}. Must be one of {allowed_roles}")
        return v


class AddUserToOrganizationResponse(BaseModel):
    """Response schema for adding a user to an organization."""

    user_id: int = Field(
        ..., description="ID of the user added to the organization", example=789
    )

    organization_id: int = Field(..., description="ID of the organization", example=123)

    role: str = Field(
        ..., description="Role of the user in the organization", example="member"
    )

    added_at: datetime = Field(
        ...,
        description="Timestamp when the user was added",
        example="2024-01-20T14:30:00Z",
    )


class OrganizationUsageStats(BaseModel):
    """Response schema for organization usage statistics."""

    current_month: Dict[str, int] = Field(
        ...,
        description="Usage statistics for the current month",
        example={
            "streams_processed": 45,
            "total_api_calls": 1250,
            "storage_used_gb": 15,
        },
    )

    # Plan limits and usage percentages removed - unlimited access for now
    # Usage is still tracked for statistics but no limits are enforced
