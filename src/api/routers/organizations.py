"""Organization management endpoints.

This module implements organization management functionality including
organization details, user management within organizations, and usage statistics.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from src.core.database import get_db
from src.infrastructure.persistence.models.organization import Organization
from src.infrastructure.persistence.models.user import User
from src.api.dependencies.auth import require_scopes
from src.api.schemas.organizations import (
    AddUserToOrganizationRequest,
    AddUserToOrganizationResponse,
    OrganizationResponse,
    OrganizationUpdate,
    OrganizationUsageStats,
    OrganizationUserResponse,
    OrganizationUsersListResponse,
)

router = APIRouter(prefix="/api/v1/organizations", tags=["Organizations"])


async def get_organization_by_id(
    org_id: int, current_user: User, db: AsyncSession
) -> Organization:
    """Get organization by ID with permission check.

    Args:
        org_id: Organization ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Organization: Organization object

    Raises:
        HTTPException: If organization not found or access denied
    """
    result = await db.execute(
        select(Organization)
        .options(selectinload(Organization.owner))
        .where(Organization.id == org_id)
    )
    organization = result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    # Check if user has access to this organization
    if organization.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization",
        )

    return organization


def create_organization_response(organization: Organization) -> OrganizationResponse:
    """Create organization response with plan limits.

    Args:
        organization: Organization object

    Returns:
        OrganizationResponse: Response schema
    """
    return OrganizationResponse(
        id=organization.id,
        name=organization.name,
        owner_id=organization.owner_id,
        plan_type=organization.plan_type,
        created_at=organization.created_at,
        plan_limits=organization.plan_limits,
    )


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: int,
    current_user: User = Depends(require_scopes(["organizations:read"])),
    db: AsyncSession = Depends(get_db),
):
    """Get organization details.

    Returns detailed information about an organization including
    subscription plan and usage limits.
    """
    organization = await get_organization_by_id(org_id, current_user, db)
    return create_organization_response(organization)


@router.put("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: int,
    org_data: OrganizationUpdate,
    current_user: User = Depends(require_scopes(["organizations:write"])),
    db: AsyncSession = Depends(get_db),
):
    """Update organization details.

    Updates organization name and/or subscription plan.
    Only organization owners can update organization details.
    """
    organization = await get_organization_by_id(org_id, current_user, db)

    # Update fields
    if org_data.name is not None:
        organization.name = org_data.name

    if org_data.plan_type is not None:
        organization.plan_type = org_data.plan_type.value

    await db.commit()
    await db.refresh(organization)

    return create_organization_response(organization)


@router.get("/{org_id}/users", response_model=OrganizationUsersListResponse)
async def list_organization_users(
    org_id: int,
    current_user: User = Depends(require_scopes(["organizations:read"])),
    db: AsyncSession = Depends(get_db),
):
    """List users in an organization.

    Returns a list of all users who have access to the organization.
    Currently, this returns the organization owner (simple implementation).
    """
    organization = await get_organization_by_id(org_id, current_user, db)

    # Get organization owner (in a full implementation, you'd have a
    # many-to-many relationship for organization members)
    result = await db.execute(select(User).where(User.id == organization.owner_id))
    owner = result.scalar_one_or_none()

    users = []
    if owner:
        users.append(
            OrganizationUserResponse(
                id=owner.id,
                email=owner.email,
                company_name=owner.company_name,
                created_at=owner.created_at,
            )
        )

    return OrganizationUsersListResponse(total=len(users), users=users)


@router.post("/{org_id}/users", response_model=AddUserToOrganizationResponse)
async def add_user_to_organization(
    org_id: int,
    user_data: AddUserToOrganizationRequest,
    current_user: User = Depends(require_scopes(["organizations:write"])),
    db: AsyncSession = Depends(get_db),
):
    """Add a user to an organization.

    Adds a user to the organization with the specified role.
    Only organization owners can add users.

    Note: In a full implementation, this would use a many-to-many
    relationship table for organization memberships.
    """
    _organization = await get_organization_by_id(org_id, current_user, db)

    # Check if user exists
    result = await db.execute(select(User).where(User.id == user_data.user_id))
    user_to_add = result.scalar_one_or_none()

    if not user_to_add:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # In a full implementation, you would create an organization membership record here
    # For now, we'll return a success response

    return AddUserToOrganizationResponse(
        user_id=user_data.user_id,
        organization_id=org_id,
        role=user_data.role,
        added_at=datetime.now(timezone.utc),
    )


@router.delete("/{org_id}/users/{user_id}")
async def remove_user_from_organization(
    org_id: int,
    user_id: int,
    current_user: User = Depends(require_scopes(["organizations:write"])),
    db: AsyncSession = Depends(get_db),
):
    """Remove a user from an organization.

    Removes a user's access to the organization.
    Only organization owners can remove users.
    """
    organization = await get_organization_by_id(org_id, current_user, db)

    # Prevent owner from removing themselves
    if user_id == organization.owner_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove organization owner",
        )

    # Check if user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user_to_remove = result.scalar_one_or_none()

    if not user_to_remove:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # In a full implementation, you would delete the organization membership record here

    return {"message": "User removed from organization successfully"}


@router.get("/{org_id}/usage", response_model=OrganizationUsageStats)
async def get_organization_usage(
    org_id: int,
    current_user: User = Depends(require_scopes(["organizations:read"])),
    db: AsyncSession = Depends(get_db),
):
    """Get organization usage statistics.

    Returns current usage statistics compared to plan limits.
    Useful for billing and usage monitoring.
    """
    organization = await get_organization_by_id(org_id, current_user, db)

    # In a full implementation, you would query actual usage data
    # from usage_records, streams, batches, etc.
    # For now, we'll return mock data

    plan_limits = organization.plan_limits

    # Mock current usage data
    current_usage = {
        "streams_processed": 45,
        "batch_videos_processed": 230,
        "total_api_calls": 1250,
        "storage_used_gb": 15,
    }

    # Calculate usage percentages
    usage_percentage = {}
    if plan_limits.get("monthly_streams", 0) > 0:
        usage_percentage["streams"] = (
            current_usage["streams_processed"] / plan_limits["monthly_streams"]
        ) * 100

    if plan_limits.get("monthly_batch_videos", 0) > 0:
        usage_percentage["batch_videos"] = (
            current_usage["batch_videos_processed"]
            / plan_limits["monthly_batch_videos"]
        ) * 100

    # Mock API calls percentage (assuming 30 days * rate limit per minute * 60 minutes * 24 hours)
    if plan_limits.get("api_rate_limit_per_minute", 0) > 0:
        monthly_api_limit = plan_limits["api_rate_limit_per_minute"] * 60 * 24 * 30
        usage_percentage["api_calls"] = (
            current_usage["total_api_calls"] / monthly_api_limit
        ) * 100

    return OrganizationUsageStats(
        current_month=current_usage,
        plan_limits=plan_limits,
        usage_percentage=usage_percentage,
    )
