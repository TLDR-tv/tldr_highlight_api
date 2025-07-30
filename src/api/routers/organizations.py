"""Organization management endpoints.

This module implements organization management functionality including
organization details, user management within organizations, and usage statistics.
"""


from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.use_cases import get_organization_management_use_case
from src.api.mappers.organization_mapper import OrganizationMapper
from src.api.schemas.organizations import (
    AddUserToOrganizationRequest,
    AddUserToOrganizationResponse,
    OrganizationResponse,
    OrganizationUpdate,
    OrganizationUsageStats,
    OrganizationUsersListResponse,
)
from src.application.use_cases.organization_management import (
    OrganizationManagementUseCase,
)
from src.application.use_cases.base import ResultStatus
from src.domain.entities.user import User

router = APIRouter(prefix="/api/v1/organizations", tags=["Organizations"])

mapper = OrganizationMapper()


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: int,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
):
    """Get organization details.

    Returns detailed information about an organization including
    subscription plan and usage limits.
    """
    request = mapper.to_get_organization_request(org_id, current_user.id)
    result = await use_case.get_organization(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to get organization",
        )

    return mapper.to_organization_response(result.organization)


@router.put("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: int,
    org_data: OrganizationUpdate,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
):
    """Update organization details.

    Updates organization name and/or subscription plan.
    Only organization owners can update organization details.
    """
    request = mapper.to_update_organization_request(org_id, current_user.id, org_data)
    result = await use_case.update_organization(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0]
            if result.errors
            else "Failed to update organization",
        )

    return mapper.to_organization_response(result.organization)


@router.get("/{org_id}/users", response_model=OrganizationUsersListResponse)
async def list_organization_users(
    org_id: int,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
):
    """List users in an organization.

    Returns a list of all users who have access to the organization.
    """
    request = mapper.to_list_members_request(org_id, current_user.id)
    result = await use_case.list_members(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0]
            if result.errors
            else "Failed to list organization users",
        )

    return mapper.to_organization_users_list_response(result.members)


@router.post("/{org_id}/users", response_model=AddUserToOrganizationResponse)
async def add_user_to_organization(
    org_id: int,
    user_data: AddUserToOrganizationRequest,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
):
    """Add a user to an organization.

    Adds a user to the organization with the specified role.
    Only organization owners can add users.
    """
    request = mapper.to_add_member_request(org_id, current_user.id, user_data)
    result = await use_case.add_member(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.errors[0]
            if result.errors
            else "Organization or user not found",
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization",
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.errors[0] if result.errors else "Invalid request",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0]
            if result.errors
            else "Failed to add user to organization",
        )

    return mapper.to_add_user_response(result.user, org_id, user_data.role or "member")


@router.delete("/{org_id}/users/{user_id}")
async def remove_user_from_organization(
    org_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
):
    """Remove a user from an organization.

    Removes a user's access to the organization.
    Only organization owners can remove users.
    """
    request = mapper.to_remove_member_request(org_id, current_user.id, user_id)
    result = await use_case.remove_member(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization or user not found",
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization",
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.errors[0]
            if result.errors
            else "Cannot remove organization owner",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0]
            if result.errors
            else "Failed to remove user from organization",
        )

    return {"message": "User removed from organization successfully"}


@router.get("/{org_id}/usage", response_model=OrganizationUsageStats)
async def get_organization_usage(
    org_id: int,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
):
    """Get organization usage statistics.

    Returns current usage statistics compared to plan limits.
    Useful for billing and usage monitoring.
    """
    # Get organization to access plan limits
    org_request = mapper.to_get_organization_request(org_id, current_user.id)
    org_result = await use_case.get_organization(org_request)

    if org_result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    elif org_result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization",
        )
    elif not org_result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=org_result.errors[0]
            if org_result.errors
            else "Failed to get organization",
        )

    # Get usage statistics
    usage_request = mapper.to_get_usage_stats_request(org_id, current_user.id)
    usage_result = await use_case.get_usage_stats(usage_request)

    if not usage_result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=usage_result.errors[0]
            if usage_result.errors
            else "Failed to get usage statistics",
        )

    # Combine usage data with plan limits
    usage_data = {
        "total_streams": usage_result.total_streams or 0,
        "total_batch_videos": 0,  # Not implemented in use case yet
        "total_api_calls": usage_result.total_api_calls or 0,
        "storage_used_gb": usage_result.storage_used_gb or 0,
    }

    plan_limits = org_result.organization.get_plan_limits()

    return mapper.to_organization_usage_stats(usage_data, plan_limits)
