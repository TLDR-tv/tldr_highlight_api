"""Organization management endpoints.

This module implements organization management functionality including
organization details, user management within organizations, and usage statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.use_cases import get_organization_management_use_case
from src.api.mappers.organization_mapper import (
    organization_to_response,
    organization_update_to_request,
    user_to_add_response,
    add_user_request_to_domain,
    users_to_list_response,
    create_get_request,
    create_remove_member_request,
    create_list_members_request,
    create_usage_stats_request,
    usage_stats_to_response,
)
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


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: int,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
) -> OrganizationResponse:
    """Get organization details.

    Returns detailed information about an organization including
    subscription plan and usage limits.
    """
    if not current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found"
        )
    request = create_get_request(org_id)
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

    if not result.organization:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Organization data not available",
        )
    return organization_to_response(result.organization)


@router.put("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: int,
    org_data: OrganizationUpdate,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
) -> OrganizationResponse:
    """Update organization details.

    Updates organization name and/or subscription plan.
    Only organization owners can update organization details.
    """
    if not current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found"
        )
    request = organization_update_to_request(org_id, org_data)
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

    if not result.organization:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Organization data not available",
        )
    return organization_to_response(result.organization)


@router.get("/{org_id}/users", response_model=OrganizationUsersListResponse)
async def list_organization_users(
    org_id: int,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
) -> OrganizationUsersListResponse:
    """List users in an organization.

    Returns a list of all users who have access to the organization.
    """
    if not current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found"
        )
    request = create_list_members_request(org_id, page=1, per_page=100)
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

    return users_to_list_response(
        result.members, total=len(result.members), page=1, per_page=100
    )


@router.post("/{org_id}/users", response_model=AddUserToOrganizationResponse)
async def add_user_to_organization(
    org_id: int,
    user_data: AddUserToOrganizationRequest,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
) -> AddUserToOrganizationResponse:
    """Add a user to an organization.

    Adds a user to the organization with the specified role.
    Only organization owners can add users.
    """
    if not current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found"
        )
    request = add_user_request_to_domain(org_id, user_data)
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

    if not result.user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User data not available",
        )
    return user_to_add_response(result.user)


@router.delete("/{org_id}/users/{user_id}")
async def remove_user_from_organization(
    org_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    use_case: OrganizationManagementUseCase = Depends(
        get_organization_management_use_case
    ),
) -> dict[str, str]:
    """Remove a user from an organization.

    Removes a user's access to the organization.
    Only organization owners can remove users.
    """
    if not current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found"
        )
    request = create_remove_member_request(org_id, user_id)
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
) -> OrganizationUsageStats:
    """Get organization usage statistics.

    Returns current usage statistics compared to plan limits.
    Useful for billing and usage monitoring.
    """
    # Get organization to access plan limits
    if not current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found"
        )
    org_request = create_get_request(org_id)
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
    usage_request = create_usage_stats_request(org_id)
    usage_result = await use_case.get_usage_stats(usage_request)

    if not usage_result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=usage_result.errors[0]
            if usage_result.errors
            else "Failed to get usage statistics",
        )

    # Collect usage data (no limits enforced)
    usage_data = {
        "current_month_streams": usage_result.total_streams or 0,
        "current_storage_gb": usage_result.storage_used_gb or 0.0,
        "active_streams": 0,  # Would need to be calculated separately
        "total_highlights": usage_result.total_highlights or 0,
        "total_processing_minutes": usage_result.total_processing_minutes or 0.0,
    }

    return usage_stats_to_response(usage_data, org_result.organization)
