"""Pythonic mapping functions for organization API DTOs and domain objects."""

from typing import List
from datetime import datetime, timezone

from src.api.schemas.organizations import (
    OrganizationResponse,
    OrganizationUpdate,
    OrganizationUserResponse,
    OrganizationUsersListResponse,
    AddUserToOrganizationRequest,
    AddUserToOrganizationResponse,
    OrganizationUsageStats,
)
from src.application.use_cases.organization_management import (
    GetOrganizationRequest,
    UpdateOrganizationRequest,
    AddMemberRequest,
    RemoveMemberRequest,
    ListMembersRequest,
    GetUsageStatsRequest,
)
from src.domain.entities.organization import Organization
from src.domain.entities.user import User


def organization_to_response(organization: Organization) -> OrganizationResponse:
    """Convert Organization domain entity to OrganizationResponse DTO."""
    return OrganizationResponse(
        id=organization.id,
        company_name=str(organization.company_name),
        plan=organization.plan,
        is_active=organization.is_active,
        webhook_url=str(organization.webhook_url) if organization.webhook_url else None,
        webhook_secret=organization.webhook_secret,
        settings=organization.settings,
        created_at=organization.created_at.to_datetime(),
        updated_at=organization.updated_at.to_datetime(),
        member_count=len(organization.member_ids),
        monthly_stream_limit=organization.get_monthly_stream_limit(),
        concurrent_stream_limit=organization.get_concurrent_stream_limit(),
        storage_quota_gb=organization.get_storage_quota_gb(),
    )


def organization_update_to_request(
    org_id: int, dto: OrganizationUpdate
) -> UpdateOrganizationRequest:
    """Convert OrganizationUpdate DTO to UpdateOrganizationRequest."""
    return UpdateOrganizationRequest(
        organization_id=org_id,
        company_name=dto.company_name,
        webhook_url=dto.webhook_url,
        webhook_secret=dto.webhook_secret,
        settings=dto.settings,
    )


def user_to_organization_user_response(user: User) -> OrganizationUserResponse:
    """Convert User domain entity to OrganizationUserResponse DTO."""
    return OrganizationUserResponse(
        id=user.id,
        email=str(user.email),
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at.to_datetime(),
        last_login=user.last_login.to_datetime() if user.last_login else None,
    )


def users_to_list_response(
    users: List[User], total: int, page: int, per_page: int
) -> OrganizationUsersListResponse:
    """Convert list of users to paginated response."""
    return OrganizationUsersListResponse(
        users=[user_to_organization_user_response(user) for user in users],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page if per_page > 0 else 0,
    )


def add_user_request_to_domain(
    org_id: int, dto: AddUserToOrganizationRequest
) -> AddMemberRequest:
    """Convert AddUserToOrganizationRequest DTO to AddMemberRequest."""
    return AddMemberRequest(
        organization_id=org_id,
        user_email=dto.user_email,
        send_invitation=dto.send_invitation,
    )


def user_to_add_response(user: User) -> AddUserToOrganizationResponse:
    """Convert User to AddUserToOrganizationResponse."""
    return AddUserToOrganizationResponse(
        user=user_to_organization_user_response(user),
        message="User successfully added to organization",
    )


def create_get_request(org_id: int) -> GetOrganizationRequest:
    """Create GetOrganizationRequest."""
    return GetOrganizationRequest(organization_id=org_id)


def create_remove_member_request(org_id: int, user_id: int) -> RemoveMemberRequest:
    """Create RemoveMemberRequest."""
    return RemoveMemberRequest(organization_id=org_id, user_id=user_id)


def create_list_members_request(
    org_id: int, page: int, per_page: int
) -> ListMembersRequest:
    """Create ListMembersRequest."""
    return ListMembersRequest(
        organization_id=org_id,
        page=page,
        per_page=per_page,
    )


def create_usage_stats_request(
    org_id: int, start_date: datetime = None, end_date: datetime = None
) -> GetUsageStatsRequest:
    """Create GetUsageStatsRequest with date range."""
    if start_date is None:
        # Default to current month
        now = datetime.now(timezone.utc)
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if end_date is None:
        end_date = datetime.now(timezone.utc)

    return GetUsageStatsRequest(
        organization_id=org_id,
        start_date=start_date,
        end_date=end_date,
    )


def usage_stats_to_response(stats: dict, org: Organization) -> OrganizationUsageStats:
    """Convert usage stats dict to OrganizationUsageStats response."""
    return OrganizationUsageStats(
        organization_id=org.id,
        current_month_streams=stats.get("current_month_streams", 0),
        monthly_stream_limit=org.get_monthly_stream_limit(),
        current_storage_gb=stats.get("current_storage_gb", 0.0),
        storage_quota_gb=org.get_storage_quota_gb(),
        active_streams=stats.get("active_streams", 0),
        concurrent_stream_limit=org.get_concurrent_stream_limit(),
        total_highlights=stats.get("total_highlights", 0),
        total_processing_minutes=stats.get("total_processing_minutes", 0.0),
    )
