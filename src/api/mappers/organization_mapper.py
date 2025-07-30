"""Organization mapper for converting between API DTOs and domain objects."""

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


class OrganizationMapper:
    """Maps between organization API DTOs and domain entities."""

    @staticmethod
    def to_get_organization_request(
        organization_id: int, user_id: int
    ) -> GetOrganizationRequest:
        """Convert parameters to GetOrganizationRequest."""
        return GetOrganizationRequest(
            requester_id=user_id, organization_id=organization_id
        )

    @staticmethod
    def to_organization_response(organization: Organization) -> OrganizationResponse:
        """Convert Organization domain entity to response DTO."""
        return OrganizationResponse(
            id=organization.id,
            name=organization.name.value,
            owner_id=organization.owner_id,
            plan_type=organization.plan_type.value,
            created_at=organization.created_at.value,
        )

    @staticmethod
    def to_update_organization_request(
        organization_id: int, user_id: int, update_dto: OrganizationUpdate
    ) -> UpdateOrganizationRequest:
        """Convert update DTO to domain request."""
        return UpdateOrganizationRequest(
            requester_id=user_id,
            organization_id=organization_id,
            name=update_dto.name,
            settings={"plan_type": update_dto.plan_type.value}
            if update_dto.plan_type
            else None,
        )

    @staticmethod
    def to_add_member_request(
        organization_id: int, user_id: int, add_user_dto: AddUserToOrganizationRequest
    ) -> AddMemberRequest:
        """Convert add user DTO to domain request."""
        # Note: In the current implementation, we're using user_id instead of email
        # This would need to be adjusted based on the actual domain service implementation
        return AddMemberRequest(
            requester_id=user_id,
            organization_id=organization_id,
            user_email=f"user_{add_user_dto.user_id}@example.com",  # Placeholder
            role=add_user_dto.role or "member",
        )

    @staticmethod
    def to_add_user_response(
        user: User, organization_id: int, role: str
    ) -> AddUserToOrganizationResponse:
        """Convert domain result to add user response DTO."""
        return AddUserToOrganizationResponse(
            user_id=user.id,
            organization_id=organization_id,
            role=role,
            added_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def to_remove_member_request(
        organization_id: int, requester_id: int, user_id_to_remove: int
    ) -> RemoveMemberRequest:
        """Convert parameters to RemoveMemberRequest."""
        return RemoveMemberRequest(
            requester_id=requester_id,
            organization_id=organization_id,
            user_id=user_id_to_remove,
        )

    @staticmethod
    def to_list_members_request(
        organization_id: int, user_id: int
    ) -> ListMembersRequest:
        """Convert parameters to ListMembersRequest."""
        return ListMembersRequest(requester_id=user_id, organization_id=organization_id)

    @staticmethod
    def to_organization_user_response(user: User) -> OrganizationUserResponse:
        """Convert User domain entity to response DTO."""
        return OrganizationUserResponse(
            id=user.id,
            email=user.email.value,
            company_name=user.company_name.value,
            created_at=user.created_at.value,
        )

    @staticmethod
    def to_organization_users_list_response(
        users: List[User],
    ) -> OrganizationUsersListResponse:
        """Convert list of users to response DTO."""
        user_responses = [
            OrganizationMapper.to_organization_user_response(user) for user in users
        ]
        return OrganizationUsersListResponse(total=len(users), users=user_responses)

    @staticmethod
    def to_get_usage_stats_request(
        organization_id: int, user_id: int
    ) -> GetUsageStatsRequest:
        """Convert parameters to GetUsageStatsRequest."""
        return GetUsageStatsRequest(
            requester_id=user_id, organization_id=organization_id
        )

    @staticmethod
    def to_organization_usage_stats(
        usage_result: dict
    ) -> OrganizationUsageStats:
        """Convert usage result to response DTO (unlimited limits)."""
        # Calculate current month usage
        current_usage = {
            "streams_processed": usage_result.get("total_streams", 0),
            "batch_videos_processed": usage_result.get("total_batch_videos", 0),
            "total_api_calls": usage_result.get("total_api_calls", 0),
            "storage_used_gb": usage_result.get("storage_used_gb", 0),
        }

        return OrganizationUsageStats(
            current_month=current_usage,
        )
