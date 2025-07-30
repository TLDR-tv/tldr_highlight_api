"""Organization management use cases."""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.organization import Organization, PlanType
from src.domain.entities.user import User
from src.domain.value_objects.company_name import CompanyName
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.services.organization_management_service import OrganizationManagementService
from src.domain.exceptions import (
    EntityNotFoundError,
    DuplicateEntityError,
    BusinessRuleViolation,
    UnauthorizedAccessError
)


@dataclass
class GetOrganizationRequest:
    """Request to get organization details."""
    requester_id: int
    organization_id: int


@dataclass
class GetOrganizationResult(UseCaseResult):
    """Result of getting organization."""
    organization: Optional[Organization] = None
    member_count: Optional[int] = None
    usage_stats: Optional[dict] = None


@dataclass
class UpdateOrganizationRequest:
    """Request to update organization."""
    requester_id: int
    organization_id: int
    name: Optional[str] = None
    settings: Optional[dict] = None


@dataclass
class UpdateOrganizationResult(UseCaseResult):
    """Result of organization update."""
    organization: Optional[Organization] = None


@dataclass
class AddMemberRequest:
    """Request to add member to organization."""
    requester_id: int
    organization_id: int
    user_email: str
    role: str = "member"


@dataclass
class AddMemberResult(UseCaseResult):
    """Result of adding member."""
    user: Optional[User] = None


@dataclass
class RemoveMemberRequest:
    """Request to remove member from organization."""
    requester_id: int
    organization_id: int
    user_id: int


@dataclass
class RemoveMemberResult(UseCaseResult):
    """Result of removing member."""
    pass


@dataclass
class ListMembersRequest:
    """Request to list organization members."""
    requester_id: int
    organization_id: int


@dataclass
class ListMembersResult(UseCaseResult):
    """Result of listing members."""
    members: List[User] = None
    total: Optional[int] = None


@dataclass
class GetUsageStatsRequest:
    """Request to get organization usage statistics."""
    requester_id: int
    organization_id: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class GetUsageStatsResult(UseCaseResult):
    """Result of getting usage stats."""
    total_streams: Optional[int] = None
    total_highlights: Optional[int] = None
    total_api_calls: Optional[int] = None
    storage_used_gb: Optional[float] = None
    quota_usage: Optional[dict] = None


@dataclass
class UpgradePlanRequest:
    """Request to upgrade organization plan."""
    requester_id: int
    organization_id: int
    new_plan: str


@dataclass
class UpgradePlanResult(UseCaseResult):
    """Result of plan upgrade."""
    organization: Optional[Organization] = None
    previous_plan: Optional[str] = None


class OrganizationManagementUseCase(UseCase[GetOrganizationRequest, GetOrganizationResult]):
    """Use case for organization management operations."""
    
    def __init__(
        self,
        org_repo: OrganizationRepository,
        user_repo: UserRepository,
        org_service: OrganizationManagementService
    ):
        """Initialize organization management use case.
        
        Args:
            org_repo: Repository for organization operations
            user_repo: Repository for user operations
            org_service: Organization management domain service
        """
        self.org_repo = org_repo
        self.user_repo = user_repo
        self.org_service = org_service
    
    async def get_organization(self, request: GetOrganizationRequest) -> GetOrganizationResult:
        """Get organization details.
        
        Args:
            request: Get organization request
            
        Returns:
            Organization details result
        """
        try:
            # Get organization
            org = await self.org_repo.get(request.organization_id)
            if not org:
                return GetOrganizationResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["Organization not found"]
                )
            
            # Check permission
            if not await self.org_service.user_has_access(request.requester_id, org.id):
                return GetOrganizationResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Access denied"]
                )
            
            # Get member count
            members = await self.user_repo.get_by_organization(request.organization_id)
            member_count = len(members)
            
            # Get basic usage stats
            usage_stats = await self.org_service.get_current_usage(request.organization_id)
            
            return GetOrganizationResult(
                status=ResultStatus.SUCCESS,
                organization=org,
                member_count=member_count,
                usage_stats=usage_stats,
                message="Organization retrieved successfully"
            )
            
        except Exception as e:
            return GetOrganizationResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to get organization: {str(e)}"]
            )
    
    async def update_organization(self, request: UpdateOrganizationRequest) -> UpdateOrganizationResult:
        """Update organization details.
        
        Args:
            request: Update organization request
            
        Returns:
            Update result
        """
        try:
            # Get organization
            org = await self.org_repo.get(request.organization_id)
            if not org:
                return UpdateOrganizationResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["Organization not found"]
                )
            
            # Check permission (must be owner)
            if org.owner_id != request.requester_id:
                return UpdateOrganizationResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Only organization owner can update details"]
                )
            
            # Update name if provided
            if request.name:
                try:
                    new_name = CompanyName(request.name)
                    org = org.change_name(new_name)
                except ValueError as e:
                    return UpdateOrganizationResult(
                        status=ResultStatus.VALIDATION_ERROR,
                        errors=[str(e)]
                    )
            
            # Update settings if provided
            if request.settings:
                org = org.update_settings(request.settings)
            
            # Save changes
            saved_org = await self.org_repo.save(org)
            
            return UpdateOrganizationResult(
                status=ResultStatus.SUCCESS,
                organization=saved_org,
                message="Organization updated successfully"
            )
            
        except DuplicateEntityError as e:
            return UpdateOrganizationResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)]
            )
        except Exception as e:
            return UpdateOrganizationResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to update organization: {str(e)}"]
            )
    
    async def add_member(self, request: AddMemberRequest) -> AddMemberResult:
        """Add member to organization.
        
        Args:
            request: Add member request
            
        Returns:
            Add member result
        """
        try:
            # Use organization service to add member
            user = await self.org_service.add_member(
                organization_id=request.organization_id,
                user_email=request.user_email,
                added_by_id=request.requester_id,
                role=request.role
            )
            
            return AddMemberResult(
                status=ResultStatus.SUCCESS,
                user=user,
                message="Member added successfully"
            )
            
        except EntityNotFoundError as e:
            return AddMemberResult(
                status=ResultStatus.NOT_FOUND,
                errors=[str(e)]
            )
        except UnauthorizedAccessError as e:
            return AddMemberResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=[str(e)]
            )
        except BusinessRuleViolation as e:
            return AddMemberResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)]
            )
        except Exception as e:
            return AddMemberResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to add member: {str(e)}"]
            )
    
    async def remove_member(self, request: RemoveMemberRequest) -> RemoveMemberResult:
        """Remove member from organization.
        
        Args:
            request: Remove member request
            
        Returns:
            Remove member result
        """
        try:
            # Use organization service to remove member
            await self.org_service.remove_member(
                organization_id=request.organization_id,
                user_id=request.user_id,
                removed_by_id=request.requester_id
            )
            
            return RemoveMemberResult(
                status=ResultStatus.SUCCESS,
                message="Member removed successfully"
            )
            
        except EntityNotFoundError as e:
            return RemoveMemberResult(
                status=ResultStatus.NOT_FOUND,
                errors=[str(e)]
            )
        except UnauthorizedAccessError as e:
            return RemoveMemberResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=[str(e)]
            )
        except BusinessRuleViolation as e:
            return RemoveMemberResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)]
            )
        except Exception as e:
            return RemoveMemberResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to remove member: {str(e)}"]
            )
    
    async def list_members(self, request: ListMembersRequest) -> ListMembersResult:
        """List organization members.
        
        Args:
            request: List members request
            
        Returns:
            List of members
        """
        try:
            # Check permission
            if not await self.org_service.user_has_access(request.requester_id, request.organization_id):
                return ListMembersResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Access denied"]
                )
            
            # Get members
            members = await self.user_repo.get_by_organization(request.organization_id)
            
            return ListMembersResult(
                status=ResultStatus.SUCCESS,
                members=members,
                total=len(members),
                message=f"Found {len(members)} members"
            )
            
        except Exception as e:
            return ListMembersResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to list members: {str(e)}"]
            )
    
    async def get_usage_stats(self, request: GetUsageStatsRequest) -> GetUsageStatsResult:
        """Get organization usage statistics.
        
        Args:
            request: Get usage stats request
            
        Returns:
            Usage statistics
        """
        try:
            # Check permission
            if not await self.org_service.user_has_access(request.requester_id, request.organization_id):
                return GetUsageStatsResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Access denied"]
                )
            
            # Get detailed usage stats
            usage_summary = await self.org_service.get_usage_summary(
                organization_id=request.organization_id,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Get current usage vs quotas
            current_usage = await self.org_service.get_current_usage(request.organization_id)
            
            return GetUsageStatsResult(
                status=ResultStatus.SUCCESS,
                total_streams=usage_summary.get("total_streams", 0),
                total_highlights=usage_summary.get("total_highlights", 0),
                total_api_calls=usage_summary.get("total_api_calls", 0),
                storage_used_gb=usage_summary.get("storage_used_gb", 0.0),
                quota_usage=current_usage,
                message="Usage statistics retrieved successfully"
            )
            
        except Exception as e:
            return GetUsageStatsResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to get usage stats: {str(e)}"]
            )
    
    async def upgrade_plan(self, request: UpgradePlanRequest) -> UpgradePlanResult:
        """Upgrade organization plan.
        
        Args:
            request: Upgrade plan request
            
        Returns:
            Upgrade result
        """
        try:
            # Parse plan type
            try:
                new_plan_type = PlanType(request.new_plan.upper())
            except ValueError:
                return UpgradePlanResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=[f"Invalid plan type: {request.new_plan}"]
                )
            
            # Use organization service to upgrade plan
            org = await self.org_service.upgrade_plan(
                organization_id=request.organization_id,
                new_plan=new_plan_type,
                upgraded_by_id=request.requester_id
            )
            
            return UpgradePlanResult(
                status=ResultStatus.SUCCESS,
                organization=org,
                previous_plan=request.new_plan,  # Would be tracked in service
                message="Plan upgraded successfully"
            )
            
        except EntityNotFoundError as e:
            return UpgradePlanResult(
                status=ResultStatus.NOT_FOUND,
                errors=[str(e)]
            )
        except UnauthorizedAccessError as e:
            return UpgradePlanResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=[str(e)]
            )
        except BusinessRuleViolation as e:
            return UpgradePlanResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)]
            )
        except Exception as e:
            return UpgradePlanResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to upgrade plan: {str(e)}"]
            )
    
    async def execute(self, request: GetOrganizationRequest) -> GetOrganizationResult:
        """Execute get organization (default use case method).
        
        Args:
            request: Get organization request
            
        Returns:
            Organization result
        """
        return await self.get_organization(request)