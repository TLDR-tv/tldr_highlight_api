"""Organization workflow application service.

This application service handles organization lifecycle, member management,
and plan upgrades. Orchestrates between domain entities and repositories.
"""

from typing import Dict, Any, List
import logfire

from src.domain.entities.organization import Organization, PlanType
from src.domain.entities.usage_record import UsageRecord, UsageType
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp
from src.domain.exceptions import (
    EntityNotFoundError,
    DuplicateEntityError,
    BusinessRuleViolation,
    UnauthorizedOperation,
)
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.webhook_repository import WebhookRepository


class OrganizationWorkflow:
    """Application service for organization management.
    
    Handles organization lifecycle, member management, and plan changes.
    All limits are unlimited for first client.
    """
    
    def __init__(
        self,
        org_repo: OrganizationRepository,
        user_repo: UserRepository,
        usage_repo: UsageRecordRepository,
        stream_repo: StreamRepository,
        api_key_repo: APIKeyRepository,
        webhook_repo: WebhookRepository,
    ):
        """Initialize organization workflow.
        
        Args:
            org_repo: Repository for organization operations
            user_repo: Repository for user operations
            usage_repo: Repository for usage tracking
            stream_repo: Repository for stream operations
            api_key_repo: Repository for API key operations
            webhook_repo: Repository for webhook operations
        """
        self.org_repo = org_repo
        self.user_repo = user_repo
        self.usage_repo = usage_repo
        self.stream_repo = stream_repo
        self.api_key_repo = api_key_repo
        self.webhook_repo = webhook_repo
        self.logger = logfire.get_logger(__name__)
    
    async def create_organization(
        self, owner_id: int, name: str, plan_type: PlanType = PlanType.STARTER
    ) -> Organization:
        """Create a new organization.
        
        Args:
            owner_id: User ID of the organization owner
            name: Organization name
            plan_type: Initial plan type
            
        Returns:
            Created organization
            
        Raises:
            EntityNotFoundError: If owner doesn't exist
            DuplicateEntityError: If organization name already exists
            BusinessRuleViolation: If user already owns an organization
        """
        # Validate owner exists
        owner = await self.user_repo.get_by_id(owner_id)
        if not owner:
            raise EntityNotFoundError(f"User {owner_id} not found")
        
        # Check if user already owns an organization
        existing_orgs = await self.org_repo.get_by_owner(owner_id)
        if existing_orgs:
            raise BusinessRuleViolation(
                "User already owns an organization",
                entity_type="User",
                entity_id=owner_id
            )
        
        # Check if organization name is taken
        company_name = CompanyName(name)
        existing = await self.org_repo.get_by_name(company_name.value)
        if existing:
            raise DuplicateEntityError(
                entity_type="Organization",
                existing_value=name
            )
        
        # Create organization using factory method
        org = Organization.create(
            name=company_name,
            owner_id=owner_id,
            plan_type=plan_type
        )
        
        # Save organization
        saved_org = await self.org_repo.save(org)
        
        # The repository should handle raising the creation event after getting ID
        
        self.logger.info(
            "Organization created",
            organization_id=saved_org.id,
            owner_id=owner_id,
            plan_type=plan_type.value
        )
        
        return saved_org
    
    async def add_member(
        self, organization_id: int, user_id: int, added_by_user_id: int
    ) -> Organization:
        """Add a member to an organization.
        
        Args:
            organization_id: Organization ID
            user_id: User ID to add
            added_by_user_id: User ID performing the action
            
        Returns:
            Updated organization
            
        Raises:
            EntityNotFoundError: If organization or user doesn't exist
            UnauthorizedOperation: If user cannot add members
            BusinessRuleViolation: If business rules violated
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")
        
        # Validate user exists
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")
        
        # Check if user is already in another organization
        user_orgs = await self.org_repo.get_by_member(user_id)
        if user_orgs:
            raise BusinessRuleViolation(
                "User is already a member of another organization",
                entity_type="User",
                entity_id=user_id
            )
        
        # Add member using aggregate method (handles authorization and rules)
        org.add_member(user_id, added_by_user_id)
        
        # Save updated organization
        saved_org = await self.org_repo.save(org)
        
        self.logger.info(
            "Member added to organization",
            organization_id=organization_id,
            user_id=user_id,
            added_by=added_by_user_id
        )
        
        return saved_org
    
    async def remove_member(
        self, organization_id: int, user_id: int, removed_by_user_id: int
    ) -> Organization:
        """Remove a member from an organization.
        
        Args:
            organization_id: Organization ID
            user_id: User ID to remove
            removed_by_user_id: User ID performing the action
            
        Returns:
            Updated organization
            
        Raises:
            EntityNotFoundError: If organization doesn't exist
            UnauthorizedOperation: If user cannot remove members
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")
        
        # Remove member using aggregate method (handles authorization)
        org.remove_member(user_id, removed_by_user_id)
        
        # Save updated organization
        saved_org = await self.org_repo.save(org)
        
        self.logger.info(
            "Member removed from organization",
            organization_id=organization_id,
            user_id=user_id,
            removed_by=removed_by_user_id
        )
        
        return saved_org
    
    async def upgrade_plan(
        self, organization_id: int, new_plan: PlanType, upgraded_by_user_id: int
    ) -> Organization:
        """Upgrade organization plan.
        
        Args:
            organization_id: Organization ID
            new_plan: New plan type
            upgraded_by_user_id: User ID performing the upgrade
            
        Returns:
            Updated organization
            
        Raises:
            EntityNotFoundError: If organization doesn't exist
            UnauthorizedOperation: If user cannot upgrade plan
            BusinessRuleViolation: If downgrade attempted
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")
        
        # Check if it's an upgrade (not downgrade)
        plan_order = {
            PlanType.STARTER: 1,
            PlanType.PROFESSIONAL: 2,
            PlanType.ENTERPRISE: 3,
            PlanType.CUSTOM: 4,
        }
        
        if plan_order.get(new_plan, 0) < plan_order.get(org.plan_type, 0):
            raise BusinessRuleViolation(
                "Plan downgrades must be handled through support",
                entity_type="Organization",
                entity_id=organization_id
            )
        
        # Upgrade plan using aggregate method (handles authorization)
        old_plan = org.plan_type
        org.upgrade_plan(new_plan, upgraded_by_user_id)
        
        # Save updated organization
        saved_org = await self.org_repo.save(org)
        
        # Create usage record for tracking
        usage_record = UsageRecord(
            id=None,
            user_id=upgraded_by_user_id,
            organization_id=organization_id,
            usage_type=UsageType.API_CALL,
            resource_type="plan_upgrade",
            quantity=1,
            unit="upgrade",
            billable=False,  # Plan upgrades are not usage-based billing
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )
        await self.usage_repo.save(usage_record)
        
        self.logger.info(
            "Organization plan upgraded",
            organization_id=organization_id,
            old_plan=old_plan.value,
            new_plan=new_plan.value,
            upgraded_by=upgraded_by_user_id
        )
        
        return saved_org
    
    async def get_usage_statistics(self, organization_id: int) -> Dict[str, Any]:
        """Get current usage statistics (no limits enforced).
        
        Args:
            organization_id: Organization ID
            
        Returns:
            Dictionary with usage statistics for tracking
            
        Raises:
            EntityNotFoundError: If organization doesn't exist
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")
        
        # Get current usage for various resources (for statistics only)
        
        # API Keys
        api_keys = await self.api_key_repo.get_by_user(org.owner_id, active_only=True)
        api_key_count = len(api_keys)
        
        # Webhooks
        webhooks = await self.webhook_repo.get_by_user(org.owner_id)
        webhook_count = len(webhooks)
        
        # Concurrent streams
        active_streams = await self.stream_repo.get_active_streams()
        org_member_ids = [org.owner_id] + org.member_ids
        concurrent_streams = sum(
            1 for s in active_streams if s.user_id in org_member_ids
        )
        
        # Monthly processing minutes
        start_of_month = Timestamp.now().start_of_month()
        end_of_month = Timestamp.now().end_of_month()
        
        monthly_usage = await self.usage_repo.get_by_organization_and_period(
            organization_id, start_of_month.value, end_of_month.value
        )
        
        processing_minutes = sum(
            record.quantity for record in monthly_usage
            if record.usage_type == UsageType.STREAM_PROCESSING
        )
        
        # Return usage statistics (no limits enforced - unlimited for all)
        return {
            "api_keys": {"used": api_key_count},
            "webhooks": {"used": webhook_count},
            "concurrent_streams": {"used": concurrent_streams},
            "monthly_processing_minutes": {"used": processing_minutes},
            "team_members": {"used": org.member_count},
            "note": "All limits are unlimited for first client",
        }
    
    async def get_organization_analytics(
        self, organization_id: int, days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for an organization.
        
        Args:
            organization_id: Organization ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")
        
        # Calculate date range
        end_date = Timestamp.now()
        start_date = end_date.subtract_days(days)
        
        # Get usage by type
        usage_records = await self.usage_repo.get_by_organization_and_period(
            organization_id, start_date.value, end_date.value
        )
        
        # Aggregate by usage type
        usage_by_type = {}
        for record in usage_records:
            usage_type = record.usage_type.value
            if usage_type not in usage_by_type:
                usage_by_type[usage_type] = {
                    "quantity": 0.0,
                    "unit": record.unit,
                    "count": 0,
                    "total_cost": 0.0,
                }
            
            usage_by_type[usage_type]["quantity"] += record.quantity
            usage_by_type[usage_type]["count"] += 1
            usage_by_type[usage_type]["total_cost"] += record.total_cost or 0
        
        # Get member activity
        member_activity = []
        for member_id in [org.owner_id] + org.member_ids:
            user = await self.user_repo.get_by_id(member_id)
            if user:
                # Get user's recent streams
                user_streams = await self.stream_repo.get_by_user(
                    member_id, created_after=start_date
                )
                
                # Find last activity
                last_active = None
                if user_streams:
                    last_active = max(s.created_at for s in user_streams)
                
                member_activity.append({
                    "user_id": member_id,
                    "email": user.email.value,
                    "is_owner": member_id == org.owner_id,
                    "streams_created": len(user_streams),
                    "last_active": last_active.iso_string if last_active else None,
                })
        
        # Compile analytics
        return {
            "organization_id": organization_id,
            "period_start": start_date.iso_string,
            "period_end": end_date.iso_string,
            "usage_by_type": usage_by_type,
            "member_activity": member_activity,
            "usage_statistics": await self.get_usage_statistics(organization_id),
            "total_members": org.member_count,
            "plan_type": org.plan_type.value,
            "is_active": org.is_subscription_active,
        }
    
    async def deactivate_organization(
        self, organization_id: int, reason: str
    ) -> Organization:
        """Deactivate an organization.
        
        Args:
            organization_id: Organization ID
            reason: Reason for deactivation
            
        Returns:
            Updated organization
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")
        
        # Deactivate using aggregate method
        org.deactivate(reason)
        
        # Save updated organization
        saved_org = await self.org_repo.save(org)
        
        self.logger.info(
            "Organization deactivated",
            organization_id=organization_id,
            reason=reason
        )
        
        return saved_org