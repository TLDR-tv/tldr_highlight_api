"""Organization management domain service.

This service handles organization lifecycle, member management,
plan upgrades, and quota enforcement.
"""

from typing import Dict, Any

from src.domain.services.base import BaseDomainService
from src.domain.entities.organization import Organization, PlanType
from src.domain.entities.usage_record import UsageRecord, UsageType
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.webhook_repository import WebhookRepository
from src.domain.exceptions import (
    EntityNotFoundError,
    DuplicateEntityError,
    BusinessRuleViolation,
    QuotaExceededError,
    UnauthorizedAccessError,
)


class OrganizationManagementService(BaseDomainService):
    """Domain service for organization management.

    Handles organization lifecycle, member management, plan changes,
    and enforces business rules around quotas and limits.
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
        """Initialize organization management service.

        Args:
            org_repo: Repository for organization operations
            user_repo: Repository for user operations
            usage_repo: Repository for usage tracking
            stream_repo: Repository for stream operations
            api_key_repo: Repository for API key operations
            webhook_repo: Repository for webhook operations
        """
        super().__init__()
        self.org_repo = org_repo
        self.user_repo = user_repo
        self.usage_repo = usage_repo
        self.stream_repo = stream_repo
        self.api_key_repo = api_key_repo
        self.webhook_repo = webhook_repo

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
        owner = await self.user_repo.get(owner_id)
        if not owner:
            raise EntityNotFoundError(f"User {owner_id} not found")

        # Check if user already owns an organization
        existing_orgs = await self.org_repo.get_by_owner(owner_id)
        if existing_orgs:
            raise BusinessRuleViolation(
                "User already owns an organization. Multiple organizations per user not supported."
            )

        # Check if organization name is taken
        company_name = CompanyName(name)
        existing = await self.org_repo.get_by_name(company_name)
        if existing:
            raise DuplicateEntityError(f"Organization name '{name}' already exists")

        # Create organization
        org = Organization(
            id=None,
            name=company_name,
            owner_id=owner_id,
            plan_type=plan_type,
            subscription_started_at=Timestamp.now(),
            is_active=True,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )

        # Save organization
        saved_org = await self.org_repo.save(org)

        self.logger.info(f"Created organization {saved_org.id} for user {owner_id}")

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
            UnauthorizedAccessError: If user cannot add members
            QuotaExceededError: If member limit exceeded
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        # Check authorization
        if added_by_user_id != org.owner_id:
            raise UnauthorizedAccessError("Only organization owner can add members")

        # Validate user exists
        user = await self.user_repo.get(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")

        # Check if user is already in another organization
        user_orgs = await self.org_repo.get_by_member(user_id)
        if user_orgs:
            raise BusinessRuleViolation(
                "User is already a member of another organization"
            )

        # Add member (will check quota)
        try:
            updated_org = org.add_member(user_id)
        except ValueError as e:
            if "member limit" in str(e):
                raise QuotaExceededError(str(e))
            raise BusinessRuleViolation(str(e))

        # Save updated organization
        saved_org = await self.org_repo.save(updated_org)

        self.logger.info(f"Added user {user_id} to organization {organization_id}")

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
            UnauthorizedAccessError: If user cannot remove members
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        # Check authorization
        if removed_by_user_id != org.owner_id and removed_by_user_id != user_id:
            raise UnauthorizedAccessError(
                "Only organization owner or the member themselves can remove members"
            )

        # Remove member
        updated_org = org.remove_member(user_id)

        # Save updated organization
        saved_org = await self.org_repo.save(updated_org)

        self.logger.info(f"Removed user {user_id} from organization {organization_id}")

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
            UnauthorizedAccessError: If user cannot upgrade plan
            BusinessRuleViolation: If downgrade attempted
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        # Check authorization
        if upgraded_by_user_id != org.owner_id:
            raise UnauthorizedAccessError("Only organization owner can change plans")

        # Check if it's an upgrade (not downgrade)
        plan_order = {
            PlanType.STARTER: 1,
            PlanType.PROFESSIONAL: 2,
            PlanType.ENTERPRISE: 3,
            PlanType.CUSTOM: 4,
        }

        if plan_order.get(new_plan, 0) < plan_order.get(org.plan_type, 0):
            raise BusinessRuleViolation(
                "Plan downgrades must be handled through support"
            )

        # Upgrade plan
        upgraded_org = org.upgrade_plan(new_plan)

        # Save updated organization
        saved_org = await self.org_repo.save(upgraded_org)

        # Create usage record for plan change
        usage_record = UsageRecord(
            id=None,
            user_id=upgraded_by_user_id,
            organization_id=organization_id,
            usage_type=UsageType.API_CALL,
            resource_type="plan_upgrade",
            quantity=1,
            unit="upgrade",
            metadata={"from_plan": org.plan_type.value, "to_plan": new_plan.value},
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )
        await self.usage_repo.save(usage_record)

        self.logger.info(
            f"Upgraded organization {organization_id} from {org.plan_type.value} to {new_plan.value}"
        )

        return saved_org

    async def check_and_enforce_quotas(self, organization_id: int) -> Dict[str, Any]:
        """Check current usage against plan quotas.

        Args:
            organization_id: Organization ID

        Returns:
            Dictionary with quota status for each resource

        Raises:
            EntityNotFoundError: If organization doesn't exist
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        plan_limits = org.plan_limits

        # Get current usage for various resources

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

        monthly_usage = await self.usage_repo.calculate_usage_totals(
            org.owner_id, UsageType.STREAM_PROCESSING, start_of_month, end_of_month
        )
        processing_minutes = monthly_usage.get("total_quantity", 0)

        # API rate limit is enforced at runtime, just report the limit

        quota_status = {
            "api_keys": {
                "used": api_key_count,
                "limit": plan_limits.api_keys,
                "available": plan_limits.api_keys - api_key_count,
                "at_limit": api_key_count >= plan_limits.api_keys,
            },
            "webhooks": {
                "used": webhook_count,
                "limit": plan_limits.webhook_endpoints,
                "available": plan_limits.webhook_endpoints - webhook_count,
                "at_limit": webhook_count >= plan_limits.webhook_endpoints,
            },
            "concurrent_streams": {
                "used": concurrent_streams,
                "limit": plan_limits.concurrent_streams,
                "available": plan_limits.concurrent_streams - concurrent_streams,
                "at_limit": concurrent_streams >= plan_limits.concurrent_streams,
            },
            "monthly_processing_minutes": {
                "used": processing_minutes,
                "limit": plan_limits.monthly_processing_minutes,
                "available": max(
                    0, plan_limits.monthly_processing_minutes - processing_minutes
                ),
                "at_limit": processing_minutes
                >= plan_limits.monthly_processing_minutes,
            },
            "team_members": {
                "used": org.member_count,
                "limit": plan_limits.team_members,
                "available": plan_limits.team_members - org.member_count,
                "at_limit": org.member_count >= plan_limits.team_members,
            },
            "api_rate_limit_per_minute": {
                "limit": plan_limits.api_rate_limit_per_minute
            },
        }

        return quota_status

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
        # Get base analytics from repository
        analytics = await self.org_repo.get_organization_analytics(
            organization_id, days
        )

        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        # Add usage analytics
        start_date = Timestamp.now().subtract_days(days)
        end_date = Timestamp.now()

        # Get usage by type
        usage_by_type = await self.usage_repo.aggregate_by_type(
            org.owner_id, start_date, end_date
        )

        # Get member activity
        member_activity = []
        for member_id in [org.owner_id] + org.member_ids:
            user = await self.user_repo.get(member_id)
            if user:
                # Get user's recent streams
                user_streams = await self.stream_repo.get_by_date_range(
                    start_date, end_date, member_id
                )

                member_activity.append(
                    {
                        "user_id": member_id,
                        "email": user.email.value,
                        "is_owner": member_id == org.owner_id,
                        "streams_created": len(user_streams),
                        "last_active": max(
                            (s.created_at for s in user_streams), default=None
                        ),
                    }
                )

        # Combine analytics
        analytics["usage_by_type"] = {
            usage_type.value: data for usage_type, data in usage_by_type.items()
        }
        analytics["member_activity"] = member_activity
        analytics["quota_status"] = await self.check_and_enforce_quotas(organization_id)

        return analytics

    async def handle_subscription_renewal(
        self, organization_id: int, extend_days: int = 30
    ) -> Organization:
        """Handle subscription renewal for an organization.

        Args:
            organization_id: Organization ID
            extend_days: Number of days to extend subscription

        Returns:
            Updated organization
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        # Calculate new subscription end date
        current_end = org.subscription_ends_at or Timestamp.now()
        new_end = current_end.add_days(extend_days)

        # Update organization
        renewed_org = Organization(
            id=org.id,
            name=org.name,
            owner_id=org.owner_id,
            plan_type=org.plan_type,
            member_ids=org.member_ids,
            custom_limits=org.custom_limits,
            settings=org.settings,
            subscription_started_at=org.subscription_started_at,
            subscription_ends_at=new_end,
            is_active=True,
            created_at=org.created_at,
            updated_at=Timestamp.now(),
        )

        # Save updated organization
        saved_org = await self.org_repo.save(renewed_org)

        self.logger.info(
            f"Renewed subscription for organization {organization_id} until {new_end.iso_string}"
        )

        return saved_org

    async def deactivate_expired_subscriptions(self) -> int:
        """Deactivate organizations with expired subscriptions.

        Returns:
            Number of organizations deactivated
        """
        # Get organizations with expired subscriptions
        expired_orgs = await self.org_repo.get_expiring_soon(days=0)

        deactivated_count = 0
        for org in expired_orgs:
            if org.subscription_ends_at and org.subscription_ends_at.is_before(
                Timestamp.now()
            ):
                # Deactivate organization
                deactivated_org = org.deactivate()
                await self.org_repo.save(deactivated_org)
                deactivated_count += 1

                self.logger.info(f"Deactivated expired organization {org.id}")

        return deactivated_count
