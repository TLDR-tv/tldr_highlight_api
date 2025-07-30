"""Organization repository implementation."""

from typing import Optional, List, Dict
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, or_

from src.domain.repositories.organization_repository import (
    OrganizationRepository as IOrganizationRepository,
)
from src.domain.entities.organization import Organization, PlanType
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp
from src.domain.exceptions import EntityNotFoundError
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.organization import (
    Organization as OrganizationModel,
)
from src.infrastructure.persistence.models.user import User as UserModel
from src.infrastructure.persistence.mappers.organization_mapper import (
    OrganizationMapper,
)


class OrganizationRepository(
    BaseRepository[Organization, OrganizationModel, int], IOrganizationRepository
):
    """Concrete implementation of OrganizationRepository using SQLAlchemy."""

    def __init__(self, session):
        """Initialize OrganizationRepository with session."""
        super().__init__(
            session=session, model_class=OrganizationModel, mapper=OrganizationMapper()
        )

    async def get_by_owner(self, owner_id: int) -> List[Organization]:
        """Get organizations owned by a user.

        Args:
            owner_id: User ID of the owner

        Returns:
            List of organizations owned by the user
        """
        stmt = (
            select(OrganizationModel)
            .where(OrganizationModel.owner_id == owner_id)
            .order_by(OrganizationModel.created_at.desc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_member(self, user_id: int) -> List[Organization]:
        """Get organizations where user is a member.

        Args:
            user_id: User ID

        Returns:
            List of organizations where user is a member
        """
        # This requires a many-to-many relationship or JSON field query
        # Assuming we have a members relationship or use JSON containment
        stmt = (
            select(OrganizationModel)
            .join(UserModel, OrganizationModel.members)
            .where(UserModel.id == user_id)
            .order_by(OrganizationModel.created_at.desc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_name(self, name: CompanyName) -> Optional[Organization]:
        """Get organization by exact name.

        Args:
            name: Company name

        Returns:
            Organization if found, None otherwise
        """
        stmt = select(OrganizationModel).where(OrganizationModel.name == name.value)

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        return self.mapper.to_domain(model) if model else None

    async def search_by_name(self, query: str) -> List[Organization]:
        """Search organizations by name (partial match).

        Args:
            query: Search query for organization name

        Returns:
            List of organizations matching the search query
        """
        search_term = f"%{query.lower()}%"

        stmt = (
            select(OrganizationModel)
            .where(func.lower(OrganizationModel.name).like(search_term))
            .order_by(
                # Exact matches first, then partial matches
                func.lower(OrganizationModel.name) == query.lower(),
                OrganizationModel.name,
            )
            .limit(50)
        )  # Reasonable limit for search results

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_plan_type(self, plan_type: PlanType) -> List[Organization]:
        """Get all organizations with specific plan type.

        Args:
            plan_type: Plan type to filter by

        Returns:
            List of organizations with the specified plan type
        """
        stmt = (
            select(OrganizationModel)
            .where(OrganizationModel.plan_type == plan_type.value)
            .order_by(OrganizationModel.created_at.desc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_active_organizations(self) -> List[Organization]:
        """Get all active organizations.

        Returns:
            List of active organizations
        """
        stmt = (
            select(OrganizationModel)
            .where(
                and_(
                    OrganizationModel.is_active.is_(True),
                    or_(
                        OrganizationModel.subscription_ends_at.is_(None),
                        OrganizationModel.subscription_ends_at > func.now(),
                    ),
                )
            )
            .order_by(OrganizationModel.created_at.desc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def count_by_plan_type(self) -> Dict[PlanType, int]:
        """Get count of organizations by plan type.

        Returns:
            Dictionary mapping plan types to counts
        """
        stmt = select(OrganizationModel.plan_type, func.count()).group_by(
            OrganizationModel.plan_type
        )

        result = await self.session.execute(stmt)
        counts = {}

        for row in result:
            plan_type = PlanType(row[0])
            count = row[1]
            counts[plan_type] = count

        # Ensure all plan types are represented (with 0 if no orgs)
        for plan_type in PlanType:
            if plan_type not in counts:
                counts[plan_type] = 0

        return counts

    async def get_expiring_soon(self, days: int = 30) -> List[Organization]:
        """Get organizations with subscriptions expiring soon.

        Args:
            days: Number of days to look ahead for expiring subscriptions

        Returns:
            List of organizations with subscriptions expiring within the timeframe
        """
        expiration_cutoff = datetime.utcnow() + timedelta(days=days)

        stmt = (
            select(OrganizationModel)
            .where(
                and_(
                    OrganizationModel.is_active.is_(True),
                    OrganizationModel.subscription_ends_at.isnot(None),
                    OrganizationModel.subscription_ends_at <= expiration_cutoff,
                    OrganizationModel.subscription_ends_at > func.now(),
                )
            )
            .order_by(OrganizationModel.subscription_ends_at.asc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_with_usage_stats(
        self, organization_id: int
    ) -> Optional[Organization]:
        """Get organization with usage statistics.

        Args:
            organization_id: Organization ID

        Returns:
            Organization with usage statistics if found, None otherwise
        """
        # Get the organization first
        org = await self.get(organization_id)
        if not org:
            return None

        # This would typically load related usage data
        # For now, we'll just return the organization
        # In a real implementation, you might eager load usage records
        stmt = (
            select(OrganizationModel)
            .where(OrganizationModel.id == organization_id)
            .options(
                # Add selectinload for related usage data when available
                # selectinload(OrganizationModel.usage_records)
            )
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        return self.mapper.to_domain(model) if model else None

    async def get_organization_analytics(
        self, organization_id: int, days: int = 30
    ) -> Dict[str, any]:
        """Get analytics data for an organization.

        Args:
            organization_id: Organization ID
            days: Number of days to look back for analytics

        Returns:
            Dictionary with organization analytics data
        """
        org = await self.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        # Calculate time window
        start_date = datetime.utcnow() - timedelta(days=days)

        # Get member count (this would be from a members table or relationship)
        # For now, we'll use the stored member_ids from the domain entity
        current_members = len(org.member_ids) + 1  # +1 for owner

        # Get plan utilization
        plan_limits = org.plan_limits

        # In a real implementation, you'd query usage records, streams, etc.
        # For now, return basic analytics structure
        return {
            "organization_id": organization_id,
            "name": org.name.value,
            "plan_type": org.plan_type.value,
            "is_active": org.is_active,
            "subscription_status": "active"
            if org.is_subscription_active
            else "inactive",
            "member_stats": {
                "current_members": current_members,
                "max_members": plan_limits.team_members,
                "utilization_percentage": (current_members / plan_limits.team_members)
                * 100,
            },
            "plan_limits": {
                "api_rate_limit_per_minute": plan_limits.api_rate_limit_per_minute,
                "monthly_processing_minutes": plan_limits.monthly_processing_minutes,
                "concurrent_streams": plan_limits.concurrent_streams,
                "webhook_endpoints": plan_limits.webhook_endpoints,
                "api_keys": plan_limits.api_keys,
                "team_members": plan_limits.team_members,
            },
            "created_at": org.created_at.iso_string,
            "subscription_started_at": org.subscription_started_at.iso_string
            if org.subscription_started_at
            else None,
            "subscription_ends_at": org.subscription_ends_at.iso_string
            if org.subscription_ends_at
            else None,
        }

    async def get_organizations_by_usage(
        self, min_usage_minutes: int = 0, period_days: int = 30
    ) -> List[Organization]:
        """Get organizations by usage criteria.

        Args:
            min_usage_minutes: Minimum usage in minutes
            period_days: Period to check usage over

        Returns:
            List of organizations meeting usage criteria
        """
        # This would typically join with usage records
        # For now, return active organizations as a placeholder
        return await self.get_active_organizations()

    async def bulk_update_subscriptions(self, updates: List[Dict]) -> int:
        """Bulk update subscription information for multiple organizations.

        Args:
            updates: List of dictionaries with organization updates
                   Each dict should have 'id' and update fields

        Returns:
            Number of organizations updated
        """
        if not updates:
            return 0

        updated_count = 0

        for update_data in updates:
            org_id = update_data.get("id")
            if not org_id:
                continue

            # Get existing organization
            org = await self.get(org_id)
            if not org:
                continue

            # Create updated organization
            updated_org = Organization(
                id=org.id,
                name=org.name,
                owner_id=org.owner_id,
                plan_type=PlanType(update_data.get("plan_type", org.plan_type.value)),
                member_ids=org.member_ids.copy(),
                custom_limits=update_data.get("custom_limits", org.custom_limits),
                settings=update_data.get("settings", org.settings),
                subscription_started_at=Timestamp(
                    update_data["subscription_started_at"]
                )
                if update_data.get("subscription_started_at")
                else org.subscription_started_at,
                subscription_ends_at=Timestamp(update_data["subscription_ends_at"])
                if update_data.get("subscription_ends_at")
                else org.subscription_ends_at,
                is_active=update_data.get("is_active", org.is_active),
                created_at=org.created_at,
                updated_at=Timestamp.now(),
            )

            # Save updated organization
            await self.save(updated_org)
            updated_count += 1

        return updated_count

    async def cleanup_inactive_organizations(self, inactive_days: int = 365) -> int:
        """Clean up organizations that have been inactive for a long time.

        Args:
            inactive_days: Number of days of inactivity before cleanup

        Returns:
            Number of organizations cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=inactive_days)

        # Find inactive organizations
        stmt = select(OrganizationModel).where(
            and_(
                OrganizationModel.is_active.is_(False),
                or_(
                    OrganizationModel.subscription_ends_at < cutoff_date,
                    OrganizationModel.updated_at < cutoff_date,
                ),
            )
        )

        result = await self.session.execute(stmt)
        inactive_orgs = list(result.scalars().unique())

        # In a real implementation, you'd want to:
        # 1. Archive data rather than delete
        # 2. Notify stakeholders
        # 3. Clean up related resources

        # For now, just count what would be cleaned up
        return len(inactive_orgs)
