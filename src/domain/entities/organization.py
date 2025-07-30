"""Organization domain entity."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.company_name import CompanyName


class PlanType(Enum):
    """Organization subscription plan types."""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class PlanLimits:
    """Value object for plan limits."""

    api_rate_limit_per_minute: int
    monthly_processing_minutes: int
    concurrent_streams: int
    webhook_endpoints: int
    api_keys: int
    team_members: int

    @classmethod
    def for_plan(cls, plan_type: PlanType) -> "PlanLimits":
        """Get limits for a specific plan type."""
        limits = {
            PlanType.STARTER: cls(
                api_rate_limit_per_minute=60,
                monthly_processing_minutes=1000,
                concurrent_streams=1,
                webhook_endpoints=2,
                api_keys=2,
                team_members=3,
            ),
            PlanType.PROFESSIONAL: cls(
                api_rate_limit_per_minute=300,
                monthly_processing_minutes=10000,
                concurrent_streams=5,
                webhook_endpoints=10,
                api_keys=10,
                team_members=10,
            ),
            PlanType.ENTERPRISE: cls(
                api_rate_limit_per_minute=1000,
                monthly_processing_minutes=100000,
                concurrent_streams=20,
                webhook_endpoints=50,
                api_keys=50,
                team_members=100,
            ),
            PlanType.CUSTOM: cls(
                api_rate_limit_per_minute=2000,
                monthly_processing_minutes=1000000,
                concurrent_streams=100,
                webhook_endpoints=100,
                api_keys=100,
                team_members=1000,
            ),
        }
        return limits[plan_type]


@dataclass
class Organization(Entity[int]):
    """Domain entity representing an organization.

    Organizations group users together and define subscription
    plans and usage limits.
    """

    name: CompanyName
    owner_id: int
    plan_type: PlanType

    # Member management
    member_ids: List[int] = field(default_factory=list)

    # Custom settings
    custom_limits: Optional[Dict[str, int]] = None
    settings: Dict[str, Any] = field(default_factory=dict)

    # Subscription details
    subscription_started_at: Optional[Timestamp] = None
    subscription_ends_at: Optional[Timestamp] = None
    is_active: bool = True

    @property
    def plan_limits(self) -> PlanLimits:
        """Get usage limits based on plan."""
        base_limits = PlanLimits.for_plan(self.plan_type)

        # Apply custom limits if available
        if self.custom_limits:
            for key, value in self.custom_limits.items():
                if hasattr(base_limits, key):
                    setattr(base_limits, key, value)

        return base_limits

    @property
    def is_subscription_active(self) -> bool:
        """Check if subscription is currently active."""
        if not self.is_active:
            return False

        if self.subscription_ends_at:
            return not self.subscription_ends_at.is_before(Timestamp.now())

        return True

    @property
    def member_count(self) -> int:
        """Get current member count including owner."""
        return len(self.member_ids) + 1  # +1 for owner

    def can_add_member(self) -> bool:
        """Check if organization can add more members."""
        return self.member_count < self.plan_limits.team_members

    def add_member(self, user_id: int) -> "Organization":
        """Add a member to the organization."""
        if user_id == self.owner_id:
            raise ValueError("Owner is already a member")

        if user_id in self.member_ids:
            return self

        if not self.can_add_member():
            raise ValueError(
                f"Organization has reached member limit of {self.plan_limits.team_members}"
            )

        new_member_ids = self.member_ids.copy()
        new_member_ids.append(user_id)

        return Organization(
            id=self.id,
            name=self.name,
            owner_id=self.owner_id,
            plan_type=self.plan_type,
            member_ids=new_member_ids,
            custom_limits=self.custom_limits.copy() if self.custom_limits else None,
            settings=self.settings.copy(),
            subscription_started_at=self.subscription_started_at,
            subscription_ends_at=self.subscription_ends_at,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def remove_member(self, user_id: int) -> "Organization":
        """Remove a member from the organization."""
        if user_id not in self.member_ids:
            return self

        new_member_ids = [uid for uid in self.member_ids if uid != user_id]

        return Organization(
            id=self.id,
            name=self.name,
            owner_id=self.owner_id,
            plan_type=self.plan_type,
            member_ids=new_member_ids,
            custom_limits=self.custom_limits.copy() if self.custom_limits else None,
            settings=self.settings.copy(),
            subscription_started_at=self.subscription_started_at,
            subscription_ends_at=self.subscription_ends_at,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def upgrade_plan(self, new_plan: PlanType) -> "Organization":
        """Upgrade organization to a new plan."""
        if new_plan == self.plan_type:
            return self

        return Organization(
            id=self.id,
            name=self.name,
            owner_id=self.owner_id,
            plan_type=new_plan,
            member_ids=self.member_ids.copy(),
            custom_limits=self.custom_limits.copy() if self.custom_limits else None,
            settings=self.settings.copy(),
            subscription_started_at=Timestamp.now(),
            subscription_ends_at=self.subscription_ends_at,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def deactivate(self) -> "Organization":
        """Deactivate the organization."""
        return Organization(
            id=self.id,
            name=self.name,
            owner_id=self.owner_id,
            plan_type=self.plan_type,
            member_ids=self.member_ids.copy(),
            custom_limits=self.custom_limits.copy() if self.custom_limits else None,
            settings=self.settings.copy(),
            subscription_started_at=self.subscription_started_at,
            subscription_ends_at=self.subscription_ends_at,
            is_active=False,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )
