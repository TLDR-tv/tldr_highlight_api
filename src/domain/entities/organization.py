"""Organization domain entity."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from src.domain.entities.base import AggregateRoot
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.company_name import CompanyName
from src.domain.exceptions import (
    BusinessRuleViolation,
    QuotaExceeded,
    UnauthorizedOperation,
    InvalidStateTransition
)
from src.domain.events import (
    OrganizationCreatedEvent,
    MemberAddedEvent,
    MemberRemovedEvent,
    PlanUpgradedEvent,
    OrganizationDeactivatedEvent
)


class PlanType(Enum):
    """Organization subscription plan types."""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass(frozen=True)
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
class Organization(AggregateRoot[int]):
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
            # Create new instance with overrides since PlanLimits is frozen
            overrides = {}
            for key, value in self.custom_limits.items():
                if hasattr(base_limits, key):
                    overrides[key] = value
            
            if overrides:
                # Get all current values
                current_values = {
                    'api_rate_limit_per_minute': base_limits.api_rate_limit_per_minute,
                    'monthly_processing_minutes': base_limits.monthly_processing_minutes,
                    'concurrent_streams': base_limits.concurrent_streams,
                    'webhook_endpoints': base_limits.webhook_endpoints,
                    'api_keys': base_limits.api_keys,
                    'team_members': base_limits.team_members,
                }
                # Apply overrides
                current_values.update(overrides)
                return PlanLimits(**current_values)

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

    @property
    def can_add_member(self) -> bool:
        """Check if organization can add more members."""
        return self.member_count < self.plan_limits.team_members

    def add_member(self, user_id: int, added_by_user_id: int) -> None:
        """Add a member to the organization.
        
        This follows Pythonic DDD patterns where aggregates protect invariants
        and raise domain events.
        """
        # Business rule: Only owner can add members
        if added_by_user_id != self.owner_id:
            raise UnauthorizedOperation(
                operation="add_member",
                reason="Only the owner can add members",
                user_id=added_by_user_id
            )
        
        # Business rule: Owner is implicitly a member
        if user_id == self.owner_id:
            raise BusinessRuleViolation(
                "Owner is already a member",
                entity_type="Organization",
                entity_id=self.id
            )

        # Business rule: No duplicate members
        if user_id in self.member_ids:
            return  # Idempotent operation

        # Business rule: Respect plan limits
        if not self.can_add_member:
            raise QuotaExceeded(
                resource="team_members",
                limit=self.plan_limits.team_members,
                current=self.member_count,
                organization_id=self.id
            )

        # Update state
        self.member_ids.append(user_id)
        self.updated_at = Timestamp.now()
        
        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                MemberAddedEvent(
                    organization_id=self.id,
                    user_id=user_id,
                    added_by_user_id=added_by_user_id
                )
            )

    def remove_member(self, user_id: int, removed_by_user_id: int) -> None:
        """Remove a member from the organization."""
        # Business rule: Only owner can remove members
        if removed_by_user_id != self.owner_id:
            raise UnauthorizedOperation(
                operation="remove_member",
                reason="Only the owner can remove members",
                user_id=removed_by_user_id
            )
        
        # Business rule: Member must exist
        if user_id not in self.member_ids:
            return  # Idempotent operation

        # Update state
        self.member_ids.remove(user_id)
        self.updated_at = Timestamp.now()
        
        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                MemberRemovedEvent(
                    organization_id=self.id,
                    user_id=user_id,
                    removed_by_user_id=removed_by_user_id
                )
            )

    def upgrade_plan(self, new_plan: PlanType, upgraded_by_user_id: int) -> None:
        """Upgrade organization to a new plan."""
        # Business rule: Only owner can upgrade plan
        if upgraded_by_user_id != self.owner_id:
            raise UnauthorizedOperation(
                operation="upgrade_plan",
                reason="Only the owner can upgrade the plan",
                user_id=upgraded_by_user_id
            )
        
        # Business rule: Must be a different plan
        if new_plan == self.plan_type:
            return  # No change needed
        
        old_plan = self.plan_type
        
        # Update state
        self.plan_type = new_plan
        self.subscription_started_at = Timestamp.now()
        self.updated_at = Timestamp.now()
        
        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                PlanUpgradedEvent(
                    organization_id=self.id,
                    old_plan=old_plan.value,
                    new_plan=new_plan.value,
                    upgraded_by_user_id=upgraded_by_user_id
                )
            )

    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate the organization."""
        # Business rule: Can't deactivate already inactive org
        if not self.is_active:
            raise InvalidStateTransition(
                entity_type="Organization",
                entity_id=self.id,
                from_state="inactive",
                to_state="inactive",
                allowed_states=["active"]
            )
        
        # Update state
        self.is_active = False
        self.updated_at = Timestamp.now()
        
        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                OrganizationDeactivatedEvent(
                    organization_id=self.id,
                    reason=reason
                )
            )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Organization({self.name.value} - {self.plan_type.value})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"Organization(id={self.id}, name={self.name.value!r}, "
            f"plan={self.plan_type.value}, members={self.member_count}, "
            f"active={self.is_active})"
        )
    
    @classmethod
    def create(
        cls,
        name: CompanyName,
        owner_id: int,
        plan_type: PlanType = PlanType.STARTER
    ) -> "Organization":
        """Factory method to create a new organization.
        
        This is the Pythonic way to handle entity creation with
        proper initialization and event raising.
        """
        org = cls(
            id=None,  # Will be assigned by repository
            name=name,
            owner_id=owner_id,
            plan_type=plan_type,
            member_ids=[],
            custom_limits=None,
            settings={},
            subscription_started_at=Timestamp.now(),
            subscription_ends_at=None,
            is_active=True
        )
        
        # Note: We can't raise the creation event here since we don't have an ID yet
        # The repository should handle this after persisting
        
        return org
