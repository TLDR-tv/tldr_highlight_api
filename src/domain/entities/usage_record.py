"""Usage record domain entity."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.duration import Duration


class UsageType(Enum):
    """Type of usage being recorded."""

    STREAM_PROCESSING = "stream_processing"
    API_CALL = "api_call"
    WEBHOOK_DELIVERY = "webhook_delivery"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"


@dataclass
class UsageRecord(Entity[int]):
    """Domain entity representing resource usage.

    Usage records track consumption of various resources
    for billing and analytics purposes.
    """

    user_id: int
    organization_id: Optional[int]
    usage_type: UsageType

    # What was consumed
    resource_id: Optional[int] = None  # ID of stream, etc.
    resource_type: Optional[str] = None  # "stream", etc.

    # Quantity consumed
    quantity: float = 1.0
    unit: str = "unit"  # "minutes", "requests", "GB", etc.

    # When it was consumed
    period_start: Timestamp = None
    period_end: Optional[Timestamp] = None

    # Billing information
    billable: bool = True
    rate: Optional[float] = None  # Cost per unit
    total_cost: Optional[float] = None

    # Metadata
    api_key_id: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def __post_init__(self):
        """Set period_start if not provided."""
        if self.period_start is None:
            object.__setattr__(self, "period_start", Timestamp.now())
        super().__post_init__()

    @property
    def duration(self) -> Optional[Duration]:
        """Calculate duration of usage period."""
        if self.period_start and self.period_end:
            return self.period_end.duration_since(self.period_start)
        return None

    @property
    def is_complete(self) -> bool:
        """Check if usage period is complete."""
        return self.period_end is not None

    def complete(self, quantity: Optional[float] = None) -> "UsageRecord":
        """Mark usage as complete."""
        if self.is_complete:
            return self

        final_quantity = quantity if quantity is not None else self.quantity
        final_cost = None
        if self.rate and self.billable:
            final_cost = final_quantity * self.rate

        return UsageRecord(
            id=self.id,
            user_id=self.user_id,
            organization_id=self.organization_id,
            usage_type=self.usage_type,
            resource_id=self.resource_id,
            resource_type=self.resource_type,
            quantity=final_quantity,
            unit=self.unit,
            period_start=self.period_start,
            period_end=Timestamp.now(),
            billable=self.billable,
            rate=self.rate,
            total_cost=final_cost,
            api_key_id=self.api_key_id,
            ip_address=self.ip_address,
            user_agent=self.user_agent,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def update_quantity(self, new_quantity: float) -> "UsageRecord":
        """Update the quantity consumed."""
        new_cost = None
        if self.rate and self.billable:
            new_cost = new_quantity * self.rate

        return UsageRecord(
            id=self.id,
            user_id=self.user_id,
            organization_id=self.organization_id,
            usage_type=self.usage_type,
            resource_id=self.resource_id,
            resource_type=self.resource_type,
            quantity=new_quantity,
            unit=self.unit,
            period_start=self.period_start,
            period_end=self.period_end,
            billable=self.billable,
            rate=self.rate,
            total_cost=new_cost,
            api_key_id=self.api_key_id,
            ip_address=self.ip_address,
            user_agent=self.user_agent,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def mark_non_billable(self, reason: Optional[str] = None) -> "UsageRecord":
        """Mark usage as non-billable."""
        return UsageRecord(
            id=self.id,
            user_id=self.user_id,
            organization_id=self.organization_id,
            usage_type=self.usage_type,
            resource_id=self.resource_id,
            resource_type=self.resource_type,
            quantity=self.quantity,
            unit=self.unit,
            period_start=self.period_start,
            period_end=self.period_end,
            billable=False,
            rate=self.rate,
            total_cost=0.0,
            api_key_id=self.api_key_id,
            ip_address=self.ip_address,
            user_agent=self.user_agent,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    @property
    def estimated_cost(self) -> float:
        """Calculate estimated cost based on current quantity and rate."""
        if not self.billable or not self.rate:
            return 0.0
        return self.quantity * self.rate

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"UsageRecord({self.usage_type.value}: {self.quantity} {self.unit})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"UsageRecord(id={self.id}, type={self.usage_type.value}, "
            f"quantity={self.quantity}, billable={self.billable})"
        )

    @classmethod
    def for_stream_processing(
        cls,
        user_id: int,
        stream_id: int,
        duration_minutes: float,
        organization_id: Optional[int] = None,
        rate: Optional[float] = None,
    ) -> "UsageRecord":
        """Create usage record for stream processing."""
        return cls(
            id=None,
            user_id=user_id,
            organization_id=organization_id,
            usage_type=UsageType.STREAM_PROCESSING,
            resource_id=stream_id,
            resource_type="stream",
            quantity=duration_minutes,
            unit="minutes",
            billable=True,
            rate=rate,
            total_cost=None,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )

    @classmethod
    def for_api_call(
        cls,
        user_id: int,
        api_key_id: int,
        endpoint: str,
        ip_address: str,
        organization_id: Optional[int] = None,
        rate: Optional[float] = None,
    ) -> "UsageRecord":
        """Create usage record for API call."""
        return cls(
            id=None,
            user_id=user_id,
            organization_id=organization_id,
            usage_type=UsageType.API_CALL,
            resource_type=endpoint,
            quantity=1,
            unit="request",
            billable=True,
            rate=rate,
            total_cost=None,
            api_key_id=api_key_id,
            ip_address=ip_address,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )
