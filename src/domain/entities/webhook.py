"""Webhook domain entity."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from secrets import token_urlsafe

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.url import Url


class WebhookEvent(Enum):
    """Webhook event types."""

    STREAM_STARTED = "stream.started"
    STREAM_COMPLETED = "stream.completed"
    STREAM_FAILED = "stream.failed"
    HIGHLIGHT_DETECTED = "highlight.detected"


class WebhookStatus(Enum):
    """Webhook status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


@dataclass
class WebhookDelivery:
    """Value object representing a webhook delivery attempt."""

    delivered_at: Timestamp
    status_code: int
    response_time_ms: int
    error_message: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Check if delivery was successful."""
        return 200 <= self.status_code < 300


@dataclass(kw_only=True)
class Webhook(Entity[int]):
    """Domain entity representing a webhook endpoint.

    Webhooks allow users to receive real-time notifications
    about events in their streams and highlights.
    """

    url: Url
    user_id: int
    events: List[WebhookEvent]

    # Configuration
    secret: str  # For signature verification
    description: Optional[str] = None

    # State
    status: WebhookStatus = WebhookStatus.ACTIVE

    # Delivery tracking
    last_delivery: Optional[WebhookDelivery] = None
    consecutive_failures: int = 0
    total_deliveries: int = 0
    successful_deliveries: int = 0

    # Custom headers
    custom_headers: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def generate_secret() -> str:
        """Generate a webhook secret for signature verification."""
        return token_urlsafe(32)

    @property
    def is_active(self) -> bool:
        """Check if webhook is active and ready to receive events."""
        return self.status == WebhookStatus.ACTIVE

    @property
    def success_rate(self) -> float:
        """Calculate webhook delivery success rate."""
        if self.total_deliveries == 0:
            return 1.0
        return self.successful_deliveries / self.total_deliveries

    def subscribes_to(self, event: WebhookEvent) -> bool:
        """Check if webhook subscribes to specific event."""
        return event in self.events

    def record_delivery(self, delivery: WebhookDelivery) -> "Webhook":
        """Record a delivery attempt."""
        new_total = self.total_deliveries + 1
        new_successful = self.successful_deliveries + (
            1 if delivery.is_successful else 0
        )
        new_failures = 0 if delivery.is_successful else self.consecutive_failures + 1

        # Auto-deactivate after too many failures
        new_status = self.status
        if new_failures >= 10:
            new_status = WebhookStatus.FAILED

        return Webhook(
            id=self.id,
            url=self.url,
            user_id=self.user_id,
            events=self.events.copy(),
            secret=self.secret,
            description=self.description,
            status=new_status,
            last_delivery=delivery,
            consecutive_failures=new_failures,
            total_deliveries=new_total,
            successful_deliveries=new_successful,
            custom_headers=self.custom_headers.copy(),
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def add_event(self, event: WebhookEvent) -> "Webhook":
        """Subscribe to additional event."""
        if event in self.events:
            return self

        new_events = self.events.copy()
        new_events.append(event)

        return Webhook(
            id=self.id,
            url=self.url,
            user_id=self.user_id,
            events=new_events,
            secret=self.secret,
            description=self.description,
            status=self.status,
            last_delivery=self.last_delivery,
            consecutive_failures=self.consecutive_failures,
            total_deliveries=self.total_deliveries,
            successful_deliveries=self.successful_deliveries,
            custom_headers=self.custom_headers.copy(),
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def remove_event(self, event: WebhookEvent) -> "Webhook":
        """Unsubscribe from event."""
        if event not in self.events:
            return self

        new_events = [e for e in self.events if e != event]

        return Webhook(
            id=self.id,
            url=self.url,
            user_id=self.user_id,
            events=new_events,
            secret=self.secret,
            description=self.description,
            status=self.status,
            last_delivery=self.last_delivery,
            consecutive_failures=self.consecutive_failures,
            total_deliveries=self.total_deliveries,
            successful_deliveries=self.successful_deliveries,
            custom_headers=self.custom_headers.copy(),
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def activate(self) -> "Webhook":
        """Activate the webhook."""
        return Webhook(
            id=self.id,
            url=self.url,
            user_id=self.user_id,
            events=self.events.copy(),
            secret=self.secret,
            description=self.description,
            status=WebhookStatus.ACTIVE,
            last_delivery=self.last_delivery,
            consecutive_failures=0,  # Reset failures on activation
            total_deliveries=self.total_deliveries,
            successful_deliveries=self.successful_deliveries,
            custom_headers=self.custom_headers.copy(),
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def deactivate(self) -> "Webhook":
        """Deactivate the webhook."""
        return Webhook(
            id=self.id,
            url=self.url,
            user_id=self.user_id,
            events=self.events.copy(),
            secret=self.secret,
            description=self.description,
            status=WebhookStatus.INACTIVE,
            last_delivery=self.last_delivery,
            consecutive_failures=self.consecutive_failures,
            total_deliveries=self.total_deliveries,
            successful_deliveries=self.successful_deliveries,
            custom_headers=self.custom_headers.copy(),
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def add_header(self, key: str, value: str) -> "Webhook":
        """Add custom header for webhook requests."""
        new_headers = self.custom_headers.copy()
        new_headers[key] = value

        return Webhook(
            id=self.id,
            url=self.url,
            user_id=self.user_id,
            events=self.events.copy(),
            secret=self.secret,
            description=self.description,
            status=self.status,
            last_delivery=self.last_delivery,
            consecutive_failures=self.consecutive_failures,
            total_deliveries=self.total_deliveries,
            successful_deliveries=self.successful_deliveries,
            custom_headers=new_headers,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Webhook({self.url.value} - {self.status.value})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"Webhook(id={self.id}, url={self.url.value!r}, "
            f"events={len(self.events)}, status={self.status.value})"
        )
