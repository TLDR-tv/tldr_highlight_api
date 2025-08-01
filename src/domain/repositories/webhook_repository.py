"""Webhook repository protocol."""

from typing import Protocol, List, Optional

from src.domain.repositories.base import Repository
from src.domain.entities.webhook import Webhook, WebhookEvent, WebhookStatus
from src.domain.value_objects.url import Url


class WebhookRepository(Repository[Webhook, int], Protocol):
    """Repository protocol for Webhook entities.

    Extends the base repository with webhook-specific operations.
    """

    async def get_by_user(
        self, user_id: int, status: Optional[WebhookStatus] = None
    ) -> List[Webhook]:
        """Get webhooks for a user, optionally filtered by status."""
        ...

    async def get_by_event(
        self, event: WebhookEvent, active_only: bool = True
    ) -> List[Webhook]:
        """Get all webhooks subscribed to an event."""
        ...

    async def get_by_url(
        self, url: Url, user_id: Optional[int] = None
    ) -> List[Webhook]:
        """Get webhooks by URL."""
        ...

    async def get_active_for_event(
        self, event: WebhookEvent, user_id: int
    ) -> List[Webhook]:
        """Get active webhooks for a specific event and user."""
        ...

    async def get_failed_webhooks(self, failure_threshold: int = 5) -> List[Webhook]:
        """Get webhooks with consecutive failures above threshold."""
        ...

    async def count_by_status(self, user_id: Optional[int] = None) -> dict:
        """Get count of webhooks by status."""
        ...

    async def get_delivery_stats(self, webhook_id: int) -> dict:
        """Get delivery statistics for a webhook."""
        ...

    async def cleanup_failed_webhooks(self, failure_threshold: int = 10) -> int:
        """Clean up webhooks with too many failures."""
        ...

    async def reset_failure_count(self, webhook_id: int) -> Optional[Webhook]:
        """Reset consecutive failure count for a webhook."""
        ...
