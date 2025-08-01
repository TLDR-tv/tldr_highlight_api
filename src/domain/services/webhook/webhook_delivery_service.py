"""Webhook delivery domain service.

This service handles webhook event delivery, retry logic,
and delivery tracking.
"""

import json
import hmac
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from aiohttp import ClientTimeout

import logfire
from src.domain.entities.webhook import (
    Webhook,
    WebhookEvent,
    WebhookDelivery,
    WebhookStatus,
)
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.webhook_repository import WebhookRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.exceptions import EntityNotFoundError


class WebhookDeliveryService:
    """Domain service for webhook delivery.

    Handles webhook event delivery, retry logic, signature generation,
    and delivery tracking.
    """

    # Delivery configuration
    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_RETRY_ATTEMPTS = 5
    INITIAL_RETRY_DELAY_SECONDS = 60
    MAX_RETRY_DELAY_SECONDS = 3600  # 1 hour

    def __init__(
        self,
        webhook_repo: WebhookRepository,
        stream_repo: StreamRepository,
        highlight_repo: HighlightRepository,
        http_client: Optional[aiohttp.ClientSession] = None,
    ):
        """Initialize webhook delivery service.

        Args:
            webhook_repo: Repository for webhook operations
            stream_repo: Repository for stream operations
            highlight_repo: Repository for highlight operations
            http_client: Optional HTTP client for deliveries
        """
        self.webhook_repo = webhook_repo
        self.stream_repo = stream_repo
        self.highlight_repo = highlight_repo
        self._http_client = http_client
        self.logger = logfire.get_logger(__name__)

    async def trigger_event(
        self,
        event: WebhookEvent,
        user_id: int,
        resource_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Trigger webhook event for all subscribed endpoints.

        Args:
            event: Event type to trigger
            user_id: User ID associated with the event
            resource_id: ID of the resource (stream, highlight, etc.)
            metadata: Optional additional event metadata

        Returns:
            List of delivery IDs for tracking
        """
        # Get active webhooks for this event and user
        webhooks = await self.webhook_repo.get_active_for_event(event, user_id)

        if not webhooks:
            self.logger.info(
                f"No active webhooks for event {event.value} and user {user_id}"
            )
            return []

        # Build event payload
        payload = await self._build_event_payload(event, resource_id, metadata)

        # Queue deliveries
        delivery_ids = []
        for webhook in webhooks:
            delivery_id = await self._queue_delivery(webhook, event, payload)
            delivery_ids.append(delivery_id)

        self.logger.info(
            f"Triggered {len(delivery_ids)} webhook deliveries for event {event.value}"
        )

        return delivery_ids

    async def deliver_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Dict[str, Any],
        attempt: int = 1,
    ) -> WebhookDelivery:
        """Deliver webhook with retry logic.

        Args:
            webhook: Webhook to deliver to
            event: Event type
            payload: Event payload
            attempt: Current attempt number

        Returns:
            Delivery result
        """
        start_time = datetime.utcnow()

        try:
            # Prepare request
            headers = self._prepare_headers(webhook, payload)

            # Get or create HTTP client
            client = await self._get_http_client()

            # Send request with timeout
            timeout = ClientTimeout(total=self.DEFAULT_TIMEOUT_SECONDS)
            async with client.post(
                webhook.url.value, json=payload, headers=headers, timeout=timeout
            ) as response:
                response_time_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

                # Create delivery record
                delivery = WebhookDelivery(
                    delivered_at=Timestamp.now(),
                    status_code=response.status,
                    response_time_ms=response_time_ms,
                    error_message=None
                    if 200 <= response.status < 300
                    else f"HTTP {response.status}",
                )

                # Update webhook with delivery result
                updated_webhook = webhook.record_delivery(delivery)
                await self.webhook_repo.save(updated_webhook)

                # Log result
                if delivery.is_successful:
                    self.logger.info(
                        f"Successfully delivered webhook {webhook.id} "
                        f"(attempt {attempt}, {response_time_ms}ms)"
                    )
                else:
                    self.logger.warning(
                        f"Failed to deliver webhook {webhook.id}: HTTP {response.status} "
                        f"(attempt {attempt})"
                    )

                # Schedule retry if needed
                if not delivery.is_successful and attempt < self.MAX_RETRY_ATTEMPTS:
                    await self._schedule_retry(webhook, event, payload, attempt + 1)

                return delivery

        except asyncio.TimeoutError:
            response_time_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            # Create failed delivery record
            delivery = WebhookDelivery(
                delivered_at=Timestamp.now(),
                status_code=0,
                response_time_ms=response_time_ms,
                error_message="Request timeout",
            )

            # Update webhook
            updated_webhook = webhook.record_delivery(delivery)
            await self.webhook_repo.save(updated_webhook)

            self.logger.error(
                f"Webhook {webhook.id} delivery timed out (attempt {attempt})"
            )

            # Schedule retry
            if attempt < self.MAX_RETRY_ATTEMPTS:
                await self._schedule_retry(webhook, event, payload, attempt + 1)

            return delivery

        except Exception as e:
            response_time_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

            # Create failed delivery record
            delivery = WebhookDelivery(
                delivered_at=Timestamp.now(),
                status_code=0,
                response_time_ms=response_time_ms,
                error_message=str(e),
            )

            # Update webhook
            updated_webhook = webhook.record_delivery(delivery)
            await self.webhook_repo.save(updated_webhook)

            self.logger.error(
                f"Error delivering webhook {webhook.id}: {e} (attempt {attempt})"
            )

            # Schedule retry
            if attempt < self.MAX_RETRY_ATTEMPTS:
                await self._schedule_retry(webhook, event, payload, attempt + 1)

            return delivery

    async def test_webhook(self, webhook_id: int, user_id: int) -> WebhookDelivery:
        """Send a test delivery to a webhook.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for authorization check

        Returns:
            Delivery result

        Raises:
            EntityNotFoundError: If webhook not found
            UnauthorizedAccessError: If user doesn't own webhook
        """
        # Get webhook
        webhook = await self.webhook_repo.get(webhook_id)
        if not webhook:
            raise EntityNotFoundError(f"Webhook {webhook_id} not found")

        if webhook.user_id != user_id:
            from src.domain.exceptions import UnauthorizedAccessError

            raise UnauthorizedAccessError("You don't have access to this webhook")

        # Create test payload
        test_payload = {
            "event": "test",
            "timestamp": Timestamp.now().iso_string,
            "data": {
                "message": "This is a test webhook delivery",
                "webhook_id": webhook.id,
                "webhook_url": webhook.url.value,
            },
            "webhook_id": str(webhook.id),
            "delivery_id": f"test-{Timestamp.now().value.timestamp()}",
        }

        # Deliver test webhook (no retries for tests)
        return await self.deliver_webhook(
            webhook,
            WebhookEvent.STREAM_STARTED,  # Use any event type for test
            test_payload,
            attempt=self.MAX_RETRY_ATTEMPTS,  # Prevent retries
        )

    async def get_webhook_health(self, webhook_id: int) -> Dict[str, Any]:
        """Get health metrics for a webhook.

        Args:
            webhook_id: Webhook ID

        Returns:
            Dictionary with health metrics
        """
        webhook = await self.webhook_repo.get(webhook_id)
        if not webhook:
            raise EntityNotFoundError(f"Webhook {webhook_id} not found")

        # Get delivery statistics
        stats = await self.webhook_repo.get_delivery_stats(webhook_id)

        # Calculate health score (0-100)
        health_score = 100

        # Penalize for failures
        if webhook.consecutive_failures > 0:
            health_score -= min(webhook.consecutive_failures * 10, 50)

        # Penalize for low success rate
        if webhook.success_rate < 0.95:
            health_score -= int((1 - webhook.success_rate) * 30)

        # Penalize if inactive
        if webhook.status != WebhookStatus.ACTIVE:
            health_score -= 20

        health_score = max(0, health_score)

        # Determine health status
        if health_score >= 90:
            health_status = "healthy"
        elif health_score >= 70:
            health_status = "degraded"
        elif health_score >= 50:
            health_status = "unhealthy"
        else:
            health_status = "critical"

        return {
            "webhook_id": webhook_id,
            "health_score": health_score,
            "health_status": health_status,
            "status": webhook.status.value,
            "consecutive_failures": webhook.consecutive_failures,
            "success_rate": webhook.success_rate,
            "total_deliveries": webhook.total_deliveries,
            "last_delivery": stats["last_delivery"],
            "recommendations": self._get_health_recommendations(webhook, health_score),
        }

    async def cleanup_failed_webhooks(self, failure_threshold: int = 10) -> int:
        """Clean up webhooks with too many failures.

        Args:
            failure_threshold: Number of consecutive failures before cleanup

        Returns:
            Number of webhooks cleaned up
        """
        return await self.webhook_repo.cleanup_failed_webhooks(failure_threshold)

    async def reset_webhook(self, webhook_id: int, user_id: int) -> Webhook:
        """Reset a failed webhook to active status.

        Args:
            webhook_id: Webhook ID
            user_id: User ID for authorization

        Returns:
            Updated webhook
        """
        webhook = await self.webhook_repo.get(webhook_id)
        if not webhook:
            raise EntityNotFoundError(f"Webhook {webhook_id} not found")

        if webhook.user_id != user_id:
            from src.domain.exceptions import UnauthorizedAccessError

            raise UnauthorizedAccessError("You don't have access to this webhook")

        # Reset and reactivate
        reset_webhook = await self.webhook_repo.reset_failure_count(webhook_id)

        self.logger.info(f"Reset webhook {webhook_id} to active status")

        return reset_webhook

    # Private helper methods

    async def _build_event_payload(
        self,
        event: WebhookEvent,
        resource_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build event payload based on event type."""
        base_payload = {
            "event": event.value,
            "timestamp": Timestamp.now().iso_string,
            "data": {},
            "metadata": metadata or {},
        }

        # Add event-specific data
        if event == WebhookEvent.STREAM_STARTED:
            stream = await self.stream_repo.get(resource_id)
            if stream:
                base_payload["data"] = {
                    "stream_id": stream.id,
                    "title": stream.title,
                    "url": stream.url.value,
                    "platform": stream.platform.value,
                    "started_at": stream.started_at.iso_string
                    if stream.started_at
                    else None,
                }

        elif event == WebhookEvent.STREAM_COMPLETED:
            stream = await self.stream_repo.get(resource_id)
            if stream:
                highlights = await self.highlight_repo.get_by_stream(stream.id)
                base_payload["data"] = {
                    "stream_id": stream.id,
                    "title": stream.title,
                    "duration_seconds": stream.duration_seconds,
                    "highlight_count": len(highlights),
                    "completed_at": stream.completed_at.iso_string
                    if stream.completed_at
                    else None,
                }

        elif event == WebhookEvent.HIGHLIGHT_DETECTED:
            highlight = await self.highlight_repo.get(resource_id)
            if highlight:
                base_payload["data"] = {
                    "highlight_id": highlight.id,
                    "stream_id": highlight.stream_id,
                    "title": highlight.title,
                    "confidence_score": highlight.confidence_score.value,
                    "duration_seconds": highlight.duration.value,
                    "type": highlight.highlight_type.value,
                }

        return base_payload

    def _prepare_headers(
        self, webhook: Webhook, payload: Dict[str, Any]
    ) -> Dict[str, str]:
        """Prepare HTTP headers including signature."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TLDRHighlightAPI/1.0",
            "X-Webhook-Event": payload.get("event", "unknown"),
            "X-Webhook-Delivery": payload.get("delivery_id", ""),
            "X-Webhook-Timestamp": str(int(datetime.utcnow().timestamp())),
        }

        # Add custom headers
        headers.update(webhook.custom_headers)

        # Generate signature
        signature = self._generate_signature(webhook.secret, payload)
        headers["X-Webhook-Signature"] = signature

        return headers

    def _generate_signature(self, secret: str, payload: Dict[str, Any]) -> str:
        """Generate HMAC signature for webhook payload."""
        # Create canonical payload string
        canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            secret.encode("utf-8"), canonical_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return f"sha256={signature}"

    async def _get_http_client(self) -> aiohttp.ClientSession:
        """Get or create HTTP client session."""
        if not self._http_client:
            self._http_client = aiohttp.ClientSession()
        return self._http_client

    async def _queue_delivery(
        self, webhook: Webhook, event: WebhookEvent, payload: Dict[str, Any]
    ) -> str:
        """Queue webhook for delivery."""
        # Generate delivery ID
        delivery_id = f"{webhook.id}-{event.value}-{Timestamp.now().value.timestamp()}"
        payload["webhook_id"] = str(webhook.id)
        payload["delivery_id"] = delivery_id

        # In a real implementation, this would queue to a message broker
        # For now, deliver immediately in background
        asyncio.create_task(self.deliver_webhook(webhook, event, payload))

        return delivery_id

    async def _schedule_retry(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Dict[str, Any],
        attempt: int,
    ):
        """Schedule webhook retry with exponential backoff."""
        # Calculate delay with exponential backoff
        delay = min(
            self.INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 2)),
            self.MAX_RETRY_DELAY_SECONDS,
        )

        self.logger.info(
            f"Scheduling retry for webhook {webhook.id} in {delay} seconds (attempt {attempt})"
        )

        # In a real implementation, this would use a job queue
        # For now, use asyncio delay
        await asyncio.sleep(delay)
        await self.deliver_webhook(webhook, event, payload, attempt)

    def _get_health_recommendations(
        self, webhook: Webhook, health_score: int
    ) -> List[str]:
        """Get recommendations for improving webhook health."""
        recommendations = []

        if webhook.consecutive_failures > 5:
            recommendations.append(
                "Webhook has many consecutive failures. Check endpoint availability."
            )

        if webhook.success_rate < 0.9:
            recommendations.append(
                f"Success rate is low ({webhook.success_rate:.1%}). "
                "Review endpoint implementation and error logs."
            )

        if webhook.status == WebhookStatus.FAILED:
            recommendations.append(
                "Webhook is in failed state. Use reset endpoint to reactivate after fixing issues."
            )

        if not webhook.last_delivery:
            recommendations.append(
                "No deliveries recorded yet. Send a test webhook to verify configuration."
            )

        if health_score < 50:
            recommendations.append(
                "Consider replacing this webhook with a new endpoint if issues persist."
            )

        return recommendations
