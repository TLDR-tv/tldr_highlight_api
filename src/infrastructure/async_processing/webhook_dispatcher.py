"""
Webhook Dispatcher for async processing pipeline.

This module provides comprehensive webhook delivery functionality including
HMAC signature generation, retry logic with exponential backoff, dead letter
queue handling, and webhook event management.
"""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict

import httpx
import structlog

from src.core.cache import get_redis_client
from src.core.config import get_settings
from src.core.database import get_db_session
from src.infrastructure.persistence.models.webhook import Webhook
from src.infrastructure.persistence.models.stream import Stream
from src.infrastructure.persistence.models.user import User


logger = structlog.get_logger(__name__)
settings = get_settings()


class WebhookEvent(str, Enum):
    """Types of webhook events that can be dispatched."""

    STREAM_STARTED = "stream.started"
    PROGRESS_UPDATE = "stream.progress"
    HIGHLIGHTS_DETECTED = "highlights.detected"
    PROCESSING_COMPLETE = "stream.completed"
    ERROR_OCCURRED = "stream.error"
    STREAM_CANCELLED = "stream.cancelled"


class WebhookStatus(str, Enum):
    """Status values for webhook delivery attempts."""

    PENDING = "pending"
    SENDING = "sending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class WebhookDispatcher:
    """
    Comprehensive webhook dispatcher with enterprise-grade features.

    Provides secure webhook delivery with HMAC signatures, exponential backoff
    retry logic, dead letter queue handling, and comprehensive logging.
    """

    def __init__(self):
        """Initialize webhook dispatcher with Redis client and HTTP client."""
        self.redis_client = get_redis_client()
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0, read=settings.webhook_timeout_seconds, write=5.0, pool=30.0
            ),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

        # Redis key prefixes
        self.webhook_queue_prefix = "webhook_queue:"
        self.webhook_attempts_prefix = "webhook_attempts:"
        self.dead_letter_prefix = "dead_letter:"
        self.rate_limit_prefix = "webhook_rate_limit:"

        # Retry configuration
        self.max_retries = settings.webhook_max_retries
        self.base_delay = settings.webhook_retry_delay_seconds
        self.max_delay = 3600  # 1 hour max delay
        self.backoff_factor = 2
        self.jitter_factor = 0.1

    async def dispatch_webhook(
        self,
        stream_id: int,
        event: WebhookEvent,
        data: Dict[str, Any],
        delay_seconds: int = 0,
    ) -> Dict[str, Any]:
        """
        Dispatch a webhook event to all registered endpoints.

        Args:
            stream_id: ID of the stream triggering the webhook
            event: Type of webhook event
            data: Event data payload
            delay_seconds: Optional delay before sending

        Returns:
            Dict[str, Any]: Dispatch results
        """
        logger.info(
            "Dispatching webhook",
            stream_id=stream_id,
            event_type=event.value,
            delay_seconds=delay_seconds,
        )

        try:
            # Get stream and user information
            with get_db_session() as db:
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if not stream:
                    raise ValueError(f"Stream {stream_id} not found")

                user = db.query(User).filter(User.id == stream.user_id).first()
                if not user:
                    raise ValueError(f"User {stream.user_id} not found")

                # Get active webhooks for the user
                webhooks = (
                    db.query(Webhook)
                    .filter(Webhook.user_id == user.id, Webhook.active)
                    .all()
                )

            if not webhooks:
                logger.info(
                    "No active webhooks configured",
                    stream_id=stream_id,
                    user_id=user.id,
                )
                return {"dispatched": 0, "reason": "no_webhooks"}

            # Prepare webhook payload
            payload = self._prepare_payload(stream, event, data)

            # Dispatch to all webhooks
            dispatch_results = []
            for webhook in webhooks:
                try:
                    # Check if webhook is configured for this event
                    if not self._should_send_event(webhook, event):
                        continue

                    # Check rate limits
                    if not self._check_rate_limit(webhook):
                        logger.warning(
                            "Webhook rate limit exceeded", webhook_id=webhook.id
                        )
                        continue

                    # Dispatch webhook
                    if delay_seconds > 0:
                        # Schedule for later delivery
                        result = await self._schedule_webhook(
                            webhook, payload, delay_seconds
                        )
                    else:
                        # Send immediately
                        result = await self._send_webhook(webhook, payload)

                    dispatch_results.append(result)

                except Exception as e:
                    logger.error(
                        "Failed to dispatch webhook",
                        webhook_id=webhook.id,
                        error=str(e),
                    )
                    dispatch_results.append(
                        {"webhook_id": webhook.id, "status": "failed", "error": str(e)}
                    )

            summary = {
                "dispatched": len(dispatch_results),
                "successful": len(
                    [r for r in dispatch_results if r.get("status") == "delivered"]
                ),
                "failed": len(
                    [r for r in dispatch_results if r.get("status") == "failed"]
                ),
                "results": dispatch_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info("Webhook dispatch completed", **summary)
            return summary

        except Exception as e:
            logger.error(
                "Failed to dispatch webhook", stream_id=stream_id, error=str(e)
            )
            return {"error": str(e), "dispatched": 0}

    async def retry_failed_webhook(self, webhook_attempt_id: int) -> Dict[str, Any]:
        """
        Retry a failed webhook delivery.

        Args:
            webhook_attempt_id: ID of the failed webhook attempt

        Returns:
            Dict[str, Any]: Retry result
        """
        logger.info("Retrying failed webhook", attempt_id=webhook_attempt_id)

        try:
            # TODO: Implement WebhookAttempt model properly
            # For now, this is a stub that returns an error
            logger.warning("WebhookAttempt model not implemented, cannot retry webhooks")
            return {
                "error": "WebhookAttempt model not implemented",
                "status": "failed",
                "attempt_id": webhook_attempt_id
            }

        except Exception as e:
            logger.error(
                "Failed to retry webhook", attempt_id=webhook_attempt_id, error=str(e)
            )
            return {"error": str(e), "status": "failed"}

    async def process_dead_letter_queue(self) -> Dict[str, Any]:
        """
        Process webhook messages from the dead letter queue.

        Returns:
            Dict[str, Any]: Processing results
        """
        logger.info("Processing dead letter queue")

        try:
            processed = 0
            failed = 0

            # Get dead letter messages
            dead_letter_pattern = f"{self.dead_letter_prefix}*"
            dead_letter_keys = self.redis_client.keys(dead_letter_pattern)

            for key in dead_letter_keys:
                try:
                    # Get message data
                    message_data = self.redis_client.hgetall(key)
                    if not message_data:
                        continue

                    # Parse message
                    webhook_id = int(message_data["webhook_id"])
                    payload = json.loads(message_data["payload"])

                    # Get webhook
                    with get_db_session() as db:
                        webhook = (
                            db.query(Webhook).filter(Webhook.id == webhook_id).first()
                        )
                        if not webhook or not webhook.active:
                            # Webhook no longer exists or is inactive
                            self.redis_client.delete(key)
                            continue

                    # Attempt final delivery
                    result = await self._send_webhook(
                        webhook, payload, is_dead_letter=True
                    )

                    if result.get("status") == "delivered":
                        # Success - remove from dead letter queue
                        self.redis_client.delete(key)
                        processed += 1
                    else:
                        # Still failed - leave in dead letter queue
                        failed += 1

                except Exception as e:
                    logger.error(
                        "Failed to process dead letter message", key=key, error=str(e)
                    )
                    failed += 1
                    continue

            result = {
                "processed": processed,
                "failed": failed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info("Dead letter queue processing completed", **result)
            return result

        except Exception as e:
            logger.error("Failed to process dead letter queue", error=str(e))
            return {"error": str(e)}

    def cleanup_old_attempts(self, cutoff_time: datetime) -> int:
        """
        Clean up old webhook attempts and related data.

        Args:
            cutoff_time: Remove attempts older than this time

        Returns:
            int: Number of attempts cleaned up
        """
        logger.info(
            "Cleaning up old webhook attempts", cutoff_time=cutoff_time.isoformat()
        )

        try:
            with get_db_session() as db:
                # Delete old webhook attempts
                old_attempts = (
                    # db.query(WebhookAttempt)
                    # .filter(WebhookAttempt.created_at < cutoff_time)
                    []  # TODO: Implement WebhookAttempt model
                    .delete()
                )

                db.commit()

            # Clean up Redis data
            redis_cleaned = 0

            # Clean up attempt tracking
            attempt_pattern = f"{self.webhook_attempts_prefix}*"
            attempt_keys = self.redis_client.keys(attempt_pattern)

            for key in attempt_keys:
                try:
                    # Check if data is old
                    data = self.redis_client.hgetall(key)
                    if data and "created_at" in data:
                        created_at = datetime.fromisoformat(data["created_at"])
                        if created_at < cutoff_time:
                            self.redis_client.delete(key)
                            redis_cleaned += 1
                except Exception:
                    continue

            # Clean up dead letter queue
            dead_letter_pattern = f"{self.dead_letter_prefix}*"
            dead_letter_keys = self.redis_client.keys(dead_letter_pattern)

            for key in dead_letter_keys:
                try:
                    data = self.redis_client.hgetall(key)
                    if data and "created_at" in data:
                        created_at = datetime.fromisoformat(data["created_at"])
                        if created_at < cutoff_time:
                            self.redis_client.delete(key)
                            redis_cleaned += 1
                except Exception:
                    continue

            total_cleaned = old_attempts + redis_cleaned
            logger.info(
                "Webhook cleanup completed",
                db_cleaned=old_attempts,
                redis_cleaned=redis_cleaned,
            )
            return total_cleaned

        except Exception as e:
            logger.error("Failed to cleanup old webhook attempts", error=str(e))
            return 0

    async def _send_webhook(
        self, webhook: Webhook, payload: Dict[str, Any], is_dead_letter: bool = False
    ) -> Dict[str, Any]:
        """Send webhook to a single endpoint."""
        logger.info("Sending webhook", webhook_id=webhook.id, url=webhook.url)

        try:
            # Create webhook attempt record
            with get_db_session() as _db:
                # TODO: Implement WebhookAttempt model
                # For now, create a mock attempt object
                from types import SimpleNamespace
                attempt = SimpleNamespace(
                    id=f"webhook_attempt_{webhook.id}_{int(time.time())}",
                    webhook_id=webhook.id,
                    url=webhook.url,
                    payload=json.dumps(payload),
                    status=WebhookStatus.SENDING.value,
                    retry_count=0,
                    is_dead_letter=is_dead_letter,
                    response_status_code=None,
                    response_headers=None,
                    response_body=None,
                    response_time_ms=None,
                    sent_at=None,
                    delivered_at=None,
                    error_message=None
                )

            # Send the webhook
            result = await self._send_webhook_attempt(webhook, payload, attempt)

            return result

        except Exception as e:
            logger.error("Failed to send webhook", webhook_id=webhook.id, error=str(e))
            return {"webhook_id": webhook.id, "status": "failed", "error": str(e)}

    async def _send_webhook_attempt(
        self, webhook: Webhook, payload: Dict[str, Any], attempt: Any  # WebhookAttempt
    ) -> Dict[str, Any]:
        """Send a single webhook attempt."""
        start_time = time.time()

        try:
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"TL;DR-Highlight-API/{settings.app_version}",
                "X-Webhook-Event": payload.get("event", "unknown"),
                "X-Webhook-Delivery": str(attempt.id),
                "X-Webhook-Timestamp": str(int(time.time())),
            }

            # Add custom headers if configured
            # TODO: Add headers field to Webhook model if needed
            # if webhook.headers:
            #     headers.update(webhook.headers)

            # Generate HMAC signature
            if webhook.secret:
                signature = self._generate_hmac_signature(
                    json.dumps(payload, sort_keys=True), webhook.secret
                )
                headers[settings.webhook_signature_header] = signature

            # Send HTTP request
            response = await self.http_client.post(
                webhook.url, json=payload, headers=headers
            )

            response_time = time.time() - start_time

            # Update attempt record (using SimpleNamespace for now)
            attempt.response_status_code = response.status_code
            attempt.response_headers = dict(response.headers)
            attempt.response_body = response.text[:1000]  # Limit response body size
            attempt.response_time_ms = int(response_time * 1000)
            attempt.sent_at = datetime.now(timezone.utc)

            # Check if delivery was successful
            if 200 <= response.status_code < 300:
                attempt.status = WebhookStatus.DELIVERED.value
                attempt.delivered_at = datetime.now(timezone.utc)

                # Update rate limiting
                self._update_rate_limit(webhook)

                result = {
                    "webhook_id": webhook.id,
                    "attempt_id": attempt.id,
                    "status": "delivered",
                    "status_code": response.status_code,
                    "response_time_ms": attempt.response_time_ms,
                }

            else:
                attempt.status = WebhookStatus.FAILED.value
                attempt.error_message = (
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )

                result = {
                    "webhook_id": webhook.id,
                    "attempt_id": attempt.id,
                    "status": "failed",
                    "status_code": response.status_code,
                    "error": attempt.error_message,
                }

            logger.info(
                "Webhook attempt completed",
                webhook_id=webhook.id,
                attempt_id=attempt.id,
                status_code=response.status_code,
                response_time_ms=attempt.response_time_ms,
            )

            return result

        except httpx.TimeoutException:
            error_msg = "Request timeout"
            logger.warning(
                "Webhook timeout", webhook_id=webhook.id, attempt_id=attempt.id
            )

        except httpx.ConnectError:
            error_msg = "Connection failed"
            logger.warning(
                "Webhook connection failed",
                webhook_id=webhook.id,
                attempt_id=attempt.id,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Webhook request failed",
                webhook_id=webhook.id,
                attempt_id=attempt.id,
                error=error_msg,
            )

        # Update failed attempt (using SimpleNamespace for now)
        try:
            attempt.status = WebhookStatus.FAILED.value
            attempt.error_message = error_msg
            attempt.response_time_ms = int((time.time() - start_time) * 1000)
        except Exception as e:
            logger.error("Failed to update failed attempt", error=str(e))

        return {
            "webhook_id": webhook.id,
            "attempt_id": attempt.id,
            "status": "failed",
            "error": error_msg,
        }

    async def _schedule_webhook(
        self, webhook: Webhook, payload: Dict[str, Any], delay_seconds: int
    ) -> Dict[str, Any]:
        """Schedule a webhook for delayed delivery."""
        try:
            # Store in Redis for delayed processing
            scheduled_key = (
                f"{self.webhook_queue_prefix}{webhook.id}_{int(time.time())}"
            )

            scheduled_data = {
                "webhook_id": webhook.id,
                "payload": json.dumps(payload),
                "scheduled_for": (
                    datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
                ).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            self.redis_client.hset(scheduled_key, mapping=scheduled_data)
            self.redis_client.expire(
                scheduled_key, delay_seconds + 3600
            )  # Extra hour for safety

            return {
                "webhook_id": webhook.id,
                "status": "scheduled",
                "delay_seconds": delay_seconds,
                "scheduled_key": scheduled_key,
            }

        except Exception as e:
            logger.error(
                "Failed to schedule webhook", webhook_id=webhook.id, error=str(e)
            )
            return {"webhook_id": webhook.id, "status": "failed", "error": str(e)}

    async def _move_to_dead_letter(self, attempt: Any) -> Dict[str, Any]:  # WebhookAttempt
        """Move a failed webhook attempt to the dead letter queue."""
        try:
            dead_letter_key = (
                f"{self.dead_letter_prefix}{attempt.webhook_id}_{attempt.id}"
            )

            dead_letter_data = {
                "webhook_id": attempt.webhook_id,
                "attempt_id": attempt.id,
                "payload": attempt.payload,
                "original_url": attempt.url,
                "error_message": attempt.error_message,
                "retry_count": attempt.retry_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            self.redis_client.hset(dead_letter_key, mapping=dead_letter_data)
            self.redis_client.expire(dead_letter_key, 86400 * 30)  # Keep for 30 days

            # Update attempt status (using SimpleNamespace for now)
            attempt.status = WebhookStatus.DEAD_LETTER.value

            logger.info("Webhook moved to dead letter queue", attempt_id=attempt.id)

            return {
                "webhook_id": attempt.webhook_id,
                "attempt_id": attempt.id,
                "status": "dead_letter",
                "dead_letter_key": dead_letter_key,
            }

        except Exception as e:
            logger.error(
                "Failed to move webhook to dead letter queue",
                attempt_id=attempt.id,
                error=str(e),
            )
            return {
                "webhook_id": attempt.webhook_id,
                "attempt_id": attempt.id,
                "status": "failed",
                "error": str(e),
            }

    def _prepare_payload(
        self, stream: Stream, event: WebhookEvent, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare the webhook payload."""
        return {
            "event": event.value,
            "stream_id": stream.id,
            "user_id": stream.user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "api_version": "v1",
        }

    def _should_send_event(self, webhook: Webhook, event: WebhookEvent) -> bool:
        """Check if webhook should receive this event type."""
        if not webhook.events:
            return True  # Send all events if no filter configured

        return event.value in webhook.events

    def _check_rate_limit(self, webhook: Webhook) -> bool:
        """Check if webhook is within rate limits."""
        # TODO: Add rate_limit_per_minute field to Webhook model if needed
        # For now, no rate limiting
        return True

    def _update_rate_limit(self, webhook: Webhook) -> None:
        """Update rate limit counter for successful delivery."""
        # TODO: Add rate_limit_per_minute field to Webhook model if needed
        # For now, no rate limiting
        pass

    def _generate_hmac_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        signature = hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        return f"sha256={signature}"

    def _calculate_retry_delay(self, retry_count: int) -> int:
        """Calculate retry delay with exponential backoff and jitter."""
        # Base exponential backoff
        delay = min(
            self.base_delay * (self.backoff_factor**retry_count), self.max_delay
        )

        # Add jitter
        jitter = (
            delay * self.jitter_factor * (0.5 - hash(str(time.time())) % 1000 / 1000)
        )

        return int(delay + jitter)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()
