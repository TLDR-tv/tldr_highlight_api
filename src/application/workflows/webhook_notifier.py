"""Webhook notifications for B2B AI highlighting - clean Pythonic implementation.

This module handles event notifications to customer endpoints,
allowing real-time integration with their systems.
"""

import json
import hmac
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiohttp
import logfire

from src.domain.entities.webhook import Webhook, WebhookEvent
from src.domain.exceptions import EntityNotFoundError


@dataclass
class WebhookNotifier:
    """Sends notifications to customer webhooks.
    
    Simple, reliable webhook delivery for B2B integrations.
    """
    
    webhook_repo: Any  # Duck typing
    stream_repo: Any
    highlight_repo: Any
    
    # Configuration
    timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        self.logger = logfire.get_logger(__name__)
    
    async def notify(
        self,
        event: WebhookEvent,
        user_id: int,
        resource_id: int,
        data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Send webhook notifications for an event.
        
        Args:
            event: Type of event that occurred
            user_id: User who owns the resource
            resource_id: ID of the resource (stream, highlight, etc.)
            data: Additional event data
            
        Returns:
            List of delivery IDs for tracking
        """
        # Get active webhooks
        webhooks = await self.webhook_repo.get_active_for_event(event, user_id)
        if not webhooks:
            return []
        
        # Build payload
        payload = await self._build_payload(event, resource_id, data)
        
        # Send to each webhook
        delivery_ids = []
        async with aiohttp.ClientSession() as session:
            for webhook in webhooks:
                delivery_id = await self._send_webhook(
                    session, webhook, event, payload
                )
                if delivery_id:
                    delivery_ids.append(delivery_id)
        
        return delivery_ids
    
    async def notify_stream_complete(
        self,
        stream_id: int,
        user_id: int
    ) -> List[str]:
        """Notify that stream processing is complete."""
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        # Get highlights
        highlights = await self.highlight_repo.get_by_stream(stream_id)
        
        data = {
            "stream_id": stream_id,
            "status": "completed",
            "highlights_found": len(highlights),
            "duration_seconds": stream.duration.total_seconds() if stream.duration else 0,
            "highlights": [
                {
                    "id": h.id,
                    "start_time": h.start_time.seconds,
                    "end_time": h.end_time.seconds,
                    "confidence": h.confidence_score.value,
                    "types": h.highlight_types,
                    "title": h.title,
                }
                for h in highlights[:10]  # First 10
            ],
        }
        
        return await self.notify(
            WebhookEvent.PROCESSING_COMPLETE,
            user_id,
            stream_id,
            data
        )
    
    async def notify_highlight_detected(
        self,
        highlight_id: int,
        stream_id: int,
        user_id: int
    ) -> List[str]:
        """Notify that a new highlight was detected."""
        highlight = await self.highlight_repo.get(highlight_id)
        if not highlight:
            return []
        
        data = {
            "highlight_id": highlight_id,
            "stream_id": stream_id,
            "start_time": highlight.start_time.seconds,
            "end_time": highlight.end_time.seconds,
            "confidence": highlight.confidence_score.value,
            "types": highlight.highlight_types,
            "title": highlight.title,
            "description": highlight.description,
        }
        
        return await self.notify(
            WebhookEvent.HIGHLIGHT_DETECTED,
            user_id,
            stream_id,
            data
        )
    
    # Private methods
    
    async def _build_payload(
        self,
        event: WebhookEvent,
        resource_id: int,
        data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build webhook payload."""
        return {
            "event": event.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resource_id": resource_id,
            "data": data or {},
        }
    
    async def _send_webhook(
        self,
        session: aiohttp.ClientSession,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Dict[str, Any]
    ) -> Optional[str]:
        """Send a single webhook."""
        # Generate signature
        body = json.dumps(payload)
        signature = self._generate_signature(webhook.secret, body)
        
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Event": event.value,
        }
        
        try:
            async with session.post(
                webhook.url.value,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                success = 200 <= response.status < 300
                
                # Log delivery
                delivery_id = await self.webhook_repo.log_delivery(
                    webhook_id=webhook.id,
                    event=event,
                    payload=payload,
                    response_status=response.status,
                    response_body=await response.text(),
                    success=success,
                )
                
                if not success:
                    self.logger.warning(
                        f"Webhook delivery failed: {response.status}",
                        extra={
                            "webhook_id": webhook.id,
                            "url": webhook.url.value,
                            "status": response.status,
                        }
                    )
                
                return delivery_id
                
        except Exception as e:
            self.logger.error(
                f"Webhook delivery error: {e}",
                extra={
                    "webhook_id": webhook.id,
                    "url": webhook.url.value,
                    "error": str(e),
                }
            )
            
            # Log failed delivery
            return await self.webhook_repo.log_delivery(
                webhook_id=webhook.id,
                event=event,
                payload=payload,
                response_status=0,
                response_body=str(e),
                success=False,
            )
    
    def _generate_signature(self, secret: str, body: str) -> str:
        """Generate HMAC signature for webhook."""
        return hmac.new(
            secret.encode(),
            body.encode(),
            hashlib.sha256
        ).hexdigest()


@dataclass
class WebhookRetryHandler:
    """Handles webhook retry logic separately.
    
    This can be run as a background task to retry failed deliveries.
    """
    
    webhook_repo: Any
    notifier: WebhookNotifier
    
    # Retry configuration
    retry_delays: List[int] = field(
        default_factory=lambda: [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h
    )
    
    async def retry_failed_deliveries(self) -> int:
        """Retry recent failed webhook deliveries.
        
        Returns:
            Number of deliveries retried
        """
        # Get failed deliveries from last hour
        failed = await self.webhook_repo.get_failed_deliveries(
            since_minutes=60,
            max_attempts=len(self.retry_delays)
        )
        
        retried = 0
        for delivery in failed:
            if await self._should_retry(delivery):
                await self._retry_delivery(delivery)
                retried += 1
        
        return retried
    
    async def _should_retry(self, delivery) -> bool:
        """Check if delivery should be retried."""
        if delivery.attempt_count >= len(self.retry_delays):
            return False
        
        # Check if enough time has passed
        delay = self.retry_delays[delivery.attempt_count - 1]
        time_since = (datetime.now(timezone.utc) - delivery.created_at).total_seconds()
        
        return time_since >= delay
    
    async def _retry_delivery(self, delivery) -> bool:
        """Retry a single delivery."""
        webhook = await self.webhook_repo.get(delivery.webhook_id)
        if not webhook or webhook.status != "active":
            return False
        
        # Resend using notifier
        async with aiohttp.ClientSession() as session:
            result = await self.notifier._send_webhook(
                session,
                webhook,
                delivery.event,
                delivery.payload
            )
            
            # Update attempt count
            await self.webhook_repo.update_delivery_attempts(
                delivery.id,
                delivery.attempt_count + 1
            )
            
            return result is not None