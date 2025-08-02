"""Webhook delivery tasks."""

from typing import Dict, Optional
from uuid import UUID
import httpx
from structlog import get_logger

from worker.app import celery_app
from shared.infrastructure.storage.repositories import OrganizationRepository
from shared.infrastructure.database.database import Database
from shared.infrastructure.config import get_settings

logger = get_logger()


@celery_app.task(
    bind=True,
    name="send_highlight_webhook",
    max_retries=5,
    default_retry_delay=60,
)
def send_highlight_webhook(
    self,
    organization_id: str,
    highlight_data: Dict,
) -> Dict:
    """Send webhook notification for detected highlight.
    
    Args:
        organization_id: UUID of the organization
        highlight_data: Highlight information
        
    Returns:
        Webhook delivery result

    """
    import asyncio
    
    try:
        result = asyncio.run(
            _send_webhook_async(
                organization_id,
                "highlight.detected",
                highlight_data,
            )
        )
        return result
        
    except Exception as exc:
        logger.error(
            "Webhook delivery failed",
            organization_id=organization_id,
            event_type="highlight.detected",
            error=str(exc),
            exc_info=True,
        )
        
        # Exponential backoff
        countdown = self.default_retry_delay * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown, max_retries=5)


@celery_app.task(
    bind=True,
    name="send_stream_webhook",
    max_retries=3,
    default_retry_delay=30,
)
def send_stream_webhook(
    self,
    organization_id: str,
    event_type: str,
    stream_data: Dict,
) -> Dict:
    """Send webhook notification for stream events.
    
    Args:
        organization_id: UUID of the organization
        event_type: Type of event (stream.started, stream.completed, etc.)
        stream_data: Stream information
        
    Returns:
        Webhook delivery result

    """
    import asyncio
    
    try:
        result = asyncio.run(
            _send_webhook_async(
                organization_id,
                event_type,
                stream_data,
            )
        )
        return result
        
    except Exception as exc:
        logger.error(
            "Webhook delivery failed",
            organization_id=organization_id,
            event_type=event_type,
            error=str(exc),
            exc_info=True,
        )
        
        # Exponential backoff
        countdown = self.default_retry_delay * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)


@celery_app.task(
    name="send_progress_update",
    ignore_result=True,
)
def send_progress_update(stream_id: str, segments_processed: int) -> None:
    """Send progress update webhook.
    
    Args:
        stream_id: UUID of the stream
        segments_processed: Number of segments processed

    """
    import asyncio
    
    asyncio.run(
        _send_progress_update_async(stream_id, segments_processed)
    )


async def _send_webhook_async(
    organization_id: str,
    event_type: str,
    data: Dict,
) -> Dict:
    """Send webhook notification to organization."""
    settings = get_settings()
    database = Database(settings.database_url)
    await database.connect()
    
    try:
        async with database.session() as session:
            org_repo = OrganizationRepository(session)
            organization = await org_repo.get_by_id(UUID(organization_id))
            
            if not organization or not organization.webhook_url:
                logger.warning(
                    "No webhook URL configured",
                    organization_id=organization_id,
                )
                return {"status": "skipped", "reason": "no_webhook_url"}
        
        # Prepare webhook payload
        payload = {
            "event": event_type,
            "timestamp": data.get("detected_at", data.get("created_at")),
            "organization_id": organization_id,
            "data": data,
        }
        
        # Sign webhook payload
        signature = _generate_webhook_signature(
            payload,
            organization.webhook_secret,
        )
        
        # Send webhook
        async with httpx.AsyncClient() as client:
            response = await client.post(
                organization.webhook_url,
                json=payload,
                headers={
                    "X-Webhook-Signature": signature,
                    "X-Event-Type": event_type,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            
            response.raise_for_status()
            
            logger.info(
                "Webhook delivered successfully",
                organization_id=organization_id,
                event_type=event_type,
                status_code=response.status_code,
            )
            
            return {
                "status": "delivered",
                "status_code": response.status_code,
                "response": response.text[:500],  # First 500 chars
            }
            
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Webhook delivery HTTP error",
            organization_id=organization_id,
            event_type=event_type,
            status_code=exc.response.status_code,
            response=exc.response.text[:500],
        )
        raise
        
    except Exception as exc:
        logger.error(
            "Webhook delivery error",
            organization_id=organization_id,
            event_type=event_type,
            error=str(exc),
            exc_info=True,
        )
        raise
        
    finally:
        await database.disconnect()


async def _send_progress_update_async(
    stream_id: str,
    segments_processed: int,
) -> None:
    """Send progress update for stream processing."""
    settings = get_settings()
    database = Database(settings.database_url)
    await database.connect()
    
    try:
        async with database.session() as session:
            from shared.infrastructure.storage.repositories import StreamRepository
            
            stream_repo = StreamRepository(session)
            stream = await stream_repo.get_by_id(UUID(stream_id))
            
            if stream:
                await _send_webhook_async(
                    str(stream.organization_id),
                    "stream.progress",
                    {
                        "stream_id": stream_id,
                        "segments_processed": segments_processed,
                        "status": "processing",
                    },
                )
                
    finally:
        await database.disconnect()


def _generate_webhook_signature(payload: Dict, secret: str) -> str:
    """Generate HMAC signature for webhook payload."""
    import hmac
    import hashlib
    import json
    
    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    signature = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    
    return f"sha256={signature}"