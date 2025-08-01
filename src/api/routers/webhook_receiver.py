"""Webhook receiver endpoints for external events."""

from typing import Dict, Any, Optional
import logging
from fastapi import APIRouter, Request, Header, Depends, BackgroundTasks

from src.api.schemas.webhook_models import (
    WebhookResponse,
    StreamStartedWebhookEvent,
    WebhookEventType,
)
from src.api.dependencies.use_cases import get_webhook_processing_use_case
from src.application.use_cases.webhook_processing import (
    WebhookProcessingUseCase,
    ProcessWebhookRequest,
)
from src.application.use_cases.base import ResultStatus
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhook-receiver"])


@router.post("/receive/stream", response_model=WebhookResponse)
async def receive_stream_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    webhook_use_case: WebhookProcessingUseCase = Depends(
        get_webhook_processing_use_case
    ),
    x_api_key: Optional[str] = Header(None),
    x_webhook_signature: Optional[str] = Header(None),
    x_webhook_timestamp: Optional[str] = Header(None),
) -> WebhookResponse:
    """Receive webhook for stream events (start, stop, update).

    This platform-agnostic endpoint accepts webhooks for stream ingestion triggers.
    Authentication is via API key header.

    Expected payload format:
    {
        "event_id": "unique-event-id",
        "event_type": "stream.started",
        "timestamp": "2024-01-01T00:00:00Z",
        "stream_url": "rtmp://example.com/stream",
        "metadata": {
            "title": "Stream Title",
            "description": "Stream Description",
            "external_stream_id": "external-id",
            "external_user_id": "user-id",
            "external_username": "username",
            "tags": ["tag1", "tag2"],
            "custom_data": {}
        }
    }
    """
    try:
        # Get raw body and parsed JSON
        raw_body = await request.body()
        payload = await request.json()

        # Build headers dict for verification
        headers = {
            "x-webhook-signature": x_webhook_signature,
            "x-webhook-timestamp": x_webhook_timestamp,
            "x-api-key": x_api_key,
        }
        headers = {k: v for k, v in headers.items() if v is not None}

        # Validate webhook event structure
        try:
            # Parse the webhook event
            event_type = payload.get(
                "event_type", WebhookEventType.STREAM_STARTED.value
            )

            # For stream started events, validate the payload
            if event_type == WebhookEventType.STREAM_STARTED.value:
                StreamStartedWebhookEvent(
                    event_id=payload.get(
                        "event_id", f"webhook_{datetime.utcnow().timestamp()}"
                    ),
                    event_type=event_type,
                    timestamp=payload.get("timestamp", datetime.utcnow()),
                    stream_url=payload["stream_url"],
                    api_key=x_api_key,
                    metadata=payload.get("metadata", {}),
                )
        except Exception as e:
            return WebhookResponse(
                success=False, message=f"Invalid webhook payload: {str(e)}"
            )

        # Process webhook
        process_request = ProcessWebhookRequest(
            platform="custom",  # All webhooks are now platform-agnostic
            payload=payload,
            raw_payload=raw_body,
            headers=headers,
            api_key=x_api_key,
        )

        # Process asynchronously in background
        background_tasks.add_task(
            _process_webhook_async, webhook_use_case, process_request
        )

        # Return immediate success
        return WebhookResponse(
            success=True,
            message="Webhook received and queued for processing",
            event_id=payload.get("event_id"),
        )

    except Exception as e:
        logger.error(f"Error receiving webhook: {e}")
        return WebhookResponse(
            success=False, message=f"Error processing webhook: {str(e)}"
        )


@router.get("/receive/health")
async def webhook_receiver_health() -> Dict[str, Any]:
    """Health check endpoint for webhook receiver service."""
    return {
        "status": "healthy",
        "service": "webhook_receiver",
        "endpoints": [
            "/webhooks/receive/stream",
        ],
    }


async def _process_webhook_async(
    webhook_use_case: WebhookProcessingUseCase, process_request: ProcessWebhookRequest
):
    """Process webhook asynchronously in background."""
    try:
        result = await webhook_use_case.execute(process_request)

        if result.status == ResultStatus.SUCCESS:
            if result.duplicate:
                logger.info(f"Duplicate webhook ignored: {result.event_id}")
            else:
                logger.info(
                    f"Webhook processed successfully: {result.event_id}, stream_id: {result.stream_id}"
                )
        else:
            logger.error(f"Webhook processing failed: {result.errors}")

    except Exception as e:
        logger.error(f"Error in background webhook processing: {e}")
