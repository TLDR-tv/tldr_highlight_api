"""Webhook receiver endpoints for external events."""

from typing import Dict, Any, Optional
import logging
from fastapi import APIRouter, Request, Response, Header, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.models.webhook_models import (
    WebhookResponse,
    HundredMSWebhookPayload,
    TwitchWebhookPayload,
    StreamStartedWebhookEvent,
    WebhookPlatform
)
from src.api.dependencies import get_webhook_processing_use_case
from src.application.use_cases.webhook_processing import (
    WebhookProcessingUseCase,
    ProcessWebhookRequest
)
from src.application.use_cases.base import ResultStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhook-receiver"])


@router.post("/receive/stream", response_model=WebhookResponse)
async def receive_generic_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    webhook_use_case: WebhookProcessingUseCase = Depends(get_webhook_processing_use_case),
    x_api_key: Optional[str] = Header(None),
    x_webhook_signature: Optional[str] = Header(None),
    x_webhook_timestamp: Optional[str] = Header(None),
    x_webhook_platform: Optional[str] = Header(None)
) -> WebhookResponse:
    """Receive generic webhook for stream events.
    
    This endpoint accepts webhooks from any platform with a standardized format.
    Authentication can be via API key header or webhook signature.
    """
    try:
        # Get raw body and parsed JSON
        raw_body = await request.body()
        payload = await request.json()
        
        # Determine platform
        platform = x_webhook_platform or WebhookPlatform.CUSTOM.value
        
        # Build headers dict
        headers = {
            "x-webhook-signature": x_webhook_signature,
            "x-webhook-timestamp": x_webhook_timestamp,
            "x-api-key": x_api_key,
            "x-webhook-platform": platform
        }
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Process webhook
        process_request = ProcessWebhookRequest(
            platform=platform,
            payload=payload,
            raw_payload=raw_body,
            headers=headers,
            api_key=x_api_key
        )
        
        # Process asynchronously in background
        background_tasks.add_task(
            _process_webhook_async,
            webhook_use_case,
            process_request
        )
        
        # Return immediate success
        return WebhookResponse(
            success=True,
            message="Webhook received and queued for processing",
            event_id=payload.get("event_id")
        )
        
    except Exception as e:
        logger.error(f"Error receiving webhook: {e}")
        return WebhookResponse(
            success=False,
            message=f"Error processing webhook: {str(e)}"
        )


@router.post("/receive/100ms", response_model=WebhookResponse)
async def receive_100ms_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    webhook_use_case: WebhookProcessingUseCase = Depends(get_webhook_processing_use_case),
    x_webhook_signature: Optional[str] = Header(None),
    x_webhook_timestamp: Optional[str] = Header(None)
) -> WebhookResponse:
    """Receive webhook from 100ms.
    
    This endpoint is specifically for 100ms webhooks with their event format.
    """
    try:
        # Get raw body and parsed JSON
        raw_body = await request.body()
        payload = await request.json()
        
        # Validate it's a 100ms webhook
        try:
            webhook_payload = HundredMSWebhookPayload(**payload)
        except Exception as e:
            return WebhookResponse(
                success=False,
                message=f"Invalid 100ms webhook format: {str(e)}"
            )
        
        # Build headers dict
        headers = {
            "x-webhook-signature": x_webhook_signature,
            "x-webhook-timestamp": x_webhook_timestamp
        }
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Process webhook
        process_request = ProcessWebhookRequest(
            platform=WebhookPlatform.HUNDREDMS.value,
            payload=payload,
            raw_payload=raw_body,
            headers=headers
        )
        
        # Process asynchronously in background
        background_tasks.add_task(
            _process_webhook_async,
            webhook_use_case,
            process_request
        )
        
        # Return immediate success
        return WebhookResponse(
            success=True,
            message="100ms webhook received and queued for processing",
            event_id=webhook_payload.id
        )
        
    except Exception as e:
        logger.error(f"Error receiving 100ms webhook: {e}")
        return WebhookResponse(
            success=False,
            message=f"Error processing 100ms webhook: {str(e)}"
        )


@router.post("/receive/twitch", response_model=WebhookResponse)
async def receive_twitch_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    webhook_use_case: WebhookProcessingUseCase = Depends(get_webhook_processing_use_case),
    twitch_eventsub_message_signature: Optional[str] = Header(None),
    twitch_eventsub_message_timestamp: Optional[str] = Header(None),
    twitch_eventsub_message_type: Optional[str] = Header(None)
) -> WebhookResponse:
    """Receive webhook from Twitch EventSub.
    
    This endpoint handles Twitch EventSub webhooks including verification challenges.
    """
    try:
        # Get raw body and parsed JSON
        raw_body = await request.body()
        payload = await request.json()
        
        # Handle Twitch verification challenge
        if twitch_eventsub_message_type == "webhook_callback_verification":
            challenge = payload.get("challenge")
            if challenge:
                return Response(content=challenge, media_type="text/plain")
        
        # Handle revocation notification
        if twitch_eventsub_message_type == "revocation":
            logger.warning(f"Twitch webhook revoked: {payload}")
            return WebhookResponse(
                success=True,
                message="Revocation acknowledged"
            )
        
        # Validate it's a Twitch webhook
        try:
            webhook_payload = TwitchWebhookPayload(**payload)
        except Exception as e:
            return WebhookResponse(
                success=False,
                message=f"Invalid Twitch webhook format: {str(e)}"
            )
        
        # Build headers dict
        headers = {
            "twitch-eventsub-message-signature": twitch_eventsub_message_signature,
            "twitch-eventsub-message-timestamp": twitch_eventsub_message_timestamp,
            "twitch-eventsub-message-type": twitch_eventsub_message_type
        }
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Process webhook
        process_request = ProcessWebhookRequest(
            platform=WebhookPlatform.TWITCH.value,
            payload=payload,
            raw_payload=raw_body,
            headers=headers
        )
        
        # Process asynchronously in background
        background_tasks.add_task(
            _process_webhook_async,
            webhook_use_case,
            process_request
        )
        
        # Return immediate success
        return WebhookResponse(
            success=True,
            message="Twitch webhook received and queued for processing",
            event_id=webhook_payload.event.get("id")
        )
        
    except Exception as e:
        logger.error(f"Error receiving Twitch webhook: {e}")
        return WebhookResponse(
            success=False,
            message=f"Error processing Twitch webhook: {str(e)}"
        )


@router.get("/receive/health")
async def webhook_receiver_health() -> Dict[str, Any]:
    """Health check endpoint for webhook receiver service."""
    return {
        "status": "healthy",
        "service": "webhook_receiver",
        "endpoints": [
            "/webhooks/receive/stream",
            "/webhooks/receive/100ms",
            "/webhooks/receive/twitch"
        ]
    }


async def _process_webhook_async(
    webhook_use_case: WebhookProcessingUseCase,
    process_request: ProcessWebhookRequest
):
    """Process webhook asynchronously in background."""
    try:
        result = await webhook_use_case.execute(process_request)
        
        if result.status == ResultStatus.SUCCESS:
            if result.duplicate:
                logger.info(f"Duplicate webhook ignored: {result.event_id}")
            else:
                logger.info(f"Webhook processed successfully: {result.event_id}, stream_id: {result.stream_id}")
        else:
            logger.error(f"Webhook processing failed: {result.errors}")
            
    except Exception as e:
        logger.error(f"Error in background webhook processing: {e}")