"""Webhooks router for the TL;DR Highlight API.

This module provides endpoints for webhook configuration and management.
Full implementation will be provided in later phases.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.common import COMMON_RESPONSES, StatusResponse
from src.infrastructure.database import get_db

router = APIRouter()


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Webhooks service status",
    description="Check if webhooks service is operational",
    responses=COMMON_RESPONSES,
)
async def webhooks_status() -> StatusResponse:
    """Get webhooks service status.

    Returns:
        StatusResponse: Webhooks service status
    """
    from datetime import datetime

    return StatusResponse(
        status="Webhooks service operational", timestamp=datetime.utcnow()
    )


@router.post(
    "/",
    summary="Create webhook",
    description="Create a new webhook configuration (placeholder)",
    responses=COMMON_RESPONSES,
)
async def create_webhook(db: AsyncSession = Depends(get_db)):
    """Create a new webhook configuration.

    This endpoint will be fully implemented in later phases.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Webhook creation will be implemented in later phases",
    )


@router.get(
    "/",
    summary="List webhooks",
    description="List webhook configurations (placeholder)",
    responses=COMMON_RESPONSES,
)
async def list_webhooks(db: AsyncSession = Depends(get_db)):
    """List webhook configurations for the authenticated user.

    This endpoint will be fully implemented in later phases.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Webhook listing will be implemented in later phases",
    )


@router.get(
    "/{webhook_id}",
    summary="Get webhook details",
    description="Get details of a specific webhook (placeholder)",
    responses=COMMON_RESPONSES,
)
async def get_webhook(webhook_id: str, db: AsyncSession = Depends(get_db)):
    """Get webhook configuration details.

    This endpoint will be fully implemented in later phases.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Webhook details will be implemented in later phases",
    )


@router.put(
    "/{webhook_id}",
    summary="Update webhook",
    description="Update webhook configuration (placeholder)",
    responses=COMMON_RESPONSES,
)
async def update_webhook(webhook_id: str, db: AsyncSession = Depends(get_db)):
    """Update webhook configuration.

    This endpoint will be fully implemented in later phases.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Webhook updates will be implemented in later phases",
    )


@router.delete(
    "/{webhook_id}",
    summary="Delete webhook",
    description="Delete webhook configuration (placeholder)",
    responses=COMMON_RESPONSES,
)
async def delete_webhook(webhook_id: str, db: AsyncSession = Depends(get_db)):
    """Delete webhook configuration.

    This endpoint will be fully implemented in later phases.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Webhook deletion will be implemented in later phases",
    )


@router.post(
    "/{webhook_id}/test",
    summary="Test webhook",
    description="Send test webhook delivery (placeholder)",
    responses=COMMON_RESPONSES,
)
async def test_webhook(webhook_id: str, db: AsyncSession = Depends(get_db)):
    """Send test webhook delivery.

    This endpoint will be fully implemented in later phases.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Webhook testing will be implemented in later phases",
    )


@router.get(
    "/{webhook_id}/events",
    summary="Get webhook events",
    description="Get webhook delivery history",
    responses=COMMON_RESPONSES,
)
async def get_webhook_events(
    webhook_id: int,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    use_case: WebhookConfigurationUseCase = Depends(get_webhook_configuration_use_case),
) -> Dict:
    """Get webhook delivery events.

    Returns the delivery history for a webhook.
    """
    request = mapper.to_get_webhook_events_request(
        current_user.id, webhook_id, page, per_page
    )
    result = await use_case.get_webhook_events(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this webhook",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0]
            if result.errors
            else "Failed to get webhook events",
        )

    return {
        "events": result.events,
        "total": result.total,
        "success_rate": result.success_rate,
        "page": page,
        "per_page": per_page,
    }
