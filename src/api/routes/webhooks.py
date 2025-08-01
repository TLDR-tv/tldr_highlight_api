"""Webhook management endpoints."""

from fastapi import APIRouter, Depends

from ..dependencies import get_current_organization, require_scope
from ....domain.models.api_key import APIScopes

router = APIRouter()


@router.post("/configure")
async def configure_webhook(
    webhook_url: str,
    organization=Depends(get_current_organization),
    api_key=Depends(require_scope(APIScopes.WEBHOOKS_WRITE)),
):
    """Configure webhook for the organization."""
    # TODO: Implement
    return {"message": "Webhook configuration not implemented"}


@router.post("/receive/{webhook_id}")
async def receive_webhook(webhook_id: str):
    """Receive incoming webhook."""
    # TODO: Implement webhook reception
    return {"received": True}
