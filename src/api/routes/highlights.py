"""Highlight management endpoints."""

from fastapi import APIRouter, Depends
from uuid import UUID

from ..dependencies import get_current_organization, require_scope
from ...domain.models.api_key import APIScopes

router = APIRouter()


@router.get("/")
async def list_highlights(
    organization=Depends(get_current_organization),
    api_key=Depends(require_scope(APIScopes.HIGHLIGHTS_READ)),
):
    """List highlights for the organization."""
    # TODO: Implement
    return {"highlights": [], "organization_id": str(organization.id)}


@router.get("/{highlight_id}")
async def get_highlight(
    highlight_id: UUID,
    organization=Depends(get_current_organization),
    api_key=Depends(require_scope(APIScopes.HIGHLIGHTS_READ)),
):
    """Get highlight details."""
    # TODO: Implement
    return {"message": "Highlight retrieval not implemented"}
