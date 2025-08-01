"""Stream management endpoints."""

from fastapi import APIRouter, Depends
from uuid import UUID

from ..dependencies import get_current_organization, require_scope
from ...domain.models.api_key import APIScopes

router = APIRouter()


@router.post("/")
async def create_stream(
    stream_url: str,
    organization=Depends(get_current_organization),
    api_key=Depends(require_scope(APIScopes.STREAMS_WRITE)),
):
    """Create a new stream for processing."""
    # TODO: Implement stream creation
    return {
        "message": "Stream creation not implemented",
        "organization_id": str(organization.id),
    }


@router.get("/{stream_id}")
async def get_stream(
    stream_id: UUID,
    organization=Depends(get_current_organization),
    api_key=Depends(require_scope(APIScopes.STREAMS_READ)),
):
    """Get stream details."""
    # TODO: Implement
    return {"message": "Stream retrieval not implemented"}
