"""Organization management endpoints."""

from fastapi import APIRouter, Depends

from ..dependencies import get_current_organization, require_scope
from ...domain.models.api_key import APIScopes

router = APIRouter()


@router.get("/me")
async def get_current_org(
    organization=Depends(get_current_organization),
    api_key=Depends(require_scope(APIScopes.ORG_READ)),
):
    """Get current organization details."""
    return {
        "id": str(organization.id),
        "name": organization.name,
        "slug": organization.slug,
        "usage": {
            "total_streams": organization.total_streams_processed,
            "total_highlights": organization.total_highlights_generated,
            "total_seconds": organization.total_processing_seconds,
        },
    }
