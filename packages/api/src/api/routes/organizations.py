"""Organization management endpoints."""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_current_organization,
    require_scope,
    get_session,
    require_user,
    require_admin_user,
    get_organization_service,
    get_organization_repository,
    get_user_repository,
    get_api_key_repository,
    get_api_key_service,
)
from ..schemas.organization import (
    OrganizationResponse,
    OrganizationUpdateRequest,
    OrganizationUsageResponse,
    WebhookSecretResponse,
    WakeWordRequest,
)
from ..schemas.user import UserListResponse, UserResponse
from ..schemas.api_key import APIKeyResponse, APIKeyListResponse, APIKeyCreateRequest, APIKeyCreateResponse
from shared.domain.models.api_key import APIScopes
from shared.domain.models.user import User
from api.services.organization_service import OrganizationService
from shared.infrastructure.storage.repositories import (
    OrganizationRepository,
    UserRepository,
    APIKeyRepository,
)
from shared.infrastructure.security.api_key_service import APIKeyService

router = APIRouter()


# API key authenticated endpoints
@router.get("/me", response_model=OrganizationResponse)
async def get_current_org(
    organization=Depends(get_current_organization),
    api_key=Depends(require_scope(APIScopes.ORG_READ)),
):
    """Get current organization details (API key auth)."""
    return OrganizationResponse.model_validate(organization)


# User JWT authenticated endpoints
@router.get("/current", response_model=OrganizationResponse)
async def get_current_org_for_user(
    current_user: User = Depends(require_user),
    org_repository: OrganizationRepository = Depends(get_organization_repository),
):
    """Get current organization details (user auth)."""
    org = await org_repository.get(current_user.organization_id)

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    return OrganizationResponse.model_validate(org)


@router.put("/current", response_model=OrganizationResponse)
async def update_organization(
    request: OrganizationUpdateRequest,
    current_user: User = Depends(require_admin_user),
    org_service: OrganizationService = Depends(get_organization_service),
):
    """Update organization details (admin only)."""
    try:
        updated_org = await org_service.update_organization(
            organization_id=current_user.organization_id,
            name=request.name,
            webhook_url=request.webhook_url,
        )
        return OrganizationResponse.model_validate(updated_org)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/current/usage", response_model=OrganizationUsageResponse)
async def get_organization_usage(
    current_user: User = Depends(require_user),
    org_service: OrganizationService = Depends(get_organization_service),
):
    """Get organization usage statistics."""
    try:
        usage_stats = await org_service.get_usage_stats(current_user.organization_id)
        return OrganizationUsageResponse(**usage_stats)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.put("/current/webhook", response_model=OrganizationResponse)
async def configure_webhook(
    webhook_url: str,
    current_user: User = Depends(require_admin_user),
    org_service: OrganizationService = Depends(get_organization_service),
):
    """Configure webhook URL (admin only)."""
    try:
        updated_org = await org_service.update_organization(
            organization_id=current_user.organization_id,
            webhook_url=webhook_url,
        )
        return OrganizationResponse.model_validate(updated_org)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/current/webhook/secret", response_model=WebhookSecretResponse)
async def regenerate_webhook_secret(
    current_user: User = Depends(require_admin_user),
    org_service: OrganizationService = Depends(get_organization_service),
):
    """Regenerate webhook secret (admin only)."""
    try:
        new_secret = await org_service.regenerate_webhook_secret(
            organization_id=current_user.organization_id
        )
        return WebhookSecretResponse(webhook_secret=new_secret)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/current/users", response_model=UserListResponse)
async def list_organization_users(
    current_user: User = Depends(require_user),
    user_repository: UserRepository = Depends(get_user_repository),
):
    """List all users in the organization."""
    users = await user_repository.list_by_organization(current_user.organization_id)

    return UserListResponse(
        users=[UserResponse.model_validate(u) for u in users],
        total=len(users),
    )


@router.post("/current/wake-words", response_model=OrganizationResponse)
async def add_wake_word(
    request: WakeWordRequest,
    current_user: User = Depends(require_admin_user),
    org_service: OrganizationService = Depends(get_organization_service),
):
    """Add a custom wake word (admin only)."""
    try:
        updated_org = await org_service.add_wake_word(
            organization_id=current_user.organization_id,
            wake_word=request.wake_word,
        )
        return OrganizationResponse.model_validate(updated_org)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/current/wake-words/{wake_word}", response_model=OrganizationResponse)
async def remove_wake_word(
    wake_word: str,
    current_user: User = Depends(require_admin_user),
    org_service: OrganizationService = Depends(get_organization_service),
):
    """Remove a custom wake word (admin only)."""
    try:
        updated_org = await org_service.remove_wake_word(
            organization_id=current_user.organization_id,
            wake_word=wake_word,
        )
        return OrganizationResponse.model_validate(updated_org)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/current/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    current_user: User = Depends(require_admin_user),
    api_key_repository: APIKeyRepository = Depends(get_api_key_repository),
):
    """List all API keys for the organization (admin only)."""
    keys = await api_key_repository.list_by_organization(current_user.organization_id)

    api_key_responses = [
        APIKeyResponse(
            id=key.id,
            name=key.name,
            prefix=key.prefix,
            scopes=list(key.scopes),
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            is_active=key.is_active,
        )
        for key in keys
    ]

    return APIKeyListResponse(
        api_keys=api_key_responses,
        total=len(keys),
    )


@router.post("/current/api-keys", response_model=APIKeyCreateResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: User = Depends(require_admin_user),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Create a new API key for the organization (admin only)."""
    try:
        # Generate the API key
        raw_key, api_key_entity = await api_key_service.generate_api_key(
            organization_id=current_user.organization_id,
            name=request.name,
            scopes=set(request.scopes),
            created_by_user_id=current_user.id,
        )
        
        # Return the response with the raw key (only shown once)
        return APIKeyCreateResponse(
            api_key=APIKeyResponse(
                id=api_key_entity.id,
                name=api_key_entity.name,
                prefix=api_key_entity.prefix,
                scopes=list(api_key_entity.scopes),
                created_at=api_key_entity.created_at,
                last_used_at=api_key_entity.last_used_at,
                is_active=api_key_entity.is_active,
            ),
            raw_key=raw_key,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/current/api-keys/{key_id}", response_model=dict)
async def delete_api_key(
    key_id: UUID,
    current_user: User = Depends(require_admin_user),
    api_key_repository: APIKeyRepository = Depends(get_api_key_repository),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Delete an API key (admin only)."""
    # First check if the key exists and belongs to the organization
    api_key = await api_key_repository.get(key_id)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    if api_key.organization_id != current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    # Revoke the key (soft delete)
    await api_key_service.revoke_api_key(key_id)
    
    return {"detail": "API key revoked successfully"}
