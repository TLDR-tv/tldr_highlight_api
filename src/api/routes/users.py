"""User management endpoints."""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_session,
    get_current_user,
    require_user,
    require_admin_user,
    get_user_service,
    get_user_repository,
)
from ..schemas.user import (
    UserResponse,
    UserCreateRequest,
    UserUpdateRequest,
    UserRoleUpdateRequest,
    UserListResponse,
)
from ..schemas.auth import PasswordChangeRequest, MessageResponse
from ...application.services.user_service import UserService
from ...infrastructure.storage.repositories import UserRepository
from ...domain.models.user import User

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(require_user),
):
    """Get current user profile."""
    return UserResponse.model_validate(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    request: UserUpdateRequest,
    current_user: User = Depends(require_user),
    user_service: UserService = Depends(get_user_service),
):
    """Update current user profile."""
    try:
        updated_user = await user_service.update_profile(
            user_id=current_user.id,
            name=request.name,
            email=request.email,
        )
        return UserResponse.model_validate(updated_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.put("/me/password", response_model=MessageResponse)
async def change_current_user_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(require_user),
    user_service: UserService = Depends(get_user_service),
):
    """Change current user password."""
    try:
        success = await user_service.change_password(
            user_id=current_user.id,
            old_password=request.current_password,
            new_password=request.new_password,
        )
        return MessageResponse(message="Password changed successfully")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=UserListResponse)
async def list_organization_users(
    current_user: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    """List all users in the organization (admin only)."""

    users = await user_service.list_organization_users(current_user.organization_id)
    return UserListResponse(
        users=[UserResponse.model_validate(u) for u in users],
        total=len(users),
    )


@router.post("", response_model=UserResponse)
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    """Create a new user in the organization (admin only)."""

    try:
        new_user = await user_service.create_user(
            organization_id=current_user.organization_id,
            email=request.email,
            name=request.name,
            password=request.password,
            role=request.role,
        )
        return UserResponse.model_validate(new_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    current_user: User = Depends(require_user),
    user_repository: UserRepository = Depends(get_user_repository),
):
    """Get user details."""
    # Users can view their own profile or admins can view any user in their org
    if user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view other users",
        )

    user = await user_repository.get(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Ensure user is from same organization
    if user.organization_id != current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not found",
        )

    return UserResponse.model_validate(user)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    request: UserUpdateRequest,
    current_user: User = Depends(require_user),
    user_service: UserService = Depends(get_user_service),
):
    """Update user profile (self or admin only)."""
    # Users can update their own profile or admins can update any user in their org
    if user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot update other users",
        )

    try:
        updated_user = await user_service.update_profile(
            user_id=user_id,
            name=request.name,
            email=request.email,
        )

        # Ensure user is from same organization
        if updated_user.organization_id != current_user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not found",
            )

        return UserResponse.model_validate(updated_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.put("/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: UUID,
    request: UserRoleUpdateRequest,
    current_user: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    """Update user role (admin only)."""

    try:
        updated_user = await user_service.update_user_role(
            user_id=user_id,
            role=request.role,
            admin_user_id=current_user.id,
        )
        return UserResponse.model_validate(updated_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/{user_id}", response_model=MessageResponse)
async def deactivate_user(
    user_id: UUID,
    current_user: User = Depends(require_admin_user),
    user_service: UserService = Depends(get_user_service),
):
    """Deactivate user (admin only)."""

    try:
        success = await user_service.deactivate_user(
            user_id=user_id,
            admin_user_id=current_user.id,
        )
        return MessageResponse(message="User deactivated successfully")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
