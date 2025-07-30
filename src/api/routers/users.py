"""User management endpoints.

This module implements user management functionality including
user profile management and admin user operations.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.infrastructure.database import get_db
from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.organization import Organization
from src.infrastructure.persistence.models.user import User
from src.api.dependencies.auth import (
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)
from src.api.schemas.users import (
    PasswordChangeRequest,
    PasswordChangeResponse,
    UserListResponse,
    UserProfileResponse,
    UserResponse,
    UserUpdate,
)

router = APIRouter(prefix="/api/v1/users", tags=["Users"])


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get current user's profile information.

    Returns detailed profile information for the authenticated user
    including API key count and organization ownership.
    """
    # Get API keys count
    api_keys_result = await db.execute(
        select(APIKey)
        .where(APIKey.user_id == current_user.id)
        .where(APIKey.active)
    )
    api_keys_count = len(api_keys_result.scalars().all())

    # Get organizations count
    orgs_result = await db.execute(
        select(Organization).where(Organization.owner_id == current_user.id)
    )
    orgs_count = len(orgs_result.scalars().all())

    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        company_name=current_user.company_name,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        api_keys_count=api_keys_count,
        organizations_count=orgs_count,
        last_login=None,  # Would be tracked in a full implementation
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user's profile information.

    Allows users to update their email, company name, and password.
    """
    # Check if email is being changed and if it's already taken
    if user_data.email and user_data.email != current_user.email:
        result = await db.execute(select(User).where(User.email == user_data.email))
        existing_user = result.scalar_one_or_none()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address is already in use",
            )
        current_user.email = user_data.email

    # Update company name
    if user_data.company_name:
        current_user.company_name = user_data.company_name

    # Update password if provided
    if user_data.password:
        current_user.password_hash = hash_password(user_data.password)

    # Update timestamp
    current_user.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        company_name=current_user.company_name,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
    )


@router.post("/me/change-password", response_model=PasswordChangeResponse)
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change current user's password.

    Requires the current password for verification before setting
    the new password.
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Check that new password is different
    if password_data.current_password == password_data.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password",
        )

    # Update password
    current_user.password_hash = hash_password(password_data.new_password)
    current_user.updated_at = datetime.now(timezone.utc)

    await db.commit()

    return PasswordChangeResponse(
        message="Password changed successfully", changed_at=current_user.updated_at
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Get user details by ID (admin only).

    Returns user information for the specified user ID.
    Only accessible to users with admin permissions.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return UserResponse(
        id=user.id,
        email=user.email,
        company_name=user.company_name,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Update user details by ID (admin only).

    Allows admins to update any user's information including
    email, company name, and password.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Check if email is being changed and if it's already taken
    if user_data.email and user_data.email != user.email:
        email_result = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        existing_user = email_result.scalar_one_or_none()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address is already in use",
            )
        user.email = user_data.email

    # Update company name
    if user_data.company_name:
        user.company_name = user_data.company_name

    # Update password if provided
    if user_data.password:
        user.password_hash = hash_password(user_data.password)

    # Update timestamp
    user.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        company_name=user.company_name,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.get("", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """List all users (admin only).

    Returns a paginated list of all users in the system.
    Only accessible to users with admin permissions.
    """
    # Calculate offset
    offset = (page - 1) * per_page

    # Get total count
    count_result = await db.execute(select(User).count())
    total = count_result.scalar()

    # Get users for current page
    result = await db.execute(
        select(User).order_by(User.created_at.desc()).offset(offset).limit(per_page)
    )
    users = result.scalars().all()

    user_responses = [
        UserResponse(
            id=user.id,
            email=user.email,
            company_name=user.company_name,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
        for user in users
    ]

    return UserListResponse(
        total=total, page=page, per_page=per_page, users=user_responses
    )


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user (admin only).

    Permanently deletes a user and all associated data.
    Only accessible to users with admin permissions.
    """
    # Prevent admin from deleting themselves
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    await db.delete(user)
    await db.commit()

    return {"message": "User deleted successfully"}
