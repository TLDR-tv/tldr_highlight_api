"""User management endpoints.

This module implements user management functionality including
user profile management using domain-driven design principles.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.use_cases import (
    get_user_management_use_case,
    get_authentication_use_case
)
from src.api.schemas.users import (
    PasswordChangeRequest,
    PasswordChangeResponse,
    UserListResponse,
    UserProfileResponse,
    UserResponse,
    UserUpdate,
)
from src.application.use_cases.user_management import (
    UserManagementUseCase,
    UpdateProfileRequest,
    ChangePasswordRequest,
    GetProfileRequest
)
from src.application.use_cases.authentication import (
    AuthenticationUseCase,
    ListAPIKeysResult
)
from src.application.use_cases.base import ResultStatus
from src.domain.entities.user import User

router = APIRouter(prefix="/api/v1/users", tags=["Users"])


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    user_mgmt_use_case: UserManagementUseCase = Depends(get_user_management_use_case)
) -> UserProfileResponse:
    """Get current user's profile information.

    Returns detailed profile information for the authenticated user
    including API key count and organization ownership.
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication"
        )
    
    # Get profile with stats
    request = GetProfileRequest(user_id=current_user.id)
    result = await user_mgmt_use_case.get_profile(request)
    
    if result.status == ResultStatus.SUCCESS:
        if not result.user or result.user.id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response from user management service"
            )
        return UserProfileResponse(
            id=result.user.id,
            email=result.user.email.value,
            company_name=result.user.company_name.value,
            created_at=result.user.created_at.value,
            updated_at=result.user.updated_at.value,
            api_keys_count=result.api_keys_count or 0,
            organizations_count=result.organizations_count or 0,
            last_login=None,  # Would be tracked in a full implementation
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to get profile"
        )


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    user_mgmt_use_case: UserManagementUseCase = Depends(get_user_management_use_case)
) -> UserResponse:
    """Update current user's profile information.

    Allows users to update their email, company name, and password.
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication"
        )
    
    # Create update request
    request = UpdateProfileRequest(
        user_id=current_user.id,
        email=user_data.email,
        company_name=user_data.company_name,
        password=user_data.password
    )
    
    # Execute update
    result = await user_mgmt_use_case.update_profile(request)
    
    if result.status == ResultStatus.SUCCESS:
        if not result.user or result.user.id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response from user management service"
            )
        return UserResponse(
            id=result.user.id,
            email=result.user.email.value,
            company_name=result.user.company_name.value,
            created_at=result.user.created_at.value,
            updated_at=result.user.updated_at.value,
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.errors[0] if result.errors else "Validation error"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to update profile"
        )


@router.post("/me/change-password", response_model=PasswordChangeResponse)
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    user_mgmt_use_case: UserManagementUseCase = Depends(get_user_management_use_case)
) -> PasswordChangeResponse:
    """Change current user's password.

    Requires the current password for verification before setting
    the new password.
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication"
        )
    
    # Create change password request
    request = ChangePasswordRequest(
        user_id=current_user.id,
        current_password=password_data.current_password,
        new_password=password_data.new_password
    )
    
    # Execute password change
    result = await user_mgmt_use_case.change_password(request)
    
    if result.status == ResultStatus.SUCCESS:
        return PasswordChangeResponse(
            message="Password changed successfully",
            changed_at=datetime.utcnow()
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.errors[0] if result.errors else "Validation error"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to change password"
        )


# TODO: Implement admin-only endpoints using domain services
# For now, these endpoints are removed as they require admin functionality
# which is not yet implemented in the domain layer
