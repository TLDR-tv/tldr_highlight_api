"""Authentication endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import HTTPBearer

from typing import Union
from ..dependencies import get_user_service, get_organization_service, get_settings_dep
from ..schemas.auth import (
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    RegisterOrganizationRequest,
    RegisterOrganizationResponse,
    PasswordResetRequest,
    PasswordResetResponse,
    PasswordResetWithTokenResponse,
    PasswordResetConfirmRequest,
    PasswordChangeRequest,
    MessageResponse,
)
from ..schemas.user import UserResponse
from ..schemas.organization import OrganizationResponse
from ...application.services.user_service import UserService
from ...application.services.organization_service import OrganizationService
from ...infrastructure.config import Settings

router = APIRouter()
security = HTTPBearer(auto_error=False)


@router.post("/register", response_model=RegisterOrganizationResponse)
async def register_organization(
    request: RegisterOrganizationRequest,
    org_service: OrganizationService = Depends(get_organization_service),
):
    """Register a new organization with owner user."""
    try:
        org, user = await org_service.create_organization(
            name=request.organization_name,
            owner_email=request.owner_email,
            owner_name=request.owner_name,
            owner_password=request.owner_password,
            webhook_url=request.webhook_url,
        )

        return RegisterOrganizationResponse(
            message="Organization registered successfully",
            organization=OrganizationResponse.model_validate(org).model_dump(),
            user=UserResponse.model_validate(user).model_dump(),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    response: Response,
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    """User login endpoint."""
    user, access_token, refresh_token = await user_service.authenticate(
        email=request.email,
        password=request.password,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Set refresh token as httpOnly cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=settings.environment == "production",
        samesite="lax",
        max_age=86400 * 7,  # 7 days
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_expiry_seconds,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(
    request: RefreshTokenRequest,
    response: Response,
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    """Refresh access token using refresh token."""
    new_access_token, new_refresh_token = await user_service.refresh_tokens(
        refresh_token=request.refresh_token,
    )

    if not new_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    # Update refresh token cookie
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=settings.environment == "production",
        samesite="lax",
        max_age=86400 * 7,  # 7 days
    )

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.jwt_expiry_seconds,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response):
    """Logout user by clearing cookies."""
    response.delete_cookie("refresh_token")
    return MessageResponse(message="Logged out successfully")


@router.post(
    "/forgot-password",
    response_model=Union[PasswordResetResponse, PasswordResetWithTokenResponse],
)
async def forgot_password(
    request: PasswordResetRequest,
    user_service: UserService = Depends(get_user_service),
):
    """Request password reset."""
    # Always return success to prevent email enumeration
    reset_token = await user_service.request_password_reset(email=request.email)

    # In a real system, send email with reset token
    # For development, we'll return it in the response
    if reset_token:
        return PasswordResetWithTokenResponse(
            message="Password reset email sent",
            reset_token=reset_token,  # Remove this in production!
        )

    return PasswordResetResponse(message="Password reset email sent")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    request: PasswordResetConfirmRequest,
    user_service: UserService = Depends(get_user_service),
):
    """Reset password with token."""
    try:
        success = await user_service.reset_password(
            token=request.token,
            new_password=request.new_password,
        )

        if success:
            return MessageResponse(message="Password reset successfully")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to reset password",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
