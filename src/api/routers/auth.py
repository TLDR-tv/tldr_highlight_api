"""Authentication router for the TL;DR Highlight API.

This module provides endpoints for API key management and authentication.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from src.api.schemas.common import COMMON_RESPONSES, StatusResponse
from src.api.schemas.auth import (
    APIKeyCreate,
    APIKeyCreateResponse,
    APIKeyListResponse,
    LoginRequest as LoginRequestDTO,
    LoginResponse,
)
from src.api.schemas.users import UserRegistrationRequest, UserRegistrationResponse
from src.api.mappers.auth_mapper import RegisterMapper, LoginMapper, APIKeyMapper
from src.api.dependencies import (
    get_user_registration_use_case,
    get_user_login_use_case,
    get_api_key_management_use_case,
    CurrentUser,
)
from src.infrastructure.security import create_access_token
from src.application.use_cases.user_registration import UserRegistrationUseCase
from src.application.use_cases.user_login import UserLoginUseCase
from src.application.use_cases.api_key_management import APIKeyManagementUseCase
from src.domain.exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    ValidationError,
    AuthenticationError,
)

router = APIRouter()


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Authentication service status",
    description="Check if authentication service is operational",
    responses=COMMON_RESPONSES,
)
async def auth_status() -> StatusResponse:
    """Get authentication service status.

    Returns:
        StatusResponse: Authentication service status
    """
    return StatusResponse(
        status="Authentication service operational",
        timestamp=datetime.now(timezone.utc),
    )


@router.post(
    "/register",
    response_model=UserRegistrationResponse,
    summary="Register new user",
    description="Register a new user and organization with initial API key",
    responses=COMMON_RESPONSES,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    request: UserRegistrationRequest,
    registration_use_case: UserRegistrationUseCase = Depends(
        get_user_registration_use_case
    ),
) -> UserRegistrationResponse:
    """Register a new user.

    Creates a new user account, organization, and initial API key.
    """
    try:
        # Convert DTO to domain request
        mapper = RegisterMapper()
        domain_request = mapper.to_domain(request)

        # Execute use case
        result = await registration_use_case.execute(domain_request)

        if not result.is_success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.errors[0] if result.errors else "Registration failed",
            )

        # Ensure registration succeeded with valid user ID
        if result.user_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed - invalid user ID",
            )

        # Create access token for immediate login
        access_token = create_access_token(
            user_id=result.user_id,
            email=request.email,
            scopes=["streams:read", "streams:write", "highlights:read"],
        )

        return UserRegistrationResponse(
            id=result.user_id,
            email=request.email,
            company_name=request.company_name,
            created_at=datetime.now(timezone.utc),
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,
        )

    except DuplicateEntityError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="User login",
    description="Authenticate user with email and password",
    responses=COMMON_RESPONSES,
)
async def login(
    request: LoginRequestDTO,
    login_use_case: UserLoginUseCase = Depends(get_user_login_use_case),
) -> LoginResponse:
    """Authenticate user and return access token."""
    try:
        # Convert DTO to domain request
        login_mapper = LoginMapper()
        domain_request = login_mapper.to_domain(request)

        # Execute use case
        result = await login_use_case.execute(domain_request)

        if not result.is_success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.errors[0] if result.errors else "Invalid credentials",
            )

        # Ensure login succeeded with valid user ID
        if result.user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Login failed - invalid user ID",
            )

        # Create access token
        access_token = create_access_token(
            user_id=result.user_id,
            email=request.email,
            scopes=["streams:read", "streams:write", "highlights:read"],
        )

        return login_mapper.to_dto(result, access_token)

    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post(
    "/api-keys",
    response_model=APIKeyCreateResponse,
    summary="Create API key",
    description="Create a new API key for the authenticated user",
    responses=COMMON_RESPONSES,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    request: APIKeyCreate,
    current_user: CurrentUser,
    api_key_use_case: APIKeyManagementUseCase = Depends(
        get_api_key_management_use_case
    ),
) -> APIKeyCreateResponse:
    """Create a new API key.

    The key is shown only once in the response and must be saved by the user.
    """
    try:
        # Convert scopes to domain format
        api_key_mapper = APIKeyMapper()
        domain_scopes = api_key_mapper.to_domain_scopes(request.scopes)

        # Create API key
        from src.application.use_cases.api_key_management import CreateAPIKeyRequest

        create_request = CreateAPIKeyRequest(
            user_id=current_user.id,
            name=request.name,
            scopes=domain_scopes,
            expires_at=request.expires_at,
        )
        result = await api_key_use_case.create_api_key(create_request)

        if not result.is_success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.errors[0]
                if result.errors
                else "Failed to create API key",
            )

        # Ensure API key was created successfully
        if result.api_key is None or result.key is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create API key - invalid response",
            )

        # Get the created API key and convert to response
        api_key = result.api_key
        return api_key_mapper.to_create_response_dto(api_key, result.key)

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )


@router.get(
    "/api-keys",
    response_model=APIKeyListResponse,
    summary="List API keys",
    description="List all API keys for the authenticated user",
    responses=COMMON_RESPONSES,
)
async def list_api_keys(
    current_user: CurrentUser,
    api_key_use_case: APIKeyManagementUseCase = Depends(
        get_api_key_management_use_case
    ),
) -> APIKeyListResponse:
    """List API keys for the authenticated user."""
    # Get API keys from use case
    result = await api_key_use_case.list_api_keys(current_user.id)

    if not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to list API keys",
        )

    # Convert to DTOs
    api_key_mapper = APIKeyMapper()
    key_dtos = [api_key_mapper.to_dto(key) for key in result.api_keys]

    return APIKeyListResponse(total=len(key_dtos), keys=key_dtos)


@router.delete(
    "/api-keys/{key_id}",
    summary="Revoke API key",
    description="Revoke an API key by ID",
    responses=COMMON_RESPONSES,
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_api_key(
    key_id: int,
    current_user: CurrentUser,
    api_key_use_case: APIKeyManagementUseCase = Depends(
        get_api_key_management_use_case
    ),
) -> None:
    """Revoke an API key.

    The key will be deactivated and can no longer be used.
    """
    try:
        result = await api_key_use_case.revoke_api_key(
            user_id=current_user.id, api_key_id=key_id
        )

        if not result.is_success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.errors[0]
                if result.errors
                else "Failed to revoke API key",
            )

    except EntityNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )


@router.post(
    "/api-keys/{key_id}/rotate",
    response_model=APIKeyCreateResponse,
    summary="Rotate API key",
    description="Rotate an API key to generate a new key value",
    responses=COMMON_RESPONSES,
)
async def rotate_api_key(
    key_id: int,
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
    api_key_use_case: APIKeyManagementUseCase = Depends(
        get_api_key_management_use_case
    ),
) -> APIKeyCreateResponse:
    """Rotate an API key.

    Generates a new key value while keeping the same permissions.
    The old key remains valid for a grace period.
    """
    try:
        result = await api_key_use_case.rotate_api_key(
            user_id=current_user.id, api_key_id=key_id
        )

        if not result.is_success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.errors[0]
                if result.errors
                else "Failed to rotate API key",
            )

        # Schedule old key revocation after grace period
        # This would be handled by a background task in production

        # Ensure API key was rotated successfully
        if result.api_key is None or result.key is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to rotate API key - invalid response",
            )

        # Get the rotated API key and convert to response
        api_key = result.api_key
        api_key_mapper = APIKeyMapper()
        return api_key_mapper.to_create_response_dto(api_key, result.key)

    except EntityNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )
