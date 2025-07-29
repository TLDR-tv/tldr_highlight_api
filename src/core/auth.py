"""Authentication dependencies for FastAPI.

This module provides FastAPI dependencies for:
- API key authentication
- Organization context extraction
- Permission requirement decorators
- Current user context management
- Rate limiting enforcement
"""

import logging
from typing import List, Optional, Union

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import get_db
from src.core.cache import cache
from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.organization import Organization
from src.infrastructure.persistence.models.user import User
from src.services.auth import AuthService
from src.services.rate_limit import RateLimitService

logger = logging.getLogger(__name__)


# Custom exceptions
class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class InsufficientPermissions(Exception):
    """Raised when user lacks required permissions."""

    pass


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, reset_time: Optional[int] = None):
        super().__init__(message)
        self.reset_time = reset_time


# Service instances
auth_service = AuthService(cache)
rate_limit_service = RateLimitService(cache)


def get_api_key_from_header(request: Request) -> str:
    """Extract API key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        str: The API key

    Raises:
        HTTPException: If API key is missing or invalid format
    """
    api_key = request.headers.get(settings.api_key_header)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Include it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    api_key = api_key.strip()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key cannot be empty.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


async def get_current_api_key(
    api_key: str = Depends(get_api_key_from_header),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(lambda: auth_service),
) -> APIKey:
    """Get and validate the current API key.

    Args:
        api_key: API key from header
        db: Database session
        auth_service: Authentication service

    Returns:
        APIKey: Validated API key object

    Raises:
        HTTPException: If API key is invalid or expired
    """
    try:
        validated_key = await auth_service.validate_api_key(api_key, db)

        if not validated_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return validated_key

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error.",
        )


async def get_current_user(api_key: APIKey = Depends(get_current_api_key)) -> User:
    """Get the current user from the API key.

    Args:
        api_key: Validated API key

    Returns:
        User: The user associated with the API key

    Raises:
        HTTPException: If user is not found
    """
    if not api_key.user:
        logger.error(f"API key {api_key.id} has no associated user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User data not found.",
        )

    return api_key.user


async def get_current_organization(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(lambda: auth_service),
) -> Organization:
    """Get the current user's organization.

    Args:
        user: Current user
        db: Database session
        auth_service: Authentication service

    Returns:
        Organization: The user's organization

    Raises:
        HTTPException: If organization is not found
    """
    try:
        organization = await auth_service.get_user_organization(user.id, db)

        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found. Please contact support.",
            )

        return organization

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving organization data.",
        )


def require_scope(required_scopes: Union[str, List[str]]):
    """Create a dependency that requires specific scopes.

    Args:
        required_scopes: Required scope(s) for the endpoint

    Returns:
        Dependency function that checks permissions
    """
    # Normalize to list
    if isinstance(required_scopes, str):
        required_scopes = [required_scopes]

    async def check_permissions(
        api_key: APIKey = Depends(get_current_api_key),
        auth_service: AuthService = Depends(lambda: auth_service),
    ) -> APIKey:
        """Check if the API key has required permissions.

        Args:
            api_key: Validated API key
            auth_service: Authentication service

        Returns:
            APIKey: The API key if permissions are valid

        Raises:
            HTTPException: If permissions are insufficient
        """
        try:
            # Check each required scope
            for scope in required_scopes:
                if not await auth_service.has_permission(api_key, scope):
                    logger.warning(
                        f"API key {api_key.id} denied access to scope '{scope}'. "
                        f"Available scopes: {api_key.scopes}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Required scope: {scope}",
                    )

            return api_key

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Permission check error.",
            )

    return check_permissions


async def check_rate_limit(
    api_key: APIKey = Depends(get_current_api_key),
    rate_limit_service: RateLimitService = Depends(lambda: rate_limit_service),
) -> APIKey:
    """Check rate limit for the current API key.

    Args:
        api_key: Validated API key
        rate_limit_service: Rate limiting service

    Returns:
        APIKey: The API key if rate limit is not exceeded

    Raises:
        HTTPException: If rate limit is exceeded
    """
    try:
        is_allowed, remaining = await rate_limit_service.check_rate_limit(api_key)

        if not is_allowed:
            # Get detailed rate limit info for headers
            rate_info = await rate_limit_service.get_rate_limit_info(api_key)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please slow down your requests.",
                headers={
                    "X-RateLimit-Limit": str(rate_info["max_requests_per_minute"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info["reset_time"]),
                    "Retry-After": "60",
                },
            )

        return api_key

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking rate limit: {e}")
        # Fail open on rate limit service errors
        return api_key


async def check_burst_limit(
    api_key: APIKey = Depends(get_current_api_key),
    rate_limit_service: RateLimitService = Depends(lambda: rate_limit_service),
) -> APIKey:
    """Check burst rate limit for the current API key.

    Args:
        api_key: Validated API key
        rate_limit_service: Rate limiting service

    Returns:
        APIKey: The API key if burst limit is not exceeded

    Raises:
        HTTPException: If burst limit is exceeded
    """
    try:
        is_allowed, remaining = await rate_limit_service.check_burst_limit(api_key)

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Burst rate limit exceeded. Please reduce request frequency.",
                headers={
                    "X-RateLimit-Burst-Limit": "20",
                    "X-RateLimit-Burst-Remaining": "0",
                    "Retry-After": "10",
                },
            )

        return api_key

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking burst limit: {e}")
        # Fail open on rate limit service errors
        return api_key


# Convenience dependencies combining multiple checks
async def authenticated_user(
    api_key: APIKey = Depends(get_current_api_key),
    _: APIKey = Depends(check_rate_limit),
) -> User:
    """Get authenticated user with rate limiting.

    Combines API key validation, rate limiting, and user extraction.

    Returns:
        User: The authenticated user
    """
    return api_key.user


async def authenticated_organization(
    user: User = Depends(authenticated_user),
    organization: Organization = Depends(get_current_organization),
) -> Organization:
    """Get authenticated user's organization with rate limiting.

    Returns:
        Organization: The authenticated user's organization
    """
    return organization


def require_authenticated_scope(required_scopes: Union[str, List[str]]):
    """Create a dependency combining authentication, rate limiting, and scope checking.

    Args:
        required_scopes: Required scope(s) for the endpoint

    Returns:
        Dependency function with full authentication stack
    """
    scope_check = require_scope(required_scopes)

    async def full_auth_check(
        api_key: APIKey = Depends(scope_check), _: APIKey = Depends(check_rate_limit)
    ) -> APIKey:
        """Full authentication check with scope and rate limiting.

        Returns:
            APIKey: Validated API key with required permissions
        """
        return api_key

    return full_auth_check


# Optional authentication (for public endpoints with optional features)
async def optional_authentication(
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(lambda: auth_service),
) -> Optional[APIKey]:
    """Optional authentication for endpoints that work with or without auth.

    Args:
        request: FastAPI request object
        db: Database session
        auth_service: Authentication service

    Returns:
        Optional[APIKey]: API key if provided and valid, None otherwise
    """
    try:
        api_key = request.headers.get(settings.api_key_header)
        if not api_key or not api_key.strip():
            return None

        validated_key = await auth_service.validate_api_key(api_key.strip(), db)
        return validated_key

    except Exception as e:
        logger.debug(f"Optional authentication failed: {e}")
        return None


# Admin-only dependencies
admin_required = require_authenticated_scope("admin")
read_required = require_authenticated_scope("read")
write_required = require_authenticated_scope("write")
streams_required = require_authenticated_scope("streams")
batches_required = require_authenticated_scope("batches")
webhooks_required = require_authenticated_scope("webhooks")
analytics_required = require_authenticated_scope("analytics")
