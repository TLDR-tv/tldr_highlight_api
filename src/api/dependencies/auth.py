"""Authentication dependencies for FastAPI.

This module provides FastAPI-specific authentication dependencies,
keeping API concerns separate from infrastructure and business logic.
"""

import logging
from typing import Optional, Union, List, Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.config import settings
from src.infrastructure.database import get_db
from src.infrastructure.security import APIKeyValidator
from src.infrastructure.cache import get_rate_limiter, RateLimiter
from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.user import User
from src.infrastructure.persistence.models.organization import Organization

logger = logging.getLogger(__name__)


def get_api_key_from_header(request: Request) -> str:
    """Extract API key from request headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        The API key string
        
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
) -> APIKey:
    """Get and validate the current API key.
    
    Args:
        api_key: API key from header
        db: Database session
        
    Returns:
        Validated API key object
        
    Raises:
        HTTPException: If API key is invalid or expired
    """
    try:
        validator = APIKeyValidator()
        result = await validator.validate_api_key(api_key, db)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.error_message or "Invalid or expired API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        return result.api_key
        
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
        The user associated with the API key
        
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
) -> Organization:
    """Get the current user's organization.
    
    Args:
        user: Current user
        
    Returns:
        The user's organization
        
    Raises:
        HTTPException: If organization is not found
    """
    if not user.owned_organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found. Please contact support.",
        )
    
    return user.owned_organizations[0]


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
    ) -> APIKey:
        """Check if the API key has required permissions.
        
        Args:
            api_key: Validated API key
            
        Returns:
            The API key if permissions are valid
            
        Raises:
            HTTPException: If permissions are insufficient
        """
        try:
            validator = APIKeyValidator()
            
            # Check each required scope
            for scope in required_scopes:
                if not await validator.has_permission(api_key, scope):
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
) -> APIKey:
    """Check rate limit for the current API key.
    
    Args:
        api_key: Validated API key
        
    Returns:
        The API key if rate limit is not exceeded
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    try:
        rate_limiter = await get_rate_limiter()
        validator = APIKeyValidator()
        
        # Get rate limit for this API key
        max_requests = await validator.get_rate_limit(api_key)
        
        # Check rate limit
        result = await rate_limiter.is_allowed(
            key=f"api_key:{api_key.id}",
            max_requests=max_requests,
            window_seconds=60  # Per minute
        )
        
        if not result.is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please slow down your requests.",
                headers={
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(result.reset_time),
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
) -> APIKey:
    """Check burst rate limit for the current API key.
    
    Args:
        api_key: Validated API key
        
    Returns:
        The API key if burst limit is not exceeded
        
    Raises:
        HTTPException: If burst limit is exceeded
    """
    try:
        rate_limiter = await get_rate_limiter()
        
        # Check burst limit (20 requests per 10 seconds)
        result = await rate_limiter.is_allowed(
            key=f"burst:{api_key.id}",
            max_requests=20,
            window_seconds=10
        )
        
        if not result.is_allowed:
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
        The authenticated user
    """
    return api_key.user


async def authenticated_organization(
    user: User = Depends(authenticated_user),
    organization: Organization = Depends(get_current_organization),
) -> Organization:
    """Get authenticated user's organization with rate limiting.
    
    Returns:
        The authenticated user's organization
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
        api_key: APIKey = Depends(scope_check),
        _: APIKey = Depends(check_rate_limit)
    ) -> APIKey:
        """Full authentication check with scope and rate limiting.
        
        Returns:
            Validated API key with required permissions
        """
        return api_key
    
    return full_auth_check


# Optional authentication (for public endpoints with optional features)
async def optional_authentication(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[APIKey]:
    """Optional authentication for endpoints that work with or without auth.
    
    Args:
        request: FastAPI request object
        db: Database session
        
    Returns:
        API key if provided and valid, None otherwise
    """
    try:
        api_key = request.headers.get(settings.api_key_header)
        if not api_key or not api_key.strip():
            return None
        
        validator = APIKeyValidator()
        result = await validator.validate_api_key(api_key.strip(), db)
        
        return result.api_key if result.is_valid else None
        
    except Exception as e:
        logger.debug(f"Optional authentication failed: {e}")
        return None


# Type aliases for cleaner dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalCurrentUser = Annotated[Optional[User], Depends(optional_authentication)]

# Pre-defined scope requirements for common use cases
admin_required = require_authenticated_scope("admin")
read_required = require_authenticated_scope("read")
write_required = require_authenticated_scope("write")
streams_required = require_authenticated_scope("streams")
batches_required = require_authenticated_scope("batches")
webhooks_required = require_authenticated_scope("webhooks")
analytics_required = require_authenticated_scope("analytics")