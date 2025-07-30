"""API key validation infrastructure.

This module provides API key validation as an infrastructure service,
separating security concerns from business logic.
"""

import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.user import User
from src.infrastructure.persistence.models.organization import Organization
from src.infrastructure.cache import get_redis_cache, RedisCache
from .auth_utils import verify_api_key

logger = logging.getLogger(__name__)


@dataclass
class APIKeyValidationResult:
    """Result of API key validation."""
    is_valid: bool
    api_key: Optional[APIKey] = None
    user: Optional[User] = None
    organization: Optional[Organization] = None
    error_message: Optional[str] = None


class APIKeyValidator:
    """Service for validating API keys against the database.
    
    This class provides infrastructure-level API key validation,
    caching results for performance while maintaining security.
    """
    
    def __init__(self, cache: Optional[RedisCache] = None):
        """Initialize validator with optional cache."""
        self._cache = cache

    @property
    async def cache(self) -> RedisCache:
        """Get cache instance, creating if necessary."""
        if self._cache is None:
            self._cache = await get_redis_cache()
        return self._cache

    async def validate_api_key(
        self, 
        api_key: str, 
        db: AsyncSession
    ) -> APIKeyValidationResult:
        """Validate an API key against the database.
        
        Args:
            api_key: The API key to validate
            db: Database session
            
        Returns:
            APIKeyValidationResult with validation status and details
        """
        try:
            # First try to get from cache
            cache_key = f"api_key:{api_key[:16]}"  # Use prefix for security
            cache = await self.cache
            cached_result = await cache.get(cache_key)

            if cached_result:
                # Get full API key from database using cached ID
                api_key_obj = await self._get_api_key_by_id(
                    cached_result["id"], 
                    db
                )
                
                if api_key_obj and self._is_api_key_valid(api_key_obj, api_key):
                    await self._update_last_used(api_key_obj, db)
                    return APIKeyValidationResult(
                        is_valid=True,
                        api_key=api_key_obj,
                        user=api_key_obj.user,
                        organization=self._get_user_organization(api_key_obj.user)
                    )

            # Cache miss or invalid cached data - query database
            api_key_obj = await self._find_valid_api_key(api_key, db)
            
            if api_key_obj:
                # Cache the result
                await cache.set(
                    cache_key,
                    {"id": api_key_obj.id},
                    ttl=300,  # 5 minutes
                )
                await self._update_last_used(api_key_obj, db)
                
                return APIKeyValidationResult(
                    is_valid=True,
                    api_key=api_key_obj,
                    user=api_key_obj.user,
                    organization=self._get_user_organization(api_key_obj.user)
                )
            
            return APIKeyValidationResult(
                is_valid=False,
                error_message="Invalid or expired API key"
            )

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return APIKeyValidationResult(
                is_valid=False,
                error_message="API key validation error"
            )

    async def _get_api_key_by_id(
        self, 
        api_key_id: int, 
        db: AsyncSession
    ) -> Optional[APIKey]:
        """Get API key by ID with user and organization data."""
        stmt = (
            select(APIKey)
            .options(
                selectinload(APIKey.user).selectinload(User.owned_organizations)
            )
            .where(APIKey.id == api_key_id)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def _find_valid_api_key(
        self, 
        api_key: str, 
        db: AsyncSession
    ) -> Optional[APIKey]:
        """Find valid API key in database."""
        stmt = (
            select(APIKey)
            .options(
                selectinload(APIKey.user).selectinload(User.owned_organizations)
            )
            .where(APIKey.active.is_(True))
        )
        result = await db.execute(stmt)
        api_keys = result.scalars().all()

        # Check each active API key
        for api_key_obj in api_keys:
            if self._is_api_key_valid(api_key_obj, api_key):
                return api_key_obj
                
        return None

    def _is_api_key_valid(self, api_key_obj: APIKey, provided_key: str) -> bool:
        """Check if provided key matches the stored key and is valid."""
        # Check if key is active
        if not api_key_obj.active:
            return False

        # Check if key has expired
        if api_key_obj.is_expired():
            return False

        # Verify the key
        return verify_api_key(provided_key, api_key_obj.key)

    def _get_user_organization(self, user: User) -> Optional[Organization]:
        """Get the first organization owned by the user."""
        if user.owned_organizations:
            return user.owned_organizations[0]
        return None

    async def _update_last_used(self, api_key: APIKey, db: AsyncSession) -> None:
        """Update the last used timestamp for an API key."""
        try:
            api_key.last_used_at = datetime.now(timezone.utc)
            await db.commit()
        except Exception as e:
            logger.error(f"Error updating API key last used: {e}")
            await db.rollback()

    async def has_permission(self, api_key: APIKey, required_scope: str) -> bool:
        """Check if an API key has the required permission scope.
        
        Args:
            api_key: The API key object
            required_scope: The required permission scope
            
        Returns:
            True if the key has the required scope
        """
        # Admin scope grants all permissions
        if "admin" in api_key.scopes:
            return True
            
        return required_scope in api_key.scopes

    async def get_rate_limit(self, api_key: APIKey) -> int:
        """Get the rate limit for an API key based on the user's plan.
        
        Args:
            api_key: The API key object
            
        Returns:
            Rate limit per minute
        """
        try:
            # Get user's organization
            if api_key.user.owned_organizations:
                org = api_key.user.owned_organizations[0]
                return org.plan_limits.get("api_rate_limit_per_minute", 60)
                
            # Default rate limit if no organization
            return 60
            
        except Exception as e:
            logger.error(f"Error getting rate limit for API key: {e}")
            return 60  # Default fallback

    async def validate_scopes(self, scopes: List[str]) -> bool:
        """Validate that provided scopes are valid.
        
        Args:
            scopes: List of scopes to validate
            
        Returns:
            True if all scopes are valid
        """
        valid_scopes = {
            "read",      # Read access to resources
            "write",     # Create and update resources
            "delete",    # Delete resources
            "streams",   # Stream processing
            "batches",   # Batch processing
            "webhooks",  # Webhook management
            "analytics", # Analytics access
            "admin",     # Full administrative access
        }
        
        return all(scope in valid_scopes for scope in scopes)