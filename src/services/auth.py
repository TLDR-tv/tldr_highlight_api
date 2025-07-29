"""Authentication service for the TL;DR Highlight API.

This module provides the core authentication service that handles:
- API key validation and verification
- Permission/scope checking
- Organization-based multi-tenancy validation
- Rate limiting tracking per API key
- JWT token generation and validation for sessions
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.cache import RedisCache
from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.organization import Organization
from src.infrastructure.persistence.models.user import User
from src.utils.auth import generate_api_key, hash_api_key, verify_api_key

logger = logging.getLogger(__name__)


class AuthService:
    """Service for handling authentication and authorization."""

    def __init__(self, cache: RedisCache):
        """Initialize authentication service.

        Args:
            cache: Redis cache instance for caching auth data
        """
        self.cache = cache

    async def validate_api_key(
        self, api_key: str, db: AsyncSession
    ) -> Optional[APIKey]:
        """Validate an API key against the database.

        Args:
            api_key: The API key to validate
            db: Database session

        Returns:
            APIKey: The validated API key object if valid, None otherwise
        """
        try:
            # First try to get from cache
            cache_key = f"api_key:{api_key[:16]}"  # Use prefix for security
            cached_result = await self.cache.get(cache_key)

            if cached_result:
                # Get full API key from database using cached ID
                stmt = (
                    select(APIKey)
                    .options(
                        selectinload(APIKey.user).selectinload(User.owned_organizations)
                    )
                    .where(APIKey.id == cached_result["id"])
                )
                result = await db.execute(stmt)
                api_key_obj = result.scalar_one_or_none()

                if api_key_obj and self._is_api_key_valid(api_key_obj, api_key):
                    await self.update_last_used(api_key_obj, db)
                    return api_key_obj

            # Cache miss or invalid cached data - query database
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
                    # Cache the result
                    await self.cache.set(
                        cache_key,
                        {"id": api_key_obj.id},
                        ttl=300,  # 5 minutes
                    )
                    await self.update_last_used(api_key_obj, db)
                    return api_key_obj

            return None

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    def _is_api_key_valid(self, api_key_obj: APIKey, provided_key: str) -> bool:
        """Check if provided key matches the stored key and is valid.

        Args:
            api_key_obj: API key object from database
            provided_key: API key provided in request

        Returns:
            bool: True if valid, False otherwise
        """
        # Check if key is active
        if not api_key_obj.active:
            return False

        # Check if key has expired
        if api_key_obj.is_expired():
            return False

        # Verify the key
        return verify_api_key(provided_key, api_key_obj.key)

    async def has_permission(self, api_key: APIKey, required_scope: str) -> bool:
        """Check if an API key has the required permission scope.

        Args:
            api_key: The API key object
            required_scope: The required permission scope

        Returns:
            bool: True if the key has the required scope, False otherwise
        """
        # Admin scope grants all permissions
        if "admin" in api_key.scopes:
            return True

        return required_scope in api_key.scopes

    async def get_user_organization(
        self, user_id: int, db: AsyncSession
    ) -> Optional[Organization]:
        """Get the organization for a user.

        Args:
            user_id: The user ID
            db: Database session

        Returns:
            Organization: The user's organization if found, None otherwise
        """
        try:
            stmt = select(Organization).where(Organization.owner_id == user_id)
            result = await db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user organization: {e}")
            return None

    async def update_last_used(self, api_key: APIKey, db: AsyncSession) -> None:
        """Update the last used timestamp for an API key.

        Args:
            api_key: The API key to update
            db: Database session
        """
        try:
            api_key.last_used_at = datetime.now(timezone.utc)
            await db.commit()
        except Exception as e:
            logger.error(f"Error updating API key last used: {e}")
            await db.rollback()

    async def create_api_key(
        self,
        user_id: int,
        name: str,
        scopes: List[str],
        db: AsyncSession,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Create a new API key for a user.

        Args:
            user_id: The user ID
            name: Human-readable name for the key
            scopes: List of permission scopes
            db: Database session
            expires_at: Optional expiration timestamp

        Returns:
            dict: Dictionary containing the new API key details
        """
        try:
            # Generate new API key
            raw_key = generate_api_key(32)
            hashed_key = hash_api_key(raw_key)

            # Create API key object
            api_key = APIKey(
                key=hashed_key,
                name=name,
                user_id=user_id,
                scopes=scopes,
                active=True,
                expires_at=expires_at,
            )

            db.add(api_key)
            await db.commit()
            await db.refresh(api_key)

            # Return key details (including raw key for one-time display)
            return {
                "id": api_key.id,
                "key": raw_key,  # Only returned once
                "masked_key": f"{raw_key[:8]}...{raw_key[-4:]}",
                "name": name,
                "scopes": scopes,
                "created_at": api_key.created_at,
                "expires_at": api_key.expires_at,
            }

        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            await db.rollback()
            raise

    async def revoke_api_key(
        self, api_key_id: int, user_id: int, db: AsyncSession
    ) -> bool:
        """Revoke an API key.

        Args:
            api_key_id: The API key ID to revoke
            user_id: The user ID (for authorization)
            db: Database session

        Returns:
            bool: True if successfully revoked, False otherwise
        """
        try:
            stmt = select(APIKey).where(
                APIKey.id == api_key_id, APIKey.user_id == user_id
            )
            result = await db.execute(stmt)
            api_key = result.scalar_one_or_none()

            if not api_key:
                return False

            api_key.active = False
            await db.commit()

            # Invalidate cache
            cache_key = "api_key:*"  # Could be more specific
            await self.cache.delete(cache_key)

            return True

        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            await db.rollback()
            return False

    async def list_user_api_keys(
        self, user_id: int, db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """List all API keys for a user.

        Args:
            user_id: The user ID
            db: Database session

        Returns:
            list: List of API key dictionaries (without raw keys)
        """
        try:
            stmt = select(APIKey).where(APIKey.user_id == user_id)
            result = await db.execute(stmt)
            api_keys = result.scalars().all()

            return [
                {
                    "id": api_key.id,
                    "name": api_key.name,
                    "masked_key": f"...{api_key.key[-8:]}",  # Show last 8 chars of hash
                    "scopes": api_key.scopes,
                    "active": api_key.active,
                    "created_at": api_key.created_at,
                    "expires_at": api_key.expires_at,
                    "last_used_at": api_key.last_used_at,
                }
                for api_key in api_keys
            ]

        except Exception as e:
            logger.error(f"Error listing user API keys: {e}")
            return []

    async def rate_limit_for_key(self, api_key: APIKey) -> int:
        """Get the rate limit for an API key based on the user's organization plan.

        Args:
            api_key: The API key object

        Returns:
            int: Rate limit per minute
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
            bool: True if all scopes are valid, False otherwise
        """
        valid_scopes = {
            "read",  # Read access to resources
            "write",  # Create and update resources
            "delete",  # Delete resources
            "streams",  # Stream processing
            "batches",  # Batch processing
            "webhooks",  # Webhook management
            "analytics",  # Analytics access
            "admin",  # Full administrative access
        }

        return all(scope in valid_scopes for scope in scopes)

