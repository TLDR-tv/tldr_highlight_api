"""Domain-driven authentication service.

This service handles authentication and authorization using domain entities
and repository interfaces, following clean architecture principles.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from secrets import token_urlsafe

from src.core.cache import RedisCache
from src.domain.entities.api_key import APIKey
from src.domain.entities.user import User
from src.domain.entities.organization import Organization
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.value_objects.timestamp import Timestamp
from src.domain.exceptions import EntityNotFoundError, DuplicateEntityError
from src.utils.auth import hash_api_key, verify_api_key

logger = logging.getLogger(__name__)


class AuthService:
    """Domain-driven authentication service.
    
    This service handles authentication and authorization using domain entities
    and follows dependency injection principles for better testability.
    """
    
    def __init__(
        self,
        api_key_repo: APIKeyRepository,
        user_repo: UserRepository,
        org_repo: OrganizationRepository,
        cache: RedisCache
    ):
        """Initialize authentication service.
        
        Args:
            api_key_repo: Repository for API key operations
            user_repo: Repository for user operations  
            org_repo: Repository for organization operations
            cache: Redis cache instance for caching auth data
        """
        self.api_key_repo = api_key_repo
        self.user_repo = user_repo
        self.org_repo = org_repo
        self.cache = cache
    
    async def validate_api_key(self, api_key_str: str) -> Optional[APIKey]:
        """Validate an API key.
        
        Args:
            api_key_str: The raw API key string
            
        Returns:
            APIKey domain entity if valid, None otherwise
        """
        try:
            # First try to get from cache
            cache_key = f"api_key:{api_key_str[:16]}"  # Use prefix for security
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                # Get API key by ID from cache
                api_key = await self.api_key_repo.get(cached_result["id"])
                if api_key and self._is_api_key_valid(api_key, api_key_str):
                    await self._update_last_used(api_key)
                    return api_key
            
            # Cache miss - try direct lookup by hash
            key_hash = hash_api_key(api_key_str)
            api_key = await self.api_key_repo.get_by_key_hash(key_hash)
            
            if api_key and self._is_api_key_valid(api_key, api_key_str):
                # Cache the result
                await self.cache.set(
                    cache_key,
                    {"id": api_key.id},
                    ttl=300  # 5 minutes
                )
                await self._update_last_used(api_key)
                return api_key
            
            return None
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None
    
    def _is_api_key_valid(self, api_key: APIKey, provided_key: str) -> bool:
        """Check if provided key matches the stored key and is valid.
        
        Args:
            api_key: API key domain entity
            provided_key: Raw API key provided in request
            
        Returns:
            True if valid, False otherwise
        """
        # Check if key is valid (active and not expired)
        if not api_key.is_valid:
            return False
        
        # Verify the key hash
        return verify_api_key(provided_key, api_key.key_hash)
    
    async def _update_last_used(self, api_key: APIKey) -> None:
        """Update the last used timestamp for an API key.
        
        Args:
            api_key: The API key to update
        """
        try:
            updated_key = api_key.record_usage()
            await self.api_key_repo.save(updated_key)
        except Exception as e:
            logger.error(f"Error updating API key last used: {e}")
    
    async def has_permission(self, api_key: APIKey, required_scope: str) -> bool:
        """Check if an API key has the required permission scope.
        
        Args:
            api_key: The API key domain entity
            required_scope: The required permission scope
            
        Returns:
            True if the key has the required scope, False otherwise
        """
        return api_key.has_scope(required_scope)
    
    async def get_user_organization(self, user_id: int) -> Optional[Organization]:
        """Get the organization for a user.
        
        Args:
            user_id: The user ID
            
        Returns:
            Organization domain entity if found, None otherwise
        """
        try:
            organizations = await self.org_repo.get_by_owner(user_id)
            return organizations[0] if organizations else None
        except Exception as e:
            logger.error(f"Error getting user organization: {e}")
            return None
    
    async def create_api_key(
        self,
        user_id: int,
        name: str,
        scopes: List[str],
        description: Optional[str] = None,
        expires_at: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Create a new API key for a user.
        
        Args:
            user_id: The user ID
            name: Human-readable name for the key
            scopes: List of permission scopes
            description: Optional description
            expires_at: Optional expiration timestamp
            
        Returns:
            Dictionary containing the new API key details
            
        Raises:
            EntityNotFoundError: If user doesn't exist
            ValueError: If scopes are invalid
        """
        try:
            # Validate user exists
            user = await self.user_repo.get(user_id)
            if not user:
                raise EntityNotFoundError(f"User {user_id} not found")
            
            # Validate scopes
            if not self._validate_scopes(scopes):
                raise ValueError("Invalid scopes provided")
            
            # Generate new API key
            raw_key = APIKey.generate_key()
            key_hash = hash_api_key(raw_key)
            
            # Create API key domain entity
            now = Timestamp.now()
            api_key = APIKey(
                id=None,
                name=name,
                key_hash=key_hash,
                user_id=user_id,
                scopes=scopes,
                description=description,
                expires_at=expires_at,
                last_used_at=None,
                rate_limit_override=None,
                allowed_ips=[],
                is_active=True,
                created_at=now,
                updated_at=now
            )
            
            # Save to repository
            saved_key = await self.api_key_repo.save(api_key)
            
            # Return key details (including raw key for one-time display)
            return {
                "id": saved_key.id,
                "key": raw_key,  # Only returned once
                "masked_key": f"{raw_key[:8]}...{raw_key[-4:]}",
                "name": name,
                "scopes": scopes,
                "description": description,
                "created_at": saved_key.created_at.iso_string,
                "expires_at": saved_key.expires_at.iso_string if saved_key.expires_at else None,
            }
            
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise
    
    async def revoke_api_key(self, api_key_id: int, user_id: int) -> bool:
        """Revoke an API key.
        
        Args:
            api_key_id: The API key ID to revoke
            user_id: The user ID (for authorization)
            
        Returns:
            True if successfully revoked, False otherwise
        """
        try:
            api_key = await self.api_key_repo.get(api_key_id)
            if not api_key or api_key.user_id != user_id:
                return False
            
            # Deactivate the key
            deactivated_key = api_key.deactivate()
            await self.api_key_repo.save(deactivated_key)
            
            # Invalidate cache
            cache_pattern = "api_key:*"
            await self.cache.delete(cache_pattern)
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False
    
    async def list_user_api_keys(self, user_id: int) -> List[Dict[str, Any]]:
        """List all API keys for a user.
        
        Args:
            user_id: The user ID
            
        Returns:
            List of API key dictionaries (without raw keys)
        """
        try:
            api_keys = await self.api_key_repo.get_by_user(user_id, active_only=False)
            
            return [
                {
                    "id": key.id,
                    "name": key.name,
                    "masked_key": f"...{key.key_hash[-8:]}",  # Show last 8 chars of hash
                    "scopes": key.scopes,
                    "description": key.description,
                    "is_active": key.is_active,
                    "created_at": key.created_at.iso_string,
                    "expires_at": key.expires_at.iso_string if key.expires_at else None,
                    "last_used_at": key.last_used_at.iso_string if key.last_used_at else None,
                }
                for key in api_keys
            ]
            
        except Exception as e:
            logger.error(f"Error listing user API keys: {e}")
            return []
    
    async def get_rate_limit_for_key(self, api_key: APIKey) -> int:
        """Get the rate limit for an API key based on the user's organization plan.
        
        Args:
            api_key: The API key domain entity
            
        Returns:
            Rate limit per minute
        """
        try:
            # Check for override first
            if api_key.rate_limit_override:
                return api_key.rate_limit_override
            
            # Get user's organization
            organization = await self.get_user_organization(api_key.user_id)
            if organization:
                return organization.plan_limits.api_rate_limit_per_minute
            
            # Default rate limit if no organization
            return 60
            
        except Exception as e:
            logger.error(f"Error getting rate limit for API key: {e}")
            return 60  # Default fallback
    
    def _validate_scopes(self, scopes: List[str]) -> bool:
        """Validate that provided scopes are valid.
        
        Args:
            scopes: List of scopes to validate
            
        Returns:
            True if all scopes are valid, False otherwise
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
    
    async def get_api_key_usage_stats(
        self, 
        api_key_id: int, 
        since: Optional[Timestamp] = None
    ) -> Dict[str, Any]:
        """Get usage statistics for an API key.
        
        Args:
            api_key_id: API key ID
            since: Optional timestamp to get stats since
            
        Returns:
            Dictionary with usage statistics
        """
        try:
            return await self.api_key_repo.get_usage_stats(api_key_id, since)
        except Exception as e:
            logger.error(f"Error getting API key usage stats: {e}")
            return {
                "total_requests": 0,
                "by_type": {},
                "since": since.iso_string if since else None
            }