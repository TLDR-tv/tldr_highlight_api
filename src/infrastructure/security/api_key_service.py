"""API key generation and validation service."""
import secrets
import string
from typing import Optional, Tuple
from uuid import UUID
from passlib.context import CryptContext

from ...domain.models.api_key import APIKey, APIScopes
from ...domain.protocols import APIKeyRepository, AuthenticationService


class APIKeyService(AuthenticationService):
    """Service for API key management."""
    
    def __init__(self, repository: APIKeyRepository):
        """Initialize with repository."""
        self.repository = repository
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.key_prefix = "tldr"
        self.key_length = 32
    
    async def generate_api_key(
        self, 
        organization_id: UUID, 
        name: str,
        scopes: Optional[set[str]] = None,
        description: Optional[str] = None,
        created_by_user_id: Optional[UUID] = None
    ) -> Tuple[str, APIKey]:
        """
        Generate new API key.
        Returns (raw_key, key_entity) tuple.
        The raw key is only available at creation time.
        """
        # Generate secure random key
        alphabet = string.ascii_letters + string.digits
        raw_key_part = ''.join(secrets.choice(alphabet) for _ in range(self.key_length))
        
        # Create identifiable prefix
        prefix = f"{self.key_prefix}_{raw_key_part[:8]}"
        
        # Full key format: prefix_rest_of_key
        full_key = f"{prefix}_{raw_key_part[8:]}"
        
        # Hash the full key for storage
        key_hash = self.pwd_context.hash(full_key)
        
        # Create API key entity
        api_key = APIKey(
            organization_id=organization_id,
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            scopes=scopes or APIScopes.default_scopes(),
            description=description,
            created_by_user_id=created_by_user_id
        )
        
        # Save to repository
        saved_key = await self.repository.add(api_key)
        
        return full_key, saved_key
    
    async def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key and return entity if valid."""
        # Extract prefix from key
        if not api_key.startswith(f"{self.key_prefix}_"):
            return None
        
        try:
            parts = api_key.split("_", 2)
            if len(parts) != 3:
                return None
            
            prefix = f"{parts[0]}_{parts[1]}"
        except (ValueError, IndexError):
            return None
        
        # Look up key by prefix
        key_entity = await self.repository.get_by_prefix(prefix)
        if not key_entity:
            return None
        
        # Verify the full key matches the hash
        if not self.pwd_context.verify(api_key, key_entity.key_hash):
            return None
        
        # Check if key is valid
        if not key_entity.is_valid:
            return None
        
        # Record usage
        key_entity.record_usage()
        await self.repository.update(key_entity)
        
        return key_entity
    
    async def revoke_api_key(self, key_id: UUID) -> None:
        """Revoke an API key."""
        key_entity = await self.repository.get(key_id)
        if key_entity:
            key_entity.revoke()
            await self.repository.update(key_entity)
    
    async def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    async def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(password, hashed)