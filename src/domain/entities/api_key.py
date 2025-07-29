"""API Key domain entity."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from secrets import token_urlsafe

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp


@dataclass
class APIKey(Entity[int]):
    """Domain entity representing an API key.
    
    API keys are used for authenticating requests to the
    TL;DR Highlight API.
    """
    
    name: str
    key_hash: str
    user_id: int
    scopes: List[str] = field(default_factory=list)
    
    # Optional metadata
    description: Optional[str] = None
    expires_at: Optional[Timestamp] = None
    last_used_at: Optional[Timestamp] = None
    
    # Rate limiting
    rate_limit_override: Optional[int] = None  # Requests per minute
    
    # Security
    allowed_ips: List[str] = field(default_factory=list)
    is_active: bool = True
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new API key."""
        return f"tldr_{token_urlsafe(32)}"
    
    @property
    def permissions(self) -> Dict[str, bool]:
        """Get all permissions for this API key."""
        all_permissions = {
            "read": False,
            "write": False,
            "delete": False,
            "streams": False,
            "batches": False,
            "webhooks": False,
            "analytics": False,
            "admin": False,
        }
        
        # Admin scope grants all permissions
        if "admin" in self.scopes:
            return {perm: True for perm in all_permissions}
        
        # Set individual permissions based on scopes
        for scope in self.scopes:
            if scope in all_permissions:
                all_permissions[scope] = True
        
        return all_permissions
    
    @property
    def is_expired(self) -> bool:
        """Check if API key has expired."""
        if not self.expires_at:
            return False
        return self.expires_at.is_before(Timestamp.now())
    
    @property
    def is_valid(self) -> bool:
        """Check if API key is valid for use."""
        return self.is_active and not self.is_expired
    
    def has_permission(self, permission: str) -> bool:
        """Check if API key has specific permission."""
        return self.permissions.get(permission, False)
    
    def has_scope(self, scope: str) -> bool:
        """Check if API key has specific scope."""
        return scope in self.scopes or "admin" in self.scopes
    
    def add_scope(self, scope: str) -> "APIKey":
        """Add a scope to the API key."""
        if scope in self.scopes:
            return self
        
        new_scopes = self.scopes.copy()
        new_scopes.append(scope)
        
        return APIKey(
            id=self.id,
            name=self.name,
            key_hash=self.key_hash,
            user_id=self.user_id,
            scopes=new_scopes,
            description=self.description,
            expires_at=self.expires_at,
            last_used_at=self.last_used_at,
            rate_limit_override=self.rate_limit_override,
            allowed_ips=self.allowed_ips.copy(),
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now()
        )
    
    def remove_scope(self, scope: str) -> "APIKey":
        """Remove a scope from the API key."""
        if scope not in self.scopes:
            return self
        
        new_scopes = [s for s in self.scopes if s != scope]
        
        return APIKey(
            id=self.id,
            name=self.name,
            key_hash=self.key_hash,
            user_id=self.user_id,
            scopes=new_scopes,
            description=self.description,
            expires_at=self.expires_at,
            last_used_at=self.last_used_at,
            rate_limit_override=self.rate_limit_override,
            allowed_ips=self.allowed_ips.copy(),
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now()
        )
    
    def record_usage(self) -> "APIKey":
        """Update last used timestamp."""
        return APIKey(
            id=self.id,
            name=self.name,
            key_hash=self.key_hash,
            user_id=self.user_id,
            scopes=self.scopes.copy(),
            description=self.description,
            expires_at=self.expires_at,
            last_used_at=Timestamp.now(),
            rate_limit_override=self.rate_limit_override,
            allowed_ips=self.allowed_ips.copy(),
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=self.updated_at
        )
    
    def deactivate(self) -> "APIKey":
        """Deactivate the API key."""
        return APIKey(
            id=self.id,
            name=self.name,
            key_hash=self.key_hash,
            user_id=self.user_id,
            scopes=self.scopes.copy(),
            description=self.description,
            expires_at=self.expires_at,
            last_used_at=self.last_used_at,
            rate_limit_override=self.rate_limit_override,
            allowed_ips=self.allowed_ips.copy(),
            is_active=False,
            created_at=self.created_at,
            updated_at=Timestamp.now()
        )
    
    def add_ip_restriction(self, ip_address: str) -> "APIKey":
        """Add an IP address restriction."""
        if ip_address in self.allowed_ips:
            return self
        
        new_ips = self.allowed_ips.copy()
        new_ips.append(ip_address)
        
        return APIKey(
            id=self.id,
            name=self.name,
            key_hash=self.key_hash,
            user_id=self.user_id,
            scopes=self.scopes.copy(),
            description=self.description,
            expires_at=self.expires_at,
            last_used_at=self.last_used_at,
            rate_limit_override=self.rate_limit_override,
            allowed_ips=new_ips,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now()
        )