"""API Key domain entity."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from secrets import token_urlsafe
from enum import Enum

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp


class APIKeyScope(Enum):
    """Enumeration of available API key scopes."""
    # Stream operations
    STREAMS_READ = "streams:read"
    STREAMS_WRITE = "streams:write"
    STREAMS_DELETE = "streams:delete"
    
    # Highlight operations
    HIGHLIGHTS_READ = "highlights:read"
    HIGHLIGHTS_WRITE = "highlights:write"
    HIGHLIGHTS_DELETE = "highlights:delete"
    
    # Batch operations
    BATCHES_READ = "batches:read"
    BATCHES_WRITE = "batches:write"
    BATCHES_DELETE = "batches:delete"
    
    # Webhook operations
    WEBHOOKS_READ = "webhooks:read"
    WEBHOOKS_WRITE = "webhooks:write"
    WEBHOOKS_DELETE = "webhooks:delete"
    
    # Organization operations
    ORGANIZATIONS_READ = "organizations:read"
    ORGANIZATIONS_WRITE = "organizations:write"
    
    # User operations
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    
    # Admin scope
    ADMIN = "admin"


@dataclass
class APIKey(Entity[int]):
    """Domain entity representing an API key.
    
    API keys are used for authenticating requests to the
    TL;DR Highlight API.
    """
    
    name: str
    key_hash: str
    user_id: int
    scopes: List[APIKeyScope] = field(default_factory=list)
    
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
        if APIKeyScope.ADMIN in self.scopes:
            return {perm: True for perm in all_permissions}
        
        # Map scopes to permissions
        for scope in self.scopes:
            scope_value = scope.value
            if ":" in scope_value:
                resource, action = scope_value.split(":")
                if resource in all_permissions:
                    all_permissions[resource] = True
                if action in all_permissions:
                    all_permissions[action] = True
        
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
    
    def has_scope(self, scope: APIKeyScope) -> bool:
        """Check if API key has specific scope."""
        return scope in self.scopes or APIKeyScope.ADMIN in self.scopes
    
    def add_scope(self, scope: APIKeyScope) -> "APIKey":
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
    
    def remove_scope(self, scope: APIKeyScope) -> "APIKey":
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
    
    def revoke(self) -> "APIKey":
        """Revoke the API key (alias for deactivate)."""
        return self.deactivate()