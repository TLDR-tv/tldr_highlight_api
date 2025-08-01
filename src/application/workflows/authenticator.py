"""Authentication for B2B AI highlighting - clean Pythonic implementation.

This module handles API key authentication and user sessions
for enterprise customers.
"""

import secrets
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import logfire

from src.domain.entities.user import User
from src.domain.entities.api_key import APIKey
from src.domain.value_objects.email import Email
from src.domain.exceptions import (
    EntityNotFoundError,
    UnauthorizedAccessError,
    BusinessRuleViolation
)


@dataclass
class Authenticator:
    """Handles authentication for B2B customers.
    
    Supports both API key authentication (primary) and
    user/password authentication (for web portal).
    """
    
    user_repo: Any  # Duck typing
    api_key_repo: Any
    org_repo: Any
    
    def __post_init__(self):
        self.logger = logfire.get_logger(__name__)
    
    async def authenticate_api_key(
        self,
        key_string: str
    ) -> Tuple[User, APIKey]:
        """Authenticate using API key.
        
        Args:
            key_string: The API key string
            
        Returns:
            Tuple of (user, api_key) if valid
            
        Raises:
            UnauthorizedAccessError: If authentication fails
        """
        # Hash the key to look it up
        key_hash = self._hash_api_key(key_string)
        
        # Find API key
        api_key = await self.api_key_repo.get_by_hash(key_hash)
        if not api_key:
            self.logger.warning("Invalid API key attempted")
            raise UnauthorizedAccessError("Invalid API key")
        
        # Check if active
        if not api_key.is_active:
            raise UnauthorizedAccessError("API key is inactive")
        
        # Check expiration
        if api_key.is_expired:
            raise UnauthorizedAccessError("API key has expired")
        
        # Get user
        user = await self.user_repo.get(api_key.user_id)
        if not user or not user.is_active:
            raise UnauthorizedAccessError("User account is inactive")
        
        # Update last used
        api_key.record_usage()
        await self.api_key_repo.save(api_key)
        
        self.logger.info(
            f"API key authenticated for user {user.id}",
            extra={"user_id": user.id, "key_id": api_key.id}
        )
        
        return user, api_key
    
    async def authenticate_user(
        self,
        email: str,
        password: str
    ) -> User:
        """Authenticate user with email/password.
        
        For web portal access.
        """
        # Get user by email
        user = await self.user_repo.get_by_email(Email(email))
        if not user:
            # Don't reveal if email exists
            raise UnauthorizedAccessError("Invalid credentials")
        
        # Check password
        if not self._verify_password(password, user.password_hash):
            raise UnauthorizedAccessError("Invalid credentials")
        
        # Check if active
        if not user.is_active:
            raise UnauthorizedAccessError("Account is inactive")
        
        self.logger.info(
            f"User authenticated: {user.id}",
            extra={"user_id": user.id, "email": email}
        )
        
        return user
    
    async def create_api_key(
        self,
        user_id: int,
        name: str,
        permissions: Optional[list] = None,
        expires_at: Optional[datetime] = None
    ) -> Tuple[str, APIKey]:
        """Create a new API key for a user.
        
        Returns:
            Tuple of (key_string, api_key_entity)
            The key string is only returned once!
        """
        # Verify user exists
        user = await self.user_repo.get(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")
        
        # Check user's key limit (business rule)
        existing_keys = await self.api_key_repo.get_by_user(user_id, active_only=True)
        if len(existing_keys) >= 10:
            raise BusinessRuleViolation(
                "Maximum of 10 active API keys per user"
            )
        
        # Generate key
        key_string = self._generate_api_key()
        key_hash = self._hash_api_key(key_string)
        
        # Create entity
        api_key = APIKey.create(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            permissions=permissions or ["stream:create", "highlight:read"],
            expires_at=expires_at,
        )
        
        # Save
        saved_key = await self.api_key_repo.save(api_key)
        
        self.logger.info(
            f"Created API key for user {user_id}",
            extra={"user_id": user_id, "key_id": saved_key.id}
        )
        
        return key_string, saved_key
    
    async def revoke_api_key(
        self,
        key_id: int,
        user_id: int
    ) -> APIKey:
        """Revoke an API key."""
        api_key = await self.api_key_repo.get(key_id)
        if not api_key:
            raise EntityNotFoundError(f"API key {key_id} not found")
        
        # Verify ownership
        if api_key.user_id != user_id:
            raise UnauthorizedAccessError("Cannot revoke another user's key")
        
        # Revoke
        api_key.revoke()
        return await self.api_key_repo.save(api_key)
    
    async def check_permission(
        self,
        api_key: APIKey,
        permission: str
    ) -> bool:
        """Check if API key has a specific permission."""
        return api_key.has_permission(permission)
    
    # Private methods
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        # Format: tldr_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        random_part = secrets.token_urlsafe(32)
        return f"tldr_{random_part}"
    
    def _hash_api_key(self, key_string: str) -> str:
        """Hash API key for storage."""
        # Use SHA256 for API keys (fast lookups)
        import hashlib
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(
            password.encode('utf-8'),
            password_hash.encode('utf-8')
        )


@dataclass
class SessionManager:
    """Manages user sessions for web portal.
    
    Simple JWT-based sessions.
    """
    
    user_repo: Any
    secret_key: str
    token_expiry: timedelta = timedelta(hours=24)
    
    def create_session(self, user: User) -> str:
        """Create a session token for a user."""
        import jwt
        
        payload = {
            "user_id": user.id,
            "email": user.email.value,
            "exp": datetime.now(timezone.utc) + self.token_expiry,
            "iat": datetime.now(timezone.utc),
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    async def verify_session(self, token: str) -> Optional[User]:
        """Verify session token and return user."""
        import jwt
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
            
            user_id = payload.get("user_id")
            if not user_id:
                return None
            
            user = await self.user_repo.get(user_id)
            if user and user.is_active:
                return user
            
        except jwt.ExpiredSignatureError:
            pass
        except jwt.InvalidTokenError:
            pass
        
        return None