"""Security-related domain services."""

from typing import Protocol, runtime_checkable
import bcrypt
import hashlib
import secrets


@runtime_checkable
class PasswordHashingService(Protocol):
    """Service for hashing and verifying passwords."""

    def hash_password(self, password: str) -> str:
        """Hash a password using a secure algorithm.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        ...

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password to verify
            password_hash: Stored password hash
            
        Returns:
            True if password matches, False otherwise
        """
        ...


@runtime_checkable
class APIKeyHashingService(Protocol):
    """Service for hashing API keys."""

    def hash_key(self, api_key: str) -> str:
        """Hash an API key for storage.
        
        Args:
            api_key: Plain text API key
            
        Returns:
            Hashed API key
        """
        ...

    def generate_key(self) -> str:
        """Generate a new API key.
        
        Returns:
            New API key in standard format
        """
        ...


class BcryptPasswordHashingService:
    """Bcrypt implementation of password hashing service."""

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return bcrypt.hashpw(
            password.encode("utf-8"), 
            bcrypt.gensalt()
        ).decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its bcrypt hash."""
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )


class SHA256APIKeyHashingService:
    """SHA256 implementation of API key hashing service."""

    def hash_key(self, api_key: str) -> str:
        """Hash an API key using SHA256."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def generate_key(self) -> str:
        """Generate a new API key with consistent prefix."""
        return f"tldr_sk_{secrets.token_urlsafe(32)}"