"""API key domain model - secure access tokens."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4
import secrets
import string


@dataclass
class APIKey:
    """API key for accessing the system."""

    id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)
    name: str = ""  # Friendly name for the key
    key_hash: str = ""  # Hashed version of the key
    prefix: str = ""  # First 8 chars for identification

    # Permissions
    scopes: set[str] = field(
        default_factory=set
    )  # e.g., "streams:read", "highlights:write"

    # Usage tracking
    last_used_at: Optional[datetime] = None
    usage_count: int = 0

    # Lifecycle
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    revoked_at: Optional[datetime] = None

    # Metadata
    created_by_user_id: Optional[UUID] = None
    description: Optional[str] = None

    @classmethod
    def generate_key(cls) -> tuple[str, str]:
        """
        Generate a new API key.
        Returns (full_key, key_hash) tuple.
        The full key is only shown once and should be given to the user.
        """
        # Generate a secure random key
        alphabet = string.ascii_letters + string.digits
        key_length = 32
        full_key = "".join(secrets.choice(alphabet) for _ in range(key_length))

        # Create a prefix for identification (first 8 chars)
        prefix = f"tldr_{full_key[:8]}"

        # In production, we'd hash this with bcrypt or similar
        # For now, we'll use a placeholder
        key_hash = f"hashed_{full_key}"

        return f"{prefix}_{full_key}", key_hash

    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if key is valid for use."""
        return self.is_active and not self.is_expired and not self.revoked_at

    def has_scope(self, scope: str) -> bool:
        """Check if key has a specific scope."""
        return scope in self.scopes or "*" in self.scopes  # * = all permissions

    def record_usage(self) -> None:
        """Record that the key was used."""
        self.last_used_at = datetime.now(timezone.utc)
        self.usage_count += 1

    def revoke(self) -> None:
        """Revoke the API key."""
        self.is_active = False
        self.revoked_at = datetime.now(timezone.utc)

    def set_expiration(self, days: int) -> None:
        """Set key to expire after specified days."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(days=days)


# Common scopes
class APIScopes:
    """Standard API scopes."""

    # Streams
    STREAMS_READ = "streams:read"
    STREAMS_WRITE = "streams:write"

    # Highlights
    HIGHLIGHTS_READ = "highlights:read"
    HIGHLIGHTS_WRITE = "highlights:write"

    # Organizations
    ORG_READ = "organizations:read"
    ORG_ADMIN = "organizations:admin"

    # Webhooks
    WEBHOOKS_READ = "webhooks:read"
    WEBHOOKS_WRITE = "webhooks:write"

    # Full access
    ALL = "*"

    @classmethod
    def default_scopes(cls) -> set[str]:
        """Get default scopes for new API keys."""
        return {
            cls.STREAMS_READ,
            cls.STREAMS_WRITE,
            cls.HIGHLIGHTS_READ,
            cls.WEBHOOKS_READ,
        }
