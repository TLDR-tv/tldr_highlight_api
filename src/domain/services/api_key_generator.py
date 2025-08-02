"""API key generation service."""

import secrets
import string
from datetime import datetime, timezone
from typing import Tuple
from uuid import UUID, uuid4

from ..models.api_key import APIKey, APIScopes


class APIKeyGenerator:
    """Service for generating secure API keys."""

    PREFIX_LENGTH = 8
    KEY_LENGTH = 32
    ALPHABET = string.ascii_letters + string.digits

    @classmethod
    def generate_key(cls) -> Tuple[str, str]:
        """Generate a new API key and its prefix.

        Returns:
            Tuple of (full_key, prefix)
        """
        # Generate secure random key
        key = "".join(secrets.choice(cls.ALPHABET) for _ in range(cls.KEY_LENGTH))
        prefix = key[: cls.PREFIX_LENGTH]
        return key, prefix

    @classmethod
    def create_api_key(
        cls,
        organization_id: UUID,
        name: str,
        scopes: set[str] = None,
    ) -> APIKey:
        """Create a new API key entity.

        Args:
            organization_id: Organization that owns the key
            name: Descriptive name for the key
            scopes: Set of permission scopes

        Returns:
            New APIKey instance
        """
        if scopes is None:
            # Default to read-only scopes
            scopes = {
                APIScopes.STREAMS_READ,
                APIScopes.HIGHLIGHTS_READ,
                APIScopes.ORG_READ,
            }

        key, prefix = cls.generate_key()

        return APIKey(
            id=uuid4(),
            organization_id=organization_id,
            name=name,
            key=key,
            prefix=prefix,
            scopes=scopes,
            created_at=datetime.now(timezone.utc),
        )
