"""Repository protocol for organization key management.

This module defines the protocol for managing organization signing keys
in the persistence layer.
"""

from typing import Optional, List, Protocol

from src.domain.entities.organization_key import OrganizationKey, KeyStatus


class OrganizationKeyRepository(Protocol):
    """Protocol for organization key repository implementations."""

    async def create(self, key: OrganizationKey) -> OrganizationKey:
        """Create a new organization key.

        Args:
            key: OrganizationKey to create

        Returns:
            Created organization key with ID
        """
        ...

    async def get(self, key_id: str) -> Optional[OrganizationKey]:
        """Get an organization key by its public ID.

        Args:
            key_id: Public key identifier

        Returns:
            OrganizationKey if found, None otherwise
        """
        ...

    async def get_by_id(self, id: int) -> Optional[OrganizationKey]:
        """Get an organization key by its database ID.

        Args:
            id: Database ID

        Returns:
            OrganizationKey if found, None otherwise
        """
        ...

    async def get_primary_key(self, organization_id: int) -> Optional[OrganizationKey]:
        """Get the primary (active) key for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            Primary OrganizationKey if found, None otherwise
        """
        ...

    async def get_active_keys(
        self, organization_id: int, include_rotating: bool = True
    ) -> List[OrganizationKey]:
        """Get all active keys for an organization.

        Args:
            organization_id: Organization ID
            include_rotating: Include keys in rotating status

        Returns:
            List of active organization keys
        """
        ...

    async def get_by_version(
        self, organization_id: int, version: int
    ) -> Optional[OrganizationKey]:
        """Get a key by organization and version.

        Args:
            organization_id: Organization ID
            version: Key version

        Returns:
            OrganizationKey if found, None otherwise
        """
        ...

    async def update(self, key: OrganizationKey) -> OrganizationKey:
        """Update an organization key.

        Args:
            key: OrganizationKey with updated values

        Returns:
            Updated organization key
        """
        ...

    async def update_status(
        self, key_id: str, status: KeyStatus, reason: Optional[str] = None
    ) -> bool:
        """Update the status of a key.

        Args:
            key_id: Public key identifier
            status: New status
            reason: Optional reason for status change

        Returns:
            True if updated successfully
        """
        ...

    async def set_primary(self, organization_id: int, key_id: str) -> bool:
        """Set a key as the primary for an organization.

        This will unset any existing primary key.

        Args:
            organization_id: Organization ID
            key_id: Key to set as primary

        Returns:
            True if updated successfully
        """
        ...

    async def increment_usage(self, key_id: str) -> bool:
        """Increment the usage counter for a key.

        Args:
            key_id: Public key identifier

        Returns:
            True if incremented successfully
        """
        ...

    async def get_keys_for_rotation(
        self,
        days_until_expiry: int = 30,
        max_usage_count: int = 1_000_000,
        max_age_days: int = 180,
    ) -> List[OrganizationKey]:
        """Get keys that should be rotated.

        Args:
            days_until_expiry: Keys expiring within this many days
            max_usage_count: Keys with usage exceeding this count
            max_age_days: Keys older than this many days

        Returns:
            List of keys needing rotation
        """
        ...

    async def cleanup_expired_keys(self, grace_period_days: int = 30) -> int:
        """Clean up expired keys past grace period.

        Args:
            grace_period_days: Days after expiry before deletion

        Returns:
            Number of keys cleaned up
        """
        ...

    async def get_key_history(
        self, organization_id: int, limit: int = 10
    ) -> List[OrganizationKey]:
        """Get key history for an organization.

        Args:
            organization_id: Organization ID
            limit: Maximum number of keys to return

        Returns:
            List of keys ordered by creation date (newest first)
        """
        ...
