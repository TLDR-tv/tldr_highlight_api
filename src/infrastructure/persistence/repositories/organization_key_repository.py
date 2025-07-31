"""SQLAlchemy repository implementation for organization keys.

This module provides the concrete implementation of the organization key repository.
"""

from typing import Optional, List
from datetime import datetime, timedelta

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.models.organization_key import OrganizationKey as DomainOrganizationKey
from src.domain.models.organization_key import KeyStatus
from src.infrastructure.persistence.models.organization_key import OrganizationKey
from src.infrastructure.persistence.repositories.base import BaseRepository


class OrganizationKeyRepository(BaseRepository[OrganizationKey]):
    """SQLAlchemy implementation of organization key repository."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: Async database session
        """
        super().__init__(session, OrganizationKey)

    async def create(self, key: DomainOrganizationKey) -> DomainOrganizationKey:
        """Create a new organization key.

        Args:
            key: OrganizationKey to create

        Returns:
            Created organization key with ID
        """
        db_key = OrganizationKey(
            organization_id=key.organization_id,
            key_id=key.key_id,
            key_value=key.key_value,
            algorithm=key.algorithm.value,
            key_version=key.key_version,
            is_active=key.status == KeyStatus.ACTIVE,
            is_primary=key.is_primary,
            created_at=key.created_at,
            expires_at=key.expires_at,
            rotated_at=key.rotated_at,
            deactivated_at=key.deactivated_at,
            last_used_at=key.last_used_at,
            usage_count=key.usage_count,
            previous_key_id=key.previous_key_id,
            rotation_reason=key.rotation_reason,
            created_by=key.created_by,
            description=key.description,
        )

        self.session.add(db_key)
        await self.session.commit()
        await self.session.refresh(db_key)

        key.id = db_key.id
        return key

    async def get(self, key_id: str) -> Optional[DomainOrganizationKey]:
        """Get an organization key by its public ID.

        Args:
            key_id: Public key identifier

        Returns:
            OrganizationKey if found, None otherwise
        """
        result = await self.session.execute(
            select(OrganizationKey).where(OrganizationKey.key_id == key_id)
        )
        db_key = result.scalar_one_or_none()

        if db_key:
            return self._to_domain(db_key)
        return None

    async def get_by_id(self, id: int) -> Optional[DomainOrganizationKey]:
        """Get an organization key by its database ID.

        Args:
            id: Database ID

        Returns:
            OrganizationKey if found, None otherwise
        """
        db_key = await self.session.get(OrganizationKey, id)
        if db_key:
            return self._to_domain(db_key)
        return None

    async def get_primary_key(
        self, organization_id: int
    ) -> Optional[DomainOrganizationKey]:
        """Get the primary (active) key for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            Primary OrganizationKey if found, None otherwise
        """
        result = await self.session.execute(
            select(OrganizationKey).where(
                and_(
                    OrganizationKey.organization_id == organization_id,
                    OrganizationKey.is_primary.is_(True),
                    OrganizationKey.is_active.is_(True),
                )
            )
        )
        db_key = result.scalar_one_or_none()

        if db_key:
            return self._to_domain(db_key)
        return None

    async def get_active_keys(
        self, organization_id: int, include_rotating: bool = True
    ) -> List[DomainOrganizationKey]:
        """Get all active keys for an organization.

        Args:
            organization_id: Organization ID
            include_rotating: Include keys in rotating status

        Returns:
            List of active organization keys
        """
        query = select(OrganizationKey).where(
            and_(
                OrganizationKey.organization_id == organization_id,
                OrganizationKey.is_active.is_(True),
            )
        )

        result = await self.session.execute(query)
        db_keys = result.scalars().all()

        domain_keys = [self._to_domain(k) for k in db_keys]

        # Filter by status
        if include_rotating:
            return [
                k
                for k in domain_keys
                if k.status in [KeyStatus.ACTIVE, KeyStatus.ROTATING]
            ]
        else:
            return [k for k in domain_keys if k.status == KeyStatus.ACTIVE]

    async def get_by_version(
        self, organization_id: int, version: int
    ) -> Optional[DomainOrganizationKey]:
        """Get a key by organization and version.

        Args:
            organization_id: Organization ID
            version: Key version

        Returns:
            OrganizationKey if found, None otherwise
        """
        result = await self.session.execute(
            select(OrganizationKey).where(
                and_(
                    OrganizationKey.organization_id == organization_id,
                    OrganizationKey.key_version == version,
                )
            )
        )
        db_key = result.scalar_one_or_none()

        if db_key:
            return self._to_domain(db_key)
        return None

    async def update(self, key: DomainOrganizationKey) -> DomainOrganizationKey:
        """Update an organization key.

        Args:
            key: OrganizationKey with updated values

        Returns:
            Updated organization key
        """
        db_key = await self.session.get(OrganizationKey, key.id)
        if not db_key:
            raise ValueError(f"Key with ID {key.id} not found")

        db_key.is_active = key.status in [KeyStatus.ACTIVE, KeyStatus.ROTATING]
        db_key.is_primary = key.is_primary
        db_key.rotated_at = key.rotated_at
        db_key.deactivated_at = key.deactivated_at
        db_key.last_used_at = key.last_used_at
        db_key.usage_count = key.usage_count
        db_key.rotation_reason = key.rotation_reason

        await self.session.commit()
        await self.session.refresh(db_key)

        return self._to_domain(db_key)

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
        result = await self.session.execute(
            select(OrganizationKey).where(OrganizationKey.key_id == key_id)
        )
        db_key = result.scalar_one_or_none()

        if not db_key:
            return False

        db_key.is_active = status in [KeyStatus.ACTIVE, KeyStatus.ROTATING]
        if status == KeyStatus.DEACTIVATED:
            db_key.deactivated_at = datetime.utcnow()
        if reason:
            db_key.rotation_reason = reason

        await self.session.commit()
        return True

    async def set_primary(self, organization_id: int, key_id: str) -> bool:
        """Set a key as the primary for an organization.

        This will unset any existing primary key.

        Args:
            organization_id: Organization ID
            key_id: Key to set as primary

        Returns:
            True if updated successfully
        """
        # Unset current primary
        await self.session.execute(
            select(OrganizationKey)
            .where(
                and_(
                    OrganizationKey.organization_id == organization_id,
                    OrganizationKey.is_primary.is_(True),
                )
            )
            .execution_options(synchronize_session="fetch")
        )

        # Set new primary
        result = await self.session.execute(
            select(OrganizationKey).where(
                and_(
                    OrganizationKey.organization_id == organization_id,
                    OrganizationKey.key_id == key_id,
                )
            )
        )
        db_key = result.scalar_one_or_none()

        if not db_key:
            return False

        db_key.is_primary = True
        await self.session.commit()
        return True

    async def increment_usage(self, key_id: str) -> bool:
        """Increment the usage counter for a key.

        Args:
            key_id: Public key identifier

        Returns:
            True if incremented successfully
        """
        result = await self.session.execute(
            select(OrganizationKey).where(OrganizationKey.key_id == key_id)
        )
        db_key = result.scalar_one_or_none()

        if not db_key:
            return False

        db_key.usage_count = (db_key.usage_count or 0) + 1
        db_key.last_used_at = datetime.utcnow()

        await self.session.commit()
        return True

    async def get_keys_for_rotation(
        self,
        days_until_expiry: int = 30,
        max_usage_count: int = 1_000_000,
        max_age_days: int = 180,
    ) -> List[DomainOrganizationKey]:
        """Get keys that should be rotated.

        Args:
            days_until_expiry: Keys expiring within this many days
            max_usage_count: Keys with usage exceeding this count
            max_age_days: Keys older than this many days

        Returns:
            List of keys needing rotation
        """
        now = datetime.utcnow()
        expiry_threshold = now + timedelta(days=days_until_expiry)
        age_threshold = now - timedelta(days=max_age_days)

        result = await self.session.execute(
            select(OrganizationKey).where(
                and_(
                    OrganizationKey.is_active.is_(True),
                    OrganizationKey.is_primary.is_(True),
                    or_(
                        # Expiring soon
                        and_(
                            OrganizationKey.expires_at.isnot(None),
                            OrganizationKey.expires_at <= expiry_threshold,
                        ),
                        # High usage
                        OrganizationKey.usage_count >= max_usage_count,
                        # Old key
                        OrganizationKey.created_at <= age_threshold,
                    ),
                )
            )
        )
        db_keys = result.scalars().all()

        return [self._to_domain(k) for k in db_keys]

    async def cleanup_expired_keys(self, grace_period_days: int = 30) -> int:
        """Clean up expired keys past grace period.

        Args:
            grace_period_days: Days after expiry before deletion

        Returns:
            Number of keys cleaned up
        """
        threshold = datetime.utcnow() - timedelta(days=grace_period_days)

        result = await self.session.execute(
            select(OrganizationKey).where(
                and_(
                    OrganizationKey.expires_at.isnot(None),
                    OrganizationKey.expires_at <= threshold,
                    OrganizationKey.is_primary.is_(False),
                )
            )
        )
        db_keys = result.scalars().all()

        count = len(db_keys)
        for key in db_keys:
            await self.session.delete(key)

        await self.session.commit()
        return count

    async def get_key_history(
        self, organization_id: int, limit: int = 10
    ) -> List[DomainOrganizationKey]:
        """Get key history for an organization.

        Args:
            organization_id: Organization ID
            limit: Maximum number of keys to return

        Returns:
            List of keys ordered by creation date (newest first)
        """
        result = await self.session.execute(
            select(OrganizationKey)
            .where(OrganizationKey.organization_id == organization_id)
            .order_by(OrganizationKey.created_at.desc())
            .limit(limit)
        )
        db_keys = result.scalars().all()

        return [self._to_domain(k) for k in db_keys]

    def _to_domain(self, db_key: OrganizationKey) -> DomainOrganizationKey:
        """Convert database model to domain model.

        Args:
            db_key: Database organization key

        Returns:
            Domain organization key
        """
        # Determine status from database fields
        if not db_key.is_active:
            if db_key.deactivated_at:
                status = KeyStatus.DEACTIVATED
            else:
                status = KeyStatus.EXPIRED
        elif db_key.rotated_at and not db_key.is_primary:
            status = KeyStatus.ROTATING
        else:
            status = KeyStatus.ACTIVE

        return DomainOrganizationKey(
            id=db_key.id,
            organization_id=db_key.organization_id,
            key_id=db_key.key_id,
            key_value=db_key.key_value,
            algorithm=db_key.algorithm,
            key_version=db_key.key_version,
            is_primary=db_key.is_primary,
            status=status,
            created_at=db_key.created_at,
            expires_at=db_key.expires_at,
            rotated_at=db_key.rotated_at,
            deactivated_at=db_key.deactivated_at,
            last_used_at=db_key.last_used_at,
            usage_count=db_key.usage_count,
            previous_key_id=db_key.previous_key_id,
            rotation_reason=db_key.rotation_reason,
            created_by=db_key.created_by,
            description=db_key.description,
        )
