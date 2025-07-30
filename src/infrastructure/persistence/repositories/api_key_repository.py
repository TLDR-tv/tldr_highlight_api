"""API key repository implementation."""

from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from src.domain.repositories.api_key_repository import (
    APIKeyRepository as IAPIKeyRepository,
)
from src.domain.entities.api_key import APIKey
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.api_key import APIKey as APIKeyModel
from src.infrastructure.persistence.models.usage_record import (
    UsageRecord,
    UsageRecordType,
)
from src.infrastructure.persistence.mappers.api_key_mapper import APIKeyMapper


class APIKeyRepository(BaseRepository[APIKey, APIKeyModel, int], IAPIKeyRepository):
    """Concrete implementation of APIKeyRepository using SQLAlchemy."""

    def __init__(self, session):
        """Initialize APIKeyRepository with session."""
        super().__init__(
            session=session, model_class=APIKeyModel, mapper=APIKeyMapper()
        )

    async def get_by_key_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by its hash.

        Args:
            key_hash: The hashed API key

        Returns:
            APIKey domain entity if found, None otherwise
        """
        stmt = (
            select(APIKeyModel)
            .where(APIKeyModel.key_hash == key_hash)
            .options(selectinload(APIKeyModel.user))
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self.mapper.to_domain(model)

    async def get_by_user(self, user_id: int, active_only: bool = True) -> List[APIKey]:
        """Get all API keys for a user.

        Args:
            user_id: User ID
            active_only: Whether to filter only active keys

        Returns:
            List of API keys for the user
        """
        stmt = select(APIKeyModel).where(APIKeyModel.user_id == user_id)

        if active_only:
            stmt = stmt.where(APIKeyModel.is_active == True)

        stmt = stmt.order_by(APIKeyModel.created_at.desc())

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_active_keys(self) -> List[APIKey]:
        """Get all active API keys.

        Returns:
            List of all active API keys
        """
        stmt = (
            select(APIKeyModel)
            .where(APIKeyModel.is_active == True)
            .where(
                or_(
                    APIKeyModel.expires_at.is_(None),
                    APIKeyModel.expires_at > datetime.utcnow(),
                )
            )
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_expiring_soon(self, days: int = 7) -> List[APIKey]:
        """Get API keys expiring within specified days.

        Args:
            days: Number of days to look ahead

        Returns:
            List of API keys expiring soon
        """
        expiry_date = datetime.utcnow() + timedelta(days=days)

        stmt = (
            select(APIKeyModel)
            .where(
                and_(
                    APIKeyModel.expires_at.isnot(None),
                    APIKeyModel.expires_at <= expiry_date,
                    APIKeyModel.expires_at > datetime.utcnow(),
                    APIKeyModel.is_active == True,
                )
            )
            .order_by(APIKeyModel.expires_at)
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_scope(self, scope: str, active_only: bool = True) -> List[APIKey]:
        """Get all API keys with specific scope.

        Args:
            scope: The scope to search for
            active_only: Whether to filter only active keys

        Returns:
            List of API keys with the scope
        """
        # Use JSON contains for PostgreSQL
        stmt = select(APIKeyModel).where(APIKeyModel.scopes.contains(f'"{scope}"'))

        if active_only:
            stmt = stmt.where(APIKeyModel.is_active == True)

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def count_by_user(self, user_id: int, active_only: bool = True) -> int:
        """Count API keys for a user.

        Args:
            user_id: User ID
            active_only: Whether to count only active keys

        Returns:
            Count of API keys
        """
        stmt = (
            select(func.count())
            .select_from(APIKeyModel)
            .where(APIKeyModel.user_id == user_id)
        )

        if active_only:
            stmt = stmt.where(APIKeyModel.is_active == True)

        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def cleanup_expired(self) -> int:
        """Remove expired API keys.

        Returns:
            Number of keys removed
        """
        # Find expired keys
        stmt = select(APIKeyModel).where(
            and_(
                APIKeyModel.expires_at.isnot(None),
                APIKeyModel.expires_at < datetime.utcnow(),
            )
        )

        result = await self.session.execute(stmt)
        expired_keys = list(result.scalars().unique())

        # Delete them
        for key in expired_keys:
            await self.session.delete(key)

        await self.session.flush()
        return len(expired_keys)

    async def get_usage_stats(
        self, api_key_id: int, since: Optional[Timestamp] = None
    ) -> dict:
        """Get usage statistics for an API key.

        Args:
            api_key_id: API key ID
            since: Optional timestamp to get stats since

        Returns:
            Dictionary with usage statistics
        """
        # Base query for usage records
        stmt = select(
            UsageRecord.usage_type,
            func.count(UsageRecord.id).label("count"),
            func.sum(UsageRecord.quantity).label("total_quantity"),
        ).where(UsageRecord.api_key_id == api_key_id)

        if since:
            stmt = stmt.where(UsageRecord.created_at >= since.value)

        stmt = stmt.group_by(UsageRecord.usage_type)

        result = await self.session.execute(stmt)
        rows = result.all()

        # Format statistics
        stats = {
            "total_requests": 0,
            "by_type": {},
            "since": since.iso_string if since else None,
        }

        for row in rows:
            usage_type = row.usage_type
            count = row.count
            total_quantity = row.total_quantity or 0

            stats["by_type"][usage_type] = {
                "count": count,
                "total_quantity": float(total_quantity),
            }

            if usage_type == UsageRecordType.API_CALL:
                stats["total_requests"] = count

        return stats

    async def revoke_all_for_user(self, user_id: int) -> int:
        """Revoke all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            Number of keys revoked
        """
        stmt = select(APIKeyModel).where(
            and_(APIKeyModel.user_id == user_id, APIKeyModel.is_active == True)
        )

        result = await self.session.execute(stmt)
        active_keys = list(result.scalars().unique())

        # Deactivate them
        for key in active_keys:
            key.is_active = False

        await self.session.flush()
        return len(active_keys)
