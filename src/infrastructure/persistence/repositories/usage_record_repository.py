"""Usage record repository implementation."""

from typing import Optional, List, Dict, Any
from sqlalchemy import select, func, and_, or_

from src.domain.repositories.usage_record_repository import (
    UsageRecordRepository as IUsageRecordRepository,
)
from src.domain.entities.usage_record import UsageRecord, UsageType
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.usage_record import (
    UsageRecord as UsageRecordModel,
)
from src.infrastructure.persistence.mappers.usage_record_mapper import UsageRecordMapper


class UsageRecordRepository(
    BaseRepository[UsageRecord, UsageRecordModel, int], IUsageRecordRepository
):
    """Concrete implementation of UsageRecordRepository using SQLAlchemy."""

    def __init__(self, session):
        """Initialize UsageRecordRepository with session."""
        super().__init__(
            session=session, model_class=UsageRecordModel, mapper=UsageRecordMapper()
        )

    async def get_by_user(
        self,
        user_id: int,
        usage_type: Optional[UsageType] = None,
        start: Optional[Timestamp] = None,
        end: Optional[Timestamp] = None,
    ) -> List[UsageRecord]:
        """Get usage records for a user with optional filters.

        Args:
            user_id: User ID
            usage_type: Optional usage type filter
            start: Optional start timestamp
            end: Optional end timestamp

        Returns:
            List of usage records for the user
        """
        stmt = select(UsageRecordModel).where(UsageRecordModel.user_id == user_id)

        if usage_type:
            stmt = stmt.where(UsageRecordModel.usage_type == usage_type.value)

        if start:
            stmt = stmt.where(UsageRecordModel.period_start >= start.value)

        if end:
            stmt = stmt.where(
                or_(
                    UsageRecordModel.period_end <= end.value,
                    UsageRecordModel.period_end.is_(None),
                )
            )

        stmt = stmt.order_by(UsageRecordModel.period_start.desc())

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_organization(
        self,
        organization_id: int,
        usage_type: Optional[UsageType] = None,
        start: Optional[Timestamp] = None,
        end: Optional[Timestamp] = None,
    ) -> List[UsageRecord]:
        """Get usage records for an organization.

        Args:
            organization_id: Organization ID
            usage_type: Optional usage type filter
            start: Optional start timestamp
            end: Optional end timestamp

        Returns:
            List of usage records for the organization
        """
        stmt = select(UsageRecordModel).where(
            UsageRecordModel.organization_id == organization_id
        )

        if usage_type:
            stmt = stmt.where(UsageRecordModel.usage_type == usage_type.value)

        if start:
            stmt = stmt.where(UsageRecordModel.period_start >= start.value)

        if end:
            stmt = stmt.where(
                or_(
                    UsageRecordModel.period_end <= end.value,
                    UsageRecordModel.period_end.is_(None),
                )
            )

        stmt = stmt.order_by(UsageRecordModel.period_start.desc())

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_by_resource(
        self, resource_id: int, resource_type: str
    ) -> List[UsageRecord]:
        """Get usage records for a specific resource.

        Args:
            resource_id: Resource ID
            resource_type: Resource type (e.g., "stream", "batch")

        Returns:
            List of usage records for the resource
        """
        stmt = (
            select(UsageRecordModel)
            .where(
                and_(
                    UsageRecordModel.resource_id == resource_id,
                    UsageRecordModel.resource_type == resource_type,
                )
            )
            .order_by(UsageRecordModel.period_start.desc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def get_billable_usage(
        self,
        user_id: int,
        billing_period_start: Timestamp,
        billing_period_end: Timestamp,
    ) -> List[UsageRecord]:
        """Get billable usage records for a billing period.

        Args:
            user_id: User ID
            billing_period_start: Start of billing period
            billing_period_end: End of billing period

        Returns:
            List of billable usage records
        """
        stmt = (
            select(UsageRecordModel)
            .where(
                and_(
                    UsageRecordModel.user_id == user_id,
                    UsageRecordModel.billable.is_(True),
                    UsageRecordModel.period_start >= billing_period_start.value,
                    or_(
                        UsageRecordModel.period_end <= billing_period_end.value,
                        UsageRecordModel.period_end.is_(None),
                    ),
                )
            )
            .order_by(UsageRecordModel.period_start.asc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def calculate_usage_totals(
        self, user_id: int, usage_type: UsageType, start: Timestamp, end: Timestamp
    ) -> Dict[str, float]:
        """Calculate usage totals by type for a period.

        Args:
            user_id: User ID
            usage_type: Usage type to calculate
            start: Start timestamp
            end: End timestamp

        Returns:
            Dictionary with usage totals
        """
        stmt = select(
            func.sum(UsageRecordModel.quantity).label("total_quantity"),
            func.sum(UsageRecordModel.total_cost).label("total_cost"),
            func.count().label("record_count"),
            func.avg(UsageRecordModel.quantity).label("avg_quantity"),
        ).where(
            and_(
                UsageRecordModel.user_id == user_id,
                UsageRecordModel.usage_type == usage_type.value,
                UsageRecordModel.period_start >= start.value,
                or_(
                    UsageRecordModel.period_end <= end.value,
                    UsageRecordModel.period_end.is_(None),
                ),
            )
        )

        result = await self.session.execute(stmt)
        row = result.one()

        return {
            "total_quantity": float(row.total_quantity or 0),
            "total_cost": float(row.total_cost or 0),
            "record_count": row.record_count or 0,
            "average_quantity": float(row.avg_quantity or 0),
        }

    async def get_usage_by_api_key(
        self,
        api_key_id: int,
        start: Optional[Timestamp] = None,
        end: Optional[Timestamp] = None,
    ) -> List[UsageRecord]:
        """Get usage records for an API key.

        Args:
            api_key_id: API key ID
            start: Optional start timestamp
            end: Optional end timestamp

        Returns:
            List of usage records for the API key
        """
        stmt = select(UsageRecordModel).where(UsageRecordModel.api_key_id == api_key_id)

        if start:
            stmt = stmt.where(UsageRecordModel.period_start >= start.value)

        if end:
            stmt = stmt.where(
                or_(
                    UsageRecordModel.period_end <= end.value,
                    UsageRecordModel.period_end.is_(None),
                )
            )

        stmt = stmt.order_by(UsageRecordModel.period_start.desc())

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def aggregate_by_type(
        self, user_id: int, start: Timestamp, end: Timestamp
    ) -> Dict[UsageType, Dict[str, float]]:
        """Aggregate usage by type for a period.

        Args:
            user_id: User ID
            start: Start timestamp
            end: End timestamp

        Returns:
            Dictionary mapping usage types to aggregated values
        """
        stmt = (
            select(
                UsageRecordModel.usage_type,
                func.sum(UsageRecordModel.quantity).label("total_quantity"),
                func.sum(UsageRecordModel.total_cost).label("total_cost"),
                func.count().label("record_count"),
                func.avg(UsageRecordModel.quantity).label("avg_quantity"),
            )
            .where(
                and_(
                    UsageRecordModel.user_id == user_id,
                    UsageRecordModel.period_start >= start.value,
                    or_(
                        UsageRecordModel.period_end <= end.value,
                        UsageRecordModel.period_end.is_(None),
                    ),
                )
            )
            .group_by(UsageRecordModel.usage_type)
        )

        result = await self.session.execute(stmt)

        aggregated = {}
        for row in result:
            usage_type = UsageType(row.usage_type)
            aggregated[usage_type] = {
                "total_quantity": float(row.total_quantity or 0),
                "total_cost": float(row.total_cost or 0),
                "record_count": row.record_count or 0,
                "average_quantity": float(row.avg_quantity or 0),
            }

        # Ensure all usage types are represented
        for usage_type in UsageType:
            if usage_type not in aggregated:
                aggregated[usage_type] = {
                    "total_quantity": 0.0,
                    "total_cost": 0.0,
                    "record_count": 0,
                    "average_quantity": 0.0,
                }

        return aggregated

    async def get_incomplete_records(self, older_than: Timestamp) -> List[UsageRecord]:
        """Get incomplete usage records older than specified time.

        Args:
            older_than: Timestamp threshold

        Returns:
            List of incomplete usage records
        """
        stmt = (
            select(UsageRecordModel)
            .where(
                and_(
                    UsageRecordModel.period_end.is_(None),
                    UsageRecordModel.period_start < older_than.value,
                )
            )
            .order_by(UsageRecordModel.period_start.asc())
        )

        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())

        return self.mapper.to_domain_list(models)

    async def bulk_create(self, records: List[UsageRecord]) -> List[UsageRecord]:
        """Create multiple usage records at once.

        Args:
            records: List of usage record entities to create

        Returns:
            List of created usage record entities with IDs
        """
        if not records:
            return []

        try:
            # Convert to persistence models
            models = [self.mapper.to_persistence(record) for record in records]

            # Add all models to session in batches for better performance
            batch_size = 1000  # Adjust based on your database
            for i in range(0, len(models), batch_size):
                batch = models[i : i + batch_size]
                self.session.add_all(batch)
                await self.session.flush()  # Flush to get IDs

            # Refresh all models to get complete data
            for model in models:
                await self.session.refresh(model)

            # Convert back to domain entities
            return self.mapper.to_domain_list(models)

        except Exception:
            await self.session.rollback()
            raise

    async def get_usage_analytics(
        self,
        user_id: Optional[int] = None,
        organization_id: Optional[int] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get usage analytics for a user or organization.

        Args:
            user_id: Optional user ID
            organization_id: Optional organization ID
            days: Number of days to analyze

        Returns:
            Dictionary with usage analytics
        """
        from datetime import datetime, timedelta

        start_date = datetime.utcnow() - timedelta(days=days)

        # Build base query
        stmt = select(UsageRecordModel).where(
            UsageRecordModel.period_start >= start_date
        )

        if user_id:
            stmt = stmt.where(UsageRecordModel.user_id == user_id)

        if organization_id:
            stmt = stmt.where(UsageRecordModel.organization_id == organization_id)

        result = await self.session.execute(stmt)
        records = list(result.scalars().unique())

        if not records:
            return {
                "period_days": days,
                "total_records": 0,
                "by_type": {},
                "total_cost": 0.0,
                "billable_cost": 0.0,
            }

        # Aggregate by type
        by_type = {}
        total_cost = 0.0
        billable_cost = 0.0

        for record_model in records:
            usage_type = record_model.usage_type
            if usage_type not in by_type:
                by_type[usage_type] = {
                    "record_count": 0,
                    "total_quantity": 0.0,
                    "total_cost": 0.0,
                    "billable_records": 0,
                    "billable_cost": 0.0,
                }

            by_type[usage_type]["record_count"] += 1
            by_type[usage_type]["total_quantity"] += record_model.quantity or 0
            by_type[usage_type]["total_cost"] += record_model.total_cost or 0

            total_cost += record_model.total_cost or 0

            if record_model.billable:
                by_type[usage_type]["billable_records"] += 1
                by_type[usage_type]["billable_cost"] += record_model.total_cost or 0
                billable_cost += record_model.total_cost or 0

        return {
            "period_days": days,
            "total_records": len(records),
            "by_type": by_type,
            "total_cost": total_cost,
            "billable_cost": billable_cost,
            "non_billable_cost": total_cost - billable_cost,
        }

    async def get_top_consumers(
        self, usage_type: UsageType, days: int = 30, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top consumers by usage type.

        Args:
            usage_type: Usage type to analyze
            days: Number of days to analyze
            limit: Maximum number of results

        Returns:
            List of top consumers with usage data
        """
        from datetime import datetime, timedelta

        start_date = datetime.utcnow() - timedelta(days=days)

        stmt = (
            select(
                UsageRecordModel.user_id,
                func.sum(UsageRecordModel.quantity).label("total_quantity"),
                func.sum(UsageRecordModel.total_cost).label("total_cost"),
                func.count().label("record_count"),
            )
            .where(
                and_(
                    UsageRecordModel.usage_type == usage_type.value,
                    UsageRecordModel.period_start >= start_date,
                )
            )
            .group_by(UsageRecordModel.user_id)
            .order_by(func.sum(UsageRecordModel.quantity).desc())
            .limit(limit)
        )

        result = await self.session.execute(stmt)

        top_consumers = []
        for row in result:
            top_consumers.append(
                {
                    "user_id": row.user_id,
                    "total_quantity": float(row.total_quantity or 0),
                    "total_cost": float(row.total_cost or 0),
                    "record_count": row.record_count or 0,
                }
            )

        return top_consumers

    async def cleanup_old_records(
        self, older_than: Timestamp, keep_billable: bool = True
    ) -> int:
        """Clean up old usage records.

        Args:
            older_than: Timestamp before which to clean up
            keep_billable: If True, keep billable records for auditing

        Returns:
            Number of records cleaned up
        """
        # Find old records to clean up
        stmt = select(UsageRecordModel).where(
            UsageRecordModel.period_start < older_than.value
        )

        if keep_billable:
            stmt = stmt.where(UsageRecordModel.billable.is_(False))

        result = await self.session.execute(stmt)
        old_records = list(result.scalars().unique())

        # Archive before deleting (in a real implementation)
        # For now, just delete
        for record in old_records:
            await self.session.delete(record)

        await self.session.flush()
        return len(old_records)

    async def get_billing_summary(
        self,
        user_id: int,
        billing_period_start: Timestamp,
        billing_period_end: Timestamp,
    ) -> Dict[str, Any]:
        """Get billing summary for a user and period.

        Args:
            user_id: User ID
            billing_period_start: Start of billing period
            billing_period_end: End of billing period

        Returns:
            Dictionary with billing summary
        """
        # Get billable records
        billable_records = await self.get_billable_usage(
            user_id, billing_period_start, billing_period_end
        )

        if not billable_records:
            return {
                "user_id": user_id,
                "billing_period": {
                    "start": billing_period_start.iso_string,
                    "end": billing_period_end.iso_string,
                },
                "total_cost": 0.0,
                "by_type": {},
                "record_count": 0,
            }

        # Aggregate by type
        by_type = {}
        total_cost = 0.0

        for record in billable_records:
            usage_type = record.usage_type.value
            if usage_type not in by_type:
                by_type[usage_type] = {
                    "quantity": 0.0,
                    "cost": 0.0,
                    "records": 0,
                    "unit": record.unit,
                }

            by_type[usage_type]["quantity"] += record.quantity
            by_type[usage_type]["cost"] += record.total_cost or 0.0
            by_type[usage_type]["records"] += 1

            total_cost += record.total_cost or 0.0

        return {
            "user_id": user_id,
            "billing_period": {
                "start": billing_period_start.iso_string,
                "end": billing_period_end.iso_string,
            },
            "total_cost": total_cost,
            "by_type": by_type,
            "record_count": len(billable_records),
        }
