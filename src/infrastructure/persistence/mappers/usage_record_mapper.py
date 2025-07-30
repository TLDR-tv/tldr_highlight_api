"""Usage record mapper for domain entity to persistence model conversion."""

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.usage_record import UsageRecord as DomainUsageRecord, UsageType
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.usage_record import (
    UsageRecord as PersistenceUsageRecord,
)


class UsageRecordMapper(Mapper[DomainUsageRecord, PersistenceUsageRecord]):
    """Maps between UsageRecord domain entity and persistence model."""

    def to_domain(self, model: PersistenceUsageRecord) -> DomainUsageRecord:
        """Convert UsageRecord persistence model to domain entity."""
        return DomainUsageRecord(
            id=model.id,
            user_id=model.user_id,
            organization_id=model.organization_id,
            usage_type=UsageType(model.usage_type),
            resource_id=model.resource_id,
            resource_type=model.resource_type,
            quantity=model.quantity,
            unit=model.unit,
            period_start=Timestamp(model.period_start),
            period_end=Timestamp(model.period_end) if model.period_end else None,
            billable=model.billable,
            rate=model.rate,
            total_cost=model.total_cost,
            api_key_id=model.api_key_id,
            ip_address=model.ip_address,
            user_agent=model.user_agent,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at),
        )

    def to_persistence(self, entity: DomainUsageRecord) -> PersistenceUsageRecord:
        """Convert UsageRecord domain entity to persistence model."""
        model = PersistenceUsageRecord()

        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id

        model.user_id = entity.user_id
        model.organization_id = entity.organization_id
        model.usage_type = entity.usage_type.value

        # Set resource information
        model.resource_id = entity.resource_id
        model.resource_type = entity.resource_type

        # Set quantity and billing information
        model.quantity = entity.quantity
        model.unit = entity.unit
        model.billable = entity.billable
        model.rate = entity.rate
        model.total_cost = entity.total_cost

        # Set period timestamps
        model.period_start = entity.period_start.value
        model.period_end = entity.period_end.value if entity.period_end else None

        # Set metadata
        model.api_key_id = entity.api_key_id
        model.ip_address = entity.ip_address
        model.user_agent = entity.user_agent

        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value

        return model
