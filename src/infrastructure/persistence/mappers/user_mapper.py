"""User mapper for domain entity to persistence model conversion."""

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.user import User as DomainUser
from src.domain.value_objects.email import Email
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.user import User as PersistenceUser


class UserMapper(Mapper[DomainUser, PersistenceUser]):
    """Maps between User domain entity and persistence model."""

    def to_domain(self, model: PersistenceUser) -> DomainUser:
        """Convert User persistence model to domain entity."""
        return DomainUser(
            id=model.id,
            email=Email(model.email),
            company_name=CompanyName(model.company_name),
            password_hash=model.password_hash,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at),
            # Extract related IDs
            api_key_ids=[key.id for key in model.api_keys] if model.api_keys else [],
            organization_ids=[org.id for org in model.owned_organizations]
            if model.owned_organizations
            else [],
            stream_ids=[stream.id for stream in model.streams] if model.streams else [],
            batch_ids=[batch.id for batch in model.batches] if model.batches else [],
            webhook_ids=[webhook.id for webhook in model.webhooks]
            if model.webhooks
            else [],
        )

    def to_persistence(self, entity: DomainUser) -> PersistenceUser:
        """Convert User domain entity to persistence model."""
        model = PersistenceUser()

        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id

        model.email = entity.email.value
        model.company_name = entity.company_name.value
        model.password_hash = entity.password_hash

        # Set timestamps (ORM will handle these for new entities)
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value

        # Note: Related entities (api_keys, organizations, etc.)
        # should be handled by their respective repositories

        return model
