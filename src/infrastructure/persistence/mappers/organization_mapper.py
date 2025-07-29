"""Organization mapper for domain entity to persistence model conversion."""

import json
from typing import Optional

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.organization import Organization as DomainOrganization, PlanType
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.organization import Organization as PersistenceOrganization


class OrganizationMapper(Mapper[DomainOrganization, PersistenceOrganization]):
    """Maps between Organization domain entity and persistence model."""
    
    def to_domain(self, model: PersistenceOrganization) -> DomainOrganization:
        """Convert Organization persistence model to domain entity."""
        # Parse custom limits and settings
        custom_limits = json.loads(model.custom_limits) if model.custom_limits else None
        settings = json.loads(model.settings) if model.settings else {}
        
        # Extract member IDs (assuming a relationship exists)
        member_ids = []
        if hasattr(model, 'members'):
            member_ids = [member.id for member in model.members]
        
        return DomainOrganization(
            id=model.id,
            name=CompanyName(model.name),
            owner_id=model.owner_id,
            plan_type=PlanType(model.plan_type),
            member_ids=member_ids,
            custom_limits=custom_limits,
            settings=settings,
            subscription_started_at=Timestamp(model.subscription_started_at) if model.subscription_started_at else None,
            subscription_ends_at=Timestamp(model.subscription_ends_at) if model.subscription_ends_at else None,
            is_active=model.is_active,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at)
        )
    
    def to_persistence(self, entity: DomainOrganization) -> PersistenceOrganization:
        """Convert Organization domain entity to persistence model."""
        model = PersistenceOrganization()
        
        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id
        
        model.name = entity.name.value
        model.owner_id = entity.owner_id
        model.plan_type = entity.plan_type.value
        
        # Serialize custom limits and settings
        model.custom_limits = json.dumps(entity.custom_limits) if entity.custom_limits else None
        model.settings = json.dumps(entity.settings) if entity.settings else '{}'
        
        # Set subscription details
        model.subscription_started_at = entity.subscription_started_at.value if entity.subscription_started_at else None
        model.subscription_ends_at = entity.subscription_ends_at.value if entity.subscription_ends_at else None
        model.is_active = entity.is_active
        
        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value
        
        # Note: Member relationships should be handled by the repository
        
        return model