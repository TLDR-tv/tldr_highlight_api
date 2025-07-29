"""API key mapper for domain entity to persistence model conversion."""

import json
from typing import Optional

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.api_key import APIKey as DomainAPIKey
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.api_key import APIKey as PersistenceAPIKey


class APIKeyMapper(Mapper[DomainAPIKey, PersistenceAPIKey]):
    """Maps between APIKey domain entity and persistence model."""
    
    def to_domain(self, model: PersistenceAPIKey) -> DomainAPIKey:
        """Convert APIKey persistence model to domain entity."""
        # Parse scopes and allowed IPs
        scopes = json.loads(model.scopes) if model.scopes else []
        allowed_ips = json.loads(model.allowed_ips) if model.allowed_ips else []
        
        return DomainAPIKey(
            id=model.id,
            name=model.name,
            key_hash=model.key_hash,
            user_id=model.user_id,
            scopes=scopes,
            description=model.description,
            expires_at=Timestamp(model.expires_at) if model.expires_at else None,
            last_used_at=Timestamp(model.last_used_at) if model.last_used_at else None,
            rate_limit_override=model.rate_limit_override,
            allowed_ips=allowed_ips,
            is_active=model.is_active,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at)
        )
    
    def to_persistence(self, entity: DomainAPIKey) -> PersistenceAPIKey:
        """Convert APIKey domain entity to persistence model."""
        model = PersistenceAPIKey()
        
        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id
        
        model.name = entity.name
        model.key_hash = entity.key_hash
        model.user_id = entity.user_id
        
        # Serialize scopes and allowed IPs
        model.scopes = json.dumps(entity.scopes)
        model.allowed_ips = json.dumps(entity.allowed_ips) if entity.allowed_ips else '[]'
        
        # Set optional attributes
        model.description = entity.description
        model.expires_at = entity.expires_at.value if entity.expires_at else None
        model.last_used_at = entity.last_used_at.value if entity.last_used_at else None
        model.rate_limit_override = entity.rate_limit_override
        model.is_active = entity.is_active
        
        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value
        
        return model