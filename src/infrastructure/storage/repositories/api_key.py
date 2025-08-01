"""API key repository implementation."""
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ....domain.models.api_key import APIKey
from ....domain.protocols import APIKeyRepository as APIKeyRepositoryProtocol
from ..models import APIKeyModel


class APIKeyRepository:
    """SQLAlchemy implementation of API key repository."""
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session
    
    async def add(self, entity: APIKey) -> APIKey:
        """Add API key to repository."""
        model = APIKeyModel(
            id=entity.id,
            organization_id=entity.organization_id,
            name=entity.name,
            key_hash=entity.key_hash,
            prefix=entity.prefix,
            scopes=list(entity.scopes),
            last_used_at=entity.last_used_at,
            usage_count=entity.usage_count,
            is_active=entity.is_active,
            expires_at=entity.expires_at,
            revoked_at=entity.revoked_at,
            created_by_user_id=entity.created_by_user_id,
            description=entity.description,
            created_at=entity.created_at
        )
        
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        
        return self._to_entity(model)
    
    async def get(self, id: UUID) -> Optional[APIKey]:
        """Get API key by ID."""
        model = await self.session.get(APIKeyModel, id)
        return self._to_entity(model) if model else None
    
    async def get_by_prefix(self, prefix: str) -> Optional[APIKey]:
        """Get API key by prefix."""
        result = await self.session.execute(
            select(APIKeyModel).where(APIKeyModel.prefix == prefix)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None
    
    async def get_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        result = await self.session.execute(
            select(APIKeyModel).where(APIKeyModel.key_hash == key_hash)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None
    
    async def list_by_organization(self, org_id: UUID) -> list[APIKey]:
        """List API keys for an organization."""
        result = await self.session.execute(
            select(APIKeyModel)
            .where(APIKeyModel.organization_id == org_id)
            .order_by(APIKeyModel.created_at.desc())
        )
        models = result.scalars().all()
        return [self._to_entity(model) for model in models]
    
    async def update(self, entity: APIKey) -> APIKey:
        """Update existing API key."""
        model = await self.session.get(APIKeyModel, entity.id)
        if not model:
            raise ValueError(f"API key {entity.id} not found")
        
        # Update fields
        model.name = entity.name
        model.scopes = list(entity.scopes)
        model.last_used_at = entity.last_used_at
        model.usage_count = entity.usage_count
        model.is_active = entity.is_active
        model.expires_at = entity.expires_at
        model.revoked_at = entity.revoked_at
        model.description = entity.description
        
        await self.session.commit()
        await self.session.refresh(model)
        
        return self._to_entity(model)
    
    async def delete(self, id: UUID) -> None:
        """Delete API key by ID."""
        model = await self.session.get(APIKeyModel, id)
        if model:
            await self.session.delete(model)
            await self.session.commit()
    
    async def list(self, **filters) -> list[APIKey]:
        """List API keys with optional filters."""
        query = select(APIKeyModel)
        
        if "is_active" in filters:
            query = query.where(APIKeyModel.is_active == filters["is_active"])
        if "organization_id" in filters:
            query = query.where(APIKeyModel.organization_id == filters["organization_id"])
        
        result = await self.session.execute(query.order_by(APIKeyModel.created_at.desc()))
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    def _to_entity(self, model: APIKeyModel) -> APIKey:
        """Convert model to entity."""
        return APIKey(
            id=model.id,
            organization_id=model.organization_id,
            name=model.name,
            key_hash=model.key_hash,
            prefix=model.prefix,
            scopes=set(model.scopes) if model.scopes else set(),
            last_used_at=model.last_used_at,
            usage_count=model.usage_count,
            is_active=model.is_active,
            expires_at=model.expires_at,
            created_at=model.created_at,
            revoked_at=model.revoked_at,
            created_by_user_id=model.created_by_user_id,
            description=model.description
        )