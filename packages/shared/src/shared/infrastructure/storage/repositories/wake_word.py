"""Wake word repository implementation."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ....domain.models.wake_word import WakeWord
from ...database.models import WakeWordModel


class WakeWordRepository:
    """SQLAlchemy implementation of wake word repository."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def create(self, wake_word: WakeWord) -> WakeWord:
        """Create a new wake word."""
        db_wake_word = WakeWordModel(
            id=wake_word.id,
            organization_id=wake_word.organization_id,
            phrase=wake_word.phrase,
            is_active=wake_word.is_active,
            case_sensitive=wake_word.case_sensitive,
            exact_match=wake_word.exact_match,
            cooldown_seconds=wake_word.cooldown_seconds,
            max_edit_distance=wake_word.max_edit_distance,
            similarity_threshold=wake_word.similarity_threshold,
            pre_roll_seconds=wake_word.pre_roll_seconds,
            post_roll_seconds=wake_word.post_roll_seconds,
            trigger_count=wake_word.trigger_count,
            last_triggered_at=wake_word.last_triggered_at,
            created_at=wake_word.created_at,
            updated_at=wake_word.updated_at,
        )
        
        self.session.add(db_wake_word)
        await self.session.commit()
        await self.session.refresh(db_wake_word)
        
        return self._to_domain(db_wake_word)
    
    async def add(self, entity: WakeWord) -> WakeWord:
        """Add wake word to repository (alias for create)."""
        return await self.create(entity)
    
    async def get_by_id(self, wake_word_id: UUID) -> Optional[WakeWord]:
        """Get wake word by ID."""
        result = await self.session.execute(
            select(WakeWordModel).where(WakeWordModel.id == wake_word_id)
        )
        db_wake_word = result.scalar_one_or_none()
        
        return self._to_domain(db_wake_word) if db_wake_word else None
    
    async def get(self, id: UUID) -> Optional[WakeWord]:
        """Get wake word by ID (alias for get_by_id)."""
        return await self.get_by_id(id)
    
    async def get_active_by_organization(
        self, organization_id: UUID
    ) -> List[WakeWord]:
        """Get all active wake words for an organization."""
        result = await self.session.execute(
            select(WakeWordModel).where(
                and_(
                    WakeWordModel.organization_id == organization_id,
                    WakeWordModel.is_active == True,
                )
            )
        )
        db_wake_words = result.scalars().all()
        
        return [self._to_domain(w) for w in db_wake_words]
    
    async def list_by_organization(self, org_id: UUID) -> list[WakeWord]:
        """List all wake words for an organization."""
        result = await self.session.execute(
            select(WakeWordModel).where(
                WakeWordModel.organization_id == org_id
            )
        )
        db_wake_words = result.scalars().all()
        
        return [self._to_domain(w) for w in db_wake_words]
    
    async def get_active_words(self, org_id: UUID) -> list[str]:
        """Get list of active wake word phrases."""
        wake_words = await self.get_active_by_organization(org_id)
        return [w.phrase for w in wake_words]
    
    async def update(self, wake_word: WakeWord) -> WakeWord:
        """Update an existing wake word."""
        result = await self.session.execute(
            select(WakeWordModel).where(WakeWordModel.id == wake_word.id)
        )
        db_wake_word = result.scalar_one_or_none()
        
        if not db_wake_word:
            raise ValueError(f"Wake word {wake_word.id} not found")
        
        # Update fields
        db_wake_word.phrase = wake_word.phrase
        db_wake_word.is_active = wake_word.is_active
        db_wake_word.case_sensitive = wake_word.case_sensitive
        db_wake_word.exact_match = wake_word.exact_match
        db_wake_word.cooldown_seconds = wake_word.cooldown_seconds
        db_wake_word.max_edit_distance = wake_word.max_edit_distance
        db_wake_word.similarity_threshold = wake_word.similarity_threshold
        db_wake_word.pre_roll_seconds = wake_word.pre_roll_seconds
        db_wake_word.post_roll_seconds = wake_word.post_roll_seconds
        db_wake_word.trigger_count = wake_word.trigger_count
        db_wake_word.last_triggered_at = wake_word.last_triggered_at
        db_wake_word.updated_at = wake_word.updated_at
        
        await self.session.commit()
        await self.session.refresh(db_wake_word)
        
        return self._to_domain(db_wake_word)
    
    async def delete(self, wake_word_id: UUID) -> None:
        """Delete a wake word."""
        result = await self.session.execute(
            select(WakeWordModel).where(WakeWordModel.id == wake_word_id)
        )
        db_wake_word = result.scalar_one_or_none()
        
        if db_wake_word:
            await self.session.delete(db_wake_word)
            await self.session.commit()
    
    async def list(self, **filters) -> list[WakeWord]:
        """List wake words with optional filters."""
        query = select(WakeWordModel)
        
        # Apply filters
        if "organization_id" in filters:
            query = query.where(WakeWordModel.organization_id == filters["organization_id"])
        if "is_active" in filters:
            query = query.where(WakeWordModel.is_active == filters["is_active"])
        
        result = await self.session.execute(query)
        db_wake_words = result.scalars().all()
        
        return [self._to_domain(w) for w in db_wake_words]
    
    def _to_domain(self, db_wake_word: WakeWordModel) -> WakeWord:
        """Convert database model to domain model."""
        return WakeWord(
            id=db_wake_word.id,
            organization_id=db_wake_word.organization_id,
            phrase=db_wake_word.phrase,
            is_active=db_wake_word.is_active,
            case_sensitive=db_wake_word.case_sensitive,
            exact_match=db_wake_word.exact_match,
            cooldown_seconds=db_wake_word.cooldown_seconds,
            max_edit_distance=db_wake_word.max_edit_distance,
            similarity_threshold=db_wake_word.similarity_threshold,
            pre_roll_seconds=db_wake_word.pre_roll_seconds,
            post_roll_seconds=db_wake_word.post_roll_seconds,
            trigger_count=db_wake_word.trigger_count,
            last_triggered_at=db_wake_word.last_triggered_at,
            created_at=db_wake_word.created_at,
            updated_at=db_wake_word.updated_at,
        )
