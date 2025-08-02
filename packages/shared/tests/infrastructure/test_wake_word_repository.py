"""Tests for wake word repository."""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.domain.models.wake_word import WakeWord
from shared.infrastructure.storage.repositories.wake_word import WakeWordRepository
from shared.infrastructure.database.models import WakeWordModel, OrganizationModel


@pytest.mark.asyncio
class TestWakeWordRepository:
    """Test wake word repository functionality."""
    
    @pytest.fixture
    async def organization(self, db_session: AsyncSession):
        """Create test organization."""
        org = OrganizationModel(
            id=uuid4(),
            name="Test Organization",
            slug="test-org",
        )
        db_session.add(org)
        await db_session.commit()
        return org
    
    @pytest.fixture
    async def repository(self, db_session: AsyncSession):
        """Create wake word repository instance."""
        return WakeWordRepository(db_session)
    
    async def test_create_wake_word(self, repository, organization):
        """Test creating a wake word."""
        wake_word = WakeWord(
            organization_id=organization.id,
            phrase="hey assistant",
            case_sensitive=False,
            max_edit_distance=2,
            similarity_threshold=0.8,
            pre_roll_seconds=10,
            post_roll_seconds=30,
        )
        
        created = await repository.create(wake_word)
        
        assert created.id == wake_word.id
        assert created.phrase == "hey assistant"
        assert created.organization_id == organization.id
        assert created.max_edit_distance == 2
        assert created.similarity_threshold == 0.8
        assert created.pre_roll_seconds == 10
        assert created.post_roll_seconds == 30
    
    async def test_get_by_id(self, repository, organization):
        """Test getting wake word by ID."""
        wake_word = WakeWord(
            organization_id=organization.id,
            phrase="test phrase",
        )
        
        created = await repository.create(wake_word)
        fetched = await repository.get_by_id(created.id)
        
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.phrase == "test phrase"
    
    async def test_get_by_id_not_found(self, repository):
        """Test getting non-existent wake word."""
        result = await repository.get(uuid4())
        assert result is None
    
    async def test_get_active_by_organization(self, repository, organization):
        """Test getting active wake words for organization."""
        # Create multiple wake words
        active1 = await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="activate",
            is_active=True,
        ))
        
        active2 = await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="start recording",
            is_active=True,
        ))
        
        inactive = await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="deactivated",
            is_active=False,
        ))
        
        # Get active wake words
        active_words = await repository.get_active_by_organization(organization.id)
        
        assert len(active_words) == 2
        assert all(w.is_active for w in active_words)
        assert {w.phrase for w in active_words} == {"activate", "start recording"}
    
    async def test_list_by_organization(self, repository, organization):
        """Test listing all wake words for organization."""
        # Create wake words for different organizations
        org2_id = uuid4()
        
        wake1 = await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="org1 word",
        ))
        
        wake2 = await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="org1 another",
        ))
        
        wake3 = await repository.create(WakeWord(
            organization_id=org2_id,
            phrase="org2 word",
        ))
        
        # List for organization
        org_words = await repository.list_by_organization(organization.id)
        
        assert len(org_words) == 2
        assert all(w.organization_id == organization.id for w in org_words)
        assert {w.phrase for w in org_words} == {"org1 word", "org1 another"}
    
    async def test_get_active_words_phrases(self, repository, organization):
        """Test getting list of active wake word phrases."""
        await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="active one",
            is_active=True,
        ))
        
        await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="active two",
            is_active=True,
        ))
        
        await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="inactive",
            is_active=False,
        ))
        
        phrases = await repository.get_active_words(organization.id)
        
        assert len(phrases) == 2
        assert set(phrases) == {"active one", "active two"}
    
    async def test_update_wake_word(self, repository, organization):
        """Test updating a wake word."""
        wake_word = WakeWord(
            organization_id=organization.id,
            phrase="original",
            is_active=True,
            cooldown_seconds=30,
            trigger_count=5,
        )
        
        created = await repository.create(wake_word)
        
        # Update the wake word
        created.phrase = "updated phrase"
        created.is_active = False
        created.cooldown_seconds = 60
        created.trigger_count = 10
        created.last_triggered_at = datetime.now(timezone.utc)
        
        updated = await repository.update(created)
        
        assert updated.phrase == "updated phrase"
        assert updated.is_active is False
        assert updated.cooldown_seconds == 60
        assert updated.trigger_count == 10
        assert updated.last_triggered_at is not None
    
    async def test_update_nonexistent_wake_word(self, repository):
        """Test updating non-existent wake word."""
        fake_wake_word = WakeWord(
            id=uuid4(),
            organization_id=uuid4(),
            phrase="doesn't exist",
        )
        
        with pytest.raises(ValueError, match="not found"):
            await repository.update(fake_wake_word)
    
    async def test_delete_wake_word(self, repository, organization):
        """Test deleting a wake word."""
        wake_word = WakeWord(
            organization_id=organization.id,
            phrase="to be deleted",
        )
        
        created = await repository.create(wake_word)
        await repository.delete(created.id)
        
        # Verify it's deleted
        fetched = await repository.get(created.id)
        assert fetched is None
    
    async def test_delete_nonexistent_wake_word(self, repository):
        """Test deleting non-existent wake word."""
        # Should not raise exception
        await repository.delete(uuid4())
    
    async def test_list_with_filters(self, repository, organization):
        """Test listing wake words with filters."""
        org2_id = uuid4()
        
        # Create various wake words
        await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="org1 active",
            is_active=True,
        ))
        
        await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="org1 inactive",
            is_active=False,
        ))
        
        await repository.create(WakeWord(
            organization_id=org2_id,
            phrase="org2 active",
            is_active=True,
        ))
        
        # Filter by organization
        org_words = await repository.list(organization_id=organization.id)
        assert len(org_words) == 2
        assert all(w.organization_id == organization.id for w in org_words)
        
        # Filter by active status
        active_words = await repository.list(is_active=True)
        assert len(active_words) == 2
        assert all(w.is_active for w in active_words)
        
        # Combined filters
        org_active = await repository.list(
            organization_id=organization.id,
            is_active=True,
        )
        assert len(org_active) == 1
        assert org_active[0].phrase == "org1 active"
    
    async def test_unique_constraint(self, repository, organization, db_session):
        """Test unique constraint on organization + phrase."""
        # Create first wake word
        await repository.create(WakeWord(
            organization_id=organization.id,
            phrase="duplicate test",
        ))
        
        # Try to create duplicate
        duplicate = WakeWord(
            organization_id=organization.id,
            phrase="duplicate test",
        )
        
        with pytest.raises(Exception):  # Should raise integrity error
            await repository.create(duplicate)
            await db_session.commit()
    
    async def test_case_sensitivity_preservation(self, repository, organization):
        """Test that case is preserved based on case_sensitive flag."""
        # Case insensitive (should normalize)
        wake1 = WakeWord(
            organization_id=organization.id,
            phrase="HeY AsSiStAnT",
            case_sensitive=False,
        )
        created1 = await repository.create(wake1)
        assert created1.phrase == "hey assistant"
        
        # Case sensitive (should preserve)
        wake2 = WakeWord(
            organization_id=organization.id,
            phrase="HeY BoT",
            case_sensitive=True,
        )
        created2 = await repository.create(wake2)
        assert created2.phrase == "HeY BoT"
    
    async def test_timestamp_handling(self, repository, organization):
        """Test timestamp handling in repository."""
        wake_word = WakeWord(
            organization_id=organization.id,
            phrase="test timestamps",
        )
        
        created = await repository.create(wake_word)
        
        # Verify timestamps are set
        assert created.created_at is not None
        assert created.updated_at is not None
        assert created.last_triggered_at is None
        
        # Update with trigger
        created.record_trigger()
        updated = await repository.update(created)
        
        assert updated.last_triggered_at is not None
        assert updated.trigger_count == 1
        assert updated.updated_at > created.created_at
    
    async def test_configuration_limits(self, repository, organization):
        """Test configuration parameter limits are preserved."""
        wake_word = WakeWord(
            organization_id=organization.id,
            phrase="test limits",
            max_edit_distance=0,  # Minimum
            similarity_threshold=1.0,  # Maximum
            pre_roll_seconds=0,  # Minimum
            post_roll_seconds=3600,  # Large value
            cooldown_seconds=0,  # No cooldown
        )
        
        created = await repository.create(wake_word)
        
        assert created.max_edit_distance == 0
        assert created.similarity_threshold == 1.0
        assert created.pre_roll_seconds == 0
        assert created.post_roll_seconds == 3600
        assert created.cooldown_seconds == 0
    
    async def test_bulk_operations(self, repository, organization):
        """Test bulk operations performance."""
        # Create many wake words
        wake_words = []
        for i in range(50):
            wake_word = WakeWord(
                organization_id=organization.id,
                phrase=f"bulk test {i}",
                is_active=i % 2 == 0,  # Half active, half inactive
            )
            wake_words.append(await repository.create(wake_word))
        
        # Test bulk retrieval
        all_words = await repository.list_by_organization(organization.id)
        assert len(all_words) == 50
        
        active_words = await repository.get_active_by_organization(organization.id)
        assert len(active_words) == 25
        
        # Test filtering performance
        filtered = await repository.list(
            organization_id=organization.id,
            is_active=True,
        )
        assert len(filtered) == 25


class TestWakeWordRepositoryEdgeCases:
    """Test edge cases for wake word repository."""
    
    @pytest.fixture
    async def repository(self, db_session: AsyncSession):
        """Create wake word repository instance."""
        return WakeWordRepository(db_session)
    
    @pytest.mark.asyncio
    async def test_empty_phrase(self, repository):
        """Test handling empty phrase."""
        wake_word = WakeWord(
            organization_id=uuid4(),
            phrase="",  # Empty phrase
        )
        
        created = await repository.create(wake_word)
        assert created.phrase == ""
    
    @pytest.mark.asyncio
    async def test_very_long_phrase(self, repository):
        """Test handling very long phrase."""
        long_phrase = " ".join(["word"] * 100)  # Very long phrase
        wake_word = WakeWord(
            organization_id=uuid4(),
            phrase=long_phrase[:500],  # Limit to column size
        )
        
        created = await repository.create(wake_word)
        assert len(created.phrase) <= 500
    
    @pytest.mark.asyncio
    async def test_special_characters_in_phrase(self, repository):
        """Test handling special characters."""
        wake_word = WakeWord(
            organization_id=uuid4(),
            phrase="test!@#$%^&*()_+-={}[]|\\:\";<>?,./",
        )
        
        created = await repository.create(wake_word)
        assert "!@#$%^&*" in created.phrase
    
    @pytest.mark.asyncio
    async def test_unicode_in_phrase(self, repository):
        """Test handling unicode characters."""
        wake_word = WakeWord(
            organization_id=uuid4(),
            phrase="测试 テスト тест 🎯",
        )
        
        created = await repository.create(wake_word)
        assert "测试" in created.phrase
        assert "🎯" in created.phrase
    
    @pytest.mark.asyncio
    async def test_null_last_triggered_at(self, repository):
        """Test handling null last_triggered_at."""
        wake_word = WakeWord(
            organization_id=uuid4(),
            phrase="test null",
            last_triggered_at=None,
        )
        
        created = await repository.create(wake_word)
        assert created.last_triggered_at is None
        
        # Update with timestamp
        created.last_triggered_at = datetime.now(timezone.utc)
        updated = await repository.update(created)
        assert updated.last_triggered_at is not None
        
        # Update back to None
        updated.last_triggered_at = None
        updated2 = await repository.update(updated)
        assert updated2.last_triggered_at is None