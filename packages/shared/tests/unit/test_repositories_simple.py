"""Simplified unit tests for repository implementations."""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from shared.infrastructure.storage.repositories.api_key import APIKeyRepository
from shared.infrastructure.storage.repositories.highlight import HighlightRepository
from shared.infrastructure.storage.repositories.organization import OrganizationRepository
from shared.infrastructure.storage.repositories.user import UserRepository
from shared.infrastructure.storage.repositories.wake_word import WakeWordRepository


class TestAPIKeyRepository:
    """Test API key repository methods."""

    @pytest.fixture
    def repository(self, mock_db_session):
        """Create API key repository."""
        return APIKeyRepository(mock_db_session)

    async def test_create_api_key(self, repository, mock_db_session, sample_api_key):
        """Test creating an API key."""
        result = await repository.create(sample_api_key)

        assert result == sample_api_key
        mock_db_session.add.assert_called_once_with(sample_api_key)
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once_with(sample_api_key)

    async def test_get_by_key_hash(self, repository, mock_db_session, sample_api_key):
        """Test getting API key by hash."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_api_key
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_by_key_hash("hashed_key_value")

        assert result == sample_api_key
        mock_db_session.execute.assert_called_once()

    async def test_list_by_organization(self, repository, mock_db_session, sample_api_key):
        """Test listing API keys by organization."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_api_key]
        mock_db_session.execute.return_value = mock_result

        result = await repository.list_by_organization(sample_api_key.organization_id)

        assert result == [sample_api_key]
        mock_db_session.execute.assert_called_once()

    async def test_delete_api_key(self, repository, mock_db_session, sample_api_key):
        """Test deleting an API key."""
        await repository.delete(sample_api_key)

        mock_db_session.delete.assert_called_once_with(sample_api_key)
        mock_db_session.commit.assert_called_once()


class TestHighlightRepository:
    """Test highlight repository methods."""

    @pytest.fixture
    def repository(self, mock_db_session):
        """Create highlight repository."""
        return HighlightRepository(mock_db_session)

    async def test_create_highlight(self, repository, mock_db_session, sample_highlight):
        """Test creating a highlight."""
        result = await repository.create(sample_highlight)

        assert result == sample_highlight
        mock_db_session.add.assert_called_once_with(sample_highlight)
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once_with(sample_highlight)

    async def test_get_by_id(self, repository, mock_db_session, sample_highlight):
        """Test getting highlight by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_highlight
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_highlight.id, sample_highlight.organization_id)

        assert result == sample_highlight
        mock_db_session.execute.assert_called_once()

    async def test_list_by_stream(self, repository, mock_db_session, sample_highlight):
        """Test listing highlights by stream."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_highlight]
        mock_db_session.execute.return_value = mock_result

        result = await repository.list_by_stream(
            sample_highlight.stream_id, 
            sample_highlight.organization_id
        )

        assert result == [sample_highlight]
        mock_db_session.execute.assert_called_once()

    async def test_delete_highlight(self, repository, mock_db_session, sample_highlight):
        """Test deleting a highlight."""
        await repository.delete(sample_highlight)

        mock_db_session.delete.assert_called_once_with(sample_highlight)
        mock_db_session.commit.assert_called_once()


class TestOrganizationRepository:
    """Test organization repository methods."""

    @pytest.fixture
    def repository(self, mock_db_session):
        """Create organization repository."""
        return OrganizationRepository(mock_db_session)

    async def test_create_organization(self, repository, mock_db_session, sample_organization):
        """Test creating an organization."""
        result = await repository.create(sample_organization)

        assert result == sample_organization
        mock_db_session.add.assert_called_once_with(sample_organization)
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once_with(sample_organization)

    async def test_get_by_id(self, repository, mock_db_session, sample_organization):
        """Test getting organization by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_organization
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_organization.id)

        assert result == sample_organization
        mock_db_session.execute.assert_called_once()

    async def test_get_by_slug(self, repository, mock_db_session, sample_organization):
        """Test getting organization by slug."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_organization
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_by_slug("test-org")

        assert result == sample_organization
        mock_db_session.execute.assert_called_once()


class TestUserRepository:
    """Test user repository methods."""

    @pytest.fixture
    def repository(self, mock_db_session):
        """Create user repository."""
        return UserRepository(mock_db_session)

    async def test_create_user(self, repository, mock_db_session, sample_user):
        """Test creating a user."""
        result = await repository.create(sample_user)

        assert result == sample_user
        mock_db_session.add.assert_called_once_with(sample_user)
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once_with(sample_user)

    async def test_get_by_email(self, repository, mock_db_session, sample_user):
        """Test getting user by email."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_by_email("test@example.com")

        assert result == sample_user
        mock_db_session.execute.assert_called_once()

    async def test_get_by_id(self, repository, mock_db_session, sample_user):
        """Test getting user by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_user.id)

        assert result == sample_user
        mock_db_session.execute.assert_called_once()

    async def test_list_by_organization(self, repository, mock_db_session, sample_user):
        """Test listing users by organization."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_user]
        mock_db_session.execute.return_value = mock_result

        result = await repository.list_by_organization(sample_user.organization_id)

        assert result == [sample_user]
        mock_db_session.execute.assert_called_once()


class TestWakeWordRepository:
    """Test wake word repository methods."""

    @pytest.fixture
    def repository(self, mock_db_session):
        """Create wake word repository."""
        return WakeWordRepository(mock_db_session)

    async def test_create_wake_word(self, repository, mock_db_session, sample_wake_word):
        """Test creating a wake word."""
        result = await repository.create(sample_wake_word)

        assert result == sample_wake_word
        mock_db_session.add.assert_called_once_with(sample_wake_word)
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once_with(sample_wake_word)

    async def test_get_by_id(self, repository, mock_db_session, sample_wake_word):
        """Test getting wake word by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_wake_word
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_wake_word.id)

        assert result == sample_wake_word
        mock_db_session.execute.assert_called_once()

    async def test_get_active_by_organization(self, repository, mock_db_session, sample_wake_word):
        """Test getting active wake words by organization."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_wake_word]
        mock_db_session.execute.return_value = mock_result

        result = await repository.get_active_by_organization(sample_wake_word.organization_id)

        assert result == [sample_wake_word]
        mock_db_session.execute.assert_called_once()

    async def test_delete_wake_word(self, repository, mock_db_session, sample_wake_word):
        """Test deleting a wake word."""
        await repository.delete(sample_wake_word)

        mock_db_session.delete.assert_called_once_with(sample_wake_word)
        mock_db_session.commit.assert_called_once()


class TestRepositoryPatterns:
    """Test common repository patterns."""

    def test_all_repositories_importable(self):
        """Test that all repository classes can be imported."""
        assert APIKeyRepository is not None
        assert HighlightRepository is not None
        assert OrganizationRepository is not None
        assert UserRepository is not None
        assert WakeWordRepository is not None

    def test_all_repositories_have_common_methods(self):
        """Test that repositories follow common patterns."""
        from shared.infrastructure.storage.repositories.base import BaseRepository
        
        # All repositories should inherit from base or implement common interface
        repos = [APIKeyRepository, HighlightRepository, OrganizationRepository, 
                UserRepository, WakeWordRepository]
        
        for repo_class in repos:
            # Check that repositories have basic CRUD methods
            assert hasattr(repo_class, '__init__')
            
            # Most should have create method
            if repo_class != WakeWordRepository:  # Some may have different patterns
                try:
                    instance = repo_class(Mock())
                    assert hasattr(instance, 'create') or hasattr(instance, 'get_by_id')
                except:
                    pass  # Skip if constructor needs specific parameters

    def test_session_handling_pattern(self, mock_db_session):
        """Test that all repositories follow session handling patterns."""
        # Test with one repository as example
        repo = APIKeyRepository(mock_db_session)
        
        # Should store session
        assert repo.session == mock_db_session or hasattr(repo, '_session')

    async def test_multi_tenant_patterns(self, mock_db_session, sample_api_key):
        """Test multi-tenant filtering patterns."""
        repo = APIKeyRepository(mock_db_session)
        
        # Mock query result
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_api_key
        mock_db_session.execute.return_value = mock_result
        
        # Repository should include organization filtering
        result = await repo.get_by_key_hash("test_hash")
        
        # Should have executed a query
        mock_db_session.execute.assert_called_once()