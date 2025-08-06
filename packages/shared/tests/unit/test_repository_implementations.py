"""Unit tests for repository implementations with mocked database."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4, UUID
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from shared.infrastructure.storage.repositories.api_key import APIKeyRepository
from shared.infrastructure.storage.repositories.highlight import HighlightRepository
from shared.infrastructure.storage.repositories.organization import OrganizationRepository
from shared.infrastructure.storage.repositories.user import UserRepository
from shared.infrastructure.storage.repositories.wake_word import WakeWordRepository
from shared.domain.models.api_key import APIKey, APIScopes
from shared.domain.models.highlight import Highlight
from shared.domain.models.organization import Organization
from shared.domain.models.user import User, UserRole
from shared.domain.models.wake_word import WakeWord


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
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_key_hash("hashed_key_value")

        assert result == sample_api_key
        mock_session.execute.assert_called_once()

    async def test_get_by_key_hash_not_found(self, repository, mock_session):
        """Test getting API key by hash when not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_key_hash("nonexistent_hash")

        assert result is None

    async def test_list_by_organization(self, repository, mock_session, sample_api_key):
        """Test listing API keys by organization."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_api_key]
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_organization(sample_api_key.organization_id)

        assert result == [sample_api_key]
        mock_session.execute.assert_called_once()

    async def test_update_api_key(self, repository, mock_session, sample_api_key):
        """Test updating an API key."""
        sample_api_key.name = "Updated Name"
        sample_api_key.description = "Updated description"

        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.update(sample_api_key)

        assert result == sample_api_key
        assert sample_api_key.name == "Updated Name"
        assert sample_api_key.description == "Updated description"
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_api_key)

    async def test_delete_api_key(self, repository, mock_session, sample_api_key):
        """Test deleting an API key."""
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        await repository.delete(sample_api_key)

        mock_session.delete.assert_called_once_with(sample_api_key)
        mock_session.commit.assert_called_once()


class TestHighlightRepository:
    """Test highlight repository methods."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create highlight repository."""
        return HighlightRepository(mock_session)

    @pytest.fixture
    def sample_highlight(self):
        """Sample highlight for testing."""
        return Highlight(
            id=uuid4(),
            stream_id=uuid4(),
            organization_id=uuid4(),
            start_time=10.0,
            end_time=25.0,
            title="Test Highlight",
            overall_score=8.5,
            clip_path="/path/to/clip.mp4",
        )

    async def test_create_highlight(self, repository, mock_session, sample_highlight):
        """Test creating a highlight."""
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_highlight)

        assert result == sample_highlight
        mock_session.add.assert_called_once_with(sample_highlight)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_highlight)

    async def test_get_by_id(self, repository, mock_session, sample_highlight):
        """Test getting highlight by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_highlight
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_highlight.id, sample_highlight.organization_id)

        assert result == sample_highlight
        mock_session.execute.assert_called_once()

    async def test_list_by_stream(self, repository, mock_session, sample_highlight):
        """Test listing highlights by stream."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_highlight]
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_stream(
            sample_highlight.stream_id, 
            sample_highlight.organization_id
        )

        assert result == [sample_highlight]
        mock_session.execute.assert_called_once()

    async def test_list_by_organization_with_pagination(self, repository, mock_session, sample_highlight):
        """Test listing highlights by organization with pagination."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_highlight]
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_organization(
            sample_highlight.organization_id,
            page=1,
            page_size=10
        )

        assert result == [sample_highlight]
        mock_session.execute.assert_called_once()

    async def test_count_by_organization(self, repository, mock_session, sample_highlight):
        """Test counting highlights by organization."""
        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        result = await repository.count_by_organization(sample_highlight.organization_id)

        assert result == 5
        mock_session.execute.assert_called_once()

    async def test_delete_highlight(self, repository, mock_session, sample_highlight):
        """Test deleting a highlight."""
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        await repository.delete(sample_highlight)

        mock_session.delete.assert_called_once_with(sample_highlight)
        mock_session.commit.assert_called_once()


class TestOrganizationRepository:
    """Test organization repository methods."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create organization repository."""
        return OrganizationRepository(mock_session)

    @pytest.fixture
    def sample_organization(self):
        """Sample organization for testing."""
        return Organization(
            id=uuid4(),
            name="Test Organization",
            slug="test-org",
            webhook_url="https://example.com/webhook",
            webhook_secret="secret123",
        )

    async def test_create_organization(self, repository, mock_session, sample_organization):
        """Test creating an organization."""
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_organization)

        assert result == sample_organization
        mock_session.add.assert_called_once_with(sample_organization)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_organization)

    async def test_get_by_id(self, repository, mock_session, sample_organization):
        """Test getting organization by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_organization
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_organization.id)

        assert result == sample_organization
        mock_session.execute.assert_called_once()

    async def test_get_by_slug(self, repository, mock_session, sample_organization):
        """Test getting organization by slug."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_organization
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_slug("test-org")

        assert result == sample_organization
        mock_session.execute.assert_called_once()

    async def test_get_by_slug_not_found(self, repository, mock_session):
        """Test getting organization by slug when not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_slug("nonexistent-slug")

        assert result is None

    async def test_update_organization(self, repository, mock_session, sample_organization):
        """Test updating an organization."""
        sample_organization.name = "Updated Organization"
        sample_organization.webhook_url = "https://new-webhook.com/webhook"

        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.update(sample_organization)

        assert result == sample_organization
        assert sample_organization.name == "Updated Organization"
        assert sample_organization.webhook_url == "https://new-webhook.com/webhook"
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_organization)


class TestUserRepository:
    """Test user repository methods."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create user repository."""
        return UserRepository(mock_session)

    @pytest.fixture
    def sample_user(self):
        """Sample user for testing."""
        return User(
            id=uuid4(),
            organization_id=uuid4(),
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            role=UserRole.MEMBER,
            is_active=True,
        )

    async def test_create_user(self, repository, mock_session, sample_user):
        """Test creating a user."""
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_user)

        assert result == sample_user
        mock_session.add.assert_called_once_with(sample_user)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_user)

    async def test_get_by_email(self, repository, mock_session, sample_user):
        """Test getting user by email."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_email("test@example.com")

        assert result == sample_user
        mock_session.execute.assert_called_once()

    async def test_get_by_email_not_found(self, repository, mock_session):
        """Test getting user by email when not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_email("nonexistent@example.com")

        assert result is None

    async def test_get_by_id(self, repository, mock_session, sample_user):
        """Test getting user by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_user.id)

        assert result == sample_user
        mock_session.execute.assert_called_once()

    async def test_list_by_organization(self, repository, mock_session, sample_user):
        """Test listing users by organization."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_user]
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_organization(sample_user.organization_id)

        assert result == [sample_user]
        mock_session.execute.assert_called_once()

    async def test_update_user(self, repository, mock_session, sample_user):
        """Test updating a user."""
        sample_user.full_name = "Updated User"
        sample_user.role = UserRole.ADMIN

        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.update(sample_user)

        assert result == sample_user
        assert sample_user.full_name == "Updated User"
        assert sample_user.role == UserRole.ADMIN
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_user)

    async def test_authenticate_user(self, repository, mock_session, sample_user):
        """Test authenticating a user."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_email_and_organization(
            "test@example.com", 
            sample_user.organization_id
        )

        assert result == sample_user
        mock_session.execute.assert_called_once()


class TestWakeWordRepository:
    """Test wake word repository methods."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create wake word repository."""
        return WakeWordRepository(mock_session)

    @pytest.fixture
    def sample_wake_word(self):
        """Sample wake word for testing."""
        return WakeWord(
            id=uuid4(),
            organization_id=uuid4(),
            phrase="hey assistant",
            case_sensitive=False,
            max_edit_distance=2,
            similarity_threshold=0.8,
            pre_roll_seconds=10,
            post_roll_seconds=30,
            is_active=True,
        )

    async def test_create_wake_word(self, repository, mock_session, sample_wake_word):
        """Test creating a wake word."""
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.create(sample_wake_word)

        assert result == sample_wake_word
        mock_session.add.assert_called_once_with(sample_wake_word)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_wake_word)

    async def test_get_by_id(self, repository, mock_session, sample_wake_word):
        """Test getting wake word by ID."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_wake_word
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(sample_wake_word.id)

        assert result == sample_wake_word
        mock_session.execute.assert_called_once()

    async def test_get_active_by_organization(self, repository, mock_session, sample_wake_word):
        """Test getting active wake words by organization."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_wake_word]
        mock_session.execute.return_value = mock_result

        result = await repository.get_active_by_organization(sample_wake_word.organization_id)

        assert result == [sample_wake_word]
        mock_session.execute.assert_called_once()

    async def test_list_by_organization(self, repository, mock_session, sample_wake_word):
        """Test listing wake words by organization."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_wake_word]
        mock_session.execute.return_value = mock_result

        result = await repository.list_by_organization(sample_wake_word.organization_id)

        assert result == [sample_wake_word]
        mock_session.execute.assert_called_once()

    async def test_get_active_words_phrases(self, repository, mock_session, sample_wake_word):
        """Test getting active wake word phrases."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_wake_word]
        mock_session.execute.return_value = mock_result

        result = await repository.get_active_words_phrases(sample_wake_word.organization_id)

        assert result == ["hey assistant"]
        mock_session.execute.assert_called_once()

    async def test_update_wake_word(self, repository, mock_session, sample_wake_word):
        """Test updating a wake word."""
        sample_wake_word.phrase = "updated phrase"
        sample_wake_word.similarity_threshold = 0.9

        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repository.update(sample_wake_word)

        assert result == sample_wake_word
        assert sample_wake_word.phrase == "updated phrase"
        assert sample_wake_word.similarity_threshold == 0.9
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(sample_wake_word)

    async def test_delete_wake_word(self, repository, mock_session, sample_wake_word):
        """Test deleting a wake word."""
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        await repository.delete(sample_wake_word)

        mock_session.delete.assert_called_once_with(sample_wake_word)
        mock_session.commit.assert_called_once()

    async def test_update_last_triggered(self, repository, mock_session, sample_wake_word):
        """Test updating last triggered timestamp."""
        triggered_at = datetime.now(timezone.utc)
        mock_session.commit = AsyncMock()

        await repository.update_last_triggered(sample_wake_word.id, triggered_at)

        assert sample_wake_word.last_triggered_at == triggered_at
        mock_session.commit.assert_called_once()