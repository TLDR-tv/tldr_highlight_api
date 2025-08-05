"""Unit tests for organization service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
from datetime import datetime, timezone
import secrets

from api.services.organization_service import OrganizationService
from api.services.user_service import UserService
from shared.domain.models.organization import Organization
from shared.domain.models.user import User, UserRole
from shared.infrastructure.storage.repositories import OrganizationRepository


class TestOrganizationService:
    """Test suite for OrganizationService."""

    @pytest.fixture
    def mock_organization_repository(self):
        """Create a mock organization repository."""
        repository = AsyncMock(spec=OrganizationRepository)
        return repository

    @pytest.fixture
    def mock_user_service(self):
        """Create a mock user service."""
        service = AsyncMock(spec=UserService)
        return service

    @pytest.fixture
    def organization_service(self, mock_organization_repository, mock_user_service):
        """Create an organization service with mocked dependencies."""
        return OrganizationService(
            organization_repository=mock_organization_repository,
            user_service=mock_user_service,
        )

    @pytest.fixture
    def sample_organization(self):
        """Create a sample organization for testing."""
        org = Organization(
            id=uuid4(),
            name="Test Organization",
            slug="test-organization",
            webhook_url="https://example.com/webhook",
            webhook_secret="test-secret",
            is_active=True,
        )
        # Set initial usage stats
        org.total_streams_processed = 10
        org.total_highlights_generated = 50
        org.total_processing_seconds = 3600
        return org

    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id=uuid4(),
            organization_id=uuid4(),
            email="owner@example.com",
            name="Owner User",
            role=UserRole.ADMIN,
            is_active=True,
        )

    @pytest.mark.asyncio
    async def test_create_organization_success(
        self,
        organization_service,
        mock_organization_repository,
        mock_user_service,
        sample_organization,
        sample_user,
    ):
        """Test successful organization creation with owner user."""
        # Arrange
        mock_organization_repository.get_by_slug.return_value = None
        mock_organization_repository.add.return_value = sample_organization
        mock_user_service.create_user.return_value = sample_user

        # Act
        with patch("secrets.token_urlsafe", return_value="mocked-secret"):
            org, user = await organization_service.create_organization(
                name="Test Organization",
                owner_email="owner@example.com",
                owner_name="Owner User",
                owner_password="secure_password123",
                webhook_url="https://example.com/webhook",
            )

        # Assert
        assert org == sample_organization
        assert user == sample_user
        mock_organization_repository.get_by_slug.assert_called_once_with("test-organization")
        mock_organization_repository.add.assert_called_once()
        mock_user_service.create_user.assert_called_once_with(
            organization_id=sample_organization.id,
            email="owner@example.com",
            name="Owner User",
            password="secure_password123",
            role=UserRole.ADMIN,
        )

    @pytest.mark.asyncio
    async def test_create_organization_without_webhook(
        self,
        organization_service,
        mock_organization_repository,
        mock_user_service,
        sample_organization,
        sample_user,
    ):
        """Test organization creation without webhook URL."""
        # Arrange
        sample_organization.webhook_url = None
        sample_organization.webhook_secret = None
        mock_organization_repository.get_by_slug.return_value = None
        mock_organization_repository.add.return_value = sample_organization
        mock_user_service.create_user.return_value = sample_user

        # Act
        org, user = await organization_service.create_organization(
            name="Test Organization",
            owner_email="owner@example.com",
            owner_name="Owner User",
            owner_password="secure_password123",
        )

        # Assert
        assert org.webhook_url is None
        assert org.webhook_secret is None

    @pytest.mark.asyncio
    async def test_create_organization_duplicate_name(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test organization creation with duplicate name."""
        # Arrange
        mock_organization_repository.get_by_slug.return_value = sample_organization

        # Act & Assert
        with pytest.raises(ValueError, match="Organization with similar name already exists"):
            await organization_service.create_organization(
                name="Test Organization",
                owner_email="owner@example.com",
                owner_name="Owner User",
                owner_password="secure_password123",
            )

        mock_organization_repository.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_organization_user_creation_fails(
        self,
        organization_service,
        mock_organization_repository,
        mock_user_service,
        sample_organization,
    ):
        """Test organization creation rollback when user creation fails."""
        # Arrange
        mock_organization_repository.get_by_slug.return_value = None
        mock_organization_repository.add.return_value = sample_organization
        mock_user_service.create_user.side_effect = Exception("User creation failed")

        # Act & Assert
        with pytest.raises(Exception, match="User creation failed"):
            await organization_service.create_organization(
                name="Test Organization",
                owner_email="owner@example.com",
                owner_name="Owner User",
                owner_password="secure_password123",
            )

        # Verify rollback
        mock_organization_repository.delete.assert_called_once_with(sample_organization.id)

    @pytest.mark.asyncio
    async def test_update_organization_success(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test successful organization update."""
        # Arrange
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.get_by_slug.return_value = None
        mock_organization_repository.update.return_value = sample_organization

        # Act
        result = await organization_service.update_organization(
            organization_id=sample_organization.id,
            name="Updated Organization",
            webhook_url="https://new.example.com/webhook",
            is_active=False,
        )

        # Assert
        assert result == sample_organization
        assert sample_organization.name == "Updated Organization"
        assert sample_organization.webhook_url == "https://new.example.com/webhook"
        assert sample_organization.is_active is False
        mock_organization_repository.update.assert_called_once_with(sample_organization)

    @pytest.mark.asyncio
    async def test_update_organization_not_found(
        self,
        organization_service,
        mock_organization_repository,
    ):
        """Test updating non-existent organization."""
        # Arrange
        mock_organization_repository.get.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Organization not found"):
            await organization_service.update_organization(
                organization_id=uuid4(),
                name="Updated Organization",
            )

    @pytest.mark.asyncio
    async def test_update_organization_duplicate_name(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test updating organization with duplicate name."""
        # Arrange
        other_org = Organization(name="Other Organization", slug="other-organization")
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.get_by_slug.return_value = other_org

        # Act & Assert
        with pytest.raises(ValueError, match="Organization with similar name already exists"):
            await organization_service.update_organization(
                organization_id=sample_organization.id,
                name="Other Organization",
            )

    @pytest.mark.asyncio
    async def test_update_organization_webhook_secret_generation(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test webhook secret generation when adding webhook URL."""
        # Arrange
        sample_organization.webhook_url = None
        sample_organization.webhook_secret = None
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.update.return_value = sample_organization

        # Act
        with patch("secrets.token_urlsafe", return_value="new-secret"):
            result = await organization_service.update_organization(
                organization_id=sample_organization.id,
                webhook_url="https://example.com/webhook",
            )

        # Assert
        assert sample_organization.webhook_url == "https://example.com/webhook"
        assert sample_organization.webhook_secret == "new-secret"

    @pytest.mark.asyncio
    async def test_get_usage_stats_success(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test getting usage statistics."""
        # Arrange
        mock_organization_repository.get.return_value = sample_organization

        # Act
        stats = await organization_service.get_usage_stats(sample_organization.id)

        # Assert
        assert stats["organization_id"] == str(sample_organization.id)
        assert stats["name"] == "Test Organization"
        assert stats["total_streams_processed"] == 10
        assert stats["total_highlights_generated"] == 50
        assert stats["total_processing_seconds"] == 3600
        assert stats["total_processing_hours"] == 1.0
        assert stats["avg_highlights_per_stream"] == 5.0
        assert stats["avg_processing_seconds_per_stream"] == 360.0
        assert stats["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_usage_stats_zero_streams(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test usage statistics with zero streams processed."""
        # Arrange
        sample_organization.total_streams_processed = 0
        sample_organization.total_highlights_generated = 0
        sample_organization.total_processing_seconds = 0
        mock_organization_repository.get.return_value = sample_organization

        # Act
        stats = await organization_service.get_usage_stats(sample_organization.id)

        # Assert
        assert stats["avg_highlights_per_stream"] == 0
        assert stats["avg_processing_seconds_per_stream"] == 0

    @pytest.mark.asyncio
    async def test_get_usage_stats_not_found(
        self,
        organization_service,
        mock_organization_repository,
    ):
        """Test getting usage stats for non-existent organization."""
        # Arrange
        mock_organization_repository.get.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Organization not found"):
            await organization_service.get_usage_stats(uuid4())

    @pytest.mark.asyncio
    async def test_regenerate_webhook_secret_success(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test successful webhook secret regeneration."""
        # Arrange
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.update.return_value = sample_organization

        # Act
        with patch("secrets.token_urlsafe", return_value="new-secret-123"):
            new_secret = await organization_service.regenerate_webhook_secret(
                sample_organization.id
            )

        # Assert
        assert new_secret == "new-secret-123"
        assert sample_organization.webhook_secret == "new-secret-123"
        mock_organization_repository.update.assert_called_once_with(sample_organization)

    @pytest.mark.asyncio
    async def test_regenerate_webhook_secret_no_webhook(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test regenerating webhook secret without webhook URL."""
        # Arrange
        sample_organization.webhook_url = None
        mock_organization_repository.get.return_value = sample_organization

        # Act & Assert
        with pytest.raises(ValueError, match="No webhook URL configured"):
            await organization_service.regenerate_webhook_secret(sample_organization.id)

    @pytest.mark.asyncio
    async def test_regenerate_webhook_secret_not_found(
        self,
        organization_service,
        mock_organization_repository,
    ):
        """Test regenerating webhook secret for non-existent organization."""
        # Arrange
        mock_organization_repository.get.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Organization not found"):
            await organization_service.regenerate_webhook_secret(uuid4())

    @pytest.mark.asyncio
    async def test_add_wake_word_success(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test successfully adding a wake word."""
        # Arrange
        sample_organization.wake_words = {"existing", "words"}
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.update.return_value = sample_organization

        # Act
        result = await organization_service.add_wake_word(
            sample_organization.id, "NewWord"
        )

        # Assert
        assert result == sample_organization
        assert "newword" in sample_organization.wake_words
        mock_organization_repository.update.assert_called_once_with(sample_organization)

    @pytest.mark.asyncio
    async def test_add_wake_word_duplicate(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test adding duplicate wake word."""
        # Arrange
        sample_organization.wake_words = {"existing", "words"}
        mock_organization_repository.get.return_value = sample_organization

        # Act & Assert
        with pytest.raises(ValueError, match="Wake word 'Existing' already exists"):
            await organization_service.add_wake_word(
                sample_organization.id, "Existing"
            )

    @pytest.mark.asyncio
    async def test_add_wake_word_not_found(
        self,
        organization_service,
        mock_organization_repository,
    ):
        """Test adding wake word to non-existent organization."""
        # Arrange
        mock_organization_repository.get.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Organization not found"):
            await organization_service.add_wake_word(uuid4(), "test")

    @pytest.mark.asyncio
    async def test_remove_wake_word_success(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test successfully removing a wake word."""
        # Arrange
        sample_organization.wake_words = {"existing", "words", "remove"}
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.update.return_value = sample_organization

        # Act
        result = await organization_service.remove_wake_word(
            sample_organization.id, "remove"
        )

        # Assert
        assert result == sample_organization
        assert "remove" not in sample_organization.wake_words
        mock_organization_repository.update.assert_called_once_with(sample_organization)

    @pytest.mark.asyncio
    async def test_remove_wake_word_not_found(
        self,
        organization_service,
        mock_organization_repository,
    ):
        """Test removing wake word from non-existent organization."""
        # Arrange
        mock_organization_repository.get.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Organization not found"):
            await organization_service.remove_wake_word(uuid4(), "test")

    @pytest.mark.asyncio
    async def test_record_stream_usage_success(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test recording stream usage."""
        # Arrange
        initial_streams = sample_organization.total_streams_processed
        initial_highlights = sample_organization.total_highlights_generated
        initial_seconds = sample_organization.total_processing_seconds
        
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.update.return_value = sample_organization

        # Act
        await organization_service.record_stream_usage(
            organization_id=sample_organization.id,
            processing_seconds=120.5,
            highlights_count=5,
        )

        # Assert
        assert sample_organization.total_streams_processed == initial_streams + 1
        assert sample_organization.total_highlights_generated == initial_highlights + 5
        assert sample_organization.total_processing_seconds == initial_seconds + 120.5
        mock_organization_repository.update.assert_called_once_with(sample_organization)

    @pytest.mark.asyncio
    async def test_record_stream_usage_not_found(
        self,
        organization_service,
        mock_organization_repository,
    ):
        """Test recording usage for non-existent organization."""
        # Arrange
        mock_organization_repository.get.return_value = None

        # Act
        await organization_service.record_stream_usage(
            organization_id=uuid4(),
            processing_seconds=120.5,
            highlights_count=5,
        )

        # Assert - should log error but not raise exception
        mock_organization_repository.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_organization_partial_update(
        self,
        organization_service,
        mock_organization_repository,
        sample_organization,
    ):
        """Test partial organization update with only some fields."""
        # Arrange
        original_name = sample_organization.name
        original_webhook = sample_organization.webhook_url
        mock_organization_repository.get.return_value = sample_organization
        mock_organization_repository.update.return_value = sample_organization

        # Act - only update is_active
        result = await organization_service.update_organization(
            organization_id=sample_organization.id,
            is_active=False,
        )

        # Assert
        assert result.is_active is False
        assert result.name == original_name  # Unchanged
        assert result.webhook_url == original_webhook  # Unchanged

    @pytest.mark.asyncio
    async def test_create_organization_strips_whitespace(
        self,
        organization_service,
        mock_organization_repository,
        mock_user_service,
        sample_organization,
        sample_user,
    ):
        """Test that organization name is stripped of whitespace."""
        # Arrange
        mock_organization_repository.get_by_slug.return_value = None
        mock_organization_repository.add.return_value = sample_organization
        mock_user_service.create_user.return_value = sample_user

        # Act
        org, _ = await organization_service.create_organization(
            name="  Test Organization  ",
            owner_email="owner@example.com",
            owner_name="Owner User",
            owner_password="secure_password123",
        )

        # Assert
        # Check the organization that was passed to add()
        add_call_args = mock_organization_repository.add.call_args[0][0]
        assert add_call_args.name == "Test Organization"