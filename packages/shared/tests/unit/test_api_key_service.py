"""Unit tests for API key service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from shared.infrastructure.security.api_key_service import APIKeyService
from shared.domain.models.api_key import APIKey, APIScopes
from shared.domain.protocols import APIKeyRepository


class TestAPIKeyService:
    """Test suite for APIKeyService."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock API key repository."""
        repository = AsyncMock(spec=APIKeyRepository)
        return repository

    @pytest.fixture
    def api_key_service(self, mock_repository):
        """Create an API key service with mocked dependencies."""
        return APIKeyService(repository=mock_repository)

    @pytest.fixture
    def sample_api_key(self):
        """Create a sample API key for testing."""
        return APIKey(
            id=uuid4(),
            organization_id=uuid4(),
            name="Test API Key",
            key_hash="hashed_key",
            prefix="tldr_12345678",
            scopes={APIScopes.STREAMS_READ, APIScopes.STREAMS_WRITE},
            description="Test API key",
            created_by_user_id=uuid4(),
            is_active=True,
        )

    @pytest.mark.asyncio
    async def test_generate_api_key_success(
        self, api_key_service, mock_repository, sample_api_key
    ):
        """Test successful API key generation."""
        # Arrange
        org_id = uuid4()
        user_id = uuid4()
        mock_repository.add.return_value = sample_api_key

        # Act
        with patch("secrets.choice") as mock_choice:
            # Mock the random key generation
            mock_choice.side_effect = list("A" * 32)  # Predictable key for testing
            
            raw_key, key_entity = await api_key_service.generate_api_key(
                organization_id=org_id,
                name="Test Key",
                scopes={APIScopes.STREAMS_READ},
                description="Test description",
                created_by_user_id=user_id,
            )

        # Assert
        assert raw_key.startswith("tldr_AAAAAAAA_")
        assert len(raw_key) == 38  # tldr_ (5) + 8 + _ (1) + 24 = 38
        assert key_entity == sample_api_key
        
        # Verify repository was called
        mock_repository.add.assert_called_once()
        add_call_args = mock_repository.add.call_args[0][0]
        assert add_call_args.organization_id == org_id
        assert add_call_args.name == "Test Key"
        assert add_call_args.scopes == {APIScopes.STREAMS_READ}
        assert add_call_args.description == "Test description"
        assert add_call_args.created_by_user_id == user_id

    @pytest.mark.asyncio
    async def test_generate_api_key_default_scopes(
        self, api_key_service, mock_repository, sample_api_key
    ):
        """Test API key generation with default scopes."""
        # Arrange
        mock_repository.add.return_value = sample_api_key

        # Act
        raw_key, key_entity = await api_key_service.generate_api_key(
            organization_id=uuid4(),
            name="Test Key",
        )

        # Assert
        add_call_args = mock_repository.add.call_args[0][0]
        assert add_call_args.scopes == APIScopes.default_scopes()

    @pytest.mark.asyncio
    async def test_validate_api_key_success(
        self, api_key_service, mock_repository, sample_api_key
    ):
        """Test successful API key validation."""
        # Arrange
        raw_key = "tldr_12345678_remaining_key_part"
        mock_repository.get_by_prefix.return_value = sample_api_key
        mock_repository.update.return_value = sample_api_key
        
        with patch.object(api_key_service.pwd_context, "verify", return_value=True):
            # Act
            result = await api_key_service.validate_api_key(raw_key)

        # Assert
        assert result == sample_api_key
        mock_repository.get_by_prefix.assert_called_once_with("tldr_12345678")
        mock_repository.update.assert_called_once_with(sample_api_key)
        # Check that usage was recorded
        assert sample_api_key.last_used_at is not None

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid_prefix(
        self, api_key_service, mock_repository
    ):
        """Test API key validation with invalid prefix."""
        # Arrange
        raw_key = "invalid_prefix_12345678_key"

        # Act
        result = await api_key_service.validate_api_key(raw_key)

        # Assert
        assert result is None
        mock_repository.get_by_prefix.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid_format(
        self, api_key_service, mock_repository
    ):
        """Test API key validation with invalid format."""
        # Arrange
        raw_key = "tldr_missingparts"

        # Act
        result = await api_key_service.validate_api_key(raw_key)

        # Assert
        assert result is None
        mock_repository.get_by_prefix.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_api_key_not_found(
        self, api_key_service, mock_repository
    ):
        """Test API key validation when key not found."""
        # Arrange
        raw_key = "tldr_12345678_remaining_key_part"
        mock_repository.get_by_prefix.return_value = None

        # Act
        result = await api_key_service.validate_api_key(raw_key)

        # Assert
        assert result is None
        mock_repository.get_by_prefix.assert_called_once_with("tldr_12345678")

    @pytest.mark.asyncio
    async def test_validate_api_key_wrong_hash(
        self, api_key_service, mock_repository, sample_api_key
    ):
        """Test API key validation with wrong hash."""
        # Arrange
        raw_key = "tldr_12345678_remaining_key_part"
        mock_repository.get_by_prefix.return_value = sample_api_key
        
        with patch.object(api_key_service.pwd_context, "verify", return_value=False):
            # Act
            result = await api_key_service.validate_api_key(raw_key)

        # Assert
        assert result is None
        mock_repository.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_api_key_inactive(
        self, api_key_service, mock_repository, sample_api_key
    ):
        """Test API key validation with inactive key."""
        # Arrange
        raw_key = "tldr_12345678_remaining_key_part"
        sample_api_key.is_active = False
        mock_repository.get_by_prefix.return_value = sample_api_key
        
        with patch.object(api_key_service.pwd_context, "verify", return_value=True):
            # Act
            result = await api_key_service.validate_api_key(raw_key)

        # Assert
        assert result is None
        mock_repository.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_api_key_expired(
        self, api_key_service, mock_repository, sample_api_key
    ):
        """Test API key validation with expired key."""
        # Arrange
        from datetime import datetime, timezone, timedelta
        raw_key = "tldr_12345678_remaining_key_part"
        # Set expiration date in the past
        sample_api_key.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        mock_repository.get_by_prefix.return_value = sample_api_key
        
        with patch.object(api_key_service.pwd_context, "verify", return_value=True):
            # Act
            result = await api_key_service.validate_api_key(raw_key)

        # Assert
        assert result is None
        mock_repository.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_revoke_api_key_success(
        self, api_key_service, mock_repository, sample_api_key
    ):
        """Test successful API key revocation."""
        # Arrange
        key_id = sample_api_key.id
        mock_repository.get.return_value = sample_api_key
        mock_repository.update.return_value = sample_api_key

        # Act
        await api_key_service.revoke_api_key(key_id)

        # Assert
        assert sample_api_key.is_active is False
        assert sample_api_key.revoked_at is not None
        mock_repository.get.assert_called_once_with(key_id)
        mock_repository.update.assert_called_once_with(sample_api_key)

    @pytest.mark.asyncio
    async def test_revoke_api_key_not_found(
        self, api_key_service, mock_repository
    ):
        """Test API key revocation when key not found."""
        # Arrange
        key_id = uuid4()
        mock_repository.get.return_value = None

        # Act
        await api_key_service.revoke_api_key(key_id)

        # Assert
        mock_repository.get.assert_called_once_with(key_id)
        mock_repository.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_hash_password(self, api_key_service):
        """Test password hashing."""
        # Arrange
        password = "test_password"

        # Act
        hashed = await api_key_service.hash_password(password)

        # Assert
        assert hashed != password
        assert api_key_service.pwd_context.verify(password, hashed)

    @pytest.mark.asyncio
    async def test_verify_password_correct(self, api_key_service):
        """Test password verification with correct password."""
        # Arrange
        password = "test_password"
        hashed = api_key_service.pwd_context.hash(password)

        # Act
        result = await api_key_service.verify_password(password, hashed)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_password_incorrect(self, api_key_service):
        """Test password verification with incorrect password."""
        # Arrange
        password = "test_password"
        wrong_password = "wrong_password"
        hashed = api_key_service.pwd_context.hash(password)

        # Act
        result = await api_key_service.verify_password(wrong_password, hashed)

        # Assert
        assert result is False

    def test_initialization(self, mock_repository):
        """Test service initialization."""
        # Act
        service = APIKeyService(repository=mock_repository)

        # Assert
        assert service.repository == mock_repository
        assert service.key_prefix == "tldr"
        assert service.key_length == 32
        assert service.pwd_context is not None

    @pytest.mark.asyncio
    async def test_generate_api_key_with_custom_prefix_length(
        self, mock_repository, sample_api_key
    ):
        """Test API key generation with custom configuration."""
        # Arrange
        service = APIKeyService(repository=mock_repository)
        service.key_prefix = "custom"
        service.key_length = 40
        mock_repository.add.return_value = sample_api_key

        # Act
        with patch("secrets.choice") as mock_choice:
            mock_choice.side_effect = list("B" * 40)
            
            raw_key, _ = await service.generate_api_key(
                organization_id=uuid4(),
                name="Test Key",
            )

        # Assert
        assert raw_key.startswith("custom_BBBBBBBB_")
        assert len(raw_key) == 48  # custom_ (7) + 8 + _ (1) + 32 = 48

    @pytest.mark.asyncio
    async def test_validate_api_key_empty_string(
        self, api_key_service, mock_repository
    ):
        """Test API key validation with empty string."""
        # Act
        result = await api_key_service.validate_api_key("")

        # Assert
        assert result is None
        mock_repository.get_by_prefix.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_api_key_exception_handling(
        self, api_key_service, mock_repository
    ):
        """Test API key validation handles exceptions gracefully."""
        # Arrange
        raw_key = "tldr_12345678_remaining_key_part"
        mock_repository.get_by_prefix.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await api_key_service.validate_api_key(raw_key)