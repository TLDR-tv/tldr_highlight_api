"""Unit tests for API key service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from shared.infrastructure.security.api_key_service import APIKeyService
from shared.domain.models.api_key import APIKey, APIScopes


class TestAPIKeyService:
    """Test APIKeyService implementation."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock API key repository."""
        repository = AsyncMock()
        repository.add = AsyncMock()
        repository.get = AsyncMock()
        repository.get_by_prefix = AsyncMock()
        repository.update = AsyncMock()
        return repository

    @pytest.fixture
    def service(self, mock_repository):
        """Create API key service instance."""
        return APIKeyService(mock_repository)

    @pytest.fixture
    def sample_api_key(self):
        """Create sample API key entity."""
        return APIKey(
            id=uuid4(),
            organization_id=uuid4(),
            name="Test API Key",
            key_hash="$2b$12$sample_hash",
            prefix="tldr_abc12345",
            scopes={APIScopes.STREAMS_READ, APIScopes.HIGHLIGHTS_READ},
            is_active=True,
        )

    def test_service_initialization(self, mock_repository):
        """Test service initialization."""
        service = APIKeyService(mock_repository)
        
        assert service.repository == mock_repository
        assert service.key_prefix == "tldr"
        assert service.key_length == 32
        assert service.pwd_context is not None

    @pytest.mark.asyncio
    async def test_validate_api_key_malformed_key_exception(self, service, mock_repository):
        """Test validation with malformed key that causes exception."""
        # This test specifically targets the exception handling on lines 76-77
        # Test with a key that has the right prefix but causes IndexError
        # when we try to access parts[2] in the split logic
        malformed_key = "tldr_onlyonepart"  # This will split into only 2 parts, causing IndexError
        
        result = await service.validate_api_key(malformed_key)
        assert result is None
        
        # Test another case that triggers the exception handling
        malformed_key2 = "tldr_"  # This will also cause issues
        result = await service.validate_api_key(malformed_key2)
        assert result is None

    @pytest.mark.asyncio
    async def test_hash_password(self, service):
        """Test password hashing."""
        password = "test_password"
        
        with patch.object(service.pwd_context, 'hash', return_value="hashed_password") as mock_hash:
            result = await service.hash_password(password)
        
        assert result == "hashed_password"
        mock_hash.assert_called_once_with(password)

    @pytest.mark.asyncio
    async def test_verify_password(self, service):
        """Test password verification."""
        password = "test_password"
        hashed = "hashed_password"
        
        with patch.object(service.pwd_context, 'verify', return_value=True) as mock_verify:
            result = await service.verify_password(password, hashed)
        
        assert result is True
        mock_verify.assert_called_once_with(password, hashed)

    @pytest.mark.asyncio
    async def test_generate_api_key_success(self, service, mock_repository):
        """Test successful API key generation."""
        org_id = uuid4()
        user_id = uuid4()
        
        # Mock repository response
        created_key = APIKey(
            id=uuid4(),
            organization_id=org_id,
            name="Test Key",
            key_hash="$2b$12$hashed",
            prefix="tldr_abc12345",
            created_by_user_id=user_id
        )
        mock_repository.add.return_value = created_key
        
        # Mock password hashing
        with patch.object(service.pwd_context, 'hash', return_value="$2b$12$hashed"):
            full_key, saved_key = await service.generate_api_key(
                organization_id=org_id,
                name="Test Key",
                scopes={APIScopes.STREAMS_READ},
                description="Test description",
                created_by_user_id=user_id
            )
        
        # Verify key format
        assert full_key.startswith("tldr_")
        assert len(full_key) >= 38  # tldr + _ + 8 chars + _ + 24 chars = 38
        parts = full_key.split("_")
        assert len(parts) == 3
        
        # Verify repository call
        mock_repository.add.assert_called_once()
        call_args = mock_repository.add.call_args[0][0]
        assert call_args.organization_id == org_id
        assert call_args.name == "Test Key"
        assert call_args.description == "Test description"
        assert call_args.created_by_user_id == user_id
        assert call_args.scopes == {APIScopes.STREAMS_READ}
        
        assert saved_key == created_key

    @pytest.mark.asyncio
    async def test_generate_api_key_default_scopes(self, service, mock_repository):
        """Test API key generation with default scopes."""
        org_id = uuid4()
        
        created_key = APIKey(
            id=uuid4(),
            organization_id=org_id,
            name="Test Key",
            key_hash="$2b$12$hashed",
            prefix="tldr_abc12345"
        )
        mock_repository.add.return_value = created_key
        
        with patch.object(service.pwd_context, 'hash', return_value="$2b$12$hashed"):
            full_key, saved_key = await service.generate_api_key(
                organization_id=org_id,
                name="Test Key"
            )
        
        # Should use default scopes
        call_args = mock_repository.add.call_args[0][0]
        assert call_args.scopes == APIScopes.default_scopes()

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, service, mock_repository, sample_api_key):
        """Test successful API key validation."""
        test_key = "tldr_abc12345_def67890rest_of_key"
        
        # Mock repository and password verification
        mock_repository.get_by_prefix.return_value = sample_api_key
        mock_repository.update = AsyncMock()
        
        with patch.object(service.pwd_context, 'verify', return_value=True) as mock_verify:
            result = await service.validate_api_key(test_key)
        
        assert result == sample_api_key
        mock_repository.get_by_prefix.assert_called_once_with("tldr_abc12345")
        mock_verify.assert_called_once_with(test_key, sample_api_key.key_hash)
        mock_repository.update.assert_called_once_with(sample_api_key)

    @pytest.mark.asyncio
    async def test_validate_api_key_wrong_prefix(self, service):
        """Test validation with wrong prefix."""
        result = await service.validate_api_key("wrong_prefix_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_not_found(self, service, mock_repository):
        """Test validation when key not found in repository."""
        test_key = "tldr_abc12345_def67890rest_of_key"
        mock_repository.get_by_prefix.return_value = None
        
        result = await service.validate_api_key(test_key)
        
        assert result is None
        mock_repository.get_by_prefix.assert_called_once_with("tldr_abc12345")

    @pytest.mark.asyncio
    async def test_validate_api_key_hash_mismatch(self, service, mock_repository, sample_api_key):
        """Test validation with incorrect hash."""
        test_key = "tldr_abc12345_def67890rest_of_key"
        mock_repository.get_by_prefix.return_value = sample_api_key
        
        with patch.object(service.pwd_context, 'verify', return_value=False):
            result = await service.validate_api_key(test_key)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid_key(self, service, mock_repository):
        """Test validation with invalid key entity."""
        test_key = "tldr_abc12345_def67890rest_of_key"
        invalid_key = APIKey(
            id=uuid4(),
            organization_id=uuid4(),
            name="Invalid Key",
            key_hash="$2b$12$sample_hash",
            prefix="tldr_abc12345",
            is_active=False  # Invalid key
        )
        mock_repository.get_by_prefix.return_value = invalid_key
        
        with patch.object(service.pwd_context, 'verify', return_value=True):
            result = await service.validate_api_key(test_key)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_revoke_api_key_success(self, service, mock_repository, sample_api_key):
        """Test successful API key revocation."""
        key_id = sample_api_key.id
        mock_repository.get.return_value = sample_api_key
        mock_repository.update = AsyncMock()
        
        await service.revoke_api_key(key_id)
        
        mock_repository.get.assert_called_once_with(key_id)
        mock_repository.update.assert_called_once_with(sample_api_key)

    @pytest.mark.asyncio
    async def test_revoke_api_key_not_found(self, service, mock_repository):
        """Test revoking non-existent API key."""
        key_id = uuid4()
        mock_repository.get.return_value = None
        mock_repository.update = AsyncMock()
        
        await service.revoke_api_key(key_id)
        
        mock_repository.get.assert_called_once_with(key_id)
        mock_repository.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_api_key_direct_exception(self, service, mock_repository):
        """Test direct exception triggering in validate_api_key."""
        # Create a key that should cause an exception when processed
        # We'll mock the string operations that happen inside the try/except block
        from unittest.mock import patch, MagicMock
        
        # Mock the split operation to raise an exception
        mock_key = MagicMock()
        mock_key.startswith.return_value = True  # Pass the prefix check
        mock_key.split.side_effect = ValueError("Test exception")
        
        with patch('builtins.str') as mock_str:
            mock_str.return_value = mock_key
            result = await service.validate_api_key("tldr_test")
            assert result is None