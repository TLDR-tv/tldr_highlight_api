"""Unit tests for user service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
from datetime import datetime, timezone

from api.services.user_service import UserService
from api.services.auth.password_service import PasswordService
from api.services.auth.jwt_service import JWTService
from shared.domain.models.user import User, UserRole
from shared.infrastructure.storage.repositories import UserRepository
from shared.infrastructure.config.config import Settings


class TestUserService:
    """Test suite for UserService."""

    @pytest.fixture
    def mock_user_repository(self):
        """Create a mock user repository."""
        repository = AsyncMock(spec=UserRepository)
        return repository

    @pytest.fixture
    def mock_password_service(self):
        """Create a mock password service."""
        service = Mock(spec=PasswordService)
        service.hash_password.return_value = "hashed_password"
        service.verify_password.return_value = True
        service.validate_password_strength.return_value = (True, [])
        return service

    @pytest.fixture
    def mock_jwt_service(self):
        """Create a mock JWT service."""
        service = Mock(spec=JWTService)
        service.create_access_token.return_value = "access_token"
        service.create_refresh_token.return_value = "refresh_token"
        service.create_password_reset_token.return_value = "reset_token"
        service.verify_refresh_token.return_value = {"sub": str(uuid4())}
        service.verify_password_reset_token.return_value = {
            "sub": str(uuid4()),
            "email": "test@example.com"
        }
        return service

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock(spec=Settings)
        settings.environment = "development"
        return settings

    @pytest.fixture
    def user_service(
        self,
        mock_user_repository,
        mock_password_service,
        mock_jwt_service,
        mock_settings,
    ):
        """Create a user service with mocked dependencies."""
        return UserService(
            user_repository=mock_user_repository,
            password_service=mock_password_service,
            jwt_service=mock_jwt_service,
            settings=mock_settings,
        )

    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id=uuid4(),
            organization_id=uuid4(),
            email="test@example.com",
            name="Test User",
            role=UserRole.MEMBER,
            hashed_password="hashed_password",
            is_active=True,
        )

    @pytest.mark.asyncio
    async def test_create_user_success(
        self,
        user_service,
        mock_user_repository,
        mock_password_service,
        sample_user,
    ):
        """Test successful user creation."""
        # Arrange
        mock_user_repository.get_by_email.return_value = None
        mock_user_repository.add.return_value = sample_user

        # Act
        result = await user_service.create_user(
            organization_id=sample_user.organization_id,
            email="test@example.com",
            name="Test User",
            password="SecurePassword123!",
            role=UserRole.MEMBER,
        )

        # Assert
        assert result == sample_user
        mock_user_repository.get_by_email.assert_called_once_with("test@example.com")
        mock_password_service.validate_password_strength.assert_called_once_with("SecurePassword123!")
        mock_password_service.hash_password.assert_called_once_with("SecurePassword123!")
        mock_user_repository.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_email_exists(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test user creation with existing email."""
        # Arrange
        mock_user_repository.get_by_email.return_value = sample_user

        # Act & Assert
        with pytest.raises(ValueError, match="User with this email already exists"):
            await user_service.create_user(
                organization_id=sample_user.organization_id,
                email="test@example.com",
                name="Test User",
                password="SecurePassword123!",
            )

    @pytest.mark.asyncio
    async def test_create_user_weak_password(
        self,
        user_service,
        mock_user_repository,
        mock_password_service,
    ):
        """Test user creation with weak password."""
        # Arrange
        mock_user_repository.get_by_email.return_value = None
        mock_password_service.validate_password_strength.return_value = (
            False,
            ["Password too short", "No special characters"],
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid password: Password too short; No special characters"):
            await user_service.create_user(
                organization_id=uuid4(),
                email="test@example.com",
                name="Test User",
                password="weak",
            )

    @pytest.mark.asyncio
    async def test_create_user_normalizes_input(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test that email is lowercased and name is stripped."""
        # Arrange
        mock_user_repository.get_by_email.return_value = None
        mock_user_repository.add.return_value = sample_user

        # Act
        await user_service.create_user(
            organization_id=sample_user.organization_id,
            email="  TEST@EXAMPLE.COM  ",
            name="  Test User  ",
            password="SecurePassword123!",
        )

        # Assert
        add_call_args = mock_user_repository.add.call_args[0][0]
        assert add_call_args.email == "test@example.com"
        assert add_call_args.name == "Test User"

    @pytest.mark.asyncio
    async def test_authenticate_success(
        self,
        user_service,
        mock_user_repository,
        mock_password_service,
        mock_jwt_service,
        sample_user,
    ):
        """Test successful authentication."""
        # Arrange
        mock_user_repository.get_by_email.return_value = sample_user
        mock_user_repository.update.return_value = sample_user

        # Act
        user, access_token, refresh_token = await user_service.authenticate(
            email="test@example.com",
            password="password123",
        )

        # Assert
        assert user == sample_user
        assert access_token == "access_token"
        assert refresh_token == "refresh_token"
        mock_password_service.verify_password.assert_called_once_with(
            "password123", sample_user.hashed_password
        )
        mock_jwt_service.create_access_token.assert_called_once()
        mock_jwt_service.create_refresh_token.assert_called_once()
        assert sample_user.last_login_at is not None

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(
        self,
        user_service,
        mock_user_repository,
    ):
        """Test authentication with non-existent user."""
        # Arrange
        mock_user_repository.get_by_email.return_value = None

        # Act
        user, access_token, refresh_token = await user_service.authenticate(
            email="unknown@example.com",
            password="password123",
        )

        # Assert
        assert user is None
        assert access_token is None
        assert refresh_token is None

    @pytest.mark.asyncio
    async def test_authenticate_inactive_user(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test authentication with inactive user."""
        # Arrange
        sample_user.is_active = False
        mock_user_repository.get_by_email.return_value = sample_user

        # Act
        user, access_token, refresh_token = await user_service.authenticate(
            email="test@example.com",
            password="password123",
        )

        # Assert
        assert user is None
        assert access_token is None
        assert refresh_token is None

    @pytest.mark.asyncio
    async def test_authenticate_wrong_password(
        self,
        user_service,
        mock_user_repository,
        mock_password_service,
        sample_user,
    ):
        """Test authentication with wrong password."""
        # Arrange
        mock_user_repository.get_by_email.return_value = sample_user
        mock_password_service.verify_password.return_value = False

        # Act
        user, access_token, refresh_token = await user_service.authenticate(
            email="test@example.com",
            password="wrong_password",
        )

        # Assert
        assert user is None
        assert access_token is None
        assert refresh_token is None

    @pytest.mark.asyncio
    async def test_refresh_tokens_success(
        self,
        user_service,
        mock_user_repository,
        mock_jwt_service,
        sample_user,
    ):
        """Test successful token refresh."""
        # Arrange
        user_id = sample_user.id
        mock_jwt_service.verify_refresh_token.return_value = {"sub": str(user_id)}
        mock_user_repository.get.return_value = sample_user

        # Act
        new_access_token, new_refresh_token = await user_service.refresh_tokens(
            "valid_refresh_token"
        )

        # Assert
        assert new_access_token == "access_token"
        assert new_refresh_token == "refresh_token"
        mock_jwt_service.verify_refresh_token.assert_called_once_with("valid_refresh_token")
        mock_user_repository.get.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_refresh_tokens_invalid_token(
        self,
        user_service,
        mock_jwt_service,
    ):
        """Test token refresh with invalid token."""
        # Arrange
        mock_jwt_service.verify_refresh_token.return_value = None

        # Act
        new_access_token, new_refresh_token = await user_service.refresh_tokens(
            "invalid_token"
        )

        # Assert
        assert new_access_token is None
        assert new_refresh_token is None

    @pytest.mark.asyncio
    async def test_refresh_tokens_user_not_found(
        self,
        user_service,
        mock_user_repository,
        mock_jwt_service,
    ):
        """Test token refresh with non-existent user."""
        # Arrange
        user_id = uuid4()
        mock_jwt_service.verify_refresh_token.return_value = {"sub": str(user_id)}
        mock_user_repository.get.return_value = None

        # Act
        new_access_token, new_refresh_token = await user_service.refresh_tokens(
            "valid_refresh_token"
        )

        # Assert
        assert new_access_token is None
        assert new_refresh_token is None

    @pytest.mark.asyncio
    async def test_update_profile_success(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test successful profile update."""
        # Arrange
        mock_user_repository.get.return_value = sample_user
        mock_user_repository.get_by_email.return_value = None
        mock_user_repository.update.return_value = sample_user

        # Act
        result = await user_service.update_profile(
            user_id=sample_user.id,
            name="Updated Name",
            email="new@example.com",
        )

        # Assert
        assert result == sample_user
        assert sample_user.name == "Updated Name"
        assert sample_user.email == "new@example.com"

    @pytest.mark.asyncio
    async def test_update_profile_user_not_found(
        self,
        user_service,
        mock_user_repository,
    ):
        """Test profile update for non-existent user."""
        # Arrange
        mock_user_repository.get.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="User not found"):
            await user_service.update_profile(user_id=uuid4(), name="New Name")

    @pytest.mark.asyncio
    async def test_update_profile_email_exists(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test profile update with existing email."""
        # Arrange
        other_user = User(email="existing@example.com")
        mock_user_repository.get.return_value = sample_user
        mock_user_repository.get_by_email.return_value = other_user

        # Act & Assert
        with pytest.raises(ValueError, match="Email already in use"):
            await user_service.update_profile(
                user_id=sample_user.id,
                email="existing@example.com",
            )

    @pytest.mark.asyncio
    async def test_change_password_success(
        self,
        user_service,
        mock_user_repository,
        mock_password_service,
        sample_user,
    ):
        """Test successful password change."""
        # Arrange
        mock_user_repository.get.return_value = sample_user
        mock_user_repository.update.return_value = sample_user
        mock_password_service.verify_password.return_value = True
        mock_password_service.hash_password.return_value = "new_hashed_password"

        # Act
        result = await user_service.change_password(
            user_id=sample_user.id,
            old_password="old_password",
            new_password="NewSecurePassword123!",
        )

        # Assert
        assert result is True
        assert sample_user.hashed_password == "new_hashed_password"
        mock_password_service.verify_password.assert_called_once_with(
            "old_password", "hashed_password"
        )

    @pytest.mark.asyncio
    async def test_change_password_wrong_old_password(
        self,
        user_service,
        mock_user_repository,
        mock_password_service,
        sample_user,
    ):
        """Test password change with wrong old password."""
        # Arrange
        mock_user_repository.get.return_value = sample_user
        mock_password_service.verify_password.return_value = False

        # Act & Assert
        with pytest.raises(ValueError, match="Current password is incorrect"):
            await user_service.change_password(
                user_id=sample_user.id,
                old_password="wrong_password",
                new_password="NewSecurePassword123!",
            )

    @pytest.mark.asyncio
    async def test_request_password_reset_success(
        self,
        user_service,
        mock_user_repository,
        mock_jwt_service,
        sample_user,
    ):
        """Test successful password reset request."""
        # Arrange
        mock_user_repository.get_by_email.return_value = sample_user

        # Act
        with patch("api.celery_client.celery_app") as mock_celery:
            mock_task = Mock()
            mock_task.id = "task-123"
            mock_celery.send_task.return_value = mock_task
            
            result = await user_service.request_password_reset("test@example.com")

        # Assert
        assert result == "reset_token"  # In development mode
        mock_jwt_service.create_password_reset_token.assert_called_once_with(
            user_id=sample_user.id,
            email=sample_user.email,
        )
        mock_celery.send_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_password_reset_production(
        self,
        user_service,
        mock_user_repository,
        mock_jwt_service,
        mock_settings,
        sample_user,
    ):
        """Test password reset request in production mode."""
        # Arrange
        mock_settings.environment = "production"
        mock_user_repository.get_by_email.return_value = sample_user

        # Act
        with patch("api.celery_client.celery_app") as mock_celery:
            result = await user_service.request_password_reset("test@example.com")

        # Assert
        assert result is None  # Token not returned in production

    @pytest.mark.asyncio
    async def test_request_password_reset_user_not_found(
        self,
        user_service,
        mock_user_repository,
    ):
        """Test password reset request for non-existent user."""
        # Arrange
        mock_user_repository.get_by_email.return_value = None

        # Act
        result = await user_service.request_password_reset("unknown@example.com")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_reset_password_success(
        self,
        user_service,
        mock_user_repository,
        mock_jwt_service,
        mock_password_service,
        sample_user,
    ):
        """Test successful password reset."""
        # Arrange
        mock_jwt_service.verify_password_reset_token.return_value = {
            "sub": str(sample_user.id),
            "email": sample_user.email,
        }
        mock_user_repository.get.return_value = sample_user
        mock_user_repository.update.return_value = sample_user
        mock_password_service.hash_password.return_value = "new_hashed_password"

        # Act
        result = await user_service.reset_password("valid_token", "NewPassword123!")

        # Assert
        assert result is True
        assert sample_user.hashed_password == "new_hashed_password"

    @pytest.mark.asyncio
    async def test_reset_password_invalid_token(
        self,
        user_service,
        mock_jwt_service,
    ):
        """Test password reset with invalid token."""
        # Arrange
        mock_jwt_service.verify_password_reset_token.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid or expired reset token"):
            await user_service.reset_password("invalid_token", "NewPassword123!")

    @pytest.mark.asyncio
    async def test_list_organization_users(
        self,
        user_service,
        mock_user_repository,
    ):
        """Test listing organization users."""
        # Arrange
        org_id = uuid4()
        users = [Mock(spec=User), Mock(spec=User)]
        mock_user_repository.list_by_organization.return_value = users

        # Act
        result = await user_service.list_organization_users(org_id)

        # Assert
        assert result == users
        mock_user_repository.list_by_organization.assert_called_once_with(org_id)

    @pytest.mark.asyncio
    async def test_update_user_role_success(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test successful role update."""
        # Arrange
        admin_user = User(
            id=uuid4(),
            organization_id=sample_user.organization_id,
            email="admin@example.com",
            name="Admin User",
            role=UserRole.ADMIN,
        )
        mock_user_repository.get.side_effect = [admin_user, sample_user]
        mock_user_repository.update.return_value = sample_user

        # Act
        result = await user_service.update_user_role(
            user_id=sample_user.id,
            role=UserRole.ADMIN,
            admin_user_id=admin_user.id,
        )

        # Assert
        assert result == sample_user
        assert sample_user.role == UserRole.ADMIN

    @pytest.mark.asyncio
    async def test_update_user_role_not_admin(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test role update by non-admin."""
        # Arrange
        non_admin_user = User(role=UserRole.MEMBER)
        mock_user_repository.get.return_value = non_admin_user

        # Act & Assert
        with pytest.raises(ValueError, match="Only admins can change user roles"):
            await user_service.update_user_role(
                user_id=sample_user.id,
                role=UserRole.ADMIN,
                admin_user_id=non_admin_user.id,
            )

    @pytest.mark.asyncio
    async def test_update_user_role_last_admin(
        self,
        user_service,
        mock_user_repository,
    ):
        """Test preventing removal of last admin."""
        # Arrange
        org_id = uuid4()
        admin_user = User(
            id=uuid4(),
            organization_id=org_id,
            role=UserRole.ADMIN,
        )
        target_user = User(
            id=uuid4(),
            organization_id=org_id,
            role=UserRole.ADMIN,
        )
        
        mock_user_repository.get.side_effect = [admin_user, target_user]
        mock_user_repository.list_by_organization.return_value = [target_user]  # Only one admin

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot remove last admin from organization"):
            await user_service.update_user_role(
                user_id=target_user.id,
                role=UserRole.MEMBER,
                admin_user_id=admin_user.id,
            )

    @pytest.mark.asyncio
    async def test_deactivate_user_success(
        self,
        user_service,
        mock_user_repository,
        sample_user,
    ):
        """Test successful user deactivation."""
        # Arrange
        admin_user = User(
            id=uuid4(),
            organization_id=sample_user.organization_id,
            role=UserRole.ADMIN,
        )
        mock_user_repository.get.side_effect = [admin_user, sample_user]
        mock_user_repository.update.return_value = sample_user

        # Act
        result = await user_service.deactivate_user(
            user_id=sample_user.id,
            admin_user_id=admin_user.id,
        )

        # Assert
        assert result is True
        assert sample_user.is_active is False

    @pytest.mark.asyncio
    async def test_deactivate_user_self(
        self,
        user_service,
        mock_user_repository,
    ):
        """Test preventing self-deactivation."""
        # Arrange
        admin_user = User(
            id=uuid4(),
            role=UserRole.ADMIN,
        )
        mock_user_repository.get.side_effect = [admin_user, admin_user]

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot deactivate yourself"):
            await user_service.deactivate_user(
                user_id=admin_user.id,
                admin_user_id=admin_user.id,
            )

    @pytest.mark.asyncio
    async def test_deactivate_user_different_org(
        self,
        user_service,
        mock_user_repository,
    ):
        """Test preventing deactivation across organizations."""
        # Arrange
        admin_user = User(
            id=uuid4(),
            organization_id=uuid4(),
            role=UserRole.ADMIN,
        )
        target_user = User(
            id=uuid4(),
            organization_id=uuid4(),  # Different org
        )
        mock_user_repository.get.side_effect = [admin_user, target_user]

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot modify users from other organizations"):
            await user_service.deactivate_user(
                user_id=target_user.id,
                admin_user_id=admin_user.id,
            )