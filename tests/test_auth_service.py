"""Tests for authentication service."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.auth import AuthService
from src.models.api_key import APIKey
from src.models.user import User
from src.models.organization import Organization, PlanType
from src.utils.auth import hash_api_key


@pytest.fixture
def auth_service():
    """Create auth service instance."""
    cache_mock = AsyncMock()
    cache_mock.get = AsyncMock(return_value=None)
    cache_mock.set = AsyncMock(return_value=None)
    cache_mock.delete = AsyncMock(return_value=True)
    return AuthService(cache_mock)


@pytest.fixture
def db_session():
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def sample_user():
    """Create sample user."""
    return User(
        id=1,
        email="test@example.com",
        company_name="Test Company",
        password_hash="hashed_password",
    )


@pytest.fixture
def sample_organization():
    """Create sample organization."""
    return Organization(
        id=1, name="Test Org", owner_id=1, plan_type=PlanType.PROFESSIONAL.value
    )


@pytest.fixture
def sample_api_key(sample_user):
    """Create sample API key."""
    api_key = APIKey(
        id=1,
        key=hash_api_key("test_api_key_123"),
        name="Test API Key",
        user_id=sample_user.id,
        scopes=["read", "write"],
        active=True,
        created_at=datetime.now(timezone.utc),
        expires_at=None,
        last_used_at=None,
    )
    api_key.user = sample_user
    return api_key


class TestAuthService:
    """Test authentication service."""

    @pytest.mark.asyncio
    async def test_validate_api_key_valid(
        self, auth_service, db_session, sample_api_key
    ):
        """Test API key validation with valid key."""
        # Setup database mock to return the API key
        result_mock = MagicMock()
        result_mock.scalars = MagicMock()
        result_mock.scalars().all = MagicMock(return_value=[sample_api_key])

        db_session.execute = AsyncMock(return_value=result_mock)

        # Mock verify_api_key to return True
        with patch("src.services.auth.verify_api_key", return_value=True):
            result = await auth_service.validate_api_key("test_api_key_123", db_session)

        assert result is not None
        assert result.id == sample_api_key.id
        assert result.user_id == sample_api_key.user_id

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid(self, auth_service, db_session):
        """Test API key validation with invalid key."""
        # Setup database mock to return empty list
        result_mock = MagicMock()
        result_mock.scalars = MagicMock()
        result_mock.scalars().all = MagicMock(return_value=[])

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.validate_api_key("invalid_key", db_session)

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_inactive(
        self, auth_service, db_session, sample_api_key
    ):
        """Test API key validation with inactive key."""
        sample_api_key.active = False

        # Setup database mock
        result_mock = MagicMock()
        result_mock.scalars = MagicMock()
        result_mock.scalars().all = MagicMock(return_value=[sample_api_key])

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.validate_api_key("test_api_key_123", db_session)

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_expired(
        self, auth_service, db_session, sample_api_key
    ):
        """Test API key validation with expired key."""
        sample_api_key.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

        # Setup database mock
        result_mock = MagicMock()
        result_mock.scalars = MagicMock()
        result_mock.scalars().all = MagicMock(return_value=[sample_api_key])

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.validate_api_key("test_api_key_123", db_session)

        assert result is None

    @pytest.mark.asyncio
    async def test_has_permission_valid_scope(self, auth_service, sample_api_key):
        """Test permission checking with valid scope."""
        result = await auth_service.has_permission(sample_api_key, "read")
        assert result is True

    @pytest.mark.asyncio
    async def test_has_permission_invalid_scope(self, auth_service, sample_api_key):
        """Test permission checking with invalid scope."""
        result = await auth_service.has_permission(sample_api_key, "admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_has_permission_admin_scope(self, auth_service, sample_api_key):
        """Test that admin scope grants all permissions."""
        sample_api_key.scopes = ["admin"]

        result = await auth_service.has_permission(sample_api_key, "read")
        assert result is True

        result = await auth_service.has_permission(sample_api_key, "write")
        assert result is True

        result = await auth_service.has_permission(sample_api_key, "delete")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_user_organization(
        self, auth_service, db_session, sample_user, sample_organization
    ):
        """Test getting user's organization."""
        # Setup database mock
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=sample_organization)

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.get_user_organization(sample_user.id, db_session)

        assert result is not None
        assert result.id == sample_organization.id

    @pytest.mark.asyncio
    async def test_get_user_organization_not_found(
        self, auth_service, db_session, sample_user
    ):
        """Test getting user's organization when none exists."""
        # Setup database mock
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.get_user_organization(sample_user.id, db_session)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_last_used(self, auth_service, db_session, sample_api_key):
        """Test updating API key last used timestamp."""
        original_last_used = sample_api_key.last_used_at

        await auth_service.update_last_used(sample_api_key, db_session)

        assert sample_api_key.last_used_at != original_last_used
        assert sample_api_key.last_used_at is not None
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_api_key(self, auth_service, db_session, sample_user):
        """Test creating a new API key."""
        scopes = ["read", "write"]
        name = "Test Key"

        result = await auth_service.create_api_key(
            user_id=sample_user.id, name=name, scopes=scopes, db=db_session
        )

        assert result is not None
        assert result["name"] == name
        assert result["scopes"] == scopes
        assert "key" in result
        assert "masked_key" in result
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, auth_service, db_session, sample_api_key):
        """Test revoking an API key."""
        # Setup database mock
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=sample_api_key)

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.revoke_api_key(
            sample_api_key.id, sample_api_key.user_id, db_session
        )

        assert result is True
        assert sample_api_key.active is False
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_api_key_not_found(self, auth_service, db_session):
        """Test revoking an API key that doesn't exist."""
        # Setup database mock
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.revoke_api_key(999, 1, db_session)

        assert result is False

    @pytest.mark.asyncio
    async def test_list_user_api_keys(
        self, auth_service, db_session, sample_user, sample_api_key
    ):
        """Test listing user's API keys."""
        # Setup database mock
        result_mock = MagicMock()
        result_mock.scalars = MagicMock()
        result_mock.scalars().all = MagicMock(return_value=[sample_api_key])

        db_session.execute = AsyncMock(return_value=result_mock)

        result = await auth_service.list_user_api_keys(sample_user.id, db_session)

        assert len(result) == 1
        assert result[0]["id"] == sample_api_key.id
        assert result[0]["name"] == sample_api_key.name
        assert "key" not in result[0]  # Should not include raw key
        assert "masked_key" in result[0]

    @pytest.mark.asyncio
    async def test_rate_limit_for_key(
        self, auth_service, sample_api_key, sample_organization
    ):
        """Test getting rate limit for API key."""
        sample_api_key.user.owned_organizations = [sample_organization]

        rate_limit = await auth_service.rate_limit_for_key(sample_api_key)

        expected_limit = sample_organization.plan_limits["api_rate_limit_per_minute"]
        assert rate_limit == expected_limit

    @pytest.mark.asyncio
    async def test_rate_limit_for_key_default(self, auth_service, sample_api_key):
        """Test getting default rate limit when no organization."""
        sample_api_key.user.owned_organizations = []

        rate_limit = await auth_service.rate_limit_for_key(sample_api_key)

        assert rate_limit == 60  # Default rate limit
