"""Tests for authentication dependencies."""

import pytest
from fastapi import HTTPException, status
from unittest.mock import AsyncMock, MagicMock

from src.core.auth import (
    get_api_key_from_header,
    get_current_api_key,
    get_current_user,
    get_current_organization,
    require_scope,
    check_rate_limit,
    RateLimitExceeded,
    AuthenticationError,
    InsufficientPermissions,
)
from src.models.api_key import APIKey
from src.models.user import User
from src.models.organization import Organization, PlanType


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = MagicMock()
    request.headers = {"X-API-Key": "test_api_key_123"}
    return request


@pytest.fixture
def sample_user():
    """Create sample user."""
    return User(
        id=1,
        email="test@example.com",
        company_name="Test Company",
        password_hash="hashed",
    )


@pytest.fixture
def sample_organization(sample_user):
    """Create sample organization."""
    org = Organization(
        id=1,
        name="Test Org",
        owner_id=sample_user.id,
        plan_type=PlanType.PROFESSIONAL.value,
    )
    org.owner = sample_user
    return org


@pytest.fixture
def sample_api_key(sample_user):
    """Create sample API key."""
    api_key = APIKey(
        id=1,
        key="hashed_key",
        name="Test Key",
        user_id=sample_user.id,
        scopes=["read", "write"],
        active=True,
    )
    api_key.user = sample_user
    return api_key


class TestGetAPIKeyFromHeader:
    """Test API key extraction from headers."""

    def test_get_api_key_from_header_valid(self, mock_request):
        """Test extracting valid API key from header."""
        api_key = get_api_key_from_header(mock_request)
        assert api_key == "test_api_key_123"

    def test_get_api_key_from_header_missing(self):
        """Test error when API key header is missing."""
        request = MagicMock()
        request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            get_api_key_from_header(request)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "API key is required" in exc_info.value.detail

    def test_get_api_key_from_header_empty(self):
        """Test error when API key header is empty."""
        request = MagicMock()
        request.headers = {"X-API-Key": ""}

        with pytest.raises(HTTPException) as exc_info:
            get_api_key_from_header(request)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_api_key_from_header_whitespace(self):
        """Test trimming whitespace from API key."""
        request = MagicMock()
        request.headers = {"X-API-Key": "  test_api_key_123  "}

        api_key = get_api_key_from_header(request)
        assert api_key == "test_api_key_123"


class TestGetCurrentAPIKey:
    """Test current API key dependency."""

    @pytest.mark.asyncio
    async def test_get_current_api_key_valid(self, sample_api_key):
        """Test getting current API key with valid key."""
        # Mock dependencies
        auth_service = AsyncMock()
        auth_service.validate_api_key.return_value = sample_api_key

        db_session = AsyncMock()
        api_key = "test_api_key_123"

        result = await get_current_api_key(api_key, db_session, auth_service)

        assert result == sample_api_key
        auth_service.validate_api_key.assert_called_once_with(api_key, db_session)

    @pytest.mark.asyncio
    async def test_get_current_api_key_invalid(self):
        """Test error with invalid API key."""
        auth_service = AsyncMock()
        auth_service.validate_api_key.return_value = None

        db_session = AsyncMock()
        api_key = "invalid_key"

        with pytest.raises(HTTPException) as exc_info:
            await get_current_api_key(api_key, db_session, auth_service)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid or expired API key" in exc_info.value.detail


class TestGetCurrentUser:
    """Test current user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user(self, sample_api_key, sample_user):
        """Test getting current user from API key."""
        result = await get_current_user(sample_api_key)

        assert result == sample_user

    @pytest.mark.asyncio
    async def test_get_current_user_no_user(self):
        """Test error when API key has no associated user."""
        api_key = APIKey(
            id=1,
            key="hashed_key",
            name="Test Key",
            user_id=1,
            scopes=["read"],
            active=True,
        )
        api_key.user = None

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(api_key)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetCurrentOrganization:
    """Test current organization dependency."""

    @pytest.mark.asyncio
    async def test_get_current_organization(self, sample_user, sample_organization):
        """Test getting current organization from user."""
        auth_service = AsyncMock()
        auth_service.get_user_organization.return_value = sample_organization

        db_session = AsyncMock()

        result = await get_current_organization(sample_user, db_session, auth_service)

        assert result == sample_organization
        auth_service.get_user_organization.assert_called_once_with(
            sample_user.id, db_session
        )

    @pytest.mark.asyncio
    async def test_get_current_organization_not_found(self, sample_user):
        """Test error when user has no organization."""
        auth_service = AsyncMock()
        auth_service.get_user_organization.return_value = None

        db_session = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await get_current_organization(sample_user, db_session, auth_service)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Organization not found" in exc_info.value.detail


class TestRequireScope:
    """Test scope requirement dependency."""

    @pytest.mark.asyncio
    async def test_require_scope_valid(self, sample_api_key):
        """Test scope requirement with valid scope."""
        auth_service = AsyncMock()
        auth_service.has_permission.return_value = True

        dependency = require_scope("read")
        result = await dependency(sample_api_key, auth_service)

        assert result == sample_api_key
        auth_service.has_permission.assert_called_once_with(sample_api_key, "read")

    @pytest.mark.asyncio
    async def test_require_scope_invalid(self, sample_api_key):
        """Test scope requirement with invalid scope."""
        auth_service = AsyncMock()
        auth_service.has_permission.return_value = False

        dependency = require_scope("admin")

        with pytest.raises(HTTPException) as exc_info:
            await dependency(sample_api_key, auth_service)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Insufficient permissions" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_scope_multiple_scopes(self, sample_api_key):
        """Test scope requirement with multiple valid scopes."""
        auth_service = AsyncMock()
        auth_service.has_permission.side_effect = lambda key, scope: scope in [
            "read",
            "write",
        ]

        dependency = require_scope(["read", "write"])
        result = await dependency(sample_api_key, auth_service)

        assert result == sample_api_key

    @pytest.mark.asyncio
    async def test_require_scope_multiple_scopes_partial(self, sample_api_key):
        """Test scope requirement with some invalid scopes."""
        auth_service = AsyncMock()
        auth_service.has_permission.side_effect = lambda key, scope: scope == "read"

        dependency = require_scope(["read", "admin"])

        with pytest.raises(HTTPException) as exc_info:
            await dependency(sample_api_key, auth_service)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


class TestCheckRateLimit:
    """Test rate limit checking dependency."""

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, sample_api_key):
        """Test rate limit check when request is allowed."""
        rate_limit_service = AsyncMock()
        rate_limit_service.check_rate_limit.return_value = (True, 59)

        result = await check_rate_limit(sample_api_key, rate_limit_service)

        assert result == sample_api_key
        rate_limit_service.check_rate_limit.assert_called_once_with(sample_api_key)

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, sample_api_key):
        """Test rate limit check when limit is exceeded."""
        rate_limit_service = AsyncMock()
        rate_limit_service.check_rate_limit.return_value = (False, 0)
        rate_limit_service.get_rate_limit_info.return_value = {
            "max_requests_per_minute": 60,
            "current_usage": 60,
            "remaining": 0,
            "reset_time": 1234567890,
        }

        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(sample_api_key, rate_limit_service)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Rate limit exceeded" in exc_info.value.detail


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Invalid credentials")
        assert str(error) == "Invalid credentials"

    def test_insufficient_permissions(self):
        """Test InsufficientPermissions exception."""
        error = InsufficientPermissions("Need admin access")
        assert str(error) == "Need admin access"

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceeded exception."""
        error = RateLimitExceeded("Too many requests", reset_time=1234567890)
        assert str(error) == "Too many requests"
        assert error.reset_time == 1234567890
