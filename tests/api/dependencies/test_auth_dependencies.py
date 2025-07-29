"""Tests for authentication dependencies."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import jwt

from src.api.dependencies.auth import (
    AuthenticationError,
    PermissionError,
    hash_password,
    verify_password,
    generate_api_key,
    hash_api_key,
    create_access_token,
    verify_token,
    get_current_user_from_token,
    get_current_user_from_api_key,
    get_current_user,
    require_scopes,
    require_admin,
    get_optional_user,
)
from src.api.schemas.auth import TokenData
from src.models.user import User
from src.models.api_key import APIKey
from src.core.config import get_settings

settings = get_settings()


class TestPasswordUtils:
    """Test password hashing utilities."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt prefix
        assert len(hashed) > 50

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        password = "test_password123"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        password = "test_password123"
        wrong_password = "wrong_password"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False


class TestAPIKeyUtils:
    """Test API key generation and hashing."""

    def test_generate_api_key(self):
        """Test API key generation."""
        api_key = generate_api_key()
        
        assert api_key.startswith("tldr_sk_")
        assert len(api_key) == 40  # tldr_sk_ (8) + 32 chars
        
        # Generate multiple keys and ensure they're unique
        keys = [generate_api_key() for _ in range(10)]
        assert len(set(keys)) == 10

    def test_hash_api_key(self):
        """Test API key hashing."""
        api_key = "tldr_sk_test_key_12345"
        hashed = hash_api_key(api_key)
        
        assert hashed != api_key
        assert len(hashed) == 64  # SHA256 hex digest
        
        # Same key should produce same hash
        assert hash_api_key(api_key) == hashed
        
        # Different key should produce different hash
        different_key = "tldr_sk_different_key"
        assert hash_api_key(different_key) != hashed


class TestJWTTokens:
    """Test JWT token creation and verification."""

    def test_create_access_token(self):
        """Test creating access token."""
        user_id = 123
        email = "test@example.com"
        scopes = ["read", "write"]
        
        token = create_access_token(user_id, email, scopes)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify payload
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        assert payload["user_id"] == user_id
        assert payload["email"] == email
        assert payload["scopes"] == scopes
        assert "exp" in payload
        assert "iat" in payload

    def test_create_access_token_no_scopes(self):
        """Test creating access token without scopes."""
        token = create_access_token(123, "test@example.com")
        
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        assert payload["scopes"] == []

    def test_verify_token_valid(self):
        """Test verifying valid token."""
        user_id = 123
        email = "test@example.com"
        scopes = ["read"]
        
        token = create_access_token(user_id, email, scopes)
        token_data = verify_token(token)
        
        assert token_data.user_id == user_id
        assert token_data.email == email
        assert token_data.scopes == scopes

    def test_verify_token_expired(self):
        """Test verifying expired token."""
        # Create token with past expiration
        payload = {
            "user_id": 123,
            "email": "test@example.com",
            "scopes": [],
            "exp": (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp(),
            "iat": datetime.now(timezone.utc).timestamp(),
        }
        
        expired_token = jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")
        
        with pytest.raises(AuthenticationError) as exc_info:
            verify_token(expired_token)
        assert "Token has expired" in str(exc_info.value.detail)

    def test_verify_token_invalid(self):
        """Test verifying invalid token."""
        with pytest.raises(AuthenticationError) as exc_info:
            verify_token("invalid_token")
        assert "Invalid token" in str(exc_info.value.detail)

    def test_verify_token_missing_fields(self):
        """Test verifying token with missing fields."""
        # Token without user_id
        payload = {
            "email": "test@example.com",
            "scopes": [],
            "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
        }
        
        invalid_token = jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")
        
        with pytest.raises(AuthenticationError) as exc_info:
            verify_token(invalid_token)
        assert "Invalid token payload" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestUserAuthentication:
    """Test user authentication dependencies."""

    async def test_get_current_user_from_token(self, db_session: AsyncSession):
        """Test getting user from JWT token."""
        # Create mock user
        mock_user = User(id=123, email="test@example.com")
        
        # Create token
        token = create_access_token(123, "test@example.com", ["read"])
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=token
        )
        
        with patch("src.api.dependencies.auth.select") as mock_select:
            # Mock database query
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = mock_user
            db_session.execute = AsyncMock(return_value=mock_result)
            
            user = await get_current_user_from_token(credentials, db_session)
            
            assert user == mock_user
            assert user.id == 123
            assert user.email == "test@example.com"

    async def test_get_current_user_from_token_user_not_found(
        self, db_session: AsyncSession
    ):
        """Test getting user from token when user doesn't exist."""
        token = create_access_token(123, "test@example.com")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=token
        )
        
        with patch("src.api.dependencies.auth.select"):
            # Mock database query returning None
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = None
            db_session.execute = AsyncMock(return_value=mock_result)
            
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user_from_token(credentials, db_session)
            assert "User not found" in str(exc_info.value.detail)

    async def test_get_current_user_from_api_key(self, db_session: AsyncSession):
        """Test getting user from API key."""
        # Create mock objects
        mock_user = User(id=123, email="test@example.com")
        mock_api_key = APIKey(
            id=1,
            user_id=123,
            key=hash_api_key("tldr_sk_test_key"),
            active=True,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30)
        )
        mock_api_key.is_expired = MagicMock(return_value=False)
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="tldr_sk_test_key"
        )
        
        with patch("src.api.dependencies.auth.select") as mock_select:
            # Mock database queries
            mock_api_result = AsyncMock()
            mock_api_result.scalar_one_or_none.return_value = mock_api_key
            
            mock_user_result = AsyncMock()
            mock_user_result.scalar_one_or_none.return_value = mock_user
            
            db_session.execute = AsyncMock(
                side_effect=[mock_api_result, mock_user_result]
            )
            db_session.commit = AsyncMock()
            
            user, api_key = await get_current_user_from_api_key(credentials, db_session)
            
            assert user == mock_user
            assert api_key == mock_api_key
            assert api_key.last_used_at is not None

    async def test_get_current_user_from_api_key_invalid_format(
        self, db_session: AsyncSession
    ):
        """Test API key with invalid format."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_key_format"
        )
        
        with pytest.raises(AuthenticationError) as exc_info:
            await get_current_user_from_api_key(credentials, db_session)
        assert "Invalid API key format" in str(exc_info.value.detail)

    async def test_get_current_user_from_api_key_not_found(
        self, db_session: AsyncSession
    ):
        """Test API key not found in database."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="tldr_sk_nonexistent"
        )
        
        with patch("src.api.dependencies.auth.select"):
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = None
            db_session.execute = AsyncMock(return_value=mock_result)
            
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user_from_api_key(credentials, db_session)
            assert "Invalid API key" in str(exc_info.value.detail)

    async def test_get_current_user_from_api_key_inactive(
        self, db_session: AsyncSession
    ):
        """Test inactive API key."""
        mock_api_key = APIKey(
            id=1,
            user_id=123,
            key=hash_api_key("tldr_sk_test_key"),
            active=False
        )
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="tldr_sk_test_key"
        )
        
        with patch("src.api.dependencies.auth.select"):
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = mock_api_key
            db_session.execute = AsyncMock(return_value=mock_result)
            
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user_from_api_key(credentials, db_session)
            assert "API key is inactive" in str(exc_info.value.detail)

    async def test_get_current_user_from_api_key_expired(
        self, db_session: AsyncSession
    ):
        """Test expired API key."""
        mock_api_key = APIKey(
            id=1,
            user_id=123,
            key=hash_api_key("tldr_sk_test_key"),
            active=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
        mock_api_key.is_expired = MagicMock(return_value=True)
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="tldr_sk_test_key"
        )
        
        with patch("src.api.dependencies.auth.select"):
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = mock_api_key
            db_session.execute = AsyncMock(return_value=mock_result)
            
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user_from_api_key(credentials, db_session)
            assert "API key has expired" in str(exc_info.value.detail)

    async def test_get_current_user_jwt_token(self, db_session: AsyncSession):
        """Test get_current_user with JWT token."""
        mock_user = User(id=123, email="test@example.com")
        token = create_access_token(123, "test@example.com")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=token
        )
        
        with patch(
            "src.api.dependencies.auth.get_current_user_from_token",
            return_value=mock_user
        ):
            user = await get_current_user(credentials, db_session)
            assert user == mock_user

    async def test_get_current_user_api_key(self, db_session: AsyncSession):
        """Test get_current_user with API key."""
        mock_user = User(id=123, email="test@example.com")
        mock_api_key = APIKey(id=1, user_id=123)
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="tldr_sk_test_key"
        )
        
        with patch(
            "src.api.dependencies.auth.get_current_user_from_api_key",
            return_value=(mock_user, mock_api_key)
        ):
            user = await get_current_user(credentials, db_session)
            assert user == mock_user

    async def test_get_current_user_invalid_credentials(
        self, db_session: AsyncSession
    ):
        """Test get_current_user with invalid credentials."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_token"
        )
        
        with patch(
            "src.api.dependencies.auth.get_current_user_from_token",
            side_effect=AuthenticationError()
        ), patch(
            "src.api.dependencies.auth.get_current_user_from_api_key",
            side_effect=AuthenticationError()
        ):
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(credentials, db_session)
            assert "Invalid authentication credentials" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestScopeRequirements:
    """Test scope-based authorization."""

    async def test_require_scopes_with_api_key(self, db_session: AsyncSession):
        """Test requiring scopes with API key."""
        mock_user = User(id=123, email="test@example.com")
        mock_api_key = APIKey(id=1, user_id=123, scopes=["read", "write"])
        mock_api_key.has_scope = MagicMock(side_effect=lambda s: s in ["read", "write"])
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="tldr_sk_test_key"
        )
        
        check_scopes = require_scopes(["read"])
        
        with patch(
            "src.api.dependencies.auth.get_current_user_from_api_key",
            return_value=(mock_user, mock_api_key)
        ):
            user = await check_scopes(credentials, db_session)
            assert user == mock_user

    async def test_require_scopes_missing_scope(self, db_session: AsyncSession):
        """Test requiring scopes when scope is missing."""
        mock_user = User(id=123, email="test@example.com")
        mock_api_key = APIKey(id=1, user_id=123, scopes=["read"])
        mock_api_key.has_scope = MagicMock(side_effect=lambda s: s == "read")
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="tldr_sk_test_key"
        )
        
        check_scopes = require_scopes(["write"])
        
        with patch(
            "src.api.dependencies.auth.get_current_user_from_api_key",
            return_value=(mock_user, mock_api_key)
        ):
            with pytest.raises(PermissionError) as exc_info:
                await check_scopes(credentials, db_session)
            assert "Missing required scope: write" in str(exc_info.value.detail)

    async def test_require_scopes_with_jwt(self, db_session: AsyncSession):
        """Test requiring scopes with JWT token."""
        mock_user = User(id=123, email="test@example.com")
        token = create_access_token(123, "test@example.com", ["read", "write"])
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=token
        )
        
        check_scopes = require_scopes(["read"])
        
        with patch("src.api.dependencies.auth.select"):
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = mock_user
            db_session.execute = AsyncMock(return_value=mock_result)
            
            user = await check_scopes(credentials, db_session)
            assert user == mock_user

    async def test_require_admin(self, db_session: AsyncSession):
        """Test require_admin dependency."""
        mock_user = User(id=123, email="test@example.com")
        token = create_access_token(123, "test@example.com", ["admin"])
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=token
        )
        
        check_admin = require_admin()
        
        with patch("src.api.dependencies.auth.select"):
            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = mock_user
            db_session.execute = AsyncMock(return_value=mock_result)
            
            user = await check_admin(credentials, db_session)
            assert user == mock_user


@pytest.mark.asyncio
class TestOptionalAuthentication:
    """Test optional authentication."""

    async def test_get_optional_user_authenticated(self, db_session: AsyncSession):
        """Test getting optional user when authenticated."""
        mock_user = User(id=123, email="test@example.com")
        token = create_access_token(123, "test@example.com")
        
        request = MagicMock(spec=Request)
        request.headers = {"authorization": f"Bearer {token}"}
        
        with patch(
            "src.api.dependencies.auth.get_current_user",
            return_value=mock_user
        ):
            user = await get_optional_user(request, db_session)
            assert user == mock_user

    async def test_get_optional_user_no_auth_header(self, db_session: AsyncSession):
        """Test getting optional user with no auth header."""
        request = MagicMock(spec=Request)
        request.headers = {}
        
        user = await get_optional_user(request, db_session)
        assert user is None

    async def test_get_optional_user_invalid_header(self, db_session: AsyncSession):
        """Test getting optional user with invalid header format."""
        request = MagicMock(spec=Request)
        request.headers = {"authorization": "InvalidFormat"}
        
        user = await get_optional_user(request, db_session)
        assert user is None

    async def test_get_optional_user_invalid_token(self, db_session: AsyncSession):
        """Test getting optional user with invalid token."""
        request = MagicMock(spec=Request)
        request.headers = {"authorization": "Bearer invalid_token"}
        
        with patch(
            "src.api.dependencies.auth.get_current_user",
            side_effect=AuthenticationError()
        ):
            user = await get_optional_user(request, db_session)
            assert user is None


class TestExceptions:
    """Test custom exception classes."""

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Custom auth error")
        
        assert error.status_code == 401
        assert error.detail == "Custom auth error"
        assert error.headers == {"WWW-Authenticate": "Bearer"}

    def test_authentication_error_default(self):
        """Test AuthenticationError with default message."""
        error = AuthenticationError()
        
        assert error.status_code == 401
        assert error.detail == "Authentication failed"

    def test_permission_error(self):
        """Test PermissionError exception."""
        error = PermissionError("Custom permission error")
        
        assert error.status_code == 403
        assert error.detail == "Custom permission error"

    def test_permission_error_default(self):
        """Test PermissionError with default message."""
        error = PermissionError()
        
        assert error.status_code == 403
        assert error.detail == "Insufficient permissions"