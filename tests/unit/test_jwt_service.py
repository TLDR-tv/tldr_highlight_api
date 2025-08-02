"""Unit tests for JWT service."""

import pytest
from datetime import datetime, timedelta, UTC
from uuid import uuid4

from src.infrastructure.security.jwt_service import JWTService
from src.infrastructure.config import Settings


@pytest.fixture
def jwt_service():
    """Create JWT service instance."""
    settings = Settings(
        jwt_secret_key="test-secret-key",
        jwt_algorithm="HS256",
        jwt_expiry_seconds=3600,
    )
    return JWTService(settings)


class TestJWTService:
    """Test JWT service functionality."""

    def test_create_access_token(self, jwt_service: JWTService):
        """Test creating access token."""
        user_id = uuid4()
        org_id = uuid4()

        token = jwt_service.create_access_token(
            user_id=user_id,
            organization_id=org_id,
            email="test@example.com",
            role="member",
        )

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_access_token(self, jwt_service: JWTService):
        """Test verifying valid access token."""
        user_id = uuid4()
        org_id = uuid4()

        token = jwt_service.create_access_token(
            user_id=user_id,
            organization_id=org_id,
            email="test@example.com",
            role="admin",
        )

        payload = jwt_service.verify_access_token(token)

        assert payload is not None
        assert payload.sub == str(user_id)
        assert payload.org == str(org_id)
        assert payload.email == "test@example.com"
        assert payload.role == "admin"
        assert payload.type == "access"

    def test_verify_expired_access_token(self, jwt_service: JWTService):
        """Test verifying expired access token."""
        user_id = uuid4()
        org_id = uuid4()

        # Create token with negative expiry (in the past)
        token = jwt_service.create_access_token(
            user_id=user_id,
            organization_id=org_id,
            email="test@example.com",
            role="member",
            expiry_seconds=-1,  # Expired 1 second ago
        )

        payload = jwt_service.verify_access_token(token)
        assert payload is None

    def test_verify_invalid_access_token(self, jwt_service: JWTService):
        """Test verifying invalid access token."""
        payload = jwt_service.verify_access_token("invalid-token")
        assert payload is None

    def test_create_refresh_token(self, jwt_service: JWTService):
        """Test creating refresh token."""
        user_id = uuid4()

        token = jwt_service.create_refresh_token(user_id=user_id)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_refresh_token(self, jwt_service: JWTService):
        """Test verifying valid refresh token."""
        user_id = uuid4()
        jti = str(uuid4())

        token = jwt_service.create_refresh_token(
            user_id=user_id,
            jti=jti,
        )

        payload = jwt_service.verify_refresh_token(token)

        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "refresh"
        assert payload["jti"] == jti

    def test_verify_wrong_token_type(self, jwt_service: JWTService):
        """Test verifying wrong token type."""
        user_id = uuid4()

        # Create refresh token
        refresh_token = jwt_service.create_refresh_token(user_id=user_id)

        # Try to verify as access token
        payload = jwt_service.verify_access_token(refresh_token)
        assert payload is None

        # Create access token
        access_token = jwt_service.create_access_token(
            user_id=user_id,
            organization_id=uuid4(),
            email="test@example.com",
            role="member",
        )

        # Try to verify as refresh token
        payload = jwt_service.verify_refresh_token(access_token)
        assert payload is None

    def test_create_password_reset_token(self, jwt_service: JWTService):
        """Test creating password reset token."""
        user_id = uuid4()

        token = jwt_service.create_password_reset_token(
            user_id=user_id,
            email="test@example.com",
        )

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_password_reset_token(self, jwt_service: JWTService):
        """Test verifying valid password reset token."""
        user_id = uuid4()
        email = "test@example.com"

        token = jwt_service.create_password_reset_token(
            user_id=user_id,
            email=email,
        )

        payload = jwt_service.verify_password_reset_token(token)

        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["email"] == email
        assert payload["type"] == "password_reset"

    def test_token_expiry_times(self, jwt_service: JWTService):
        """Test token expiry times are set correctly."""
        user_id = uuid4()
        org_id = uuid4()

        # Access token with custom expiry
        token = jwt_service.create_access_token(
            user_id=user_id,
            organization_id=org_id,
            email="test@example.com",
            role="member",
            expiry_seconds=7200,  # 2 hours
        )

        payload = jwt_service.verify_access_token(token)
        assert payload is not None

        # Check expiry is approximately 2 hours from now
        expected_exp = datetime.now(UTC) + timedelta(seconds=7200)
        actual_exp = payload.exp

        # Allow 5 seconds tolerance
        assert abs((expected_exp - actual_exp).total_seconds()) < 5
