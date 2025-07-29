"""Tests for authentication utilities."""

from datetime import datetime, timedelta, timezone

from src.utils.auth import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
    hash_password,
    verify_password,
    create_jwt_token,
    verify_jwt_token,
    decode_jwt_token,
)


class TestAPIKeyGeneration:
    """Test API key generation and hashing."""

    def test_generate_api_key_length(self):
        """Test that generated API keys have correct length."""
        key = generate_api_key(32)
        assert len(key) == 64  # 32 bytes -> 64 hex chars

    def test_generate_api_key_uniqueness(self):
        """Test that generated API keys are unique."""
        keys = [generate_api_key(32) for _ in range(100)]
        assert len(set(keys)) == 100

    def test_generate_api_key_format(self):
        """Test that generated API keys are valid hex strings."""
        key = generate_api_key(32)
        # Should not raise ValueError
        int(key, 16)

    def test_hash_api_key(self):
        """Test API key hashing."""
        key = "test_api_key_12345"
        hashed = hash_api_key(key)

        assert hashed != key
        assert len(hashed) > len(key)
        assert hashed.startswith("$2b$")

    def test_verify_api_key_valid(self):
        """Test API key verification with valid key."""
        key = "test_api_key_12345"
        hashed = hash_api_key(key)

        assert verify_api_key(key, hashed) is True

    def test_verify_api_key_invalid(self):
        """Test API key verification with invalid key."""
        key = "test_api_key_12345"
        wrong_key = "wrong_api_key_12345"
        hashed = hash_api_key(key)

        assert verify_api_key(wrong_key, hashed) is False


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > len(password)
        assert hashed.startswith("$2b$")

    def test_verify_password_valid(self):
        """Test password verification with valid password."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_invalid(self):
        """Test password verification with invalid password."""
        password = "test_password_123"
        wrong_password = "wrong_password_123"
        hashed = hash_password(password)

        assert verify_password(wrong_password, hashed) is False

    def test_hash_password_uniqueness(self):
        """Test that same password produces different hashes."""
        password = "test_password_123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestJWTTokens:
    """Test JWT token creation and verification."""

    def test_create_jwt_token(self):
        """Test JWT token creation."""
        payload = {"user_id": 123, "email": "test@example.com"}
        token = create_jwt_token(payload)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # Header.Payload.Signature

    def test_create_jwt_token_with_expiration(self):
        """Test JWT token creation with custom expiration."""
        payload = {"user_id": 123}
        expires_delta = timedelta(hours=2)
        token = create_jwt_token(payload, expires_delta)

        decoded = decode_jwt_token(token)
        assert decoded is not None

        # Check expiration is set correctly
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        expected_exp = datetime.now(timezone.utc) + expires_delta

        # Allow 1 second tolerance
        assert abs((exp_time - expected_exp).total_seconds()) < 1

    def test_verify_jwt_token_valid(self):
        """Test JWT token verification with valid token."""
        payload = {"user_id": 123, "email": "test@example.com"}
        token = create_jwt_token(payload)

        assert verify_jwt_token(token) is True

    def test_verify_jwt_token_invalid(self):
        """Test JWT token verification with invalid token."""
        invalid_token = "invalid.jwt.token"

        assert verify_jwt_token(invalid_token) is False

    def test_verify_jwt_token_expired(self):
        """Test JWT token verification with expired token."""
        payload = {"user_id": 123}
        expires_delta = timedelta(seconds=-1)  # Already expired
        token = create_jwt_token(payload, expires_delta)

        assert verify_jwt_token(token) is False

    def test_decode_jwt_token_valid(self):
        """Test JWT token decoding with valid token."""
        payload = {"user_id": 123, "email": "test@example.com"}
        token = create_jwt_token(payload)

        decoded = decode_jwt_token(token)
        assert decoded is not None
        assert decoded["user_id"] == 123
        assert decoded["email"] == "test@example.com"
        assert "exp" in decoded
        assert "iat" in decoded

    def test_decode_jwt_token_invalid(self):
        """Test JWT token decoding with invalid token."""
        invalid_token = "invalid.jwt.token"

        decoded = decode_jwt_token(invalid_token)
        assert decoded is None

    def test_decode_jwt_token_expired(self):
        """Test JWT token decoding with expired token."""
        payload = {"user_id": 123}
        expires_delta = timedelta(seconds=-1)  # Already expired
        token = create_jwt_token(payload, expires_delta)

        decoded = decode_jwt_token(token)
        assert decoded is None
