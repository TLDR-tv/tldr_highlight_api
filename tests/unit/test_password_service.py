"""Unit tests for password service."""

import pytest

from src.infrastructure.security.password_service import PasswordService


@pytest.fixture
def password_service():
    """Create password service instance."""
    return PasswordService()


class TestPasswordService:
    """Test password service functionality."""

    def test_hash_password(self, password_service: PasswordService):
        """Test password hashing."""
        password = "SecurePassword123!"

        hashed = password_service.hash_password(password)

        assert isinstance(hashed, str)
        assert len(hashed) > 0
        assert hashed != password  # Should not be plain text
        assert hashed.startswith("$2b$")  # bcrypt hash format

    def test_verify_correct_password(self, password_service: PasswordService):
        """Test verifying correct password."""
        password = "SecurePassword123!"

        hashed = password_service.hash_password(password)
        is_valid = password_service.verify_password(password, hashed)

        assert is_valid is True

    def test_verify_incorrect_password(self, password_service: PasswordService):
        """Test verifying incorrect password."""
        password = "SecurePassword123!"
        wrong_password = "WrongPassword123!"

        hashed = password_service.hash_password(password)
        is_valid = password_service.verify_password(wrong_password, hashed)

        assert is_valid is False

    def test_different_hashes_for_same_password(
        self, password_service: PasswordService
    ):
        """Test that same password produces different hashes."""
        password = "SecurePassword123!"

        hash1 = password_service.hash_password(password)
        hash2 = password_service.hash_password(password)

        # Hashes should be different due to random salt
        assert hash1 != hash2

        # But both should verify correctly
        assert password_service.verify_password(password, hash1) is True
        assert password_service.verify_password(password, hash2) is True

    def test_validate_password_strength_valid(self, password_service: PasswordService):
        """Test validating strong password."""
        password = "SecurePass123!"

        is_valid, errors = password_service.validate_password_strength(password)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_password_too_short(self, password_service: PasswordService):
        """Test password that's too short."""
        password = "Short1!"

        is_valid, errors = password_service.validate_password_strength(password)

        assert is_valid is False
        assert len(errors) > 0
        assert any("8 characters" in error for error in errors)

    def test_validate_password_no_uppercase(self, password_service: PasswordService):
        """Test password without uppercase letter."""
        password = "password123!"

        is_valid, errors = password_service.validate_password_strength(password)

        assert is_valid is False
        assert any("uppercase" in error for error in errors)

    def test_validate_password_no_lowercase(self, password_service: PasswordService):
        """Test password without lowercase letter."""
        password = "PASSWORD123!"

        is_valid, errors = password_service.validate_password_strength(password)

        assert is_valid is False
        assert any("lowercase" in error for error in errors)

    def test_validate_password_no_number(self, password_service: PasswordService):
        """Test password without number."""
        password = "SecurePassword!"

        is_valid, errors = password_service.validate_password_strength(password)

        assert is_valid is False
        assert any("digit" in error for error in errors)

    def test_validate_password_no_special_char(self, password_service: PasswordService):
        """Test password without special character."""
        password = "SecurePassword123"

        is_valid, errors = password_service.validate_password_strength(password)

        assert is_valid is False
        assert any("special character" in error for error in errors)

    def test_validate_password_multiple_errors(self, password_service: PasswordService):
        """Test password with multiple validation errors."""
        password = "weak"

        is_valid, errors = password_service.validate_password_strength(password)

        assert is_valid is False
        assert len(errors) >= 4  # Too short, no uppercase, no number, no special char

    def test_hash_empty_password(self, password_service: PasswordService):
        """Test hashing empty password."""
        # Should still hash successfully (validation is separate)
        hashed = password_service.hash_password("")

        assert isinstance(hashed, str)
        assert password_service.verify_password("", hashed) is True

    def test_verify_with_invalid_hash(self, password_service: PasswordService):
        """Test verifying password with invalid hash."""
        from passlib.exc import UnknownHashError

        password = "SecurePassword123!"
        invalid_hash = "not-a-valid-hash"

        # Should raise an exception for invalid hash format
        with pytest.raises(UnknownHashError):
            password_service.verify_password(password, invalid_hash)
