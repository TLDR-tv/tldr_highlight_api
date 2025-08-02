"""Password hashing and verification service."""

from passlib.context import CryptContext


class PasswordService:
    """Service for secure password hashing and verification."""

    def __init__(self):
        """Initialize password context with bcrypt."""
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12,  # Good balance of security and performance
        )

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password string

        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to compare against

        Returns:
            True if password matches, False otherwise

        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def needs_rehash(self, hashed_password: str) -> bool:
        """Check if password hash needs to be updated.

        This is useful for upgrading hashes when security requirements change.

        Args:
            hashed_password: Existing password hash

        Returns:
            True if hash should be regenerated

        """
        return self.pwd_context.needs_update(hashed_password)

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """Validate password meets security requirements.

        Requirements:
        - At least 8 characters long
        - Contains at least one uppercase letter
        - Contains at least one lowercase letter
        - Contains at least one digit
        - Contains at least one special character

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, error_messages)

        """
        errors = []

        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")

        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors
