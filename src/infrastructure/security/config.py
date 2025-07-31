"""Security configuration with strong validation.

This module provides security-specific configuration with validation
to ensure secure defaults and prevent weak configurations.
"""

import secrets
import string
from typing import Optional
from pydantic import BaseModel, Field, field_validator, SecretStr
from enum import Enum


class JWTAlgorithm(str, Enum):
    """Supported JWT signing algorithms."""

    HS256 = "HS256"  # HMAC with SHA-256
    HS384 = "HS384"  # HMAC with SHA-384
    HS512 = "HS512"  # HMAC with SHA-512
    RS256 = "RS256"  # RSA with SHA-256
    RS384 = "RS384"  # RSA with SHA-384
    RS512 = "RS512"  # RSA with SHA-512


class SecurityConfig(BaseModel):
    """Security configuration with validation."""

    # JWT Configuration
    jwt_default_algorithm: JWTAlgorithm = Field(
        default=JWTAlgorithm.HS256, description="Default JWT signing algorithm"
    )
    jwt_issuer: str = Field(default="tldr-api", description="JWT issuer identifier")
    jwt_default_expiry_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Default JWT expiry in hours",
    )

    # Master signing key (for backward compatibility)
    master_signing_key: Optional[SecretStr] = Field(
        default=None, description="Master signing key for backward compatibility"
    )

    # Key requirements
    min_key_length: int = Field(
        default=32, ge=32, description="Minimum signing key length in bytes"
    )
    min_key_entropy_bits: int = Field(
        default=128, ge=128, description="Minimum key entropy in bits"
    )

    # Token security
    require_jti: bool = Field(
        default=True, description="Require JWT ID (jti) claim for replay prevention"
    )
    require_ip_validation: bool = Field(
        default=False, description="Require IP address validation for tokens"
    )
    max_token_lifetime_hours: int = Field(
        default=168,  # 1 week
        ge=1,
        le=8760,  # 1 year
        description="Maximum allowed token lifetime",
    )

    # Rate limiting
    content_access_rate_limit: int = Field(
        default=100, ge=1, description="Max content access requests per minute per IP"
    )
    token_generation_rate_limit: int = Field(
        default=10, ge=1, description="Max token generation requests per minute per org"
    )

    # Revocation settings
    enable_token_revocation: bool = Field(
        default=True, description="Enable token revocation system"
    )
    revocation_check_cache_ttl: int = Field(
        default=300,  # 5 minutes
        ge=60,
        description="TTL for revocation check cache in seconds",
    )

    @field_validator("master_signing_key")
    @classmethod
    def validate_master_key(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        """Validate master signing key strength."""
        if v is None:
            return v

        key_value = v.get_secret_value()

        # Check for weak default values
        weak_defaults = [
            "your-secret-key-here",
            "secret",
            "password",
            "123456",
            "default",
            "changeme",
        ]
        if key_value.lower() in weak_defaults:
            raise ValueError(
                "Master signing key uses a weak default value. "
                "Please generate a strong random key."
            )

        # Check minimum length
        if len(key_value) < 32:
            raise ValueError(
                f"Master signing key must be at least 32 characters, "
                f"got {len(key_value)}"
            )

        # Check entropy (simplified check)
        unique_chars = len(set(key_value))
        if unique_chars < 10:
            raise ValueError(
                "Master signing key has insufficient entropy. "
                "Use a randomly generated key."
            )

        return v

    @staticmethod
    def generate_secure_key(length: int = 32) -> str:
        """Generate a cryptographically secure random key.

        Args:
            length: Key length in bytes

        Returns:
            Secure random key as hex string
        """
        return secrets.token_hex(length)

    @staticmethod
    def validate_key_strength(key: str, min_length: int = 32) -> tuple[bool, str]:
        """Validate key strength.

        Args:
            key: Key to validate
            min_length: Minimum required length

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(key) < min_length:
            return False, f"Key must be at least {min_length} characters"

        # Check for common weak patterns
        if key.lower() in ["password", "secret", "123456", "default"]:
            return False, "Key uses a common weak value"

        # Check character diversity
        unique_chars = len(set(key))
        if unique_chars < min_length // 3:
            return False, "Key has insufficient character diversity"

        # Check for sequential patterns
        if any(
            key[i : i + 3] in string.ascii_lowercase
            or key[i : i + 3] in string.ascii_uppercase
            or key[i : i + 3] in string.digits
            for i in range(len(key) - 2)
        ):
            return False, "Key contains sequential patterns"

        return True, ""

    class Config:
        """Pydantic config."""

        use_enum_values = True
