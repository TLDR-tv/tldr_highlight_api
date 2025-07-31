"""Secure key generation and management service.

This module provides services for generating, storing, and managing
organization-specific signing keys.
"""

import secrets
import uuid
from typing import Optional, Tuple
from datetime import datetime
import logging

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from src.domain.models.organization_key import OrganizationKey, KeyAlgorithm
from src.infrastructure.security.config import SecurityConfig


logger = logging.getLogger(__name__)


class KeyEncryption:
    """Handles encryption/decryption of key values at rest."""

    def __init__(self, master_key: str):
        """Initialize with master encryption key.

        Args:
            master_key: Master key for deriving encryption keys
        """
        self.master_key = master_key.encode()

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive an encryption key from master key and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.master_key))

    def encrypt(self, key_value: str, key_id: str) -> str:
        """Encrypt a key value.

        Args:
            key_value: Raw key value to encrypt
            key_id: Key ID to use as salt component

        Returns:
            Encrypted key value
        """
        salt = key_id.encode()[:16].ljust(16, b"0")
        fernet_key = self._derive_key(salt)
        f = Fernet(fernet_key)
        return f.encrypt(key_value.encode()).decode()

    def decrypt(self, encrypted_value: str, key_id: str) -> str:
        """Decrypt a key value.

        Args:
            encrypted_value: Encrypted key value
            key_id: Key ID used as salt component

        Returns:
            Decrypted key value
        """
        salt = key_id.encode()[:16].ljust(16, b"0")
        fernet_key = self._derive_key(salt)
        f = Fernet(fernet_key)
        return f.decrypt(encrypted_value.encode()).decode()


class KeyGenerationService:
    """Service for generating and managing organization signing keys."""

    def __init__(self, security_config: SecurityConfig, master_encryption_key: str):
        """Initialize key generation service.

        Args:
            security_config: Security configuration
            master_encryption_key: Master key for encrypting stored keys
        """
        self.config = security_config
        self.encryptor = KeyEncryption(master_encryption_key)

    def generate_key_id(self, organization_id: int) -> str:
        """Generate a unique public key identifier.

        Args:
            organization_id: Organization ID

        Returns:
            Unique key identifier
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"org{organization_id}_key_{timestamp}_{unique_id}"

    def generate_signing_key(
        self,
        algorithm: KeyAlgorithm = KeyAlgorithm.HS256,
        key_length: Optional[int] = None,
    ) -> str:
        """Generate a cryptographically secure signing key.

        Args:
            algorithm: Signing algorithm
            key_length: Key length in bytes (uses config default if not specified)

        Returns:
            Secure random key
        """
        if key_length is None:
            key_length = self.config.min_key_length

        # For HMAC algorithms, generate random bytes
        if algorithm.value.startswith("HS"):
            return secrets.token_hex(key_length)

        # For RSA algorithms, we would generate key pairs
        # This is a simplified implementation
        raise NotImplementedError(f"Key generation for {algorithm} not implemented")

    def validate_key(self, key_value: str) -> Tuple[bool, Optional[str]]:
        """Validate a signing key meets security requirements.

        Args:
            key_value: Key to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check minimum length
        if len(key_value) < self.config.min_key_length:
            return (
                False,
                f"Key must be at least {self.config.min_key_length} characters",
            )

        # Check for weak patterns
        return SecurityConfig.validate_key_strength(
            key_value, self.config.min_key_length
        )

    def create_organization_key(
        self,
        organization_id: int,
        algorithm: KeyAlgorithm = KeyAlgorithm.HS256,
        expires_in_days: Optional[int] = 365,
        created_by: Optional[str] = None,
        description: Optional[str] = None,
    ) -> OrganizationKey:
        """Create a new organization signing key.

        Args:
            organization_id: Organization ID
            algorithm: Signing algorithm
            expires_in_days: Key expiry in days
            created_by: Creator identifier
            description: Key description

        Returns:
            New OrganizationKey with encrypted key value
        """
        # Generate key ID and value
        key_id = self.generate_key_id(organization_id)
        key_value = self.generate_signing_key(algorithm)

        # Validate the generated key
        is_valid, error = self.validate_key(key_value)
        if not is_valid:
            raise ValueError(f"Generated key failed validation: {error}")

        # Encrypt the key value
        encrypted_value = self.encryptor.encrypt(key_value, key_id)

        # Create domain model
        return OrganizationKey.create_new(
            organization_id=organization_id,
            key_id=key_id,
            key_value=encrypted_value,
            algorithm=algorithm,
            expires_in_days=expires_in_days,
            created_by=created_by,
            description=description
            or f"Signing key for organization {organization_id}",
        )

    def decrypt_key_value(self, org_key: OrganizationKey) -> str:
        """Decrypt an organization key value.

        Args:
            org_key: OrganizationKey with encrypted value

        Returns:
            Decrypted key value
        """
        return self.encryptor.decrypt(org_key.key_value, org_key.key_id)

    def rotate_key(
        self,
        current_key: OrganizationKey,
        reason: str,
        created_by: Optional[str] = None,
    ) -> OrganizationKey:
        """Rotate an organization key.

        Args:
            current_key: Current key to rotate
            reason: Reason for rotation
            created_by: Who initiated the rotation

        Returns:
            New organization key
        """
        # Create new key with same organization
        new_key = self.create_organization_key(
            organization_id=current_key.organization_id,
            algorithm=current_key.algorithm,
            expires_in_days=365,  # Default to 1 year
            created_by=created_by,
            description=f"Rotated from {current_key.key_id}: {reason}",
        )

        # Update key version
        new_key.key_version = current_key.key_version + 1

        # Link rotation
        current_key.rotate(new_key, reason)

        logger.info(
            f"Rotated key for organization {current_key.organization_id}: "
            f"{current_key.key_id} -> {new_key.key_id}"
        )

        return new_key

    def should_rotate_key(self, org_key: OrganizationKey) -> Tuple[bool, Optional[str]]:
        """Check if a key should be rotated.

        Args:
            org_key: Key to check

        Returns:
            Tuple of (should_rotate, reason)
        """
        # Check if expired
        if org_key.is_expired:
            return True, "Key has expired"

        # Check if nearing expiry (30 days)
        if org_key.expires_at:
            days_until_expiry = (org_key.expires_at - datetime.utcnow()).days
            if days_until_expiry < 30:
                return True, f"Key expires in {days_until_expiry} days"

        # Check usage count (rotate after 1M uses)
        if org_key.usage_count > 1_000_000:
            return True, "Key usage exceeded 1M signatures"

        # Check age (rotate after 6 months even if not expired)
        key_age = datetime.utcnow() - org_key.created_at
        if key_age.days > 180:
            return True, f"Key is {key_age.days} days old"

        return False, None
