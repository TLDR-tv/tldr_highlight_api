"""Domain model for organization signing keys.

This module defines the domain model for organization-specific signing keys
used for generating secure signed URLs.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum


class KeyAlgorithm(str, Enum):
    """Supported signing algorithms."""

    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"
    RS256 = "RS256"
    RS384 = "RS384"
    RS512 = "RS512"


class KeyStatus(str, Enum):
    """Key lifecycle status."""

    ACTIVE = "active"
    ROTATING = "rotating"
    DEACTIVATED = "deactivated"
    EXPIRED = "expired"
    COMPROMISED = "compromised"


@dataclass
class OrganizationKey:
    """Organization signing key domain model."""

    # Identity
    id: Optional[int]
    organization_id: int
    key_id: str  # Public identifier for the key

    # Key material (encrypted at rest)
    key_value: str
    algorithm: KeyAlgorithm

    # Versioning
    key_version: int
    is_primary: bool  # Primary key for new signatures

    # Lifecycle
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    rotated_at: Optional[datetime]
    deactivated_at: Optional[datetime]

    # Usage metrics
    last_used_at: Optional[datetime]
    usage_count: int

    # Rotation tracking
    previous_key_id: Optional[str]
    rotation_reason: Optional[str]

    # Metadata
    created_by: Optional[str]
    description: Optional[str]

    @property
    def is_active(self) -> bool:
        """Check if key is currently active."""
        if self.status != KeyStatus.ACTIVE:
            return False

        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False

        return True

    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        return self.expires_at and datetime.utcnow() > self.expires_at

    @property
    def can_verify(self) -> bool:
        """Check if key can be used for verification.

        Keys can verify tokens even when not active for signing,
        to support grace periods during rotation.
        """
        # Compromised keys cannot verify
        if self.status == KeyStatus.COMPROMISED:
            return False

        # Allow verification during rotation
        if self.status == KeyStatus.ROTATING:
            return True

        # Active non-expired keys can verify
        return self.is_active

    @property
    def can_sign(self) -> bool:
        """Check if key can be used for signing new tokens."""
        return self.is_active and self.is_primary

    def rotate(self, new_key: "OrganizationKey", reason: str) -> None:
        """Mark this key as rotating to a new key.

        Args:
            new_key: The new key replacing this one
            reason: Reason for rotation
        """
        self.status = KeyStatus.ROTATING
        self.is_primary = False
        self.rotated_at = datetime.utcnow()
        self.rotation_reason = reason
        new_key.previous_key_id = self.key_id

    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate this key.

        Args:
            reason: Optional reason for deactivation
        """
        self.status = KeyStatus.DEACTIVATED
        self.is_primary = False
        self.deactivated_at = datetime.utcnow()
        if reason:
            self.rotation_reason = reason

    def mark_compromised(self) -> None:
        """Mark this key as compromised."""
        self.status = KeyStatus.COMPROMISED
        self.is_primary = False
        self.deactivated_at = datetime.utcnow()
        self.rotation_reason = "Key compromised"

    def increment_usage(self) -> None:
        """Increment usage counter and update last used timestamp."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()

    @classmethod
    def create_new(
        cls,
        organization_id: int,
        key_id: str,
        key_value: str,
        algorithm: KeyAlgorithm = KeyAlgorithm.HS256,
        expires_in_days: Optional[int] = None,
        created_by: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "OrganizationKey":
        """Create a new organization key.

        Args:
            organization_id: Organization ID
            key_id: Public key identifier
            key_value: Secret key value
            algorithm: Signing algorithm
            expires_in_days: Optional expiry in days
            created_by: Optional creator identifier
            description: Optional description

        Returns:
            New OrganizationKey instance
        """
        now = datetime.utcnow()
        expires_at = None
        if expires_in_days:
            expires_at = now + timedelta(days=expires_in_days)

        return cls(
            id=None,
            organization_id=organization_id,
            key_id=key_id,
            key_value=key_value,
            algorithm=algorithm,
            key_version=1,
            is_primary=True,
            status=KeyStatus.ACTIVE,
            created_at=now,
            expires_at=expires_at,
            rotated_at=None,
            deactivated_at=None,
            last_used_at=None,
            usage_count=0,
            previous_key_id=None,
            rotation_reason=None,
            created_by=created_by,
            description=description,
        )
