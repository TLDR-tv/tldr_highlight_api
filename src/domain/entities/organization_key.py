"""Domain model for organization signing keys.

This module defines the domain model for organization-specific signing keys
used for generating secure signed URLs.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional
from enum import Enum

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp


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
class OrganizationKey(Entity[int]):
    """Organization signing key domain model."""

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
    expires_at: Optional[Timestamp] = None
    rotated_at: Optional[Timestamp] = None
    deactivated_at: Optional[Timestamp] = None

    # Usage metrics
    last_used_at: Optional[Timestamp] = None
    usage_count: int = 0

    # Rotation tracking
    previous_key_id: Optional[str] = None
    rotation_reason: Optional[str] = None

    # Metadata
    created_by: Optional[str] = None
    description: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Check if key is currently active."""
        if self.status != KeyStatus.ACTIVE:
            return False

        if self.expires_at and self.expires_at.is_before(Timestamp.now()):
            return False

        return True

    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        return self.expires_at and self.expires_at.is_before(Timestamp.now())

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
        self.rotated_at = Timestamp.now()
        self.rotation_reason = reason
        new_key.previous_key_id = self.key_id

    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate this key.

        Args:
            reason: Optional reason for deactivation
        """
        self.status = KeyStatus.DEACTIVATED
        self.is_primary = False
        self.deactivated_at = Timestamp.now()
        if reason:
            self.rotation_reason = reason

    def mark_compromised(self) -> None:
        """Mark this key as compromised."""
        self.status = KeyStatus.COMPROMISED
        self.is_primary = False
        self.deactivated_at = Timestamp.now()
        self.rotation_reason = "Key compromised"

    def increment_usage(self) -> None:
        """Increment usage counter and update last used timestamp."""
        self.usage_count += 1
        self.last_used_at = Timestamp.now()

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
        now = Timestamp.now()
        expires_at = None
        if expires_in_days:
            expires_at = Timestamp.from_datetime(
                now.value + timedelta(days=expires_in_days)
            )

        return cls(
            id=None,
            organization_id=organization_id,
            key_id=key_id,
            key_value=key_value,
            algorithm=algorithm,
            key_version=1,
            is_primary=True,
            status=KeyStatus.ACTIVE,
            expires_at=expires_at,
            rotated_at=None,
            deactivated_at=None,
            last_used_at=None,
            usage_count=0,
            previous_key_id=None,
            rotation_reason=None,
            created_by=created_by,
            description=description,
            created_at=now,
            updated_at=now,
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"OrganizationKey({self.key_id} - {self.status.value})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"OrganizationKey(id={self.id}, key_id={self.key_id!r}, "
            f"status={self.status.value}, is_primary={self.is_primary})"
        )
