"""Value object for dimension set versioning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class DimensionSetVersion:
    """Immutable value object representing a dimension set version.

    Uses semantic versioning to track changes:
    - Major: Breaking changes to dimension definitions
    - Minor: New dimensions added (backward compatible)
    - Patch: Bug fixes or clarifications to existing dimensions
    """

    major: int
    minor: int
    patch: int
    effective_date: datetime
    migration_strategy: Optional[str] = None
    changelog: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate version components."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise InvalidValueError("Version components must be non-negative")

    @property
    def version_string(self) -> str:
        """Return version as string (e.g., '1.2.3')."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, other: "DimensionSetVersion") -> bool:
        """Check if this version is backward compatible with another.

        Compatible if:
        - Same major version AND
        - This minor version >= other minor version
        """
        return self.major == other.major and self.minor >= other.minor

    def is_newer_than(self, other: "DimensionSetVersion") -> bool:
        """Check if this version is newer than another."""
        if self.major != other.major:
            return self.major > other.major
        if self.minor != other.minor:
            return self.minor > other.minor
        return self.patch > other.patch

    def increment_major(self, effective_date: datetime) -> "DimensionSetVersion":
        """Create a new major version."""
        return DimensionSetVersion(
            major=self.major + 1,
            minor=0,
            patch=0,
            effective_date=effective_date,
            migration_strategy=None,
            changelog=None,
        )

    def increment_minor(self, effective_date: datetime) -> "DimensionSetVersion":
        """Create a new minor version."""
        return DimensionSetVersion(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
            effective_date=effective_date,
            migration_strategy=self.migration_strategy,
            changelog=None,
        )

    def increment_patch(self, effective_date: datetime) -> "DimensionSetVersion":
        """Create a new patch version."""
        return DimensionSetVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
            effective_date=effective_date,
            migration_strategy=self.migration_strategy,
            changelog=None,
        )
