"""Timestamp value object."""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date, time
from typing import Union, TYPE_CHECKING

from src.domain.exceptions import InvalidValueError

if TYPE_CHECKING:
    from .duration import Duration


@dataclass(frozen=True)
class Timestamp:
    """Value object representing a point in time.

    This is an immutable value object that wraps datetime functionality
    and ensures all timestamps are timezone-aware (UTC).
    """

    value: datetime

    def __post_init__(self) -> None:
        """Validate timestamp after initialization."""
        if not isinstance(self.value, datetime):
            raise InvalidValueError(
                f"Timestamp must be a datetime object, got {type(self.value).__name__}"
            )

        # Ensure timezone awareness
        if self.value.tzinfo is None:
            # Make naive datetime UTC
            object.__setattr__(self, "value", self.value.replace(tzinfo=timezone.utc))
        elif self.value.tzinfo != timezone.utc:
            # Convert to UTC
            object.__setattr__(self, "value", self.value.astimezone(timezone.utc))

    @classmethod
    def now(cls) -> "Timestamp":
        """Create timestamp for current UTC time."""
        return cls(datetime.now(timezone.utc))

    @classmethod
    def from_unix(cls, unix_timestamp: Union[int, float]) -> "Timestamp":
        """Create timestamp from Unix timestamp (seconds since epoch)."""
        return cls(datetime.fromtimestamp(unix_timestamp, tz=timezone.utc))

    @classmethod
    def from_iso_string(cls, iso_string: str) -> "Timestamp":
        """Create timestamp from ISO 8601 string."""
        try:
            # Try parsing with timezone
            dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
            return cls(dt)
        except ValueError:
            # Try parsing without timezone and assume UTC
            dt = datetime.fromisoformat(iso_string)
            return cls(dt.replace(tzinfo=timezone.utc))

    @property
    def unix_timestamp(self) -> float:
        """Get Unix timestamp (seconds since epoch)."""
        return self.value.timestamp()

    @property
    def iso_string(self) -> str:
        """Get ISO 8601 formatted string."""
        return self.value.isoformat()

    @property
    def date(self) -> date:
        """Get the date portion."""
        return self.value.date()

    @property
    def time(self) -> time:
        """Get the time portion."""
        return self.value.time()

    def add_duration(self, duration: "Duration") -> "Timestamp":
        """Add a duration to this timestamp."""
        from src.domain.value_objects.duration import Duration

        if not isinstance(duration, Duration):
            raise TypeError(f"Expected Duration, got {type(duration).__name__}")

        delta = timedelta(seconds=duration.seconds)
        return Timestamp(self.value + delta)

    def subtract_duration(self, duration: "Duration") -> "Timestamp":
        """Subtract a duration from this timestamp."""
        from src.domain.value_objects.duration import Duration

        if not isinstance(duration, Duration):
            raise TypeError(f"Expected Duration, got {type(duration).__name__}")

        delta = timedelta(seconds=duration.seconds)
        return Timestamp(self.value - delta)

    def duration_since(self, other: "Timestamp") -> "Duration":
        """Calculate duration since another timestamp."""
        from src.domain.value_objects.duration import Duration

        if not isinstance(other, Timestamp):
            raise TypeError(f"Expected Timestamp, got {type(other).__name__}")

        if self.value < other.value:
            raise ValueError("Cannot calculate negative duration")

        delta = self.value - other.value
        return Duration(delta.total_seconds())

    def is_before(self, other: "Timestamp") -> bool:
        """Check if this timestamp is before another."""
        if not isinstance(other, Timestamp):
            raise TypeError(f"Cannot compare Timestamp with {type(other).__name__}")
        return self.value < other.value

    def is_after(self, other: "Timestamp") -> bool:
        """Check if this timestamp is after another."""
        if not isinstance(other, Timestamp):
            raise TypeError(f"Cannot compare Timestamp with {type(other).__name__}")
        return self.value > other.value

    def format(self, format_string: str) -> str:
        """Format timestamp using strftime format string."""
        return self.value.strftime(format_string)

    def __str__(self) -> str:
        """String representation returns ISO format."""
        return self.iso_string
