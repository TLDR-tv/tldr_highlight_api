"""Duration value object."""

from dataclasses import dataclass
from typing import Union

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class Duration:
    """Value object representing a time duration in seconds.
    
    This is an immutable value object that ensures durations are positive
    and provides utility methods for duration manipulation.
    """
    
    seconds: float
    
    def __post_init__(self):
        """Validate duration after initialization."""
        if not isinstance(self.seconds, (int, float)):
            raise InvalidValueError(
                f"Duration must be a number, got {type(self.seconds).__name__}"
            )
        
        if self.seconds < 0:
            raise InvalidValueError(
                f"Duration cannot be negative, got {self.seconds} seconds"
            )
    
    @classmethod
    def from_milliseconds(cls, milliseconds: Union[int, float]) -> "Duration":
        """Create duration from milliseconds."""
        return cls(milliseconds / 1000.0)
    
    @classmethod
    def from_minutes(cls, minutes: Union[int, float]) -> "Duration":
        """Create duration from minutes."""
        return cls(minutes * 60.0)
    
    @classmethod
    def from_hours(cls, hours: Union[int, float]) -> "Duration":
        """Create duration from hours."""
        return cls(hours * 3600.0)
    
    @property
    def milliseconds(self) -> float:
        """Get duration in milliseconds."""
        return self.seconds * 1000.0
    
    @property
    def minutes(self) -> float:
        """Get duration in minutes."""
        return self.seconds / 60.0
    
    @property
    def hours(self) -> float:
        """Get duration in hours."""
        return self.seconds / 3600.0
    
    @property
    def formatted(self) -> str:
        """Get human-readable duration string."""
        if self.seconds < 60:
            return f"{self.seconds:.1f}s"
        elif self.seconds < 3600:
            minutes = int(self.seconds // 60)
            seconds = int(self.seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(self.seconds // 3600)
            minutes = int((self.seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def __add__(self, other: "Duration") -> "Duration":
        """Add two durations together."""
        if not isinstance(other, Duration):
            raise TypeError(f"Cannot add Duration and {type(other).__name__}")
        return Duration(self.seconds + other.seconds)
    
    def __sub__(self, other: "Duration") -> "Duration":
        """Subtract one duration from another."""
        if not isinstance(other, Duration):
            raise TypeError(f"Cannot subtract {type(other).__name__} from Duration")
        result = self.seconds - other.seconds
        if result < 0:
            raise ValueError("Subtraction would result in negative duration")
        return Duration(result)
    
    def __str__(self) -> str:
        """String representation returns formatted duration."""
        return self.formatted
    
    def __float__(self) -> float:
        """Allow conversion to float (seconds)."""
        return self.seconds