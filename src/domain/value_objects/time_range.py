"""Value object representing a time range."""

from dataclasses import dataclass

from ..exceptions import InvalidValueError


@dataclass(frozen=True)
class TimeRange:
    """Value object representing a time range in seconds.
    
    This is used to represent the temporal bounds of video segments,
    highlights, and other time-based domain concepts.
    """
    
    start_seconds: float
    end_seconds: float
    
    def __post_init__(self):
        """Validate the time range."""
        if self.start_seconds < 0:
            raise InvalidValueError("Start time cannot be negative")
        
        if self.end_seconds < 0:
            raise InvalidValueError("End time cannot be negative")
        
        if self.end_seconds <= self.start_seconds:
            raise InvalidValueError("End time must be greater than start time")
    
    @property
    def duration_seconds(self) -> float:
        """Get the duration of this time range in seconds."""
        return self.end_seconds - self.start_seconds
    
    def overlaps_with(self, other: "TimeRange") -> bool:
        """Check if this time range overlaps with another.
        
        Args:
            other: Another time range
            
        Returns:
            True if the ranges overlap
        """
        return not (
            self.end_seconds <= other.start_seconds or
            other.end_seconds <= self.start_seconds
        )
    
    def contains(self, time_seconds: float) -> bool:
        """Check if a specific time is within this range.
        
        Args:
            time_seconds: Time in seconds
            
        Returns:
            True if the time is within this range
        """
        return self.start_seconds <= time_seconds <= self.end_seconds
    
    def extend(self, seconds: float) -> "TimeRange":
        """Create a new time range extended by the given seconds.
        
        Args:
            seconds: Seconds to extend (can be negative to shrink)
            
        Returns:
            New TimeRange instance
        """
        return TimeRange(
            max(0, self.start_seconds - seconds),
            self.end_seconds + seconds
        )
    
    def shift(self, seconds: float) -> "TimeRange":
        """Create a new time range shifted by the given seconds.
        
        Args:
            seconds: Seconds to shift (positive = forward, negative = backward)
            
        Returns:
            New TimeRange instance
        """
        return TimeRange(
            max(0, self.start_seconds + seconds),
            max(0, self.end_seconds + seconds)
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"[{self.start_seconds:.1f}s - {self.end_seconds:.1f}s]"