"""Domain entity representing a video segment.

A video segment is a chunk of a stream that can be analyzed independently
for highlight detection. This is a core domain concept.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from ..value_objects.time_range import TimeRange
from ..value_objects.file_path import FilePath
from ..exceptions import BusinessRuleViolation


@dataclass
class VideoSegment:
    """Domain entity representing a segment of video content.
    
    A video segment is created by the stream segmenter and represents
    a discrete chunk of video that can be analyzed for highlights.
    """
    
    id: UUID
    stream_id: int
    segment_index: int
    file_path: FilePath
    time_range: TimeRange
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis state
    is_analyzed: bool = False
    analysis_started_at: Optional[datetime] = None
    analysis_completed_at: Optional[datetime] = None
    analysis_error: Optional[str] = None
    
    # Detected highlights (filled after analysis)
    highlight_ids: List[int] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        stream_id: int,
        segment_index: int,
        file_path: Path,
        start_time: float,
        end_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "VideoSegment":
        """Factory method to create a new video segment.
        
        Args:
            stream_id: ID of the parent stream
            segment_index: Sequential index of this segment
            file_path: Path to the video file
            start_time: Start time in seconds relative to stream start
            end_time: End time in seconds relative to stream start
            metadata: Optional metadata about the segment
            
        Returns:
            New VideoSegment instance
            
        Raises:
            BusinessRuleViolation: If validation fails
        """
        if segment_index < 0:
            raise BusinessRuleViolation("Segment index must be non-negative")
        
        if start_time < 0:
            raise BusinessRuleViolation("Start time must be non-negative")
        
        if end_time <= start_time:
            raise BusinessRuleViolation("End time must be greater than start time")
        
        return cls(
            id=uuid4(),
            stream_id=stream_id,
            segment_index=segment_index,
            file_path=FilePath(str(file_path)),
            time_range=TimeRange(start_time, end_time),
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
    
    @property
    def duration_seconds(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.time_range.duration_seconds
    
    @property
    def start_time(self) -> float:
        """Get the start time in seconds."""
        return self.time_range.start_seconds
    
    @property
    def end_time(self) -> float:
        """Get the end time in seconds."""
        return self.time_range.end_seconds
    
    @property
    def file_exists(self) -> bool:
        """Check if the video file exists."""
        return Path(self.file_path.value).exists()
    
    @property
    def file_size_bytes(self) -> int:
        """Get the size of the video file in bytes."""
        if self.file_exists:
            return Path(self.file_path.value).stat().st_size
        return 0
    
    def start_analysis(self) -> None:
        """Mark this segment as being analyzed."""
        if self.is_analyzed:
            raise BusinessRuleViolation("Segment has already been analyzed")
        
        self.analysis_started_at = datetime.now(timezone.utc)
    
    def complete_analysis(self, highlight_ids: List[int]) -> None:
        """Mark analysis as complete with found highlights.
        
        Args:
            highlight_ids: IDs of highlights found in this segment
        """
        if not self.analysis_started_at:
            raise BusinessRuleViolation("Cannot complete analysis that hasn't started")
        
        self.is_analyzed = True
        self.analysis_completed_at = datetime.now(timezone.utc)
        self.highlight_ids = highlight_ids
        self.analysis_error = None
    
    def fail_analysis(self, error: str) -> None:
        """Mark analysis as failed with an error.
        
        Args:
            error: Error message describing what went wrong
        """
        if not self.analysis_started_at:
            raise BusinessRuleViolation("Cannot fail analysis that hasn't started")
        
        self.is_analyzed = True
        self.analysis_completed_at = datetime.now(timezone.utc)
        self.analysis_error = error
    
    def get_analysis_duration_seconds(self) -> Optional[float]:
        """Get how long the analysis took in seconds."""
        if self.analysis_started_at and self.analysis_completed_at:
            delta = self.analysis_completed_at - self.analysis_started_at
            return delta.total_seconds()
        return None
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to this segment.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def to_analysis_info(self) -> Dict[str, Any]:
        """Convert to dictionary format for AI analysis.
        
        Returns:
            Dictionary with segment information for analysis
        """
        return {
            "segment_id": str(self.id),
            "segment_index": self.segment_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration_seconds,
            "file_path": self.file_path.value,
            **self.metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"VideoSegment(index={self.segment_index}, "
            f"time={self.start_time:.1f}-{self.end_time:.1f}s, "
            f"analyzed={self.is_analyzed})"
        )