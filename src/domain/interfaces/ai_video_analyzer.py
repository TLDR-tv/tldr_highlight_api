"""Domain interface for AI-based video analysis.

This interface defines the contract that any AI video analyzer must implement,
allowing the domain layer to remain independent of specific AI providers.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Protocol, List, Dict, Any, Optional

from ..entities.dimension_set_aggregate import DimensionSetAggregate
from ..entities.highlight_agent_config import HighlightAgentConfig
from ..value_objects.dimension_scores import DimensionScores
from ..value_objects.confidence_score import ConfidenceScore
from ..value_objects.time_range import TimeRange


class HighlightCandidate:
    """A potential highlight identified by AI analysis."""
    
    def __init__(
        self,
        time_range: TimeRange,
        confidence_score: ConfidenceScore,
        dimension_scores: DimensionScores,
        title: str,
        description: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.time_range = time_range
        self.confidence_score = confidence_score
        self.dimension_scores = dimension_scores
        self.title = title
        self.description = description
        self.reasoning = reasoning
        self.metadata = metadata or {}
    
    @property
    def start_time(self) -> float:
        """Get start time in seconds."""
        return self.time_range.start_seconds
    
    @property
    def end_time(self) -> float:
        """Get end time in seconds."""
        return self.time_range.end_seconds
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.time_range.duration_seconds
    
    def meets_threshold(self, min_confidence: float) -> bool:
        """Check if this candidate meets the minimum confidence threshold."""
        return self.confidence_score.value >= min_confidence


class AIVideoAnalyzer(Protocol):
    """Protocol for AI-based video analysis.
    
    This interface abstracts the AI analysis functionality, allowing different
    AI providers (Gemini, OpenAI, custom models) to be used interchangeably.
    """
    
    @abstractmethod
    async def analyze_video(
        self,
        video_path: Path,
        dimension_set: DimensionSetAggregate,
        segment_info: Dict[str, Any],
        agent_config: Optional[HighlightAgentConfig] = None
    ) -> List[HighlightCandidate]:
        """Analyze a video file and return highlight candidates.
        
        Args:
            video_path: Path to the video file to analyze
            dimension_set: The dimension set defining what to look for
            segment_info: Metadata about the video segment (start_time, duration, etc.)
            agent_config: Optional configuration for analysis behavior
            
        Returns:
            List of highlight candidates found in the video
            
        Raises:
            AIAnalysisError: If the analysis fails
        """
        ...
    
    @abstractmethod
    async def analyze_with_multimodal_context(
        self,
        video_path: Path,
        audio_transcript: Optional[str],
        dimension_set: DimensionSetAggregate,
        segment_info: Dict[str, Any],
        agent_config: Optional[HighlightAgentConfig] = None
    ) -> List[HighlightCandidate]:
        """Analyze video with additional context from other modalities.
        
        Args:
            video_path: Path to the video file
            audio_transcript: Optional transcript of the audio
            dimension_set: The dimension set defining what to look for
            segment_info: Metadata about the video segment
            agent_config: Optional configuration for analysis behavior
            
        Returns:
            List of highlight candidates with multimodal context
        """
        ...
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources used by the analyzer.
        
        This might include removing uploaded files, closing connections, etc.
        """
        ...
    
    @abstractmethod
    def supports_batch_analysis(self) -> bool:
        """Check if this analyzer supports batch processing of multiple videos."""
        ...
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this analyzer.
        
        Returns:
            Dictionary describing what this analyzer can do, e.g.:
            {
                "max_video_duration": 300,  # seconds
                "supported_formats": ["mp4", "avi", "mov"],
                "multimodal": True,
                "batch_processing": False,
                "model_name": "gemini-1.5-flash"
            }
        """
        ...


class AIAnalysisError(Exception):
    """Exception raised when AI analysis fails."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause