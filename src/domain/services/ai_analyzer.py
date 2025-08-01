"""Simple AI video analyzer following modern Python practices.

No ABC, no Protocol - just a simple class that can be used directly
or duck-typed. YAGNI principle in action.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..entities.dimension_set_aggregate import DimensionSetAggregate
from ..entities.highlight_agent_config import HighlightAgentConfig
from ..entities.highlight import HighlightCandidate
from ..value_objects.dimension_scores import DimensionScores
from ..value_objects.confidence_score import ConfidenceScore
from ..value_objects.time_range import TimeRange


class AIAnalysisError(Exception):
    """Raised when AI analysis fails."""
    pass


@dataclass
class AIVideoAnalyzer:
    """Simple AI video analyzer using dataclass for clean initialization.
    
    Just a regular class - no inheritance needed. Other analyzers can
    duck-type this interface or be completely different.
    """
    
    model_name: str = "gemini-2.0-flash-exp"
    max_retries: int = 3
    capabilities: Dict[str, Any] = field(default_factory=lambda: {
        "max_video_duration": 300,
        "supported_formats": ["mp4", "avi", "mov", "mkv"],
        "multimodal": True,
        "batch_processing": False,
    })
    
    async def analyze_video(
        self,
        video_path: Path,
        dimension_set: DimensionSetAggregate,
        segment_info: Dict[str, Any],
        agent_config: Optional[HighlightAgentConfig] = None
    ) -> List[HighlightCandidate]:
        """Analyze a video and return highlight candidates."""
        # Default implementation that specific analyzers will override
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement analyze_video"
        )
    
    async def cleanup(self) -> None:
        """Clean up resources if needed."""
        # Most analyzers won't need this, so default is no-op
        pass
    
    def create_candidate(
        self,
        start_time: float,
        end_time: float,
        confidence: float,
        dimension_scores: Dict[str, float],
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HighlightCandidate:
        """Helper to create a highlight candidate.
        
        This is just a convenience method - not required to use.
        """
        return HighlightCandidate(
            time_range=TimeRange(start_time, end_time),
            confidence_score=ConfidenceScore(confidence),
            dimension_scores=DimensionScores(scores=dimension_scores),
            title=description[:100],
            description=description,
            reasoning=metadata.get("reasoning", "AI analysis"),
            metadata=metadata or {}
        )