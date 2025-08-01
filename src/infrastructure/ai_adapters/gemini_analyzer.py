"""Infrastructure adapter for Gemini AI video analysis.

This adapter implements the domain's AIVideoAnalyzer interface using
Google's Gemini API, keeping the domain layer independent of the
specific AI provider.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from ...domain.interfaces.ai_video_analyzer import (
    AIVideoAnalyzer,
    HighlightCandidate,
    AIAnalysisError
)
from ...domain.entities.dimension_set_aggregate import DimensionSetAggregate
from ...domain.entities.highlight_agent_config import HighlightAgentConfig
from ...domain.value_objects.confidence_score import ConfidenceScore
from ...domain.value_objects.time_range import TimeRange
from ...domain.value_objects.dimension_scores import DimensionScores
from ..content_processing.gemini_video_processor import GeminiVideoProcessor


class GeminiAIAnalyzer(AIVideoAnalyzer):
    """Gemini implementation of the AI video analyzer.
    
    This adapter translates between the domain's AI analysis interface
    and the Gemini-specific implementation details.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        max_retries: int = 3
    ):
        """Initialize the Gemini analyzer.
        
        Args:
            api_key: Gemini API key
            model_name: Gemini model to use
            max_retries: Maximum retry attempts for API calls
        """
        self.gemini_processor = GeminiVideoProcessor(
            api_key=api_key,
            model_name=model_name,
            max_retries=max_retries
        )
    
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
            segment_info: Metadata about the video segment
            agent_config: Optional configuration for analysis behavior
            
        Returns:
            List of highlight candidates found in the video
            
        Raises:
            AIAnalysisError: If the analysis fails
        """
        try:
            # Call Gemini processor with dimension set
            detected_highlights = await self.gemini_processor.analyze_video_with_dimensions(
                video_path=str(video_path),
                segment_info=segment_info,
                dimension_set=dimension_set,
                agent_config=agent_config
            )
            
            # Convert DetectedHighlight entities to HighlightCandidate objects
            candidates = []
            for highlight in detected_highlights:
                candidate = self._convert_to_candidate(highlight, segment_info)
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            raise AIAnalysisError(f"Gemini analysis failed: {str(e)}", cause=e)
    
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
        # For now, just use the standard analysis
        # In the future, we could pass the transcript to Gemini as additional context
        candidates = await self.analyze_video(
            video_path, dimension_set, segment_info, agent_config
        )
        
        # Add transcript context to metadata if available
        if audio_transcript:
            for candidate in candidates:
                candidate.metadata["audio_transcript"] = audio_transcript
        
        return candidates
    
    async def cleanup(self) -> None:
        """Clean up any resources used by the analyzer."""
        # Clean up uploaded files from Gemini
        await self.gemini_processor.cleanup_all_files()
    
    def supports_batch_analysis(self) -> bool:
        """Check if this analyzer supports batch processing."""
        # Gemini processes videos one at a time
        return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this analyzer."""
        return {
            "max_video_duration": 300,  # 5 minutes
            "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
            "multimodal": True,
            "batch_processing": False,
            "model_name": self.gemini_processor.model_name,
            "supports_dimensions": True,
            "supports_custom_prompts": True,
            "max_file_size_mb": 200
        }
    
    def _convert_to_candidate(
        self,
        highlight: Any,  # DetectedHighlight from Gemini
        segment_info: Dict[str, Any]
    ) -> HighlightCandidate:
        """Convert a Gemini DetectedHighlight to a domain HighlightCandidate.
        
        Args:
            highlight: The highlight from Gemini processor
            segment_info: Segment metadata
            
        Returns:
            Domain HighlightCandidate object
        """
        # Extract time range
        time_range = TimeRange(
            start_seconds=highlight.start_time,
            end_seconds=highlight.end_time
        )
        
        # Extract confidence score
        confidence_score = ConfidenceScore(highlight.confidence_score)
        
        # Extract dimension scores
        dimension_scores = DimensionScores(
            scores=highlight.dimension_scores or {}
        )
        
        # Create metadata
        metadata = {
            "segment_id": segment_info.get("segment_id"),
            "segment_index": segment_info.get("segment_index"),
            "analysis_method": "gemini_video",
            "model": self.gemini_processor.model_name
        }
        
        # Add any additional metadata from the highlight
        if hasattr(highlight, 'metadata') and highlight.metadata:
            metadata.update(highlight.metadata)
        
        # Create candidate
        return HighlightCandidate(
            time_range=time_range,
            confidence_score=confidence_score,
            dimension_scores=dimension_scores,
            title=highlight.title or "Untitled Highlight",
            description=highlight.description or "",
            reasoning=getattr(highlight, 'reasoning', "AI-detected highlight"),
            metadata=metadata
        )