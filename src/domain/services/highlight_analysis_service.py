"""Domain service for AI-based highlight analysis.

This is the core domain service that encapsulates the business logic
for analyzing video segments and detecting highlights. It represents
the primary value proposition of the application.
"""

import asyncio
from typing import List, Optional, Dict, Any

from ..entities.video_segment import VideoSegment
from ..entities.dimension_set_aggregate import DimensionSetAggregate
from ..entities.highlight_agent_config import HighlightAgentConfig
from ..entities.detected_highlight import DetectedHighlight
from ..interfaces.ai_video_analyzer import AIVideoAnalyzer, HighlightCandidate
from ..value_objects.confidence_score import ConfidenceScore
from ..value_objects.highlight_type import HighlightType
from ..value_objects.processing_options import ProcessingOptions
from ..exceptions import BusinessRuleViolation, DomainError


class HighlightAnalysisService:
    """Core domain service for analyzing video segments and detecting highlights.
    
    This service encapsulates the business rules and logic for:
    - Analyzing video segments using AI
    - Evaluating highlight candidates against business rules
    - Creating highlight entities from candidates
    - Managing the overall highlight detection workflow
    """
    
    def __init__(
        self,
        ai_analyzer: AIVideoAnalyzer,
        processing_options: Optional[ProcessingOptions] = None
    ):
        """Initialize the highlight analysis service.
        
        Args:
            ai_analyzer: The AI analyzer implementation to use
            processing_options: Optional processing configuration
        """
        self.ai_analyzer = ai_analyzer
        self.processing_options = processing_options or ProcessingOptions.default()
    
    async def analyze_segment(
        self,
        segment: VideoSegment,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None
    ) -> List[DetectedHighlight]:
        """Analyze a video segment and detect highlights.
        
        This is the core method that orchestrates the highlight detection process
        for a single video segment.
        
        Args:
            segment: The video segment to analyze
            dimension_set: The dimension set defining what to look for
            agent_config: Optional agent configuration for customizing behavior
            
        Returns:
            List of detected highlights in this segment
            
        Raises:
            BusinessRuleViolation: If business rules are violated
            DomainError: If analysis fails
        """
        # Validate inputs
        self._validate_segment(segment)
        self._validate_dimension_set(dimension_set)
        
        # Mark segment as being analyzed
        segment.start_analysis()
        
        try:
            # Get AI analysis results
            candidates = await self.ai_analyzer.analyze_video(
                video_path=segment.file_path.path,
                dimension_set=dimension_set,
                segment_info=segment.to_analysis_info(),
                agent_config=agent_config
            )
            
            # Filter and create highlights from candidates
            highlights = []
            for candidate in candidates:
                if self._should_create_highlight(candidate, dimension_set, agent_config):
                    highlight = self._create_highlight_from_candidate(
                        candidate=candidate,
                        segment=segment,
                        dimension_set=dimension_set,
                        agent_config=agent_config
                    )
                    highlights.append(highlight)
            
            # Apply post-processing rules
            highlights = self._apply_post_processing_rules(
                highlights, dimension_set, agent_config
            )
            
            # Mark segment as analyzed
            highlight_ids = [h.id for h in highlights]
            segment.complete_analysis(highlight_ids)
            
            return highlights
            
        except Exception as e:
            # Mark segment as failed
            segment.fail_analysis(str(e))
            raise DomainError(f"Failed to analyze segment: {e}") from e
    
    async def analyze_segments_batch(
        self,
        segments: List[VideoSegment],
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None,
        max_concurrent: int = 3
    ) -> Dict[int, List[DetectedHighlight]]:
        """Analyze multiple segments in batch.
        
        Args:
            segments: List of segments to analyze
            dimension_set: The dimension set to use
            agent_config: Optional agent configuration
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            Dictionary mapping segment index to detected highlights
        """
        results = {}
        
        # Process segments with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(segment: VideoSegment):
            async with semaphore:
                try:
                    highlights = await self.analyze_segment(
                        segment, dimension_set, agent_config
                    )
                    results[segment.segment_index] = highlights
                except Exception as e:
                    # Log error but continue with other segments
                    results[segment.segment_index] = []
        
        # Analyze all segments
        tasks = [analyze_with_semaphore(segment) for segment in segments]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def _validate_segment(self, segment: VideoSegment) -> None:
        """Validate a segment before analysis."""
        if segment.is_analyzed:
            raise BusinessRuleViolation("Segment has already been analyzed")
        
        if not segment.file_exists:
            raise BusinessRuleViolation(f"Segment file does not exist: {segment.file_path}")
        
        # Check segment duration
        min_duration = self.processing_options.min_segment_duration or 5.0
        if segment.duration_seconds < min_duration:
            raise BusinessRuleViolation(
                f"Segment duration {segment.duration_seconds}s is below "
                f"minimum {min_duration}s"
            )
    
    def _validate_dimension_set(self, dimension_set: DimensionSetAggregate) -> None:
        """Validate dimension set before analysis."""
        if not dimension_set.is_valid():
            raise BusinessRuleViolation("Dimension set is not valid")
        
        if dimension_set.dimension_count == 0:
            raise BusinessRuleViolation("Dimension set has no dimensions")
    
    def _should_create_highlight(
        self,
        candidate: HighlightCandidate,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig]
    ) -> bool:
        """Determine if a candidate should become a highlight.
        
        This method encapsulates the business rules for highlight creation.
        """
        # Check confidence threshold
        min_confidence = self.processing_options.min_confidence_threshold
        if agent_config and hasattr(agent_config, 'min_confidence_threshold'):
            min_confidence = agent_config.min_confidence_threshold
        
        if not candidate.meets_threshold(min_confidence):
            return False
        
        # Check duration constraints
        min_duration = self.processing_options.min_highlight_duration
        max_duration = self.processing_options.max_highlight_duration
        
        if candidate.duration < min_duration:
            return False
        
        if candidate.duration > max_duration:
            return False
        
        # Check dimension requirements
        if hasattr(dimension_set, 'minimum_dimensions_required'):
            scored_dimensions = len([
                score for score in candidate.dimension_scores.scores.values()
                if score > 0
            ])
            if scored_dimensions < dimension_set.minimum_dimensions_required:
                return False
        
        # Additional business rules can be added here
        
        return True
    
    def _create_highlight_from_candidate(
        self,
        candidate: HighlightCandidate,
        segment: VideoSegment,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig]
    ) -> DetectedHighlight:
        """Create a highlight entity from a candidate."""
        # Determine highlight types based on dimension scores
        highlight_types = self._determine_highlight_types(
            candidate, dimension_set, agent_config
        )
        
        # Create the highlight
        highlight = DetectedHighlight.create(
            stream_id=segment.stream_id,
            start_time=candidate.start_time,
            end_time=candidate.end_time,
            confidence_score=candidate.confidence_score.value,
            highlight_types=highlight_types,
            title=candidate.title,
            description=candidate.description,
            dimension_scores=candidate.dimension_scores.to_dict(),
            metadata={
                "segment_id": str(segment.id),
                "segment_index": segment.segment_index,
                "reasoning": candidate.reasoning,
                **candidate.metadata
            }
        )
        
        return highlight
    
    def _determine_highlight_types(
        self,
        candidate: HighlightCandidate,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig]
    ) -> List[str]:
        """Determine highlight types based on dimension scores."""
        types = []
        
        # Use type registry if available
        if agent_config and hasattr(agent_config, 'type_registry'):
            for type_def in agent_config.type_registry.get_all_types():
                if self._matches_type_criteria(
                    candidate.dimension_scores,
                    type_def.get('criteria', {})
                ):
                    types.append(type_def['id'])
        
        # Fallback to dimension-based types
        if not types:
            # Find dimensions with high scores
            for dim_id, score in candidate.dimension_scores.scores.items():
                if score >= 0.7:  # High score threshold
                    types.append(f"{dim_id}_moment")
        
        # Default type if none found
        if not types:
            types = ["highlight"]
        
        return types
    
    def _matches_type_criteria(
        self,
        dimension_scores: Dict[str, float],
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if dimension scores match type criteria."""
        for dim_id, requirement in criteria.items():
            score = dimension_scores.scores.get(dim_id, 0.0)
            
            if isinstance(requirement, dict):
                min_score = requirement.get('min', 0.0)
                max_score = requirement.get('max', 1.0)
                if not (min_score <= score <= max_score):
                    return False
            elif isinstance(requirement, (int, float)):
                if score < requirement:
                    return False
        
        return True
    
    def _apply_post_processing_rules(
        self,
        highlights: List[DetectedHighlight],
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig]
    ) -> List[DetectedHighlight]:
        """Apply post-processing rules to highlights.
        
        This includes:
        - Merging overlapping highlights
        - Removing duplicates
        - Applying business-specific rules
        """
        if not highlights:
            return highlights
        
        # Sort by start time
        highlights.sort(key=lambda h: h.start_time.value)
        
        # Merge overlapping highlights if configured
        if self.processing_options.merge_overlapping_highlights:
            highlights = self._merge_overlapping_highlights(highlights)
        
        # Apply maximum highlights per segment if configured
        max_per_segment = getattr(agent_config, 'max_highlights_per_segment', None)
        if max_per_segment and len(highlights) > max_per_segment:
            # Keep the highest confidence highlights
            highlights.sort(key=lambda h: h.confidence_score.value, reverse=True)
            highlights = highlights[:max_per_segment]
            highlights.sort(key=lambda h: h.start_time.value)
        
        return highlights
    
    def _merge_overlapping_highlights(
        self,
        highlights: List[DetectedHighlight]
    ) -> List[DetectedHighlight]:
        """Merge highlights that overlap in time."""
        if len(highlights) <= 1:
            return highlights
        
        merged = []
        current = highlights[0]
        
        for next_highlight in highlights[1:]:
            # Check if highlights overlap
            if current.end_time.value >= next_highlight.start_time.value:
                # Merge highlights
                current = self._merge_two_highlights(current, next_highlight)
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_highlight
        
        # Add the last highlight
        merged.append(current)
        
        return merged
    
    def _merge_two_highlights(
        self,
        h1: DetectedHighlight,
        h2: DetectedHighlight
    ) -> DetectedHighlight:
        """Merge two overlapping highlights."""
        # Take the union of time ranges
        start_time = min(h1.start_time.value, h2.start_time.value)
        end_time = max(h1.end_time.value, h2.end_time.value)
        
        # Use the higher confidence score
        confidence = max(h1.confidence_score.value, h2.confidence_score.value)
        
        # Merge highlight types
        types = list(set(h1.highlight_types + h2.highlight_types))
        
        # Merge dimension scores (take max for each dimension)
        merged_scores = h1.dimension_scores.copy()
        for dim_id, score in h2.dimension_scores.items():
            if dim_id not in merged_scores or score > merged_scores[dim_id]:
                merged_scores[dim_id] = score
        
        # Create merged highlight
        return DetectedHighlight.create(
            stream_id=h1.stream_id,
            start_time=start_time,
            end_time=end_time,
            confidence_score=confidence,
            highlight_types=types,
            title=h1.title,  # Keep first highlight's title
            description=f"{h1.description} {h2.description}",
            dimension_scores=merged_scores,
            metadata={
                "merged": True,
                "original_highlights": 2,
                **h1.metadata
            }
        )