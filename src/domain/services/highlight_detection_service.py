"""Highlight detection domain service.

This service orchestrates the highlight detection process using
configurable dimensions and analysis strategies.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.domain.services.base import BaseDomainService
from src.domain.entities.highlight import Highlight
from src.domain.entities.stream import Stream
from src.domain.entities.dimension_set import DimensionSet
from src.domain.entities.highlight_type_registry import HighlightTypeRegistry
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.url import Url
from src.domain.value_objects.processing_options import ProcessingOptions, FusionStrategy
from src.domain.services.analysis_strategies import AnalysisStrategy, AnalysisSegment, AnalysisResult
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.dimension_set_repository import DimensionSetRepository
from src.domain.repositories.highlight_type_registry_repository import HighlightTypeRegistryRepository
from src.domain.exceptions import EntityNotFoundError, BusinessRuleViolation


class DetectionResult:
    """Result from a single detector."""
    
    def __init__(
        self,
        start_time: float,
        end_time: float,
        confidence: float,
        detector_type: str,
        metadata: Dict[str, Any]
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence
        self.detector_type = detector_type
        self.metadata = metadata


class HighlightDetectionService(BaseDomainService):
    """Domain service for highlight detection orchestration.
    
    Uses configurable dimensions and analysis strategies to detect
    highlights in multi-modal content.
    """
    
    def __init__(
        self,
        highlight_repo: HighlightRepository,
        stream_repo: StreamRepository,
        dimension_set_repo: DimensionSetRepository,
        type_registry_repo: HighlightTypeRegistryRepository,
        analysis_strategies: Dict[str, AnalysisStrategy]
    ):
        """Initialize highlight detection service.
        
        Args:
            highlight_repo: Repository for highlight operations
            stream_repo: Repository for stream operations
            dimension_set_repo: Repository for dimension sets
            type_registry_repo: Repository for highlight type registries
            analysis_strategies: Map of strategy names to implementations
        """
        super().__init__()
        self.highlight_repo = highlight_repo
        self.stream_repo = stream_repo
        self.dimension_set_repo = dimension_set_repo
        self.type_registry_repo = type_registry_repo
        self.analysis_strategies = analysis_strategies
    
    async def process_stream_segment(
        self,
        stream_id: int,
        segment_data: Dict[str, Any],
        processing_options: ProcessingOptions
    ) -> List[Highlight]:
        """Process a stream segment to detect highlights.
        
        Args:
            stream_id: Stream ID
            segment_data: Multi-modal segment data
            processing_options: Processing configuration
            
        Returns:
            List of detected highlights
            
        Raises:
            EntityNotFoundError: If required entities don't exist
        """
        # Verify stream exists
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        # Load dimension set if specified
        dimension_set = None
        if processing_options.dimension_set_id:
            dimension_set = await self.dimension_set_repo.get(processing_options.dimension_set_id)
            if not dimension_set:
                raise EntityNotFoundError(f"DimensionSet {processing_options.dimension_set_id} not found")
        
        # Load type registry if specified
        type_registry = None
        if processing_options.type_registry_id:
            type_registry = await self.type_registry_repo.get(processing_options.type_registry_id)
            if not type_registry:
                raise EntityNotFoundError(f"TypeRegistry {processing_options.type_registry_id} not found")
        
        # Create analysis segment
        analysis_segment = AnalysisSegment(
            start_time=segment_data.get("start_time", 0.0),
            end_time=segment_data.get("end_time", 0.0),
            video_frames=segment_data.get("video_frames", []),
            audio_data=segment_data.get("audio_data", {}),
            text_data=segment_data.get("text_data", {}),
            metadata=segment_data.get("metadata", {}),
            social_data=segment_data.get("social_data", {})
        )
        
        # Get analysis strategy
        strategy = self.analysis_strategies.get(processing_options.detection_strategy.value)
        if not strategy:
            raise BusinessRuleViolation(f"Unknown analysis strategy: {processing_options.detection_strategy}")
        
        # Configure strategy with dimension set
        if dimension_set and hasattr(strategy, 'set_dimensions'):
            strategy.set_dimensions(dimension_set)
        
        # Analyze segment
        analysis_result = await strategy.analyze(analysis_segment)
        
        # Apply fusion strategy if multiple modalities
        if len(analysis_result.modality_scores) > 1:
            fused_scores = self._apply_fusion_strategy(
                analysis_result.modality_scores,
                processing_options.fusion_strategy,
                processing_options.modality_weights
            )
            analysis_result.dimension_scores = fused_scores
        
        # Check if segment qualifies as highlight
        if self._is_highlight(analysis_result, processing_options):
            highlight = await self._create_highlight(
                stream_id,
                analysis_segment,
                analysis_result,
                type_registry
            )
            return [highlight]
        
        return []
    
    async def detect_highlights_batch(
        self,
        stream_id: int,
        segments: List[Dict[str, Any]],
        processing_options: ProcessingOptions
    ) -> List[Highlight]:
        """Process multiple segments to detect highlights.
        
        Args:
            stream_id: Stream ID
            segments: List of segment data
            processing_options: Processing configuration
            
        Returns:
            List of detected highlights
        """
        highlights = []
        
        for segment_data in segments:
            segment_highlights = await self.process_stream_segment(
                stream_id,
                segment_data,
                processing_options
            )
            highlights.extend(segment_highlights)
        
        # Post-process to merge adjacent highlights
        if highlights:
            highlights = await self._post_process_highlights(highlights, processing_options)
        
        return highlights
    
    def _apply_fusion_strategy(
        self,
        modality_scores: Dict[str, Dict[str, float]],
        fusion_strategy: FusionStrategy,
        modality_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply fusion strategy to combine modality scores.
        
        Args:
            modality_scores: Scores per modality per dimension
            fusion_strategy: Strategy to use
            modality_weights: Weights for each modality
            
        Returns:
            Fused dimension scores
        """
        if fusion_strategy == FusionStrategy.WEIGHTED:
            return self._weighted_fusion(modality_scores, modality_weights)
        elif fusion_strategy == FusionStrategy.CONSENSUS:
            return self._consensus_fusion(modality_scores)
        elif fusion_strategy == FusionStrategy.CASCADE:
            return self._cascade_fusion(modality_scores, modality_weights)
        elif fusion_strategy == FusionStrategy.MAX_CONFIDENCE:
            return self._max_confidence_fusion(modality_scores)
        else:
            # Default to weighted
            return self._weighted_fusion(modality_scores, modality_weights)
    
    def _weighted_fusion(
        self,
        modality_scores: Dict[str, Dict[str, float]],
        modality_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Weighted average fusion."""
        fused_scores = {}
        
        # Get all dimensions
        all_dimensions = set()
        for scores in modality_scores.values():
            all_dimensions.update(scores.keys())
        
        # Calculate weighted average for each dimension
        for dimension in all_dimensions:
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for modality, scores in modality_scores.items():
                if dimension in scores:
                    weight = modality_weights.get(modality, 1.0)
                    weighted_sum += scores[dimension] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                fused_scores[dimension] = weighted_sum / weight_sum
        
        return fused_scores
    
    def _consensus_fusion(self, modality_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Consensus-based fusion - all modalities must agree."""
        fused_scores = {}
        
        # Get dimensions present in all modalities
        if not modality_scores:
            return fused_scores
        
        common_dimensions = set(next(iter(modality_scores.values())).keys())
        for scores in modality_scores.values():
            common_dimensions &= set(scores.keys())
        
        # Use minimum score (all must agree)
        for dimension in common_dimensions:
            min_score = min(scores.get(dimension, 0.0) for scores in modality_scores.values())
            fused_scores[dimension] = min_score
        
        return fused_scores
    
    def _cascade_fusion(
        self,
        modality_scores: Dict[str, Dict[str, float]],
        modality_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Cascade fusion - prioritize modalities by weight."""
        fused_scores = {}
        
        # Sort modalities by weight
        sorted_modalities = sorted(
            modality_scores.keys(),
            key=lambda m: modality_weights.get(m, 0.0),
            reverse=True
        )
        
        # Take scores from highest priority modality with data
        for modality in sorted_modalities:
            if modality_scores[modality]:
                return modality_scores[modality].copy()
        
        return fused_scores
    
    def _max_confidence_fusion(self, modality_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Max confidence fusion - take highest score per dimension."""
        fused_scores = {}
        
        # Get all dimensions
        all_dimensions = set()
        for scores in modality_scores.values():
            all_dimensions.update(scores.keys())
        
        # Take maximum score for each dimension
        for dimension in all_dimensions:
            max_score = max(
                scores.get(dimension, 0.0)
                for scores in modality_scores.values()
            )
            fused_scores[dimension] = max_score
        
        return fused_scores
    
    def _is_highlight(self, analysis_result: AnalysisResult, options: ProcessingOptions) -> bool:
        """Determine if analysis result qualifies as a highlight."""
        if not analysis_result.dimension_scores:
            return False
        
        # Check confidence threshold
        overall_confidence = analysis_result.overall_confidence
        if overall_confidence < options.min_confidence_threshold:
            return False
        
        # Check duration constraints
        duration = analysis_result.end_time - analysis_result.start_time
        if duration < options.min_highlight_duration:
            return False
        if duration > options.max_highlight_duration:
            return False
        
        # Check minimum dimension scores
        high_scoring_dimensions = sum(
            1 for score in analysis_result.dimension_scores.values()
            if score >= 0.7  # High score threshold
        )
        
        if high_scoring_dimensions < 2:  # Need at least 2 high-scoring dimensions
            return False
        
        return True
    
    async def _create_highlight(
        self,
        stream_id: int,
        segment: AnalysisSegment,
        analysis_result: AnalysisResult,
        type_registry: Optional[HighlightTypeRegistry]
    ) -> Highlight:
        """Create highlight entity from analysis result."""
        # Determine highlight types
        highlight_types = []
        if type_registry:
            highlight_types = type_registry.determine_types(
                analysis_result.dimension_scores,
                analysis_result.metadata
            )
        
        # Generate title and description
        title = self._generate_title(analysis_result, highlight_types)
        description = self._generate_description(analysis_result)
        
        # Create highlight entity
        highlight = Highlight(
            id=None,
            stream_id=stream_id,
            start_time=Duration(segment.start_time),
            end_time=Duration(segment.end_time),
            confidence_score=ConfidenceScore(analysis_result.overall_confidence),
            highlight_types=highlight_types,
            title=title,
            description=description,
            tags=analysis_result.suggested_tags[:10],
            sentiment_score=analysis_result.metadata.get("sentiment_score"),
            viewer_engagement=analysis_result.metadata.get("engagement_score"),
            video_analysis=segment.video_frames[0] if segment.video_frames else {},
            audio_analysis=segment.audio_data,
            chat_analysis=segment.text_data,
            processed_by="flexible_detector_v1",
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Save highlight
        return await self.highlight_repo.save(highlight)
    
    def _generate_title(self, analysis_result: AnalysisResult, highlight_types: List[str]) -> str:
        """Generate title for highlight."""
        if highlight_types:
            # Use first type as basis for title
            type_name = highlight_types[0].replace("_", " ").title()
            return f"{type_name} Moment"
        
        # Fallback based on top dimensions
        if analysis_result.dimension_scores:
            top_dimension = max(
                analysis_result.dimension_scores.items(),
                key=lambda x: x[1]
            )[0]
            return f"{top_dimension.replace('_', ' ').title()} Highlight"
        
        return "Stream Highlight"
    
    def _generate_description(self, analysis_result: AnalysisResult) -> str:
        """Generate description for highlight."""
        descriptions = []
        
        # Add dimension-based descriptions
        for dimension, score in analysis_result.dimension_scores.items():
            if score >= 0.8:
                descriptions.append(f"High {dimension.replace('_', ' ')}")
        
        # Add metadata descriptions
        if "transcription" in analysis_result.metadata:
            transcript = analysis_result.metadata["transcription"][:100]
            descriptions.append(f'"{transcript}..."')
        
        return ". ".join(descriptions) if descriptions else "Automatically detected highlight"
    
    async def _post_process_highlights(
        self,
        highlights: List[Highlight],
        options: ProcessingOptions
    ) -> List[Highlight]:
        """Post-process highlights to merge adjacent ones."""
        if len(highlights) < 2:
            return highlights
        
        # Sort by start time
        sorted_highlights = sorted(highlights, key=lambda h: h.start_time.value)
        
        # Merge adjacent highlights
        merged = []
        current = sorted_highlights[0]
        
        for next_highlight in sorted_highlights[1:]:
            gap = next_highlight.start_time.value - current.end_time.value
            
            # Merge if gap is small and types overlap
            if gap <= 5.0 and self._should_merge(current, next_highlight):
                current = self._merge_highlights(current, next_highlight)
            else:
                merged.append(current)
                current = next_highlight
        
        merged.append(current)
        return merged
    
    async def process_detection_results(
        self,
        stream_id: int,
        video_results: List[DetectionResult],
        audio_results: List[DetectionResult],
        chat_results: List[DetectionResult]
    ) -> List[Highlight]:
        """Process detection results from multiple sources (compatibility method).
        
        Args:
            stream_id: Stream ID
            video_results: Results from video detector
            audio_results: Results from audio detector
            chat_results: Results from chat detector
            
        Returns:
            List of created highlights
        """
        # Convert detection results to segments
        segments = []
        
        # Combine all results into segments
        all_results = [
            (r, "video") for r in video_results
        ] + [
            (r, "audio") for r in audio_results
        ] + [
            (r, "chat") for r in chat_results
        ]
        
        # Sort by start time
        all_results.sort(key=lambda x: x[0].start_time)
        
        # Group overlapping results into segments
        if all_results:
            current_segment = {
                "start_time": all_results[0][0].start_time,
                "end_time": all_results[0][0].end_time,
                "video_frames": [],
                "audio_data": {},
                "text_data": {},
                "metadata": {}
            }
            
            for result, modality in all_results:
                if result.start_time <= current_segment["end_time"]:
                    # Overlapping - merge
                    current_segment["end_time"] = max(current_segment["end_time"], result.end_time)
                    if modality == "video":
                        current_segment["video_frames"].append(result.metadata)
                    elif modality == "audio":
                        current_segment["audio_data"].update(result.metadata)
                    elif modality == "chat":
                        current_segment["text_data"].update(result.metadata)
                else:
                    # Non-overlapping - save current and start new
                    segments.append(current_segment)
                    current_segment = {
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                        "video_frames": [],
                        "audio_data": {},
                        "text_data": {},
                        "metadata": {}
                    }
                    if modality == "video":
                        current_segment["video_frames"].append(result.metadata)
                    elif modality == "audio":
                        current_segment["audio_data"].update(result.metadata)
                    elif modality == "chat":
                        current_segment["text_data"].update(result.metadata)
            
            segments.append(current_segment)
        
        # Get stream to determine processing options
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        # Use default processing options if not specified
        processing_options = stream.processing_options or ProcessingOptions()
        
        # Process segments using the flexible system
        return await self.detect_highlights_batch(stream_id, segments, processing_options)
    
    def _should_merge(self, h1: Highlight, h2: Highlight) -> bool:
        """Check if two highlights should be merged."""
        # Check for common types
        common_types = set(h1.highlight_types) & set(h2.highlight_types)
        if common_types:
            return True
        
        # Check for similar confidence
        confidence_diff = abs(h1.confidence_score.value - h2.confidence_score.value)
        if confidence_diff < 0.2:
            return True
        
        return False
    
    def _merge_highlights(self, h1: Highlight, h2: Highlight) -> Highlight:
        """Merge two highlights into one."""
        # Combine types
        merged_types = list(set(h1.highlight_types + h2.highlight_types))[:3]
        
        # Use higher confidence
        confidence = ConfidenceScore(max(h1.confidence_score.value, h2.confidence_score.value))
        
        # Merge tags
        merged_tags = list(set(h1.tags + h2.tags))[:10]
        
        # Average sentiment and engagement
        sentiment = None
        if h1.sentiment_score is not None and h2.sentiment_score is not None:
            sentiment = (h1.sentiment_score + h2.sentiment_score) / 2
        elif h1.sentiment_score is not None:
            sentiment = h1.sentiment_score
        elif h2.sentiment_score is not None:
            sentiment = h2.sentiment_score
        
        engagement = None
        if h1.viewer_engagement is not None and h2.viewer_engagement is not None:
            engagement = (h1.viewer_engagement + h2.viewer_engagement) / 2
        elif h1.viewer_engagement is not None:
            engagement = h1.viewer_engagement
        elif h2.viewer_engagement is not None:
            engagement = h2.viewer_engagement
        
        return Highlight(
            id=h1.id,  # Keep first highlight's ID
            stream_id=h1.stream_id,
            start_time=h1.start_time,  # Use earlier start
            end_time=h2.end_time,      # Use later end
            confidence_score=confidence,
            highlight_types=merged_types,
            title=h1.title,  # Keep first title
            description=f"{h1.description} {h2.description}".strip(),
            thumbnail_url=h1.thumbnail_url,
            clip_url=h1.clip_url,
            tags=merged_tags,
            sentiment_score=sentiment,
            viewer_engagement=engagement,
            video_analysis={**h1.video_analysis, **h2.video_analysis},
            audio_analysis={**h1.audio_analysis, **h2.audio_analysis},
            chat_analysis={**h1.chat_analysis, **h2.chat_analysis},
            processed_by=h1.processed_by,
            created_at=h1.created_at,
            updated_at=Timestamp.now()
        )