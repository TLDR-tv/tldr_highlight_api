"""Highlight detection service using multi-dimensional scoring."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID

from shared.domain.models.highlight import Highlight
from shared.domain.models.stream import Stream
from worker.services.dimension_framework import (
    ScoringContext,
    ScoringRubric,
    ScoringStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class VideoSegment:
    """Represents a video segment for analysis."""

    file_path: Path
    start_time: float  # seconds from stream start
    duration: float  # seconds
    segment_number: int

    @property
    def end_time(self) -> float:
        """Calculate end time of segment."""
        return self.start_time + self.duration

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS timestamp."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


@dataclass
class HighlightCandidate:
    """Potential highlight with scoring information and precise boundaries."""

    stream_id: UUID
    start_time: float  # Segment-based start time (fallback)
    end_time: float    # Segment-based end time (fallback)
    segments: list[VideoSegment]
    dimension_scores: dict[str, float]
    overall_score: float
    confidence: float
    metadata: dict = field(default_factory=dict)
    
    # New fields for precise boundaries
    precise_start_time: Optional[float] = None  # Precise start within segment (seconds)
    precise_end_time: Optional[float] = None    # Precise end within segment (seconds)
    boundary_confidence: float = 0.0            # Confidence in precise boundaries
    boundary_reasoning: str = ""                # Explanation for boundary selection

    @property
    def duration(self) -> float:
        """Calculate highlight duration in seconds using precise boundaries if available."""
        if self.has_precise_boundaries:
            return self.precise_end_time - self.precise_start_time
        return self.end_time - self.start_time
    
    @property
    def has_precise_boundaries(self) -> bool:
        """Check if precise boundaries are available and valid."""
        return (
            self.precise_start_time is not None 
            and self.precise_end_time is not None
            and self.precise_start_time < self.precise_end_time
        )
    
    @property
    def effective_start_time(self) -> float:
        """Get the effective start time (precise if available, otherwise segment-based)."""
        return self.precise_start_time if self.has_precise_boundaries else self.start_time
    
    @property
    def effective_end_time(self) -> float:
        """Get the effective end time (precise if available, otherwise segment-based).""" 
        return self.precise_end_time if self.has_precise_boundaries else self.end_time
    
    def validate_duration_constraints(self, min_duration: float = 15.0, max_duration: float = 90.0) -> bool:
        """Validate that the effective duration meets constraints."""
        duration = self.duration
        is_valid = min_duration <= duration <= max_duration
        
        if not is_valid:
            logger.warning(
                f"Highlight duration {duration:.1f}s outside constraints [{min_duration}-{max_duration}]"
            )
        return is_valid
    
    def adjust_to_constraints(self, min_duration: float = 15.0, max_duration: float = 90.0) -> None:
        """Adjust highlight boundaries to meet duration constraints."""
        current_duration = self.duration
        
        if current_duration < min_duration:
            # Extend the highlight symmetrically if possible
            extension_needed = min_duration - current_duration
            half_extension = extension_needed / 2
            
            if self.has_precise_boundaries:
                # Extend precise boundaries
                new_start = max(self.start_time, self.precise_start_time - half_extension)
                new_end = min(self.end_time, self.precise_end_time + half_extension)
                
                # Ensure minimum duration is met
                if new_end - new_start < min_duration:
                    if new_end + (min_duration - (new_end - new_start)) <= self.end_time:
                        new_end = new_start + min_duration
                    else:
                        new_start = max(self.start_time, new_end - min_duration)
                
                self.precise_start_time = new_start
                self.precise_end_time = new_end
                self.boundary_reasoning += f" [Extended to meet {min_duration}s minimum]"
            else:
                # Fall back to segment boundaries - already meet minimum in most cases
                logger.info(f"Highlight too short ({current_duration:.1f}s), using segment boundaries")
                
        elif current_duration > max_duration:
            # Truncate the highlight
            if self.has_precise_boundaries:
                # Keep the start, truncate the end
                self.precise_end_time = self.precise_start_time + max_duration
                self.boundary_reasoning += f" [Truncated to meet {max_duration}s maximum]"
            else:
                # Truncate segment-based boundaries
                self.end_time = self.start_time + max_duration
                logger.info(f"Highlight too long ({current_duration:.1f}s), truncated to {max_duration}s")
    
    def get_clip_offset_and_duration(self, segment_start_time: float) -> tuple[float, float]:
        """Calculate offset and duration for FFmpeg clip creation.
        
        Args:
            segment_start_time: Start time of the source video segment
            
        Returns:
            Tuple of (offset_from_segment_start, clip_duration)
        """
        if self.has_precise_boundaries:
            # Use precise boundaries relative to segment start
            offset = self.precise_start_time - segment_start_time
            duration = self.precise_end_time - self.precise_start_time
        else:
            # Use segment-based boundaries (legacy behavior)
            offset = self.start_time - segment_start_time  
            duration = self.end_time - self.start_time
        
        # Ensure non-negative offset
        offset = max(0.0, offset)
        
        return offset, duration

    def to_highlight(
        self, organization_id: UUID, clip_url: str, thumbnail_url: str
    ) -> Highlight:
        """Convert to Highlight domain model using effective timestamps."""
        highlight = Highlight(
            organization_id=organization_id,
            stream_id=self.stream_id,
            start_time=self.effective_start_time,
            end_time=self.effective_end_time,
            duration=self.duration,
            clip_path=clip_url,
            thumbnail_path=thumbnail_url,
            overall_score=self.overall_score,
        )
        
        # Add dimension scores to the highlight
        for dim_name, score in self.dimension_scores.items():
            highlight.add_dimension_score(dim_name, score, self.confidence)
        
        # Note: Precise boundaries information is embedded in the clip timing itself
        # The fact that we have precise boundaries is reflected in the accurate start/end times
        
        return highlight


class HighlightDetector:
    """Service for detecting highlights in video streams using multi-dimensional scoring."""

    def __init__(
        self,
        scoring_strategy: ScoringStrategy,
        min_highlight_duration: float = 10.0,  # seconds
        max_highlight_duration: float = 120.0,  # seconds
        overlap_threshold: float = 0.5,  # fraction of overlap to merge
    ):
        """Initialize highlight detector.

        Args:
            scoring_strategy: Strategy for scoring video segments
            min_highlight_duration: Minimum duration for a highlight
            max_highlight_duration: Maximum duration for a highlight
            overlap_threshold: Threshold for merging overlapping highlights

        """
        self.scoring_strategy = scoring_strategy
        self.min_highlight_duration = min_highlight_duration
        self.max_highlight_duration = max_highlight_duration
        self.overlap_threshold = overlap_threshold

    async def detect_highlights(
        self, stream: Stream, segments: list[VideoSegment], rubric: ScoringRubric
    ) -> list[HighlightCandidate]:
        """Detect highlights in a stream using the scoring rubric.

        Args:
            stream: Stream being analyzed
            segments: Video segments to analyze
            rubric: Scoring rubric with dimensions and thresholds

        Returns:
            List of highlight candidates

        """
        if not segments:
            logger.warning(f"No segments provided for stream {stream.id}")
            return []

        logger.info(
            f"Detecting highlights in {len(segments)} segments "
            f"for stream {stream.id} using rubric '{rubric.name}'"
        )

        # Score all segments
        scoring_context = ScoringContext()
        segment_scores = []

        for segment in segments:
            try:
                # Try structured scoring first (with precise boundaries)
                if hasattr(self.scoring_strategy, 'score_with_boundaries'):
                    logger.info(f"Using structured scoring for segment {segment.segment_number}")
                    dimension_scores, highlight_boundary = await self.scoring_strategy.score_with_boundaries(
                        content=segment.file_path,
                        rubric=rubric,
                        context=scoring_context.segment_history,
                    )
                else:
                    # Fallback to basic scoring
                    logger.info(f"Using basic scoring for segment {segment.segment_number}")
                    dimension_scores = await self.scoring_strategy.score(
                        content=segment.file_path,
                        rubric=rubric,
                        context=scoring_context.segment_history,
                    )
                    highlight_boundary = None

                # Add to context for temporal analysis
                scores_dict = {
                    name: score for name, (score, _) in dimension_scores.items()
                }
                scoring_context.add_segment_scores(scores_dict)
                scoring_context.segment_history.append(segment)

                # Calculate weighted overall score
                weights = rubric.get_normalized_weights()
                overall_score = sum(
                    scores_dict.get(dim.name, 0.0) * weights.get(dim.name, 0.0)
                    for dim in rubric.dimensions
                )

                # Calculate average confidence
                confidences = [conf for _, conf in dimension_scores.values()]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )

                segment_scores.append(
                    {
                        "segment": segment,
                        "dimension_scores": dimension_scores,
                        "overall_score": overall_score,
                        "confidence": avg_confidence,
                        "highlight_boundary": highlight_boundary,  # New field
                    }
                )

                boundary_info = ""
                if highlight_boundary:
                    boundary_info = f", boundary={highlight_boundary.start_timestamp}-{highlight_boundary.end_timestamp}"
                
                logger.debug(
                    f"Segment {segment.segment_number}: "
                    f"score={overall_score:.2f}, confidence={avg_confidence:.2f}{boundary_info}"
                )

            except Exception as e:
                logger.error(f"Failed to score segment {segment.segment_number}: {e}")
                segment_scores.append(
                    {
                        "segment": segment,
                        "dimension_scores": {},
                        "overall_score": 0.0,
                        "confidence": 0.0,
                        "highlight_boundary": None,
                    }
                )

        # Find highlight candidates based on scores
        candidates = self._find_highlight_candidates(
            segment_scores=segment_scores, rubric=rubric, stream_id=stream.id
        )

        # Merge overlapping candidates
        merged_candidates = self._merge_overlapping_candidates(candidates)

        logger.info(
            f"Found {len(merged_candidates)} highlight candidates "
            f"(merged from {len(candidates)} raw candidates)"
        )

        return merged_candidates

    def _find_highlight_candidates(
        self, segment_scores: list[dict], rubric: ScoringRubric, stream_id: UUID
    ) -> list[HighlightCandidate]:
        """Find potential highlights based on scores exceeding thresholds.
        
        This method now handles both precise boundary-based highlights and 
        traditional segment-based highlights, prioritizing precise boundaries.

        Args:
            segment_scores: List of scored segments with optional precise boundaries
            rubric: Scoring rubric with thresholds
            stream_id: ID of the stream

        Returns:
            List of highlight candidates

        """
        candidates = []
        current_segment_highlight = None

        for score_data in segment_scores:
            segment = score_data["segment"]
            overall_score = score_data["overall_score"]
            confidence = score_data["confidence"]
            highlight_boundary = score_data.get("highlight_boundary")

            # Check if segment meets highlight criteria
            meets_criteria = (
                overall_score >= rubric.highlight_threshold
                and confidence >= rubric.highlight_confidence_threshold
            )

            if meets_criteria and highlight_boundary:
                # Handle precise boundary-based highlight (new approach)
                logger.info(f"Creating precise boundary candidate for segment {segment.segment_number}")
                
                candidate = self._create_precise_candidate(
                    segment=segment,
                    score_data=score_data,
                    highlight_boundary=highlight_boundary,
                    stream_id=stream_id
                )
                
                if candidate and candidate.validate_duration_constraints(
                    self.min_highlight_duration, self.max_highlight_duration
                ):
                    # Adjust boundaries if needed to meet constraints
                    candidate.adjust_to_constraints(
                        self.min_highlight_duration, self.max_highlight_duration
                    )
                    candidates.append(candidate)
                    logger.info(
                        f"Added precise candidate: {candidate.duration:.1f}s "
                        f"({candidate.precise_start_time:.1f}-{candidate.precise_end_time:.1f})"
                    )
                else:
                    logger.warning(f"Precise candidate failed validation for segment {segment.segment_number}")
                    
            elif meets_criteria:
                # Handle segment-based highlight (fallback approach)
                if current_segment_highlight is None:
                    # Start new segment-based highlight
                    current_segment_highlight = {
                        "start_time": segment.start_time,
                        "segments": [segment],
                        "scores": [score_data],
                    }
                else:
                    # Extend current segment-based highlight
                    current_segment_highlight["segments"].append(segment)
                    current_segment_highlight["scores"].append(score_data)
            else:
                # End current segment-based highlight if exists
                if current_segment_highlight is not None:
                    candidate = self._create_segment_candidate(current_segment_highlight, stream_id)
                    if candidate:
                        candidates.append(candidate)
                        logger.info(f"Added segment-based candidate: {candidate.duration:.1f}s")
                    current_segment_highlight = None

        # Handle final segment-based highlight
        if current_segment_highlight is not None:
            candidate = self._create_segment_candidate(current_segment_highlight, stream_id)
            if candidate:
                candidates.append(candidate)
                logger.info(f"Added final segment-based candidate: {candidate.duration:.1f}s")

        logger.info(f"Created {len(candidates)} highlight candidates total")
        return candidates

    def _create_precise_candidate(
        self, 
        segment: VideoSegment,
        score_data: dict,
        highlight_boundary: 'HighlightBoundary', 
        stream_id: UUID
    ) -> Optional[HighlightCandidate]:
        """Create a highlight candidate with precise boundaries.

        Args:
            segment: Video segment containing the highlight
            score_data: Scoring data for the segment
            highlight_boundary: Precise boundary information from Gemini
            stream_id: ID of the stream

        Returns:
            HighlightCandidate with precise boundaries or None if invalid

        """
        try:
            # Convert MM:SS timestamps to absolute seconds
            boundary_start_secs, boundary_end_secs = highlight_boundary.to_seconds()
            
            # Calculate absolute times by adding to segment start
            precise_start = segment.start_time + boundary_start_secs
            precise_end = segment.start_time + boundary_end_secs
            
            # Ensure boundaries are within segment limits
            precise_start = max(segment.start_time, precise_start)
            precise_end = min(segment.end_time, precise_end)
            
            if precise_start >= precise_end:
                logger.warning(f"Invalid precise boundaries: {precise_start} >= {precise_end}")
                return None

            # Extract dimension scores  
            dimension_scores = {
                name: score for name, (score, _) in score_data["dimension_scores"].items()
            }

            # Create candidate with precise boundaries
            candidate = HighlightCandidate(
                stream_id=stream_id,
                start_time=segment.start_time,  # Segment boundaries (fallback)
                end_time=segment.end_time,
                segments=[segment],
                dimension_scores=dimension_scores,
                overall_score=score_data["overall_score"],
                confidence=score_data["confidence"],
                metadata={
                    "segment_count": 1,
                    "boundary_type": "precise",
                    "original_boundary": f"{highlight_boundary.start_timestamp}-{highlight_boundary.end_timestamp}",
                },
                # Precise boundary fields
                precise_start_time=precise_start,
                precise_end_time=precise_end,
                boundary_confidence=highlight_boundary.confidence,
                boundary_reasoning=highlight_boundary.reasoning,
            )
            
            return candidate
            
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to create precise candidate: {e}")
            return None

    def _create_segment_candidate(
        self, highlight_data: dict, stream_id: UUID
    ) -> Optional[HighlightCandidate]:
        """Create a highlight candidate from segment data (fallback method).

        Args:
            highlight_data: Dictionary with segments and scores
            stream_id: ID of the stream

        Returns:
            HighlightCandidate or None if duration constraints not met

        """
        segments = highlight_data["segments"]
        scores = highlight_data["scores"]

        if not segments:
            return None

        # Calculate duration
        start_time = segments[0].start_time
        end_time = segments[-1].end_time
        duration = end_time - start_time

        # Check duration constraints
        if duration < self.min_highlight_duration:
            logger.debug(f"Highlight too short: {duration:.1f}s")
            return None

        if duration > self.max_highlight_duration:
            # Truncate to max duration
            end_time = start_time + self.max_highlight_duration
            # Filter segments that fit within duration
            segments = [s for s in segments if s.start_time < end_time]
            scores = scores[: len(segments)]

        # Aggregate scores across segments
        all_dimension_scores = {}
        total_confidence = 0.0

        for score_data in scores:
            for dim_name, (score, conf) in score_data["dimension_scores"].items():
                if dim_name not in all_dimension_scores:
                    all_dimension_scores[dim_name] = []
                all_dimension_scores[dim_name].append(score)
            total_confidence += score_data["confidence"]

        # Calculate average scores
        avg_dimension_scores = {
            dim: sum(scores) / len(scores)
            for dim, scores in all_dimension_scores.items()
        }

        # Calculate overall score (already weighted)
        overall_scores = [s["overall_score"] for s in scores]
        avg_overall_score = sum(overall_scores) / len(overall_scores)
        avg_confidence = total_confidence / len(scores)

        return HighlightCandidate(
            stream_id=stream_id,
            start_time=start_time,
            end_time=end_time,
            segments=segments,
            dimension_scores=avg_dimension_scores,
            overall_score=avg_overall_score,
            confidence=avg_confidence,
            metadata={
                "segment_count": len(segments),
                "peak_score": max(overall_scores),
                "boundary_type": "segment_based",
            },
            # No precise boundaries for segment-based candidates
            precise_start_time=None,
            precise_end_time=None,
            boundary_confidence=0.0,
            boundary_reasoning="Fallback to segment-based boundaries",
        )

    def _merge_overlapping_candidates(
        self, candidates: list[HighlightCandidate]
    ) -> list[HighlightCandidate]:
        """Merge overlapping highlight candidates.

        Args:
            candidates: List of candidates sorted by start time

        Returns:
            List of merged candidates

        """
        if not candidates:
            return []

        # Sort by start time
        candidates.sort(key=lambda c: c.start_time)

        merged = []
        current = candidates[0]

        for next_candidate in candidates[1:]:
            # Calculate overlap
            overlap_start = max(current.start_time, next_candidate.start_time)
            overlap_end = min(current.end_time, next_candidate.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Check if significant overlap
            overlap_ratio = overlap_duration / min(
                current.duration, next_candidate.duration
            )

            if overlap_ratio >= self.overlap_threshold:
                # Merge candidates
                current = self._merge_two_candidates(current, next_candidate)
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_candidate

        # Add final candidate
        merged.append(current)

        return merged

    def _merge_two_candidates(
        self, c1: HighlightCandidate, c2: HighlightCandidate
    ) -> HighlightCandidate:
        """Merge two highlight candidates.

        Args:
            c1: First candidate
            c2: Second candidate

        Returns:
            Merged candidate

        """
        # Merge time range
        start_time = min(c1.start_time, c2.start_time)
        end_time = max(c1.end_time, c2.end_time)

        # Merge segments
        all_segments = c1.segments + c2.segments
        all_segments.sort(key=lambda s: s.start_time)
        # Remove duplicates
        unique_segments = []
        seen = set()
        for seg in all_segments:
            if seg.segment_number not in seen:
                unique_segments.append(seg)
                seen.add(seg.segment_number)

        # Weighted average of scores based on duration
        w1 = c1.duration / (c1.duration + c2.duration)
        w2 = c2.duration / (c1.duration + c2.duration)

        # Merge dimension scores
        merged_scores = {}
        all_dims = set(c1.dimension_scores.keys()) | set(c2.dimension_scores.keys())
        for dim in all_dims:
            s1 = c1.dimension_scores.get(dim, 0.0)
            s2 = c2.dimension_scores.get(dim, 0.0)
            merged_scores[dim] = w1 * s1 + w2 * s2

        # Merge overall score and confidence
        overall_score = w1 * c1.overall_score + w2 * c2.overall_score
        confidence = w1 * c1.confidence + w2 * c2.confidence

        # Merge metadata
        metadata = {
            **c1.metadata,
            **c2.metadata,
            "merged": True,
            "segment_count": len(unique_segments),
            "peak_score": max(
                c1.metadata.get("peak_score", c1.overall_score),
                c2.metadata.get("peak_score", c2.overall_score),
            ),
        }

        return HighlightCandidate(
            stream_id=c1.stream_id,
            start_time=start_time,
            end_time=end_time,
            segments=unique_segments,
            dimension_scores=merged_scores,
            overall_score=overall_score,
            confidence=confidence,
            metadata=metadata,
        )