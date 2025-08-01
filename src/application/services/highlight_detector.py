"""Highlight detection service using multi-dimensional scoring."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID

from ...domain.models.highlight import Highlight
from ...domain.models.stream import Stream
from ...domain.services.dimension_framework import (
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
    """Potential highlight with scoring information."""

    stream_id: UUID
    start_time: float
    end_time: float
    segments: list[VideoSegment]
    dimension_scores: dict[str, float]
    overall_score: float
    confidence: float
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Calculate highlight duration in seconds."""
        return self.end_time - self.start_time

    def to_highlight(
        self, organization_id: UUID, s3_url: str, thumbnail_url: str
    ) -> Highlight:
        """Convert to Highlight domain model."""
        highlight = Highlight(
            organization_id=organization_id,
            stream_id=self.stream_id,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.duration,
            s3_url=s3_url,
            thumbnail_url=thumbnail_url,
            confidence_score=self.confidence,
            metadata={
                "dimension_scores": self.dimension_scores,
                "overall_score": self.overall_score,
                **self.metadata,
            },
        )
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
                # Score the segment
                dimension_scores = await self.scoring_strategy.score(
                    content=segment.file_path,
                    rubric=rubric,
                    context=scoring_context.segment_history,
                )

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
                    }
                )

                logger.debug(
                    f"Segment {segment.segment_number}: "
                    f"score={overall_score:.2f}, confidence={avg_confidence:.2f}"
                )

            except Exception as e:
                logger.error(f"Failed to score segment {segment.segment_number}: {e}")
                segment_scores.append(
                    {
                        "segment": segment,
                        "dimension_scores": {},
                        "overall_score": 0.0,
                        "confidence": 0.0,
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

        Args:
            segment_scores: List of scored segments
            rubric: Scoring rubric with thresholds
            stream_id: ID of the stream

        Returns:
            List of highlight candidates
        """
        candidates = []
        current_highlight = None

        for score_data in segment_scores:
            segment = score_data["segment"]
            overall_score = score_data["overall_score"]
            confidence = score_data["confidence"]

            # Check if segment meets highlight criteria
            meets_criteria = (
                overall_score >= rubric.highlight_threshold
                and confidence >= rubric.highlight_confidence_threshold
            )

            if meets_criteria:
                if current_highlight is None:
                    # Start new highlight
                    current_highlight = {
                        "start_time": segment.start_time,
                        "segments": [segment],
                        "scores": [score_data],
                    }
                else:
                    # Extend current highlight
                    current_highlight["segments"].append(segment)
                    current_highlight["scores"].append(score_data)
            else:
                # End current highlight if exists
                if current_highlight is not None:
                    candidate = self._create_candidate(current_highlight, stream_id)
                    if candidate:
                        candidates.append(candidate)
                    current_highlight = None

        # Handle final highlight
        if current_highlight is not None:
            candidate = self._create_candidate(current_highlight, stream_id)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _create_candidate(
        self, highlight_data: dict, stream_id: UUID
    ) -> Optional[HighlightCandidate]:
        """Create a highlight candidate from segment data.

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
            },
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
