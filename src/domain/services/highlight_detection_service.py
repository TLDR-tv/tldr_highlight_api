"""Highlight detection domain service.

This service orchestrates the highlight detection process using
configurable dimensions and analysis strategies.
"""

from typing import List, Dict, Any, Optional

from src.domain.services.base import BaseDomainService
from src.domain.entities.highlight import Highlight
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.processing_options import (
    ProcessingOptions,
)
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.dimension_set_repository import DimensionSetRepository
from src.domain.repositories.highlight_type_registry_repository import (
    HighlightTypeRegistryRepository,
)
from src.domain.exceptions import EntityNotFoundError


class DetectionResult:
    """Result from a single detector."""

    def __init__(
        self,
        start_time: float,
        end_time: float,
        confidence: float,
        detector_type: str,
        metadata: Dict[str, Any],
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence
        self.detector_type = detector_type
        self.metadata = metadata


class HighlightDetectionService(BaseDomainService):
    """Domain service for highlight detection orchestration.

    Simplified service that coordinates highlight detection using B2BStreamAgent.
    """

    def __init__(
        self,
        highlight_repo: HighlightRepository,
        stream_repo: StreamRepository,
        dimension_set_repo: Optional[DimensionSetRepository] = None,
        type_registry_repo: Optional[HighlightTypeRegistryRepository] = None,
    ):
        """Initialize highlight detection service.

        Args:
            highlight_repo: Repository for highlight operations
            stream_repo: Repository for stream operations
            dimension_set_repo: Optional repository for dimension sets
            type_registry_repo: Optional repository for highlight type registries
        """
        super().__init__()
        self.highlight_repo = highlight_repo
        self.stream_repo = stream_repo
        self.dimension_set_repo = dimension_set_repo
        self.type_registry_repo = type_registry_repo

    async def process_stream_segment(
        self,
        stream_id: int,
        segment_data: Dict[str, Any],
        processing_options: ProcessingOptions,
    ) -> List[Highlight]:
        """Process a stream segment to detect highlights.

        This method is now deprecated and returns an empty list.
        Highlight detection is handled by B2BStreamAgent directly.

        Args:
            stream_id: Stream ID
            segment_data: Multi-modal segment data
            processing_options: Processing configuration

        Returns:
            Empty list - actual detection happens in B2BStreamAgent
        """
        # This method is kept for backward compatibility but returns empty
        # Actual highlight detection is handled by B2BStreamAgent in async tasks
        self.logger.warning(
            f"process_stream_segment called for stream {stream_id} - "
            "this method is deprecated, use B2BStreamAgent directly"
        )
        return []

    async def detect_highlights_batch(
        self,
        stream_id: int,
        segments: List[Dict[str, Any]],
        processing_options: ProcessingOptions,
    ) -> List[Highlight]:
        """Process multiple segments to detect highlights.

        This method is now deprecated and returns an empty list.
        Highlight detection is handled by B2BStreamAgent directly.

        Args:
            stream_id: Stream ID
            segments: List of segment data
            processing_options: Processing configuration

        Returns:
            Empty list - actual detection happens in B2BStreamAgent
        """
        self.logger.warning(
            f"detect_highlights_batch called for stream {stream_id} - "
            "this method is deprecated, use B2BStreamAgent directly"
        )
        return []











    async def process_detection_results(
        self,
        stream_id: int,
        video_results: List[DetectionResult],
        audio_results: List[DetectionResult],
        chat_results: List[DetectionResult],
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
        all_results = (
            [(r, "video") for r in video_results]
            + [(r, "audio") for r in audio_results]
            + [(r, "chat") for r in chat_results]
        )

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
                "metadata": {},
            }

            for result, modality in all_results:
                if result.start_time <= current_segment["end_time"]:
                    # Overlapping - merge
                    current_segment["end_time"] = max(
                        current_segment["end_time"], result.end_time
                    )
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
                        "metadata": {},
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
        return await self.detect_highlights_batch(
            stream_id, segments, processing_options
        )

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
        confidence = ConfidenceScore(
            max(h1.confidence_score.value, h2.confidence_score.value)
        )

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
            end_time=h2.end_time,  # Use later end
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
            updated_at=Timestamp.now(),
        )
