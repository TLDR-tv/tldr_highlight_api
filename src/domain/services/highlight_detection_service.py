"""Highlight detection domain service.

This service orchestrates the highlight detection process, aggregating
results from multiple detectors and creating highlight entities.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.domain.services.base import BaseDomainService
from src.domain.entities.highlight import Highlight, HighlightType
from src.domain.entities.stream import Stream
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.url import Url
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.stream_repository import StreamRepository
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
    
    Aggregates results from multiple detectors (video, audio, chat),
    applies fusion scoring, and creates highlight entities.
    """
    
    def __init__(
        self,
        highlight_repo: HighlightRepository,
        stream_repo: StreamRepository,
        min_confidence_threshold: float = 0.7,
        min_duration_seconds: float = 5.0,
        max_duration_seconds: float = 120.0
    ):
        """Initialize highlight detection service.
        
        Args:
            highlight_repo: Repository for highlight operations
            stream_repo: Repository for stream operations
            min_confidence_threshold: Minimum confidence score to create highlight
            min_duration_seconds: Minimum highlight duration
            max_duration_seconds: Maximum highlight duration
        """
        super().__init__()
        self.highlight_repo = highlight_repo
        self.stream_repo = stream_repo
        self.min_confidence_threshold = min_confidence_threshold
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
    
    async def process_detection_results(
        self,
        stream_id: int,
        video_results: List[DetectionResult],
        audio_results: List[DetectionResult],
        chat_results: List[DetectionResult]
    ) -> List[Highlight]:
        """Process detection results from multiple sources and create highlights.
        
        Args:
            stream_id: Stream ID
            video_results: Results from video detector
            audio_results: Results from audio detector
            chat_results: Results from chat detector
            
        Returns:
            List of created highlights
            
        Raises:
            EntityNotFoundError: If stream doesn't exist
        """
        # Verify stream exists
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        # Aggregate and fuse results
        fused_segments = self._fuse_detection_results(
            video_results, audio_results, chat_results
        )
        
        # Filter by confidence and duration
        valid_segments = self._filter_segments(fused_segments)
        
        # Create highlight entities
        highlights = []
        for segment in valid_segments:
            highlight = await self._create_highlight_from_segment(
                stream_id, segment
            )
            highlights.append(highlight)
        
        # Bulk create highlights
        if highlights:
            created_highlights = await self.highlight_repo.bulk_create(highlights)
            self.logger.info(
                f"Created {len(created_highlights)} highlights for stream {stream_id}"
            )
            return created_highlights
        
        return []
    
    async def detect_trending_moments(
        self,
        stream_id: int,
        chat_spike_threshold: float = 2.0,
        engagement_window_seconds: float = 30.0
    ) -> List[Highlight]:
        """Detect trending moments based on chat engagement spikes.
        
        Args:
            stream_id: Stream ID
            chat_spike_threshold: Multiplier for baseline chat rate
            engagement_window_seconds: Time window for engagement analysis
            
        Returns:
            List of trending highlights
        """
        # This would integrate with real-time chat analysis
        # For now, return empty list as placeholder
        self.logger.info(f"Detecting trending moments for stream {stream_id}")
        return []
    
    async def post_process_highlights(
        self,
        stream_id: int,
        merge_threshold_seconds: float = 10.0
    ) -> List[Highlight]:
        """Post-process highlights to merge overlapping segments.
        
        Args:
            stream_id: Stream ID
            merge_threshold_seconds: Max gap between highlights to merge
            
        Returns:
            List of processed highlights
        """
        # Get all highlights for stream
        highlights = await self.highlight_repo.get_by_stream(stream_id)
        
        if len(highlights) < 2:
            return highlights
        
        # Sort by start time
        sorted_highlights = sorted(highlights, key=lambda h: h.start_time.value)
        
        # Merge overlapping or close highlights
        merged = []
        current = sorted_highlights[0]
        
        for next_highlight in sorted_highlights[1:]:
            gap = next_highlight.start_time.value - current.end_time.value
            
            if gap <= merge_threshold_seconds:
                # Merge highlights
                current = self._merge_highlights(current, next_highlight)
            else:
                merged.append(current)
                current = next_highlight
        
        merged.append(current)
        
        # Update merged highlights
        updated = []
        for highlight in merged:
            saved = await self.highlight_repo.save(highlight)
            updated.append(saved)
        
        self.logger.info(
            f"Post-processed {len(highlights)} highlights into {len(updated)} for stream {stream_id}"
        )
        
        return updated
    
    async def rank_highlights(
        self,
        stream_id: int,
        limit: Optional[int] = None
    ) -> List[Tuple[Highlight, float]]:
        """Rank highlights by importance score.
        
        Args:
            stream_id: Stream ID
            limit: Optional limit on number of results
            
        Returns:
            List of (highlight, score) tuples sorted by score
        """
        highlights = await self.highlight_repo.get_by_stream(stream_id)
        
        # Calculate importance scores
        scored_highlights = []
        for highlight in highlights:
            score = self._calculate_importance_score(highlight)
            scored_highlights.append((highlight, score))
        
        # Sort by score descending
        scored_highlights.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit if specified
        if limit:
            scored_highlights = scored_highlights[:limit]
        
        return scored_highlights
    
    async def generate_highlight_metadata(
        self,
        highlight_id: int,
        include_thumbnails: bool = True,
        include_transcription: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for a highlight.
        
        Args:
            highlight_id: Highlight ID
            include_thumbnails: Whether to include thumbnail URLs
            include_transcription: Whether to include transcription
            
        Returns:
            Dictionary with highlight metadata
        """
        highlight = await self.highlight_repo.get(highlight_id)
        if not highlight:
            raise EntityNotFoundError(f"Highlight {highlight_id} not found")
        
        metadata = {
            "id": highlight.id,
            "title": highlight.title,
            "description": highlight.description,
            "duration_seconds": highlight.duration.value,
            "confidence_score": highlight.confidence_score.value,
            "type": highlight.highlight_type.value,
            "tags": highlight.tags,
            "sentiment": highlight.sentiment_label,
            "engagement_level": highlight.engagement_level,
        }
        
        if include_thumbnails and highlight.thumbnail_url:
            metadata["thumbnail_url"] = highlight.thumbnail_url.value
        
        if include_transcription:
            # Extract transcription from audio analysis if available
            transcription = highlight.audio_analysis.get("transcription", "")
            metadata["transcription"] = transcription
        
        return metadata
    
    # Private helper methods
    
    def _fuse_detection_results(
        self,
        video_results: List[DetectionResult],
        audio_results: List[DetectionResult],
        chat_results: List[DetectionResult]
    ) -> List[Dict[str, Any]]:
        """Fuse detection results from multiple sources."""
        # Combine all results with weights
        all_segments = []
        
        # Weight contributions from different detectors
        weights = {
            "video": 0.4,
            "audio": 0.3,
            "chat": 0.3
        }
        
        # Add weighted results
        for result in video_results:
            all_segments.append({
                "start_time": result.start_time,
                "end_time": result.end_time,
                "weighted_confidence": result.confidence * weights["video"],
                "detectors": {"video": result},
                "metadata": result.metadata
            })
        
        for result in audio_results:
            all_segments.append({
                "start_time": result.start_time,
                "end_time": result.end_time,
                "weighted_confidence": result.confidence * weights["audio"],
                "detectors": {"audio": result},
                "metadata": result.metadata
            })
        
        for result in chat_results:
            all_segments.append({
                "start_time": result.start_time,
                "end_time": result.end_time,
                "weighted_confidence": result.confidence * weights["chat"],
                "detectors": {"chat": result},
                "metadata": result.metadata
            })
        
        # Merge overlapping segments
        if not all_segments:
            return []
        
        # Sort by start time
        all_segments.sort(key=lambda x: x["start_time"])
        
        # Merge overlapping segments and combine confidence scores
        merged = []
        current = all_segments[0].copy()
        
        for segment in all_segments[1:]:
            if segment["start_time"] <= current["end_time"]:
                # Overlapping - merge
                current["end_time"] = max(current["end_time"], segment["end_time"])
                current["weighted_confidence"] += segment["weighted_confidence"]
                current["detectors"].update(segment["detectors"])
                # Merge metadata
                for key, value in segment["metadata"].items():
                    if key not in current["metadata"]:
                        current["metadata"][key] = value
            else:
                # Non-overlapping - save current and start new
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        
        return merged
    
    def _filter_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter segments by confidence and duration."""
        valid = []
        
        for segment in segments:
            duration = segment["end_time"] - segment["start_time"]
            
            # Check duration constraints
            if duration < self.min_duration_seconds:
                continue
            if duration > self.max_duration_seconds:
                # Split long segments
                segment["end_time"] = segment["start_time"] + self.max_duration_seconds
            
            # Check confidence threshold
            if segment["weighted_confidence"] >= self.min_confidence_threshold:
                valid.append(segment)
        
        return valid
    
    async def _create_highlight_from_segment(
        self,
        stream_id: int,
        segment: Dict[str, Any]
    ) -> Highlight:
        """Create highlight entity from processed segment."""
        # Determine highlight type based on detectors
        highlight_type = self._determine_highlight_type(segment["detectors"])
        
        # Generate title and description
        title = self._generate_title(segment, highlight_type)
        description = self._generate_description(segment)
        
        # Extract analysis data
        video_analysis = {}
        audio_analysis = {}
        chat_analysis = {}
        
        if "video" in segment["detectors"]:
            video_analysis = segment["detectors"]["video"].metadata
        if "audio" in segment["detectors"]:
            audio_analysis = segment["detectors"]["audio"].metadata
        if "chat" in segment["detectors"]:
            chat_analysis = segment["detectors"]["chat"].metadata
        
        # Create highlight entity
        return Highlight(
            id=None,
            stream_id=stream_id,
            start_time=Duration(segment["start_time"]),
            end_time=Duration(segment["end_time"]),
            confidence_score=ConfidenceScore(min(segment["weighted_confidence"], 1.0)),
            highlight_type=highlight_type,
            title=title,
            description=description,
            tags=self._extract_tags(segment),
            sentiment_score=chat_analysis.get("sentiment_score"),
            viewer_engagement=chat_analysis.get("engagement_score"),
            video_analysis=video_analysis,
            audio_analysis=audio_analysis,
            chat_analysis=chat_analysis,
            processed_by="fusion_detector_v1",
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
    
    def _determine_highlight_type(self, detectors: Dict[str, DetectionResult]) -> HighlightType:
        """Determine highlight type based on active detectors."""
        # Priority order for type determination
        if "video" in detectors:
            video_type = detectors["video"].metadata.get("detected_type")
            if video_type == "gameplay":
                return HighlightType.GAMEPLAY
            elif video_type == "reaction":
                return HighlightType.REACTION
        
        if "audio" in detectors:
            audio_features = detectors["audio"].metadata.get("features", {})
            if audio_features.get("is_funny", False):
                return HighlightType.FUNNY
            elif audio_features.get("is_emotional", False):
                return HighlightType.EMOTIONAL
        
        if "chat" in detectors:
            chat_sentiment = detectors["chat"].metadata.get("dominant_sentiment")
            if chat_sentiment == "excitement":
                return HighlightType.CLIMACTIC
        
        return HighlightType.CUSTOM
    
    def _generate_title(self, segment: Dict[str, Any], highlight_type: HighlightType) -> str:
        """Generate title for highlight."""
        # Use metadata to create descriptive title
        if highlight_type == HighlightType.GAMEPLAY:
            return "Epic Gameplay Moment"
        elif highlight_type == HighlightType.REACTION:
            return "Streamer Reaction"
        elif highlight_type == HighlightType.FUNNY:
            return "Hilarious Moment"
        elif highlight_type == HighlightType.EMOTIONAL:
            return "Emotional Scene"
        elif highlight_type == HighlightType.CLIMACTIC:
            return "Climactic Moment"
        else:
            return "Stream Highlight"
    
    def _generate_description(self, segment: Dict[str, Any]) -> str:
        """Generate description for highlight."""
        descriptions = []
        
        if "video" in segment["detectors"]:
            video_desc = segment["detectors"]["video"].metadata.get("description")
            if video_desc:
                descriptions.append(video_desc)
        
        if "audio" in segment["detectors"]:
            audio_desc = segment["detectors"]["audio"].metadata.get("description")
            if audio_desc:
                descriptions.append(audio_desc)
        
        if "chat" in segment["detectors"]:
            chat_desc = segment["detectors"]["chat"].metadata.get("description")
            if chat_desc:
                descriptions.append(chat_desc)
        
        return " ".join(descriptions) if descriptions else "Automatically detected highlight"
    
    def _extract_tags(self, segment: Dict[str, Any]) -> List[str]:
        """Extract tags from segment metadata."""
        tags = set()
        
        for detector_result in segment["detectors"].values():
            detector_tags = detector_result.metadata.get("tags", [])
            tags.update(detector_tags)
        
        return list(tags)[:10]  # Limit to 10 tags
    
    def _merge_highlights(self, h1: Highlight, h2: Highlight) -> Highlight:
        """Merge two highlights into one."""
        # Use higher confidence score
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
            highlight_type=h1.highlight_type,  # Keep first type
            title=h1.title,
            description=f"{h1.description} {h2.description}".strip(),
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
    
    def _calculate_importance_score(self, highlight: Highlight) -> float:
        """Calculate importance score for ranking."""
        score = 0.0
        
        # Base score from confidence
        score += highlight.confidence_score.value * 0.4
        
        # Engagement score
        if highlight.viewer_engagement:
            score += highlight.viewer_engagement * 0.3
        
        # Sentiment impact
        if highlight.sentiment_score is not None:
            # Both positive and negative extremes are interesting
            sentiment_impact = abs(highlight.sentiment_score)
            score += sentiment_impact * 0.2
        
        # Duration factor (prefer medium length)
        duration = highlight.duration.value
        if 10 <= duration <= 60:
            score += 0.1
        elif 60 < duration <= 90:
            score += 0.05
        
        return min(score, 1.0)