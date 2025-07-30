"""Content processing domain service.

This service handles the processing of multimedia content including
video frames, audio transcription, and chat analysis following DDD principles.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from src.domain.services.base import BaseDomainService
from src.domain.entities.stream import Stream
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.exceptions import (
    BusinessRuleViolation,
)


class ContentType(str, Enum):
    """Types of content that can be processed."""

    VIDEO = "video"
    AUDIO = "audio"
    CHAT = "chat"
    MULTIMODAL = "multimodal"


@dataclass
class VideoFrame:
    """Represents a video frame for processing."""

    timestamp: Timestamp
    data: bytes
    width: int
    height: int
    quality_score: float
    metadata: Dict[str, Any]


@dataclass
class AudioSegment:
    """Represents an audio segment for processing."""

    timestamp: Timestamp
    duration: float
    data: bytes
    sample_rate: int
    channels: int
    metadata: Dict[str, Any]


@dataclass
class ChatMessage:
    """Represents a chat message for analysis."""

    timestamp: Timestamp
    user_id: str
    username: str
    text: str
    sentiment_score: float
    is_highlight: bool
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Result of content processing."""

    content_type: ContentType
    timestamp: Timestamp
    confidence: ConfidenceScore
    highlight_score: float
    features: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ContentAnalysisResult:
    """Aggregated result of content analysis."""

    stream_id: int
    timestamp: Timestamp
    video_features: Optional[Dict[str, Any]]
    audio_features: Optional[Dict[str, Any]]
    chat_features: Optional[Dict[str, Any]]
    combined_score: float
    is_highlight: bool
    confidence: ConfidenceScore
    metadata: Dict[str, Any]


class ContentProcessingService(BaseDomainService):
    """Domain service for content processing orchestration.

    Handles the business logic for processing video frames, audio segments,
    and chat messages to identify potential highlights.
    """

    def __init__(self):
        """Initialize content processing service."""
        super().__init__()

        # Thresholds and configuration
        self.highlight_threshold = 0.7
        self.min_confidence = 0.5
        self.frame_quality_threshold = 0.3

    async def process_video_frame(
        self, stream: Stream, frame: VideoFrame
    ) -> ProcessingResult:
        """Process a single video frame.

        Args:
            stream: The stream being processed
            frame: Video frame to process

        Returns:
            ProcessingResult with video analysis

        Raises:
            ProcessingError: If frame processing fails
            BusinessRuleViolation: If frame doesn't meet quality requirements
        """
        # Validate frame quality
        if frame.quality_score < self.frame_quality_threshold:
            raise BusinessRuleViolation(
                f"Frame quality {frame.quality_score} below threshold {self.frame_quality_threshold}"
            )

        # In a real implementation, this would call ML models
        # For now, return a mock result
        features = {
            "scene_change": frame.quality_score > 0.8,
            "motion_intensity": frame.quality_score * 0.9,
            "visual_complexity": 0.5,
            "detected_objects": ["player", "goal", "ball"]
            if frame.quality_score > 0.7
            else [],
        }

        highlight_score = self._calculate_video_highlight_score(features)

        return ProcessingResult(
            content_type=ContentType.VIDEO,
            timestamp=frame.timestamp,
            confidence=ConfidenceScore(min(frame.quality_score + 0.1, 1.0)),
            highlight_score=highlight_score,
            features=features,
            metadata=frame.metadata,
        )

    async def process_audio_segment(
        self, stream: Stream, segment: AudioSegment
    ) -> ProcessingResult:
        """Process an audio segment.

        Args:
            stream: The stream being processed
            segment: Audio segment to process

        Returns:
            ProcessingResult with audio analysis
        """
        # In a real implementation, this would perform:
        # - Speech-to-text transcription
        # - Audio event detection
        # - Volume/energy analysis

        features = {
            "volume_peak": 0.8,
            "speech_detected": True,
            "transcription": "Goal scored! What an amazing play!",
            "audio_events": ["crowd_cheer", "commentator_excitement"],
            "energy_level": 0.9,
        }

        highlight_score = self._calculate_audio_highlight_score(features)

        return ProcessingResult(
            content_type=ContentType.AUDIO,
            timestamp=segment.timestamp,
            confidence=ConfidenceScore(0.85),
            highlight_score=highlight_score,
            features=features,
            metadata=segment.metadata,
        )

    async def process_chat_messages(
        self, stream: Stream, messages: List[ChatMessage]
    ) -> ProcessingResult:
        """Process a batch of chat messages.

        Args:
            stream: The stream being processed
            messages: Chat messages to analyze

        Returns:
            ProcessingResult with chat analysis
        """
        if not messages:
            return ProcessingResult(
                content_type=ContentType.CHAT,
                timestamp=Timestamp.now(),
                confidence=ConfidenceScore(0.0),
                highlight_score=0.0,
                features={},
                metadata={},
            )

        # Analyze chat velocity and sentiment
        avg_sentiment = sum(msg.sentiment_score for msg in messages) / len(messages)
        highlight_messages = [msg for msg in messages if msg.is_highlight]
        chat_velocity = len(messages)

        features = {
            "message_count": len(messages),
            "average_sentiment": avg_sentiment,
            "highlight_message_ratio": len(highlight_messages) / len(messages),
            "chat_velocity": chat_velocity,
            "peak_activity": chat_velocity > 50,
            "emote_density": self._calculate_emote_density(messages),
        }

        highlight_score = self._calculate_chat_highlight_score(features)

        return ProcessingResult(
            content_type=ContentType.CHAT,
            timestamp=messages[0].timestamp,  # Use first message timestamp
            confidence=ConfidenceScore(min(0.7 + (chat_velocity / 100), 0.95)),
            highlight_score=highlight_score,
            features=features,
            metadata={"message_count": len(messages)},
        )

    async def analyze_multimodal_content(
        self,
        stream: Stream,
        video_result: Optional[ProcessingResult],
        audio_result: Optional[ProcessingResult],
        chat_result: Optional[ProcessingResult],
    ) -> ContentAnalysisResult:
        """Combine multiple content types for highlight detection.

        Args:
            stream: The stream being processed
            video_result: Video processing result
            audio_result: Audio processing result
            chat_result: Chat processing result

        Returns:
            Combined analysis result
        """
        # Collect available results
        results = [r for r in [video_result, audio_result, chat_result] if r]

        if not results:
            raise BusinessRuleViolation("At least one content type must be processed")

        # Calculate combined score with weights
        weights = {
            ContentType.VIDEO: 0.4,
            ContentType.AUDIO: 0.3,
            ContentType.CHAT: 0.3,
        }

        total_score = 0.0
        total_weight = 0.0

        for result in results:
            weight = weights.get(result.content_type, 0.0)
            total_score += result.highlight_score * weight
            total_weight += weight

        combined_score = total_score / total_weight if total_weight > 0 else 0.0

        # Calculate combined confidence
        avg_confidence = sum(r.confidence.value for r in results) / len(results)

        # Determine if this is a highlight
        is_highlight = (
            combined_score >= self.highlight_threshold
            and avg_confidence >= self.min_confidence
        )

        return ContentAnalysisResult(
            stream_id=stream.id,
            timestamp=results[0].timestamp,  # Use earliest timestamp
            video_features=video_result.features if video_result else None,
            audio_features=audio_result.features if audio_result else None,
            chat_features=chat_result.features if chat_result else None,
            combined_score=combined_score,
            is_highlight=is_highlight,
            confidence=ConfidenceScore(avg_confidence),
            metadata={
                "content_types": [r.content_type.value for r in results],
                "individual_scores": {
                    r.content_type.value: r.highlight_score for r in results
                },
            },
        )

    def _calculate_video_highlight_score(self, features: Dict[str, Any]) -> float:
        """Calculate highlight score from video features."""
        score = 0.0

        if features.get("scene_change"):
            score += 0.3

        score += features.get("motion_intensity", 0) * 0.3
        score += features.get("visual_complexity", 0) * 0.2

        if features.get("detected_objects"):
            score += 0.2

        return min(score, 1.0)

    def _calculate_audio_highlight_score(self, features: Dict[str, Any]) -> float:
        """Calculate highlight score from audio features."""
        score = 0.0

        score += features.get("volume_peak", 0) * 0.2
        score += features.get("energy_level", 0) * 0.3

        if features.get("speech_detected"):
            score += 0.2

        if "crowd_cheer" in features.get("audio_events", []):
            score += 0.3

        return min(score, 1.0)

    def _calculate_chat_highlight_score(self, features: Dict[str, Any]) -> float:
        """Calculate highlight score from chat features."""
        score = 0.0

        score += features.get("average_sentiment", 0) * 0.2
        score += features.get("highlight_message_ratio", 0) * 0.3

        if features.get("peak_activity"):
            score += 0.3

        score += min(features.get("chat_velocity", 0) / 100, 0.2)

        return min(score, 1.0)

    def _calculate_emote_density(self, messages: List[ChatMessage]) -> float:
        """Calculate the density of emotes in chat messages."""
        if not messages:
            return 0.0

        # Simple implementation - count messages with common emote patterns
        emote_count = sum(
            1
            for msg in messages
            if any(
                pattern in msg.text for pattern in ["Pog", "KEKW", "LUL", ":)", ":("]
            )
        )

        return emote_count / len(messages)
