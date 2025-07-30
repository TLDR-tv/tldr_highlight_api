"""
Enhanced chat-based highlight detection with comprehensive sentiment analysis.

This module integrates the advanced sentiment analysis system with the existing
chat detector to provide more accurate highlight detection based on:
- Multi-modal sentiment analysis
- Chat velocity and acceleration
- Event impact scoring
- Emote-based sentiment
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


from .base_detector import (
    BaseDetector,
    ContentSegment,
    DetectionConfig,
    DetectionResult,
    ModalityType,
)
from .chat_detector import (
    ChatDetectionConfig,
    ChatMessage as OldChatMessage,
    ChatWindow,
)
from ..chat_adapters.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentCategory,
    WindowSentiment,
    VelocityMetrics,
    TemporalAnalyzer,
    ChatMessage as NewChatMessage,
    ChatUser,
    ChatEvent,
    ChatEventType,
)

logger = logging.getLogger(__name__)


class EnhancedChatDetectionConfig(ChatDetectionConfig):
    """
    Enhanced configuration for chat detection with sentiment analysis.
    """

    # Sentiment analysis parameters
    sentiment_window_size: float = 10.0  # Window size for sentiment analysis
    velocity_spike_weight: float = 0.3  # Weight for velocity spikes
    event_impact_weight: float = 0.2  # Weight for special events
    sentiment_category_weights: Dict[str, float] = {
        SentimentCategory.HYPE: 1.0,
        SentimentCategory.EXCITEMENT: 0.9,
        SentimentCategory.VERY_POSITIVE: 0.7,
        SentimentCategory.POSITIVE: 0.5,
        SentimentCategory.NEUTRAL: 0.2,
        SentimentCategory.NEGATIVE: 0.1,
        SentimentCategory.VERY_NEGATIVE: 0.0,
        SentimentCategory.DISAPPOINTMENT: 0.1,
        SentimentCategory.SURPRISE: 0.6,
    }

    # Minimum thresholds
    min_velocity_spike_intensity: float = 0.3
    min_event_impact: float = 0.2
    min_sentiment_confidence: float = 0.4


class ChatMessageAdapter:
    """Adapter to convert between old and new chat message formats."""

    @staticmethod
    def old_to_new(
        old_msg: OldChatMessage, platform: str = "unknown"
    ) -> NewChatMessage:
        """Convert old ChatMessage format to new format."""
        return NewChatMessage(
            id=f"msg_{old_msg.timestamp}_{old_msg.user_id}",
            user=ChatUser(
                id=old_msg.user_id,
                username=old_msg.username,
                display_name=old_msg.username,
            ),
            text=old_msg.message,
            timestamp=datetime.fromtimestamp(old_msg.timestamp, tz=timezone.utc),
            emotes=[],  # Extract from metadata if available
            metadata=old_msg.metadata,
        )

    @staticmethod
    def new_to_old(new_msg: NewChatMessage) -> OldChatMessage:
        """Convert new ChatMessage format to old format."""
        return OldChatMessage(
            timestamp=new_msg.timestamp.timestamp(),
            user_id=new_msg.user.id,
            username=new_msg.user.username,
            message=new_msg.text,
            platform=new_msg.metadata.get("platform", "unknown"),
            metadata=new_msg.metadata,
        )


class EnhancedChatDetector(BaseDetector):
    """
    Enhanced chat detector with comprehensive sentiment analysis.

    This detector combines the existing chat detection capabilities with
    advanced sentiment analysis for more accurate highlight detection.
    """

    def __init__(self, config: Optional[EnhancedChatDetectionConfig] = None):
        """Initialize the enhanced chat detector."""
        self.chat_config = config or EnhancedChatDetectionConfig()
        super().__init__(self.chat_config)

        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(
            window_size_seconds=self.chat_config.sentiment_window_size,
            min_confidence=self.chat_config.min_sentiment_confidence,
        )

        self.logger = logging.getLogger(f"{__name__}.EnhancedChatDetector")

    @property
    def modality(self) -> ModalityType:
        """Get the modality this detector handles."""
        return ModalityType.CHAT

    @property
    def algorithm_name(self) -> str:
        """Get the name of the detection algorithm."""
        return "EnhancedChatExcitementDetector"

    @property
    def algorithm_version(self) -> str:
        """Get the version of the detection algorithm."""
        return "2.0.0"

    def _validate_segment(self, segment: ContentSegment) -> bool:
        """Validate that a segment contains valid chat data."""
        if not super()._validate_segment(segment):
            return False

        # Check if data is chat messages or window
        if not isinstance(segment.data, (list, ChatWindow)):
            return False

        return True

    async def _prepare_messages(self, segment: ContentSegment) -> List[NewChatMessage]:
        """Prepare messages from segment data."""
        messages = []
        platform = segment.metadata.get("platform", "unknown")

        if isinstance(segment.data, ChatWindow):
            # Convert from ChatWindow
            for old_msg in segment.data.messages:
                messages.append(ChatMessageAdapter.old_to_new(old_msg, platform))

        elif isinstance(segment.data, list):
            for msg_data in segment.data:
                if isinstance(msg_data, OldChatMessage):
                    messages.append(ChatMessageAdapter.old_to_new(msg_data, platform))
                elif isinstance(msg_data, dict):
                    # Create from dictionary
                    old_msg = OldChatMessage(
                        timestamp=msg_data.get("timestamp", segment.start_time),
                        user_id=msg_data.get("user_id", "unknown"),
                        username=msg_data.get("username", "unknown"),
                        message=msg_data.get("message", ""),
                        platform=platform,
                        metadata=msg_data.get("metadata", {}),
                    )
                    messages.append(ChatMessageAdapter.old_to_new(old_msg, platform))

        return messages

    async def _process_events(self, segment: ContentSegment):
        """Process any chat events in the segment."""
        events = segment.metadata.get("events", [])

        for event_data in events:
            if isinstance(event_data, ChatEvent):
                await self.sentiment_analyzer.process_event(event_data)
            elif isinstance(event_data, dict):
                # Create ChatEvent from dictionary
                try:
                    event = ChatEvent(
                        id=event_data.get("id", f"evt_{datetime.now().timestamp()}"),
                        type=ChatEventType(event_data.get("type", "message")),
                        timestamp=datetime.fromtimestamp(
                            event_data.get("timestamp", segment.start_time),
                            tz=timezone.utc,
                        ),
                        data=event_data.get("data", {}),
                    )
                    await self.sentiment_analyzer.process_event(event)
                except Exception as e:
                    self.logger.warning(f"Failed to process event: {e}")

    async def _detect_features(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Detect chat-based highlight features using enhanced sentiment analysis.
        """
        chat_config = (
            config
            if isinstance(config, EnhancedChatDetectionConfig)
            else self.chat_config
        )

        try:
            # Prepare messages
            messages = await self._prepare_messages(segment)
            if not messages:
                self.logger.warning(
                    f"No valid messages in segment {segment.segment_id}"
                )
                return []

            # Process events
            await self._process_events(segment)

            # Get platform
            platform = segment.metadata.get("platform", "generic")

            # Analyze sentiment window
            start_time = datetime.fromtimestamp(segment.start_time, tz=timezone.utc)
            end_time = datetime.fromtimestamp(segment.end_time, tz=timezone.utc)

            window_sentiment = await self.sentiment_analyzer.analyze_window(
                messages, start_time, end_time, platform
            )

            # Get velocity metrics at end of window
            velocity_metrics = (
                self.sentiment_analyzer.temporal_analyzer.calculate_velocity_metrics(
                    end_time
                )
            )

            # Get event impact
            event_impact = self.sentiment_analyzer.event_tracker.get_current_impact(
                end_time
            )

            # Calculate highlight confidence
            highlight_confidence = self.sentiment_analyzer.get_highlight_confidence(
                window_sentiment, velocity_metrics, event_impact
            )

            # Calculate component scores
            sentiment_score = self._calculate_sentiment_score(
                window_sentiment, chat_config
            )
            velocity_score = self._calculate_velocity_score(
                velocity_metrics, chat_config
            )
            event_score = event_impact * chat_config.event_impact_weight

            # Combine scores
            final_score = (
                sentiment_score
                * (
                    1
                    - chat_config.velocity_spike_weight
                    - chat_config.event_impact_weight
                )
                + velocity_score * chat_config.velocity_spike_weight
                + event_score
            )

            # Apply highlight confidence as multiplier
            final_score *= highlight_confidence

            # Check thresholds
            if (
                final_score < chat_config.min_score
                or highlight_confidence < chat_config.min_confidence
            ):
                return []

            # Create detection result
            result = DetectionResult(
                segment_id=segment.segment_id,
                modality=self.modality,
                score=final_score,
                confidence=highlight_confidence,
                features={
                    # Sentiment features
                    "sentiment_score": sentiment_score,
                    "avg_sentiment": window_sentiment.avg_sentiment,
                    "sentiment_intensity": window_sentiment.intensity,
                    "sentiment_momentum": window_sentiment.momentum,
                    # Velocity features
                    "velocity_score": velocity_score,
                    "messages_per_second": velocity_metrics.messages_per_second,
                    "velocity_spike_detected": float(velocity_metrics.spike_detected),
                    "spike_intensity": velocity_metrics.spike_intensity,
                    "acceleration": velocity_metrics.acceleration,
                    # Event features
                    "event_impact": event_impact,
                    # Engagement features
                    "message_count": window_sentiment.message_count,
                    "unique_users": window_sentiment.unique_users,
                    "emote_density": window_sentiment.emote_density,
                    "keyword_density": window_sentiment.keyword_density,
                    "spam_ratio": window_sentiment.spam_ratio,
                },
                metadata={
                    "algorithm": self.algorithm_name,
                    "version": self.algorithm_version,
                    "platform": platform,
                    "window_sentiment": {
                        "category": window_sentiment.dominant_category,
                        "intensity": window_sentiment.intensity,
                        "confidence": window_sentiment.confidence,
                        "momentum": window_sentiment.momentum,
                    },
                    "velocity_metrics": {
                        "spike_detected": velocity_metrics.spike_detected,
                        "spike_intensity": velocity_metrics.spike_intensity,
                        "acceleration": velocity_metrics.acceleration,
                        "jerk": velocity_metrics.jerk,
                    },
                    "event_impact": event_impact,
                    "highlight_confidence": highlight_confidence,
                },
            )

            self.logger.debug(
                f"Enhanced chat analysis complete for segment {segment.segment_id}: "
                f"score={final_score:.3f}, confidence={highlight_confidence:.3f}, "
                f"category={window_sentiment.dominant_category}"
            )

            return [result]

        except Exception as e:
            self.logger.error(
                f"Error in enhanced chat detection for segment {segment.segment_id}: {e}"
            )
            return []

    def _calculate_sentiment_score(
        self, window: WindowSentiment, config: EnhancedChatDetectionConfig
    ) -> float:
        """Calculate score based on sentiment analysis."""
        # Get category weight
        category_weight = config.sentiment_category_weights.get(
            window.dominant_category, 0.5
        )

        # Combine with intensity and confidence
        sentiment_score = (
            category_weight * 0.5 + window.intensity * 0.3 + window.confidence * 0.2
        )

        # Apply momentum boost
        if window.momentum > 0:
            sentiment_score *= 1 + window.momentum * 0.2

        return min(1.0, sentiment_score)

    def _calculate_velocity_score(
        self, velocity: VelocityMetrics, config: EnhancedChatDetectionConfig
    ) -> float:
        """Calculate score based on velocity metrics."""
        if (
            velocity.spike_detected
            and velocity.spike_intensity >= config.min_velocity_spike_intensity
        ):
            # High score for significant spikes
            base_score = 0.7 + (0.3 * velocity.spike_intensity)
        else:
            # Lower score based on raw velocity
            base_score = min(velocity.messages_per_second / 20, 0.5)

        # Boost for positive acceleration
        if velocity.acceleration > 0:
            acceleration_boost = min(velocity.acceleration / 10, 0.2)
            base_score += acceleration_boost

        return min(1.0, base_score)

    async def process_event_stream(
        self, events: List[ChatEvent], window_size: float = 10.0
    ) -> List[DetectionResult]:
        """
        Process a stream of chat events for highlight detection.

        Args:
            events: List of chat events
            window_size: Window size for analysis in seconds

        Returns:
            List of detection results
        """
        if not events:
            return []

        # Process all events
        for event in events:
            await self.sentiment_analyzer.process_event(event)

        # Group events into time windows
        results = []
        current_window_start = events[0].timestamp
        current_window_events = []

        for event in events:
            if (event.timestamp - current_window_start).total_seconds() > window_size:
                # Process current window
                if current_window_events:
                    segment = ContentSegment(
                        start_time=current_window_start.timestamp(),
                        end_time=current_window_events[-1].timestamp.timestamp(),
                        data=[],  # Events don't have message data
                        metadata={
                            "events": current_window_events,
                            "platform": "event_stream",
                        },
                    )

                    window_results = await self._detect_features(
                        segment, self.chat_config
                    )
                    results.extend(window_results)

                # Start new window
                current_window_start = event.timestamp
                current_window_events = [event]
            else:
                current_window_events.append(event)

        # Process final window
        if current_window_events:
            segment = ContentSegment(
                start_time=current_window_start.timestamp(),
                end_time=current_window_events[-1].timestamp.timestamp(),
                data=[],
                metadata={"events": current_window_events, "platform": "event_stream"},
            )

            window_results = await self._detect_features(segment, self.chat_config)
            results.extend(window_results)

        return results

    def get_sentiment_metrics(self) -> Dict[str, Any]:
        """Get current sentiment analyzer metrics."""
        return self.sentiment_analyzer.get_metrics_summary()

    def reset_baselines(self):
        """Reset baseline metrics for the sentiment analyzer."""
        self.sentiment_analyzer.temporal_analyzer = TemporalAnalyzer()
        self.sentiment_analyzer.message_history.clear()
        self.sentiment_analyzer.window_history.clear()
        self.logger.info("Reset sentiment analyzer baselines")
