"""
Chat-based highlight detection for the TL;DR Highlight API.

This module implements sophisticated chat analysis algorithms for identifying
exciting moments through message frequency spikes, sentiment analysis, and
community engagement patterns.
"""

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pydantic import Field

from .base_detector import (
    BaseDetector,
    ContentSegment,
    DetectionConfig,
    DetectionResult,
    ModalityType,
)
from ...utils.ml_utils import text_feature_extractor
from ...utils.scoring_utils import (
    normalize_score,
    calculate_confidence,
)

logger = logging.getLogger(__name__)


class ChatDetectionConfig(DetectionConfig):
    """
    Configuration for chat-based highlight detection.

    Extends base configuration with chat-specific parameters
    for message frequency analysis, sentiment detection, and
    community engagement metrics.
    """

    # Message frequency parameters
    frequency_spike_threshold: float = Field(
        default=2.0,
        ge=1.0,
        description="Multiplier for frequency spike detection (e.g., 2x normal rate)",
    )
    frequency_weight: float = Field(
        default=0.3, ge=0.0, description="Weight for frequency features in scoring"
    )
    frequency_window_size: float = Field(
        default=10.0, gt=0.0, description="Window size for frequency analysis (seconds)"
    )

    # Sentiment analysis parameters
    sentiment_analysis_enabled: bool = Field(
        default=True, description="Enable sentiment-based excitement detection"
    )
    sentiment_weight: float = Field(
        default=0.25, ge=0.0, description="Weight for sentiment features in scoring"
    )
    positive_sentiment_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for positive sentiment detection",
    )

    # Engagement analysis parameters
    engagement_analysis_enabled: bool = Field(
        default=True, description="Enable engagement pattern analysis"
    )
    engagement_weight: float = Field(
        default=0.25, ge=0.0, description="Weight for engagement features in scoring"
    )
    unique_user_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Threshold for unique user engagement"
    )

    # Keyword and emoji analysis parameters
    keyword_emoji_weight: float = Field(
        default=0.2, ge=0.0, description="Weight for keyword/emoji features in scoring"
    )
    custom_excitement_keywords: List[str] = Field(
        default_factory=list,
        description="Custom excitement keywords for this content type",
    )
    emoji_boost_factor: float = Field(
        default=1.5, ge=1.0, description="Multiplier for emoji excitement scores"
    )

    # Message filtering parameters
    min_message_length: int = Field(
        default=3, ge=1, description="Minimum message length to consider"
    )
    exclude_bot_messages: bool = Field(
        default=True, description="Exclude bot messages from analysis"
    )
    spam_filter_enabled: bool = Field(
        default=True, description="Enable spam message filtering"
    )

    # Community dynamics parameters
    community_size_factor: float = Field(
        default=0.1,
        ge=0.0,
        description="Factor for community size in excitement calculation",
    )
    user_diversity_weight: float = Field(
        default=0.15,
        ge=0.0,
        description="Weight for user diversity in engagement scoring",
    )


@dataclass
class ChatMessage:
    """
    Represents a single chat message for analysis.

    Contains message content, metadata, and analysis results.
    """

    timestamp: float
    user_id: str
    username: str
    message: str
    platform: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Analysis results (computed lazily)
    _sentiment_score: Optional[float] = None
    _excitement_keywords: Optional[List[str]] = None
    _emoji_count: Optional[int] = None
    _spam_score: Optional[float] = None

    @property
    def message_length(self) -> int:
        """Get message length in characters."""
        return len(self.message)

    @property
    def word_count(self) -> int:
        """Get message word count."""
        return len(self.message.split())

    @property
    def is_bot(self) -> bool:
        """Check if message is from a bot."""
        bot_indicators = ["bot", "nightbot", "streamlabs", "moobot", "streamelements"]
        username_lower = self.username.lower()
        return any(indicator in username_lower for indicator in bot_indicators)

    def get_sentiment_score(self) -> float:
        """
        Get sentiment score for the message.

        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if self._sentiment_score is not None:
            return self._sentiment_score

        # Simplified sentiment analysis
        positive_words = {
            "good",
            "great",
            "awesome",
            "amazing",
            "love",
            "like",
            "best",
            "perfect",
            "excellent",
            "fantastic",
            "wonderful",
            "brilliant",
            "outstanding",
            "epic",
            "incredible",
            "beautiful",
            "nice",
            "cool",
            "sweet",
            "sick",
            "fire",
            "lit",
        }

        negative_words = {
            "bad",
            "terrible",
            "awful",
            "hate",
            "dislike",
            "worst",
            "horrible",
            "disgusting",
            "annoying",
            "boring",
            "stupid",
            "dumb",
            "trash",
            "garbage",
        }

        words = set(self.message.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))

        # Calculate sentiment score
        if positive_count + negative_count == 0:
            sentiment = 0.0
        else:
            sentiment = (positive_count - negative_count) / (
                positive_count + negative_count
            )

        self._sentiment_score = sentiment
        return sentiment

    def get_excitement_keywords(self) -> List[str]:
        """
        Get excitement keywords found in the message.

        Returns:
            List of excitement keywords found
        """
        if self._excitement_keywords is not None:
            return self._excitement_keywords

        excitement_keywords = {
            # Gaming excitement
            "wow",
            "omg",
            "holy",
            "insane",
            "crazy",
            "nuts",
            "epic",
            "sick",
            "clutch",
            "poggers",
            "pog",
            "hype",
            "lets go",
            "let's go",
            "no way",
            "unreal",
            # Positive reactions
            "yes",
            "yeah",
            "yay",
            "nice",
            "good",
            "great",
            "awesome",
            "amazing",
            "perfect",
            "brilliant",
            "fantastic",
            "incredible",
            "outstanding",
            # Surprise/shock
            "what",
            "how",
            "wtf",
            "lol",
            "lmao",
            "rofl",
            "kekw",
            "pepehands",
            # Sports/competition
            "goal",
            "win",
            "victory",
            "champion",
            "beast",
            "godlike",
            "legendary",
        }

        found_keywords = []
        message_lower = self.message.lower()

        for keyword in excitement_keywords:
            if keyword in message_lower:
                found_keywords.append(keyword)

        self._excitement_keywords = found_keywords
        return found_keywords

    def get_emoji_count(self) -> int:
        """
        Get count of emojis in the message.

        Returns:
            Number of emoji characters
        """
        if self._emoji_count is not None:
            return self._emoji_count

        # Count Unicode emoji characters (simplified)
        emoji_count = 0
        for char in self.message:
            if ord(char) > 127:  # Non-ASCII characters (includes emojis)
                emoji_count += 1

        # Count common text emoticons
        emoticons = [":)", ":(", ":D", ":P", ":o", ":|", "xD", "^^", ":-)", ":-("]
        for emoticon in emoticons:
            emoji_count += self.message.count(emoticon)

        self._emoji_count = emoji_count
        return emoji_count

    def get_spam_score(self) -> float:
        """
        Get spam likelihood score for the message.

        Returns:
            Spam score between 0 (not spam) and 1 (likely spam)
        """
        if self._spam_score is not None:
            return self._spam_score

        spam_indicators = 0
        total_indicators = 7

        # All caps
        if self.message.isupper() and len(self.message) > 5:
            spam_indicators += 1

        # Excessive punctuation
        punct_count = sum(1 for c in self.message if c in "!?.,;:")
        if punct_count > len(self.message) * 0.3:
            spam_indicators += 1

        # Repetitive characters
        if re.search(r"(.)\1{4,}", self.message):
            spam_indicators += 1

        # Very short messages
        if len(self.message.strip()) < 3:
            spam_indicators += 1

        # Common spam patterns
        spam_patterns = [
            r"\b(follow|sub|subscribe)\b",
            r"\b(giveaway|free)\b",
            r"http[s]?://",
        ]
        for pattern in spam_patterns:
            if re.search(pattern, self.message.lower()):
                spam_indicators += 1
                break

        # Excessive emojis
        if self.get_emoji_count() > len(self.message) * 0.5:
            spam_indicators += 1

        # Single repeated word
        words = self.message.split()
        if len(words) > 3 and len(set(words)) == 1:
            spam_indicators += 1

        self._spam_score = spam_indicators / total_indicators
        return self._spam_score


@dataclass
class ChatWindow:
    """
    Represents a time window of chat messages for analysis.

    Contains messages, metadata, and aggregated statistics.
    """

    start_time: float
    end_time: float
    messages: List[ChatMessage] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get window duration in seconds."""
        return self.end_time - self.start_time

    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)

    @property
    def unique_users(self) -> Set[str]:
        """Get set of unique user IDs."""
        return {msg.user_id for msg in self.messages}

    @property
    def unique_user_count(self) -> int:
        """Get count of unique users."""
        return len(self.unique_users)

    @property
    def message_rate(self) -> float:
        """Get messages per second."""
        return self.message_count / max(0.1, self.duration)

    def get_filtered_messages(self, config: ChatDetectionConfig) -> List[ChatMessage]:
        """
        Get filtered messages based on configuration.

        Args:
            config: Chat detection configuration

        Returns:
            Filtered list of messages
        """
        filtered = []

        for msg in self.messages:
            # Filter by length
            if len(msg.message) < config.min_message_length:
                continue

            # Filter bots
            if config.exclude_bot_messages and msg.is_bot:
                continue

            # Filter spam
            if config.spam_filter_enabled and msg.get_spam_score() > 0.5:
                continue

            filtered.append(msg)

        return filtered


class ChatExcitementAnalyzer:
    """
    Analyzes chat content for excitement and engagement indicators.

    Implements various algorithms for detecting chat excitement,
    including frequency spikes, sentiment analysis, and engagement patterns.
    """

    def __init__(self, config: ChatDetectionConfig):
        """
        Initialize the chat excitement analyzer.

        Args:
            config: Chat detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ChatExcitementAnalyzer")

        # Track baseline metrics for frequency spike detection
        self.baseline_message_rate = 1.0  # Messages per second
        self.baseline_user_count = 10  # Active users

    async def analyze_frequency_spikes(
        self, chat_window: ChatWindow
    ) -> Dict[str, float]:
        """
        Analyze message frequency spikes in chat window.

        Args:
            chat_window: Chat window to analyze

        Returns:
            Dictionary with frequency spike analysis results
        """
        try:
            filtered_messages = chat_window.get_filtered_messages(self.config)

            if not filtered_messages:
                return {"frequency_score": 0.0, "message_rate": 0.0}

            # Calculate current message rate
            current_rate = len(filtered_messages) / max(0.1, chat_window.duration)

            # Calculate frequency spike score
            rate_ratio = current_rate / max(0.1, self.baseline_message_rate)

            # Check if this is a significant spike
            if rate_ratio >= self.config.frequency_spike_threshold:
                frequency_score = normalize_score(
                    (rate_ratio - 1.0) / self.config.frequency_spike_threshold,
                    method="sigmoid",
                )
            else:
                frequency_score = 0.0

            # Update baseline (exponential moving average)
            alpha = 0.1
            self.baseline_message_rate = (
                alpha * current_rate + (1 - alpha) * self.baseline_message_rate
            )

            return {
                "frequency_score": frequency_score,
                "message_rate": current_rate,
                "rate_ratio": rate_ratio,
                "baseline_rate": self.baseline_message_rate,
                "filtered_message_count": len(filtered_messages),
            }

        except Exception as e:
            self.logger.error(f"Error in frequency spike analysis: {e}")
            return {"frequency_score": 0.0, "message_rate": 0.0}

    async def analyze_sentiment(self, chat_window: ChatWindow) -> Dict[str, float]:
        """
        Analyze sentiment patterns in chat window.

        Args:
            chat_window: Chat window to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.config.sentiment_analysis_enabled:
            return {"sentiment_score": 0.0, "avg_sentiment": 0.0}

        try:
            filtered_messages = chat_window.get_filtered_messages(self.config)

            if not filtered_messages:
                return {"sentiment_score": 0.0, "avg_sentiment": 0.0}

            # Calculate sentiment scores
            sentiment_scores = [msg.get_sentiment_score() for msg in filtered_messages]
            avg_sentiment = np.mean(sentiment_scores)

            # Calculate positive sentiment ratio
            positive_count = sum(
                1
                for score in sentiment_scores
                if score > self.config.positive_sentiment_threshold
            )
            positive_ratio = positive_count / len(sentiment_scores)

            # Sentiment excitement score
            sentiment_score = normalize_score(
                avg_sentiment * positive_ratio * 2,  # Scale for normalization
                method="sigmoid",
            )

            return {
                "sentiment_score": sentiment_score,
                "avg_sentiment": avg_sentiment,
                "positive_ratio": positive_ratio,
                "positive_count": positive_count,
                "sentiment_variance": np.var(sentiment_scores),
            }

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {"sentiment_score": 0.0, "avg_sentiment": 0.0}

    async def analyze_engagement(self, chat_window: ChatWindow) -> Dict[str, float]:
        """
        Analyze engagement patterns in chat window.

        Args:
            chat_window: Chat window to analyze

        Returns:
            Dictionary with engagement analysis results
        """
        if not self.config.engagement_analysis_enabled:
            return {"engagement_score": 0.0, "user_diversity": 0.0}

        try:
            filtered_messages = chat_window.get_filtered_messages(self.config)

            if not filtered_messages:
                return {"engagement_score": 0.0, "user_diversity": 0.0}

            # Calculate engagement metrics
            unique_users = {msg.user_id for msg in filtered_messages}
            unique_user_count = len(unique_users)

            # User diversity (inverse of concentration)
            user_message_counts = Counter(msg.user_id for msg in filtered_messages)
            total_messages = len(filtered_messages)

            # Calculate Gini coefficient for user distribution
            if len(user_message_counts) > 1:
                counts = sorted(user_message_counts.values())
                n = len(counts)
                cumsum = np.cumsum(counts)
                gini = (2 * np.sum((np.arange(1, n + 1)) * counts)) / (
                    n * cumsum[-1]
                ) - (n + 1) / n
                user_diversity = 1 - gini  # Higher diversity = lower Gini
            else:
                user_diversity = 0.0

            # Engagement rate compared to baseline
            current_user_rate = unique_user_count / max(0.1, chat_window.duration)
            baseline_rate = max(
                1.0, self.baseline_user_count / 60.0
            )  # Users per second
            engagement_ratio = current_user_rate / baseline_rate

            # Combined engagement score
            engagement_components = [
                normalize_score(engagement_ratio - 1.0, method="sigmoid"),
                user_diversity,
                normalize_score(
                    unique_user_count / max(1, total_messages), method="linear"
                ),
            ]

            engagement_score = np.mean(engagement_components)

            # Update baseline
            alpha = 0.1
            self.baseline_user_count = (
                alpha * unique_user_count + (1 - alpha) * self.baseline_user_count
            )

            return {
                "engagement_score": engagement_score,
                "user_diversity": user_diversity,
                "unique_user_count": unique_user_count,
                "engagement_ratio": engagement_ratio,
                "user_concentration": gini if len(user_message_counts) > 1 else 1.0,
            }

        except Exception as e:
            self.logger.error(f"Error in engagement analysis: {e}")
            return {"engagement_score": 0.0, "user_diversity": 0.0}

    async def analyze_keywords_emojis(
        self, chat_window: ChatWindow
    ) -> Dict[str, float]:
        """
        Analyze keyword and emoji excitement in chat window.

        Args:
            chat_window: Chat window to analyze

        Returns:
            Dictionary with keyword/emoji analysis results
        """
        try:
            filtered_messages = chat_window.get_filtered_messages(self.config)

            if not filtered_messages:
                return {"keyword_emoji_score": 0.0, "excitement_keyword_count": 0}

            # Count excitement indicators
            total_excitement_keywords = 0
            total_emojis = 0

            for msg in filtered_messages:
                keywords = msg.get_excitement_keywords()
                total_excitement_keywords += len(keywords)
                total_emojis += msg.get_emoji_count()

            # Calculate densities
            total_messages = len(filtered_messages)
            keyword_density = total_excitement_keywords / total_messages
            emoji_density = total_emojis / total_messages

            # Combine into excitement score
            keyword_score = normalize_score(
                keyword_density * 5, method="sigmoid"
            )  # Scale for normalization
            emoji_score = normalize_score(emoji_density * 2, method="sigmoid")

            # Apply emoji boost factor
            emoji_score *= self.config.emoji_boost_factor

            # Combined score
            keyword_emoji_score = (keyword_score + emoji_score) / 2

            return {
                "keyword_emoji_score": min(1.0, keyword_emoji_score),
                "excitement_keyword_count": total_excitement_keywords,
                "emoji_count": total_emojis,
                "keyword_density": keyword_density,
                "emoji_density": emoji_density,
            }

        except Exception as e:
            self.logger.error(f"Error in keyword/emoji analysis: {e}")
            return {"keyword_emoji_score": 0.0, "excitement_keyword_count": 0}


class ChatDetector(BaseDetector):
    """
    Chat-based highlight detector using message frequency and sentiment analysis.

    Implements sophisticated algorithms for identifying exciting moments
    in chat content through frequency spikes, sentiment analysis,
    engagement patterns, and keyword detection.
    """

    def __init__(self, config: Optional[ChatDetectionConfig] = None):
        """
        Initialize the chat detector.

        Args:
            config: Chat detection configuration
        """
        self.chat_config = config or ChatDetectionConfig()
        super().__init__(self.chat_config)

        self.excitement_analyzer = ChatExcitementAnalyzer(self.chat_config)
        self.logger = logging.getLogger(f"{__name__}.ChatDetector")

    @property
    def modality(self) -> ModalityType:
        """Get the modality this detector handles."""
        return ModalityType.CHAT

    @property
    def algorithm_name(self) -> str:
        """Get the name of the detection algorithm."""
        return "ChatExcitementDetector"

    @property
    def algorithm_version(self) -> str:
        """Get the version of the detection algorithm."""
        return "1.0.0"

    def _validate_segment(self, segment: ContentSegment) -> bool:
        """
        Validate that a segment contains valid chat data.

        Args:
            segment: Content segment to validate

        Returns:
            True if segment contains valid chat data
        """
        if not super()._validate_segment(segment):
            return False

        # Check if data is chat messages
        if not isinstance(segment.data, (list, ChatWindow)):
            return False

        # Additional chat-specific validation
        if isinstance(segment.data, list):
            if not segment.data:
                return False

            # Check first message
            first_msg = segment.data[0]
            if not isinstance(first_msg, (ChatMessage, dict)):
                return False

        return True

    def _prepare_chat_window(self, segment: ContentSegment) -> Optional[ChatWindow]:
        """
        Prepare chat window from segment data.

        Args:
            segment: Content segment with chat data

        Returns:
            ChatWindow object or None if invalid
        """
        try:
            if isinstance(segment.data, ChatWindow):
                return segment.data

            elif isinstance(segment.data, list):
                messages = []

                for msg_data in segment.data:
                    if isinstance(msg_data, ChatMessage):
                        messages.append(msg_data)
                    elif isinstance(msg_data, dict):
                        # Create ChatMessage from dictionary
                        message = ChatMessage(
                            timestamp=msg_data.get("timestamp", segment.start_time),
                            user_id=msg_data.get("user_id", "unknown"),
                            username=msg_data.get("username", "unknown"),
                            message=msg_data.get("message", ""),
                            platform=msg_data.get("platform", "unknown"),
                            metadata=msg_data.get("metadata", {}),
                        )
                        messages.append(message)

                return ChatWindow(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    messages=messages,
                )

            return None

        except Exception as e:
            self.logger.error(f"Error preparing chat window: {e}")
            return None

    async def _detect_features(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Detect chat-based highlight features in a segment.

        Args:
            segment: Chat content segment to analyze
            config: Detection configuration

        Returns:
            List of detection results for chat analysis
        """
        chat_config = (
            config if isinstance(config, ChatDetectionConfig) else self.chat_config
        )

        try:
            # Prepare chat window
            chat_window = self._prepare_chat_window(segment)
            if chat_window is None:
                self.logger.warning(f"Invalid chat segment {segment.segment_id}")
                return []

            self.logger.debug(
                f"Analyzing chat segment {segment.segment_id} "
                f"({chat_window.message_count} messages, {chat_window.unique_user_count} users)"
            )

            # Perform various analyses concurrently
            frequency_task = self.excitement_analyzer.analyze_frequency_spikes(
                chat_window
            )
            sentiment_task = self.excitement_analyzer.analyze_sentiment(chat_window)
            engagement_task = self.excitement_analyzer.analyze_engagement(chat_window)
            keyword_task = self.excitement_analyzer.analyze_keywords_emojis(chat_window)

            # Wait for all analyses to complete
            (
                frequency_results,
                sentiment_results,
                engagement_results,
                keyword_results,
            ) = await asyncio.gather(
                frequency_task, sentiment_task, engagement_task, keyword_task
            )

            # Extract ML features
            try:
                # Prepare text data for ML feature extractor
                messages_text = [
                    msg.message
                    for msg in chat_window.get_filtered_messages(chat_config)
                ]
                ml_features = await text_feature_extractor.extract_features(
                    messages_text
                )
            except Exception as e:
                self.logger.warning(f"ML feature extraction failed: {e}")
                ml_features = np.array([])

            # Combine analysis results
            all_scores = {
                "frequency": frequency_results["frequency_score"],
                "sentiment": sentiment_results["sentiment_score"],
                "engagement": engagement_results["engagement_score"],
                "keyword_emoji": keyword_results["keyword_emoji_score"],
            }

            # Calculate weighted score
            weights = {
                "frequency": chat_config.frequency_weight,
                "sentiment": chat_config.sentiment_weight,
                "engagement": chat_config.engagement_weight,
                "keyword_emoji": chat_config.keyword_emoji_weight,
            }

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            # Calculate final score
            final_score = sum(all_scores[k] * weights[k] for k in all_scores.keys())

            # Calculate confidence based on score consistency
            score_values = list(all_scores.values())
            confidence = calculate_confidence(score_values, method="consistency")

            # Community size factor
            community_factor = 1.0 + (
                chat_config.community_size_factor
                * min(1.0, chat_window.unique_user_count / 100.0)
            )
            final_score *= community_factor
            final_score = min(1.0, final_score)

            # Check significance thresholds
            if (
                final_score < chat_config.min_score
                or confidence < chat_config.min_confidence
            ):
                return []

            # Create detection result
            result = DetectionResult(
                segment_id=segment.segment_id,
                modality=self.modality,
                score=final_score,
                confidence=confidence,
                features={
                    **all_scores,
                    "message_count": chat_window.message_count,
                    "unique_users": chat_window.unique_user_count,
                    "message_rate": frequency_results.get("message_rate", 0.0),
                    "avg_sentiment": sentiment_results.get("avg_sentiment", 0.0),
                    "user_diversity": engagement_results.get("user_diversity", 0.0),
                    "excitement_keywords": keyword_results.get(
                        "excitement_keyword_count", 0
                    ),
                    "community_factor": community_factor,
                },
                metadata={
                    "algorithm": self.algorithm_name,
                    "version": self.algorithm_version,
                    "config": chat_config.dict(),
                    "frequency_analysis": frequency_results,
                    "sentiment_analysis": sentiment_results,
                    "engagement_analysis": engagement_results,
                    "keyword_analysis": keyword_results,
                    "window_duration": chat_window.duration,
                },
                algorithm_version=self.algorithm_version,
            )

            # Add ML features if available
            if ml_features.size > 0:
                result.metadata["ml_features"] = ml_features.tolist()

            self.logger.debug(
                f"Chat analysis complete for segment {segment.segment_id}: "
                f"score={final_score:.3f}, confidence={confidence:.3f}"
            )

            return [result]

        except Exception as e:
            self.logger.error(
                f"Error in chat detection for segment {segment.segment_id}: {e}"
            )
            return []

    async def detect_highlights_from_messages(
        self, messages: List[ChatMessage], window_size: float = 30.0
    ) -> List[DetectionResult]:
        """
        Detect highlights from a stream of chat messages.

        Args:
            messages: List of chat messages
            window_size: Analysis window size in seconds

        Returns:
            List of detection results
        """
        if not messages:
            return []

        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)

        # Group messages into windows
        segments = []
        current_window_messages = []
        window_start_time = sorted_messages[0].timestamp

        for message in sorted_messages:
            if message.timestamp - window_start_time >= window_size:
                # Create segment from accumulated messages
                if current_window_messages:
                    window_end_time = current_window_messages[-1].timestamp
                    chat_window = ChatWindow(
                        start_time=window_start_time,
                        end_time=window_end_time,
                        messages=current_window_messages,
                    )

                    segment = ContentSegment(
                        start_time=window_start_time,
                        end_time=window_end_time,
                        data=chat_window,
                        metadata={"message_count": len(current_window_messages)},
                    )
                    segments.append(segment)

                # Start new window
                current_window_messages = [message]
                window_start_time = message.timestamp
            else:
                current_window_messages.append(message)

        # Add final window
        if current_window_messages:
            window_end_time = current_window_messages[-1].timestamp
            chat_window = ChatWindow(
                start_time=window_start_time,
                end_time=window_end_time,
                messages=current_window_messages,
            )

            segment = ContentSegment(
                start_time=window_start_time,
                end_time=window_end_time,
                data=chat_window,
                metadata={"message_count": len(current_window_messages)},
            )
            segments.append(segment)

        # Detect highlights in segments
        return await self.detect_highlights(segments)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for chat detection."""
        base_metrics = self.get_metrics()

        # Add chat-specific metrics
        chat_metrics = {
            **base_metrics,
            "algorithm": self.algorithm_name,
            "version": self.algorithm_version,
            "config": self.chat_config.dict(),
            "baseline_message_rate": self.excitement_analyzer.baseline_message_rate,
            "baseline_user_count": self.excitement_analyzer.baseline_user_count,
        }

        return chat_metrics
