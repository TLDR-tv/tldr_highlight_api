"""
Chat and comment processing service for sentiment analysis.

This module provides comprehensive chat message processing with sentiment analysis,
engagement metrics, spam detection, and real-time chat stream processing.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Set, Union
import statistics

from pydantic import BaseModel, Field

from src.utils.nlp_utils import (
    ChatMessage,
    TextAnalysis,
    SentimentScore,
    chat_processor as nlp_chat_processor,
    topic_analyzer,
)

logger = logging.getLogger(__name__)


class ChatProcessorConfig(BaseModel):
    """Configuration for chat processor."""

    # Processing settings
    batch_size: int = Field(
        default=50, description="Number of messages to process in a batch"
    )
    max_concurrent_analysis: int = Field(
        default=10, description="Maximum concurrent message analysis tasks"
    )

    # Buffer settings
    buffer_size: int = Field(
        default=1000, description="Maximum number of messages to keep in buffer"
    )
    analysis_buffer_size: int = Field(
        default=500, description="Maximum number of analysis results to keep"
    )

    # Time windows
    engagement_window_seconds: float = Field(
        default=60.0, description="Time window for engagement analysis"
    )
    trend_window_seconds: float = Field(
        default=300.0, description="Time window for trend analysis"
    )

    # Quality filters
    min_message_length: int = Field(
        default=3, description="Minimum message length for processing"
    )
    max_message_length: int = Field(
        default=500, description="Maximum message length for processing"
    )
    toxicity_threshold: float = Field(
        default=0.7, description="Threshold for flagging toxic messages"
    )

    # Engagement metrics
    enable_engagement_tracking: bool = Field(
        default=True, description="Enable engagement metrics tracking"
    )
    enable_topic_analysis: bool = Field(
        default=True, description="Enable topic analysis"
    )
    enable_spam_detection: bool = Field(
        default=True, description="Enable spam detection"
    )

    # Rate limiting
    messages_per_second_limit: float = Field(
        default=50.0, description="Maximum messages per second to process"
    )

    # Platform-specific settings
    platform_configs: Dict[str, Dict[str, Union[str, int, float]]] = Field(
        default_factory=lambda: {
            "twitch": {
                "emote_boost": 1.2,
                "subscriber_boost": 1.5,
                "moderator_boost": 2.0,
            },
            "youtube": {"superchat_boost": 3.0, "member_boost": 1.8},
            "discord": {"reaction_boost": 1.3, "role_boost": 1.4},
        }
    )


@dataclass
class MessageBatch:
    """Batch of chat messages for processing."""

    messages: List[ChatMessage]
    timestamp: float
    platform: str
    batch_id: str


@dataclass
class EngagementMetrics:
    """Chat engagement metrics for a time window."""

    timestamp: float
    window_duration: float

    # Volume metrics
    total_messages: int
    unique_users: int
    messages_per_minute: float

    # Content metrics
    average_message_length: float
    total_characters: int

    # Sentiment metrics
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    average_sentiment: float

    # Engagement score
    engagement_score: float
    activity_level: str  # 'low', 'medium', 'high', 'very_high'

    # Quality metrics
    spam_ratio: float
    toxicity_ratio: float
    quality_score: float


@dataclass
class TrendAnalysis:
    """Trend analysis for chat content."""

    timestamp: float
    window_duration: float

    # Topic trends
    trending_topics: List[Dict[str, Union[str, float]]]
    topic_changes: List[str]

    # Sentiment trends
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    sentiment_velocity: float

    # Engagement trends
    engagement_trend: str  # 'increasing', 'decreasing', 'stable'
    engagement_velocity: float

    # User activity trends
    new_users_ratio: float
    active_users_trend: str


@dataclass
class ProcessedChatData:
    """Processed chat data with all analysis results."""

    messages: List[TextAnalysis]
    engagement_metrics: EngagementMetrics
    trend_analysis: Optional[TrendAnalysis] = None
    processing_time: float = 0.0
    batch_id: Optional[str] = None


class ChatProcessor:
    """
    Advanced chat processor for real-time sentiment and engagement analysis.

    Features:
    - Real-time message processing with sentiment analysis
    - Engagement metrics calculation
    - Trend analysis with topic detection
    - Spam and toxicity detection
    - Platform-specific optimizations
    - Configurable processing windows
    """

    def __init__(self, config: Optional[ChatProcessorConfig] = None):
        self.config = config or ChatProcessorConfig()

        # Processing buffers
        self.message_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.analysis_buffer: deque = deque(maxlen=self.config.analysis_buffer_size)
        self.engagement_history: deque = deque(
            maxlen=100
        )  # Keep last 100 engagement windows

        # User tracking
        self.user_activity: Dict[str, Dict[str, Union[float, int]]] = defaultdict(
            lambda: {"last_message": 0.0, "message_count": 0, "first_seen": time.time()}
        )
        self.active_users: Set[str] = set()

        # Platform-specific processors
        self.platform_processors = {}

        # Processing state
        self.processing_stats = {
            "total_messages_processed": 0,
            "total_processing_time": 0.0,
            "batches_processed": 0,
            "spam_messages_detected": 0,
            "toxic_messages_detected": 0,
        }

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_analysis)
        self._processing_lock = asyncio.Lock()

        # Rate limiting
        self.rate_limiter = deque(
            maxlen=int(self.config.messages_per_second_limit * 10)
        )

    async def process_message_batch(
        self, messages: List[ChatMessage]
    ) -> ProcessedChatData:
        """
        Process a batch of chat messages.

        Args:
            messages: List of chat messages to process

        Returns:
            ProcessedChatData with analysis results
        """
        start_time = time.time()
        batch_id = f"batch_{int(start_time)}_{len(messages)}"

        try:
            logger.info(
                f"Processing chat batch {batch_id} with {len(messages)} messages"
            )

            # Filter messages
            filtered_messages = self._filter_messages(messages)

            if not filtered_messages:
                logger.info("No messages to process after filtering")
                return ProcessedChatData(
                    messages=[],
                    engagement_metrics=self._create_empty_engagement_metrics(),
                    processing_time=time.time() - start_time,
                    batch_id=batch_id,
                )

            # Process messages concurrently
            analysis_tasks = []
            for message in filtered_messages:
                task = self._process_single_message(message)
                analysis_tasks.append(task)

            # Execute tasks in batches to avoid overwhelming the system
            analyses = []
            for i in range(0, len(analysis_tasks), self.config.batch_size):
                batch_tasks = analysis_tasks[i : i + self.config.batch_size]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                # Filter out exceptions
                valid_results = [
                    result
                    for result in batch_results
                    if not isinstance(result, Exception)
                ]
                analyses.extend(valid_results)

                # Yield control to event loop
                await asyncio.sleep(0)

            # Update message buffer
            async with self._processing_lock:
                self.message_buffer.extend(filtered_messages)
                self.analysis_buffer.extend(analyses)

            # Calculate engagement metrics
            engagement_metrics = await self._calculate_engagement_metrics(
                filtered_messages, analyses
            )

            # Calculate trend analysis if enabled
            trend_analysis = None
            if self.config.enable_topic_analysis and len(analyses) > 5:
                trend_analysis = await self._analyze_trends(analyses)

            # Update stats
            processing_time = time.time() - start_time
            self.processing_stats["total_messages_processed"] += len(filtered_messages)
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["batches_processed"] += 1

            # Count spam and toxic messages
            spam_count = sum(1 for analysis in analyses if self._is_spam(analysis))
            toxic_count = sum(
                1
                for analysis in analyses
                if analysis.toxicity_score > self.config.toxicity_threshold
            )

            self.processing_stats["spam_messages_detected"] += spam_count
            self.processing_stats["toxic_messages_detected"] += toxic_count

            result = ProcessedChatData(
                messages=analyses,
                engagement_metrics=engagement_metrics,
                trend_analysis=trend_analysis,
                processing_time=processing_time,
                batch_id=batch_id,
            )

            logger.info(
                f"Chat batch {batch_id} processing completed: "
                f"{len(analyses)} messages analyzed, {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing chat batch {batch_id}: {e}")
            raise

    async def process_message_stream(
        self, message_stream: AsyncGenerator[ChatMessage, None]
    ) -> AsyncGenerator[ProcessedChatData, None]:
        """
        Process messages from a real-time stream.

        Args:
            message_stream: Async generator of chat messages

        Yields:
            ProcessedChatData for each batch
        """
        message_batch = []
        last_batch_time = time.time()

        try:
            async for message in message_stream:
                # Rate limiting
                current_time = time.time()
                self.rate_limiter.append(current_time)

                # Check rate limit
                recent_messages = sum(
                    1 for t in self.rate_limiter if current_time - t <= 1.0
                )

                if recent_messages > self.config.messages_per_second_limit:
                    logger.warning("Rate limit exceeded, dropping message")
                    continue

                message_batch.append(message)

                # Process batch when it reaches size limit or time limit
                should_process = (
                    len(message_batch) >= self.config.batch_size
                    or current_time - last_batch_time >= 5.0  # Process every 5 seconds
                )

                if should_process and message_batch:
                    processed_data = await self.process_message_batch(message_batch)
                    yield processed_data

                    message_batch = []
                    last_batch_time = current_time

                # Yield control to event loop
                await asyncio.sleep(0)

            # Process remaining messages
            if message_batch:
                processed_data = await self.process_message_batch(message_batch)
                yield processed_data

        except Exception as e:
            logger.error(f"Error processing message stream: {e}")

    async def _process_single_message(self, message: ChatMessage) -> TextAnalysis:
        """
        Process a single chat message.

        Args:
            message: Chat message to process

        Returns:
            TextAnalysis result
        """
        async with self.semaphore:
            try:
                # Use the NLP chat processor
                analysis = await nlp_chat_processor.process_message(message)

                # Apply platform-specific boosts
                if message.platform in self.config.platform_configs:
                    analysis = self._apply_platform_boost(analysis, message)

                # Update user activity
                self._update_user_activity(message)

                return analysis

            except Exception as e:
                logger.error(f"Error processing message from {message.username}: {e}")
                # Return minimal analysis on error
                return TextAnalysis(
                    text=message.message,
                    timestamp=message.timestamp,
                    sentiment=SentimentScore(0.0, 0.0, 1.0, 0.0, "neutral", 0.0),
                    word_count=len(message.message.split()),
                    toxicity_score=0.0,
                )

    def _filter_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Filter messages based on quality criteria.

        Args:
            messages: Raw messages

        Returns:
            Filtered messages
        """
        filtered = []

        for message in messages:
            # Length filter
            if not (
                self.config.min_message_length
                <= len(message.message)
                <= self.config.max_message_length
            ):
                continue

            # Basic content filter (skip empty or whitespace-only)
            if not message.message.strip():
                continue

            filtered.append(message)

        return filtered

    def _apply_platform_boost(
        self, analysis: TextAnalysis, message: ChatMessage
    ) -> TextAnalysis:
        """
        Apply platform-specific engagement boosts.

        Args:
            analysis: Original analysis
            message: Chat message with platform info

        Returns:
            Modified analysis with platform boosts
        """
        platform_config = self.config.platform_configs.get(message.platform, {})

        if not platform_config:
            return analysis

        # Apply boosts based on user metadata
        boost_factor = 1.0

        if message.metadata:
            # Twitch-specific boosts
            if message.platform == "twitch":
                if message.metadata.get("is_subscriber"):
                    boost_factor *= platform_config.get("subscriber_boost", 1.0)
                if message.metadata.get("is_moderator"):
                    boost_factor *= platform_config.get("moderator_boost", 1.0)
                if message.metadata.get("has_emotes"):
                    boost_factor *= platform_config.get("emote_boost", 1.0)

            # YouTube-specific boosts
            elif message.platform == "youtube":
                if message.metadata.get("is_superchat"):
                    boost_factor *= platform_config.get("superchat_boost", 1.0)
                if message.metadata.get("is_member"):
                    boost_factor *= platform_config.get("member_boost", 1.0)

        # Apply boost to sentiment confidence (but not polarity)
        if boost_factor > 1.0:
            analysis.sentiment.confidence = min(
                analysis.sentiment.confidence * boost_factor, 1.0
            )

        return analysis

    def _update_user_activity(self, message: ChatMessage) -> None:
        """Update user activity tracking."""
        user_data = self.user_activity[message.user_id]
        user_data["last_message"] = message.timestamp
        user_data["message_count"] += 1

        # Add to active users if recently active (last 5 minutes)
        if time.time() - message.timestamp <= 300:
            self.active_users.add(message.user_id)

        # Clean up old active users
        current_time = time.time()
        inactive_users = {
            user_id
            for user_id in self.active_users
            if current_time - self.user_activity[user_id]["last_message"] > 300
        }
        self.active_users -= inactive_users

    async def _calculate_engagement_metrics(
        self, messages: List[ChatMessage], analyses: List[TextAnalysis]
    ) -> EngagementMetrics:
        """
        Calculate engagement metrics for messages.

        Args:
            messages: Original messages
            analyses: Analysis results

        Returns:
            EngagementMetrics
        """
        if not messages or not analyses:
            return self._create_empty_engagement_metrics()

        try:
            # Basic metrics
            total_messages = len(messages)
            unique_users = len(set(msg.user_id for msg in messages))

            # Time span calculation
            timestamps = [msg.timestamp for msg in messages]
            time_span = max(timestamps) - min(timestamps)
            messages_per_minute = total_messages / max(
                time_span / 60, 1 / 60
            )  # Avoid division by zero

            # Content metrics
            message_lengths = [len(msg.message) for msg in messages]
            average_message_length = statistics.mean(message_lengths)
            total_characters = sum(message_lengths)

            # Sentiment metrics
            sentiments = [analysis.sentiment for analysis in analyses]
            positive_count = sum(1 for s in sentiments if s.label == "positive")
            negative_count = sum(1 for s in sentiments if s.label == "negative")
            neutral_count = sum(1 for s in sentiments if s.label == "neutral")

            positive_ratio = positive_count / total_messages
            negative_ratio = negative_count / total_messages
            neutral_ratio = neutral_count / total_messages

            # Average sentiment (compound score)
            average_sentiment = statistics.mean([s.compound for s in sentiments])

            # Quality metrics
            spam_count = sum(1 for analysis in analyses if self._is_spam(analysis))
            toxic_count = sum(
                1
                for analysis in analyses
                if analysis.toxicity_score > self.config.toxicity_threshold
            )

            spam_ratio = spam_count / total_messages
            toxicity_ratio = toxic_count / total_messages
            quality_score = 1.0 - (spam_ratio * 0.5 + toxicity_ratio * 0.5)

            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(
                messages_per_minute,
                unique_users,
                average_message_length,
                positive_ratio,
                negative_ratio,
                quality_score,
            )

            # Determine activity level
            activity_level = self._determine_activity_level(
                engagement_score, messages_per_minute
            )

            metrics = EngagementMetrics(
                timestamp=time.time(),
                window_duration=self.config.engagement_window_seconds,
                total_messages=total_messages,
                unique_users=unique_users,
                messages_per_minute=messages_per_minute,
                average_message_length=average_message_length,
                total_characters=total_characters,
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                neutral_ratio=neutral_ratio,
                average_sentiment=average_sentiment,
                engagement_score=engagement_score,
                activity_level=activity_level,
                spam_ratio=spam_ratio,
                toxicity_ratio=toxicity_ratio,
                quality_score=quality_score,
            )

            # Add to history
            self.engagement_history.append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
            return self._create_empty_engagement_metrics()

    def _calculate_engagement_score(
        self,
        messages_per_minute: float,
        unique_users: int,
        average_message_length: float,
        positive_ratio: float,
        negative_ratio: float,
        quality_score: float,
    ) -> float:
        """Calculate overall engagement score."""
        try:
            # Normalize metrics
            message_rate_score = min(
                messages_per_minute / 20.0, 1.0
            )  # 20 msgs/min = 1.0
            user_diversity_score = min(unique_users / 20.0, 1.0)  # 20 users = 1.0
            message_quality_score = min(
                average_message_length / 30.0, 1.0
            )  # 30 chars = 1.0

            # Sentiment contribution
            sentiment_score = positive_ratio - (
                negative_ratio * 0.3
            )  # Negative hurts less

            # Weighted combination
            engagement_score = (
                message_rate_score * 0.3
                + user_diversity_score * 0.3
                + message_quality_score * 0.15
                + sentiment_score * 0.15
                + quality_score * 0.1
            )

            return max(0.0, min(engagement_score, 1.0))

        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0

    def _determine_activity_level(
        self, engagement_score: float, messages_per_minute: float
    ) -> str:
        """Determine activity level based on metrics."""
        if engagement_score >= 0.8 or messages_per_minute >= 30:
            return "very_high"
        elif engagement_score >= 0.6 or messages_per_minute >= 15:
            return "high"
        elif engagement_score >= 0.3 or messages_per_minute >= 5:
            return "medium"
        else:
            return "low"

    async def _analyze_trends(self, analyses: List[TextAnalysis]) -> TrendAnalysis:
        """
        Analyze trends in chat content.

        Args:
            analyses: List of text analyses

        Returns:
            TrendAnalysis result
        """
        try:
            # Extract text for topic analysis
            texts = [analysis.text for analysis in analyses if analysis.text]

            # Topic analysis
            trending_topics = []
            topic_changes = []

            if len(texts) >= 10:
                topics = await topic_analyzer.extract_topics(texts)
                trending_topics = topics[:5]  # Top 5 topics

                # Compare with previous topics (simplified)
                if len(self.engagement_history) > 0:
                    # This could be enhanced to track topic changes over time
                    topic_changes = ["new_topics_detected"]

            # Sentiment trend analysis
            recent_sentiments = [
                a.sentiment.compound for a in analyses[-10:]
            ]  # Last 10 messages
            older_sentiments = (
                [a.sentiment.compound for a in analyses[:-10]]
                if len(analyses) > 10
                else []
            )

            sentiment_trend = "stable"
            sentiment_velocity = 0.0

            if older_sentiments:
                recent_avg = statistics.mean(recent_sentiments)
                older_avg = statistics.mean(older_sentiments)
                sentiment_velocity = recent_avg - older_avg

                if sentiment_velocity > 0.1:
                    sentiment_trend = "improving"
                elif sentiment_velocity < -0.1:
                    sentiment_trend = "declining"

            # Engagement trend analysis
            engagement_trend = "stable"
            engagement_velocity = 0.0

            if len(self.engagement_history) >= 2:
                recent_engagement = self.engagement_history[-1].engagement_score
                previous_engagement = self.engagement_history[-2].engagement_score
                engagement_velocity = recent_engagement - previous_engagement

                if engagement_velocity > 0.1:
                    engagement_trend = "increasing"
                elif engagement_velocity < -0.1:
                    engagement_trend = "decreasing"

            # User activity trends
            total_users = len(self.user_activity)
            recent_users = sum(
                1
                for user_data in self.user_activity.values()
                if time.time() - user_data["first_seen"] <= 300  # New in last 5 minutes
            )
            new_users_ratio = recent_users / max(total_users, 1)

            active_users_trend = "stable"
            if len(self.engagement_history) >= 2:
                _current_active = len(self.active_users)
                # This could be enhanced with historical active user tracking
                active_users_trend = "stable"  # Simplified for now

            return TrendAnalysis(
                timestamp=time.time(),
                window_duration=self.config.trend_window_seconds,
                trending_topics=trending_topics,
                topic_changes=topic_changes,
                sentiment_trend=sentiment_trend,
                sentiment_velocity=sentiment_velocity,
                engagement_trend=engagement_trend,
                engagement_velocity=engagement_velocity,
                new_users_ratio=new_users_ratio,
                active_users_trend=active_users_trend,
            )

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return TrendAnalysis(
                timestamp=time.time(),
                window_duration=self.config.trend_window_seconds,
                trending_topics=[],
                topic_changes=[],
                sentiment_trend="stable",
                sentiment_velocity=0.0,
                engagement_trend="stable",
                engagement_velocity=0.0,
                new_users_ratio=0.0,
                active_users_trend="stable",
            )

    def _is_spam(self, analysis: TextAnalysis) -> bool:
        """Check if message is likely spam."""
        # Simplified spam detection - could be enhanced
        return (
            analysis.word_count < 2  # Very short messages
            or analysis.text.count("http") > 1  # Multiple URLs
            or len(set(analysis.text.lower().split()))
            < len(analysis.text.split()) * 0.3  # Too repetitive
        )

    def _create_empty_engagement_metrics(self) -> EngagementMetrics:
        """Create empty engagement metrics."""
        return EngagementMetrics(
            timestamp=time.time(),
            window_duration=self.config.engagement_window_seconds,
            total_messages=0,
            unique_users=0,
            messages_per_minute=0.0,
            average_message_length=0.0,
            total_characters=0,
            positive_ratio=0.0,
            negative_ratio=0.0,
            neutral_ratio=1.0,
            average_sentiment=0.0,
            engagement_score=0.0,
            activity_level="low",
            spam_ratio=0.0,
            toxicity_ratio=0.0,
            quality_score=1.0,
        )

    async def get_recent_engagement_metrics(
        self, limit: int = 10
    ) -> List[EngagementMetrics]:
        """Get recent engagement metrics."""
        return list(self.engagement_history)[-limit:]

    async def get_active_users(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """Get currently active users."""
        return {
            user_id: data
            for user_id, data in self.user_activity.items()
            if user_id in self.active_users
        }

    async def get_processing_stats(self) -> Dict[str, Union[int, float]]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()

        # Calculate derived metrics
        if stats["total_messages_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["batches_processed"]
            )
            stats["spam_rate"] = (
                stats["spam_messages_detected"] / stats["total_messages_processed"]
            )
            stats["toxicity_rate"] = (
                stats["toxic_messages_detected"] / stats["total_messages_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0
            stats["spam_rate"] = 0.0
            stats["toxicity_rate"] = 0.0

        stats["active_users"] = len(self.active_users)
        stats["total_users"] = len(self.user_activity)
        stats["buffer_size"] = len(self.message_buffer)
        stats["analysis_buffer_size"] = len(self.analysis_buffer)

        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up chat processor")

        # Clear buffers
        self.message_buffer.clear()
        self.analysis_buffer.clear()
        self.engagement_history.clear()

        # Clear user tracking
        self.user_activity.clear()
        self.active_users.clear()

        logger.info("Chat processor cleanup completed")


# Global chat processor instance
chat_processor = ChatProcessor()
