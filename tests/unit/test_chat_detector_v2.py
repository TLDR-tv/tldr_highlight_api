"""
Unit tests for the enhanced chat detector with sentiment analysis integration.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from src.services.highlight_detection.chat_detector_v2 import (
    EnhancedChatDetector,
    EnhancedChatDetectionConfig,
    ChatMessageAdapter,
)
from src.services.highlight_detection.chat_detector import (
    ChatMessage as OldChatMessage,
    ChatWindow,
)
from src.services.highlight_detection.base_detector import (
    ContentSegment,
    ModalityType,
)
from src.services.chat_adapters.base import ChatEvent, ChatEventType, ChatUser
from src.services.chat_adapters.sentiment_analyzer import (
    ChatMessage as NewChatMessage,
    SentimentCategory,
    WindowSentiment,
    VelocityMetrics,
)


class TestChatMessageAdapter:
    """Test message format conversion."""

    def test_old_to_new_conversion(self):
        """Test converting old format to new format."""
        old_msg = OldChatMessage(
            timestamp=1234567890.0,
            user_id="user123",
            username="testuser",
            message="Hello world!",
            platform="twitch",
            metadata={"extra": "data"},
        )

        new_msg = ChatMessageAdapter.old_to_new(old_msg, "twitch")

        assert new_msg.id == "msg_1234567890.0_user123"
        assert new_msg.user.id == "user123"
        assert new_msg.user.username == "testuser"
        assert new_msg.text == "Hello world!"
        assert isinstance(new_msg.timestamp, datetime)
        assert new_msg.metadata == {"extra": "data"}

    def test_new_to_old_conversion(self):
        """Test converting new format to old format."""
        new_msg = NewChatMessage(
            id="msg123",
            user=ChatUser(id="user123", username="testuser", display_name="TestUser"),
            text="Hello world!",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            metadata={"platform": "twitch"},
        )

        old_msg = ChatMessageAdapter.new_to_old(new_msg)

        assert old_msg.user_id == "user123"
        assert old_msg.username == "testuser"
        assert old_msg.message == "Hello world!"
        assert old_msg.platform == "twitch"
        assert isinstance(old_msg.timestamp, float)


class TestEnhancedChatDetector:
    """Test the enhanced chat detector."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EnhancedChatDetectionConfig(
            min_score=0.3,
            min_confidence=0.4,
            sentiment_window_size=10.0,
            velocity_spike_weight=0.3,
            event_impact_weight=0.2,
        )

    @pytest.fixture
    def detector(self, config):
        """Create test detector."""
        return EnhancedChatDetector(config)

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages."""
        now = datetime.now(timezone.utc).timestamp()
        return [
            OldChatMessage(
                timestamp=now + i,
                user_id=f"user{i}",
                username=f"user{i}",
                message="Amazing play! PogChamp"
                if i % 2 == 0
                else "Wow that was sick!",
                platform="twitch",
                metadata={},
            )
            for i in range(10)
        ]

    def test_detector_properties(self, detector):
        """Test detector basic properties."""
        assert detector.modality == ModalityType.CHAT
        assert detector.algorithm_name == "EnhancedChatExcitementDetector"
        assert detector.algorithm_version == "2.0.0"

    def test_validate_segment(self, detector):
        """Test segment validation."""
        # Valid segment with list
        segment = ContentSegment(start_time=0.0, end_time=10.0, data=[], metadata={})
        assert detector._validate_segment(segment) == True

        # Valid segment with ChatWindow
        window = ChatWindow(start_time=0.0, end_time=10.0, messages=[])
        segment = ContentSegment(
            start_time=0.0, end_time=10.0, data=window, metadata={}
        )
        assert detector._validate_segment(segment) == True

        # Invalid segment
        segment = ContentSegment(
            start_time=0.0, end_time=10.0, data="invalid", metadata={}
        )
        assert detector._validate_segment(segment) == False

    @pytest.mark.asyncio
    async def test_prepare_messages_from_window(self, detector, sample_messages):
        """Test preparing messages from ChatWindow."""
        window = ChatWindow(
            start_time=sample_messages[0].timestamp,
            end_time=sample_messages[-1].timestamp,
            messages=sample_messages,
        )

        segment = ContentSegment(
            start_time=window.start_time,
            end_time=window.end_time,
            data=window,
            metadata={"platform": "twitch"},
        )

        messages = await detector._prepare_messages(segment)

        assert len(messages) == len(sample_messages)
        assert all(isinstance(msg, NewChatMessage) for msg in messages)
        assert messages[0].user.id == "user0"
        assert messages[0].text == "Amazing play! PogChamp"

    @pytest.mark.asyncio
    async def test_prepare_messages_from_list(self, detector, sample_messages):
        """Test preparing messages from list."""
        segment = ContentSegment(
            start_time=sample_messages[0].timestamp,
            end_time=sample_messages[-1].timestamp,
            data=sample_messages,
            metadata={"platform": "twitch"},
        )

        messages = await detector._prepare_messages(segment)

        assert len(messages) == len(sample_messages)
        assert all(isinstance(msg, NewChatMessage) for msg in messages)

    @pytest.mark.asyncio
    async def test_process_events(self, detector):
        """Test processing chat events."""
        event = ChatEvent(
            id="evt1",
            type=ChatEventType.RAID,
            timestamp=datetime.now(timezone.utc),
            data={"viewers": 100},
        )

        segment = ContentSegment(
            start_time=0.0, end_time=10.0, data=[], metadata={"events": [event]}
        )

        # Process events
        await detector._process_events(segment)

        # Check that event was processed
        assert len(detector.sentiment_analyzer.event_tracker.active_impacts) > 0

    @pytest.mark.asyncio
    async def test_detect_features_positive_sentiment(self, detector, sample_messages):
        """Test feature detection with positive sentiment."""
        segment = ContentSegment(
            start_time=sample_messages[0].timestamp,
            end_time=sample_messages[-1].timestamp,
            data=sample_messages,
            metadata={"platform": "twitch"},
        )

        results = await detector._detect_features(segment, detector.chat_config)

        assert len(results) <= 1  # Should return 0 or 1 result

        if results:
            result = results[0]
            assert result.modality == ModalityType.CHAT
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert "sentiment_score" in result.features
            assert "velocity_score" in result.features
            assert "message_count" in result.features
            assert result.metadata["algorithm"] == "EnhancedChatExcitementDetector"

    @pytest.mark.asyncio
    async def test_detect_features_with_event(self, detector, sample_messages):
        """Test feature detection with event impact."""
        # Add a raid event
        raid_event = ChatEvent(
            id="raid1",
            type=ChatEventType.RAID,
            timestamp=datetime.fromtimestamp(
                sample_messages[5].timestamp, tz=timezone.utc
            ),
            data={"viewers": 200},
        )

        segment = ContentSegment(
            start_time=sample_messages[0].timestamp,
            end_time=sample_messages[-1].timestamp,
            data=sample_messages,
            metadata={"platform": "twitch", "events": [raid_event]},
        )

        results = await detector._detect_features(segment, detector.chat_config)

        if results:
            result = results[0]
            assert result.features["event_impact"] > 0
            assert result.metadata["event_impact"] > 0

    def test_calculate_sentiment_score(self, detector, config):
        """Test sentiment score calculation."""
        # Hype category window
        window = WindowSentiment(
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(seconds=10),
            message_count=50,
            unique_users=30,
            avg_sentiment=0.8,
            sentiment_variance=0.1,
            dominant_category=SentimentCategory.HYPE,
            intensity=0.9,
            momentum=0.5,
            confidence=0.85,
            emote_density=2.5,
            keyword_density=1.8,
            spam_ratio=0.1,
        )

        score = detector._calculate_sentiment_score(window, config)

        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high for hype category

    def test_calculate_velocity_score(self, detector, config):
        """Test velocity score calculation."""
        # With spike
        velocity_spike = VelocityMetrics(
            timestamp=datetime.now(timezone.utc),
            messages_per_second=10.0,
            acceleration=5.0,
            jerk=1.0,
            unique_users_per_second=5.0,
            emotes_per_second=3.0,
            spike_detected=True,
            spike_intensity=0.8,
        )

        score = detector._calculate_velocity_score(velocity_spike, config)

        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high for spike

        # Without spike
        velocity_normal = VelocityMetrics(
            timestamp=datetime.now(timezone.utc),
            messages_per_second=2.0,
            acceleration=0.0,
            jerk=0.0,
            unique_users_per_second=1.0,
            emotes_per_second=0.5,
            spike_detected=False,
            spike_intensity=0.0,
        )

        score_normal = detector._calculate_velocity_score(velocity_normal, config)

        assert score_normal < score  # Normal should be lower than spike

    @pytest.mark.asyncio
    async def test_process_event_stream(self, detector):
        """Test processing a stream of events."""
        now = datetime.now(timezone.utc)
        events = [
            ChatEvent(
                id=f"evt{i}",
                type=ChatEventType.MESSAGE,
                timestamp=now + timedelta(seconds=i),
                data={"message": f"Event {i}"},
            )
            for i in range(20)
        ]

        # Add a special event
        events.append(
            ChatEvent(
                id="raid",
                type=ChatEventType.RAID,
                timestamp=now + timedelta(seconds=10),
                data={"viewers": 300},
            )
        )

        results = await detector.process_event_stream(events, window_size=5.0)

        # Should create multiple windows
        assert isinstance(results, list)
        # Results depend on sentiment analysis

    def test_get_sentiment_metrics(self, detector):
        """Test getting sentiment metrics."""
        metrics = detector.get_sentiment_metrics()

        assert isinstance(metrics, dict)
        assert "total_messages_analyzed" in metrics
        assert "window_count" in metrics
        assert "active_event_impacts" in metrics

    def test_reset_baselines(self, detector):
        """Test resetting baselines."""
        # Add some data first
        detector.sentiment_analyzer.message_history.append(Mock())
        detector.sentiment_analyzer.window_history.append(Mock())

        # Reset
        detector.reset_baselines()

        # Check cleared
        assert len(detector.sentiment_analyzer.message_history) == 0
        assert len(detector.sentiment_analyzer.window_history) == 0


class TestIntegration:
    """Integration tests for enhanced chat detector."""

    @pytest.mark.asyncio
    async def test_full_detection_workflow(self):
        """Test complete detection workflow with realistic data."""
        detector = EnhancedChatDetector()

        # Create realistic chat surge
        now = datetime.now(timezone.utc).timestamp()
        messages = []

        # Normal chat
        for i in range(5):
            messages.append(
                OldChatMessage(
                    timestamp=now + i * 2,
                    user_id=f"user{i}",
                    username=f"user{i}",
                    message="Normal chat message",
                    platform="twitch",
                )
            )

        # Excitement surge
        surge_start = now + 10
        for i in range(30):
            messages.append(
                OldChatMessage(
                    timestamp=surge_start + i * 0.3,
                    user_id=f"excited_user{i}",
                    username=f"excited_user{i}",
                    message="AMAZING PLAY!!! PogChamp POGGERS"
                    if i % 2 == 0
                    else "LETS GOOO! That was INSANE!!!",
                    platform="twitch",
                )
            )

        # Create segment
        segment = ContentSegment(
            segment_id="test_segment",
            start_time=messages[0].timestamp,
            end_time=messages[-1].timestamp,
            data=messages,
            metadata={
                "platform": "twitch",
                "events": [
                    {
                        "id": "cheer1",
                        "type": "cheer",
                        "timestamp": surge_start + 5,
                        "data": {"amount": 1000},
                    }
                ],
            },
        )

        # Detect highlights
        results = await detector.detect_highlights([segment])

        # Should detect the surge as a highlight
        assert len(results) > 0

        if results:
            result = results[0]
            assert (
                result.score > 0.25
            )  # Above threshold (adjusted for realistic scoring)
            assert result.confidence > 0.3
            assert result.features["message_count"] == len(messages)
            assert (
                result.features["velocity_score"] > 0
            )  # Should detect velocity increase
            assert (
                result.features["sentiment_score"] > 0.7
            )  # Should have high sentiment
            assert (
                result.metadata["window_sentiment"]["category"]
                == SentimentCategory.HYPE
            )  # Should detect hype
