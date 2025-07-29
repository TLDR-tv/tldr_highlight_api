"""
Unit tests for the chat-video timeline synchronization system.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import numpy as np

from src.services.chat_adapters.timeline_sync import (
    TimelineSynchronizer,
    MultiStreamTimelineSynchronizer,
    SyncStrategy,
    ChatSourceType,
    TimestampOffset,
    ChatEventBuffer,
    SyncPoint,
    HighlightCandidate
)
from src.services.chat_adapters.base import ChatEvent, ChatEventType, ChatMessage, ChatUser
from src.services.chat_adapters.sentiment_analyzer import (
    SentimentCategory,
    WindowSentiment,
    VelocityMetrics
)
from src.services.chat_adapters.timeline_sync import StreamType


class TestTimestampOffset:
    """Test TimestampOffset functionality."""
    
    def test_basic_offset_application(self):
        """Test basic offset application."""
        offset = TimestampOffset(offset_seconds=5.0, confidence=0.9)
        
        # Apply offset
        original = 100.0
        adjusted = offset.apply(original)
        
        assert adjusted == 105.0
        assert offset.confidence == 0.9
    
    def test_drift_correction(self):
        """Test drift rate correction."""
        # Create offset with drift
        offset = TimestampOffset(
            offset_seconds=5.0,
            confidence=0.9,
            drift_rate=0.01  # 0.01 seconds per second drift
        )
        
        # Update last update time to past
        past_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        offset.last_update = past_time
        
        # Apply offset with drift correction
        original = 100.0
        current_time = datetime.now(timezone.utc)
        adjusted = offset.apply(original, current_time)
        
        # Should include drift correction (approximately 0.1 seconds for 10 seconds elapsed)
        assert 105.08 <= adjusted <= 105.12  # Allow for small timing variations


class TestChatEventBuffer:
    """Test ChatEventBuffer functionality."""
    
    def test_synchronization_status(self):
        """Test synchronization status detection."""
        # Create test event
        event = ChatEvent(
            id="test_1",
            type=ChatEventType.MESSAGE,
            timestamp=datetime.now(timezone.utc),
            data={}
        )
        
        # Unsynchronized buffer
        buffer = ChatEventBuffer(event=event)
        assert not buffer.is_synchronized
        
        # Synchronized with low confidence
        buffer.video_timestamp = 100.0
        buffer.sync_confidence = 0.3
        assert not buffer.is_synchronized
        
        # Synchronized with high confidence
        buffer.sync_confidence = 0.8
        assert buffer.is_synchronized


@pytest.fixture
async def timeline_synchronizer():
    """Create a test timeline synchronizer."""
    sync = TimelineSynchronizer(
        stream_id="test_stream",
        chat_source=ChatSourceType.TWITCH_EVENTSUB,
        video_type=StreamType.TWITCH_HLS,
        sync_strategy=SyncStrategy.HYBRID
    )
    
    # Mock dependencies
    sync.buffer_manager = Mock()
    sync.frame_synchronizer = Mock()
    sync.frame_synchronizer.normalize_timestamp = Mock(side_effect=lambda _, ts, __: ts)
    sync.frame_synchronizer.register_stream = Mock()
    
    sync.sentiment_analyzer = Mock()
    sync.sentiment_analyzer.process_event = AsyncMock()
    sync.sentiment_analyzer.analyze_window = AsyncMock()
    sync.sentiment_analyzer.temporal_analyzer = Mock()
    sync.sentiment_analyzer.event_tracker = Mock()
    sync.sentiment_analyzer.get_highlight_confidence = Mock(return_value=0.8)
    sync.sentiment_analyzer.window_history = []
    
    yield sync
    
    # Cleanup
    await sync.stop()


class TestTimelineSynchronizer:
    """Test TimelineSynchronizer functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, timeline_synchronizer):
        """Test synchronizer initialization."""
        sync = timeline_synchronizer
        
        assert sync.stream_id == "test_stream"
        assert sync.chat_source == ChatSourceType.TWITCH_EVENTSUB
        assert sync.video_type == StreamType.TWITCH_HLS
        assert sync.sync_strategy == SyncStrategy.HYBRID
        assert not sync.is_synchronized
        assert len(sync.event_buffer) == 0
        assert len(sync.highlight_candidates) == 0
    
    @pytest.mark.asyncio
    async def test_add_chat_event(self, timeline_synchronizer):
        """Test adding chat events."""
        sync = timeline_synchronizer
        
        # Create test message
        user = ChatUser(
            id="user1",
            username="testuser",
            display_name="TestUser"
        )
        
        message = ChatMessage(
            id="msg1",
            user=user,
            text="This is amazing!",
            timestamp=datetime.now(timezone.utc)
        )
        
        event = ChatEvent(
            id="event1",
            type=ChatEventType.MESSAGE,
            timestamp=datetime.now(timezone.utc),
            user=user,
            data={"message": message}
        )
        
        # Add event
        await sync.add_chat_event(event)
        
        # Verify event was buffered
        assert len(sync.event_buffer) == 1
        assert sync.event_buffer[0].event == event
        assert not sync.event_buffer[0].is_synchronized  # Not synchronized yet
        
        # Verify message was buffered
        assert len(sync.message_buffer) == 1
        assert sync.message_buffer[0] == message
        
        # Verify sentiment analyzer was called
        sync.sentiment_analyzer.process_event.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_timestamp_conversion(self, timeline_synchronizer):
        """Test timestamp conversion with offset."""
        sync = timeline_synchronizer
        
        # Set up synchronization
        sync.is_synchronized = True
        sync.timestamp_offset = TimestampOffset(offset_seconds=2.0, confidence=0.9)
        
        # Test conversion
        chat_time = datetime.now(timezone.utc)
        video_time = await sync._convert_to_video_timestamp(chat_time)
        
        # Should apply offset
        expected = chat_time.timestamp() + 2.0
        assert abs(video_time - expected) < 0.1
    
    @pytest.mark.asyncio
    async def test_platform_offset_detection(self, timeline_synchronizer):
        """Test platform-specific offset detection."""
        sync = timeline_synchronizer
        
        # Test Twitch offset
        offset = sync._get_platform_offset()
        assert offset == -2.0  # Twitch chat leads by 2 seconds
        
        # Test YouTube live offset
        sync.chat_source = ChatSourceType.YOUTUBE_LIVE
        sync.video_type = StreamType.YOUTUBE_HLS
        offset = sync._get_platform_offset()
        assert offset == -1.5  # YouTube chat leads by 1.5 seconds
        
        # Test YouTube VOD (no offset)
        sync.chat_source = ChatSourceType.YOUTUBE_VOD
        offset = sync._get_platform_offset()
        assert offset == 0.0
    
    @pytest.mark.asyncio
    async def test_sync_point_creation(self, timeline_synchronizer):
        """Test sync point creation from events."""
        sync = timeline_synchronizer
        
        # Create stream online event
        event = ChatEvent(
            id="stream_start",
            type=ChatEventType.STREAM_ONLINE,
            timestamp=datetime.now(timezone.utc),
            data={}
        )
        
        # Process event
        await sync._create_sync_point_from_event(event)
        
        # Verify sync point was created
        assert len(sync.sync_points) == 1
        sync_point = sync.sync_points[0]
        
        assert sync_point.event_type == ChatEventType.STREAM_ONLINE
        assert sync_point.confidence == 0.9  # High confidence for stream events
        assert sync_point.chat_timestamp == event.timestamp.timestamp()
    
    @pytest.mark.asyncio
    async def test_highlight_detection(self, timeline_synchronizer):
        """Test highlight detection from chat sentiment."""
        sync = timeline_synchronizer
        sync.is_synchronized = True
        sync.timestamp_offset = TimestampOffset(0.0, 1.0)
        
        # Create high-sentiment window
        window_sentiment = WindowSentiment(
            start_time=datetime.now(timezone.utc) - timedelta(seconds=10),
            end_time=datetime.now(timezone.utc),
            message_count=50,
            unique_users=30,
            avg_sentiment=0.8,
            sentiment_variance=0.1,
            dominant_category=SentimentCategory.HYPE,
            intensity=0.9,
            momentum=0.5,
            confidence=0.85,
            emote_density=3.0,
            keyword_density=2.0,
            spam_ratio=0.1
        )
        
        velocity_metrics = VelocityMetrics(
            timestamp=datetime.now(timezone.utc),
            messages_per_second=10.0,
            acceleration=2.0,
            jerk=0.1,
            unique_users_per_second=5.0,
            emotes_per_second=15.0,
            spike_detected=True,
            spike_intensity=0.8
        )
        
        # Set up mocks
        sync.sentiment_analyzer.analyze_window.return_value = window_sentiment
        sync.sentiment_analyzer.temporal_analyzer.calculate_velocity_metrics.return_value = velocity_metrics
        sync.sentiment_analyzer.event_tracker.get_current_impact.return_value = 0.0
        
        # Add some messages to buffer
        for i in range(10):
            user = ChatUser(id=f"user{i}", username=f"user{i}", display_name=f"User{i}")
            message = ChatMessage(
                id=f"msg{i}",
                user=user,
                text="POGGERS!",
                timestamp=datetime.now(timezone.utc)
            )
            sync.message_buffer.append(message)
        
        # Trigger analysis
        await sync._analyze_chat_window()
        
        # Should start highlight tracking
        assert sync.active_highlight is not None
        assert sync.active_highlight["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_highlight_end_detection(self, timeline_synchronizer):
        """Test highlight end detection."""
        sync = timeline_synchronizer
        sync.is_synchronized = True
        
        # Set up active highlight
        start_time = datetime.now(timezone.utc) - timedelta(seconds=20)
        sync.active_highlight = {
            "start_time": start_time,
            "end_time": datetime.now(timezone.utc) - timedelta(seconds=5),
            "video_start": 100.0,
            "sentiment": Mock(
                avg_sentiment=0.8,
                intensity=0.9,
                dominant_category=SentimentCategory.HYPE,
                unique_users=25
            ),
            "velocity": Mock(),
            "confidence": 0.85,
            "events": [],
            "peak_confidence": 0.9,
        }
        
        # Mock low recent confidence
        sync.sentiment_analyzer.window_history = [Mock()]
        sync.sentiment_analyzer.get_highlight_confidence.return_value = 0.2
        
        # Check for end
        await sync._check_highlight_end()
        
        # Should end highlight
        assert sync.active_highlight is None
        assert len(sync.highlight_candidates) == 1
        
        candidate = sync.highlight_candidates[0]
        assert candidate.confidence == 0.9
        assert candidate.duration >= 15.0  # At least 15 seconds
    
    @pytest.mark.asyncio
    async def test_highlight_merging(self, timeline_synchronizer):
        """Test highlight merging functionality."""
        sync = timeline_synchronizer
        
        # Add existing highlight
        existing = HighlightCandidate(
            start_timestamp=100.0,
            end_timestamp=120.0,
            confidence=0.8,
            sentiment_score=0.7,
            intensity=0.8,
            category=SentimentCategory.EXCITEMENT,
            chat_events=[],
            velocity_metrics=Mock(),
            metadata={"message_count": 50, "unique_users": 20}
        )
        sync.highlight_candidates.append(existing)
        
        # Create new highlight that should merge
        new_highlight = HighlightCandidate(
            start_timestamp=122.0,  # 2 second gap
            end_timestamp=140.0,
            confidence=0.85,
            sentiment_score=0.75,
            intensity=0.85,
            category=SentimentCategory.HYPE,
            chat_events=[],
            velocity_metrics=Mock(),
            metadata={"message_count": 40, "unique_users": 25}
        )
        
        # Try merge
        merged = await sync._try_merge_highlight(new_highlight)
        
        assert merged
        assert len(sync.highlight_candidates) == 1
        
        # Check merged highlight
        merged_highlight = sync.highlight_candidates[0]
        assert merged_highlight.start_timestamp == 100.0
        assert merged_highlight.end_timestamp == 140.0
        assert merged_highlight.confidence == 0.85
        assert merged_highlight.metadata["message_count"] == 90
    
    @pytest.mark.asyncio
    async def test_get_synchronized_events(self, timeline_synchronizer):
        """Test retrieving synchronized events."""
        sync = timeline_synchronizer
        
        # Add events with different sync states
        for i in range(5):
            event = ChatEvent(
                id=f"event{i}",
                type=ChatEventType.MESSAGE,
                timestamp=datetime.now(timezone.utc),
                data={}
            )
            
            buffer = ChatEventBuffer(event=event)
            if i < 3:  # First 3 are synchronized
                buffer.video_timestamp = 100.0 + i * 10.0
                buffer.sync_confidence = 0.8
            
            sync.event_buffer.append(buffer)
        
        # Get synchronized events in range
        events = await sync.get_synchronized_events(105.0, 115.0)
        
        assert len(events) == 2  # Events at 110.0 and 110.0
        assert all(e.is_synchronized for e in events)
        assert all(105.0 <= e.video_timestamp <= 115.0 for e in events)


class TestMultiStreamTimelineSynchronizer:
    """Test MultiStreamTimelineSynchronizer functionality."""
    
    @pytest.fixture
    async def multi_synchronizer(self):
        """Create a test multi-stream synchronizer."""
        sync = MultiStreamTimelineSynchronizer()
        
        # Mock dependencies
        sync.buffer_manager = Mock()
        sync.frame_synchronizer = Mock()
        
        yield sync
        
        # Cleanup
        for stream_id in list(sync.stream_synchronizers.keys()):
            await sync.remove_stream(stream_id)
    
    @pytest.mark.asyncio
    async def test_add_remove_streams(self, multi_synchronizer):
        """Test adding and removing streams."""
        sync = multi_synchronizer
        
        # Add stream
        stream_sync = await sync.add_stream(
            "stream1",
            ChatSourceType.TWITCH_EVENTSUB,
            StreamType.TWITCH_HLS
        )
        
        assert "stream1" in sync.stream_synchronizers
        assert stream_sync.stream_id == "stream1"
        
        # Add another stream
        await sync.add_stream(
            "stream2",
            ChatSourceType.YOUTUBE_LIVE,
            StreamType.YOUTUBE_HLS
        )
        
        assert len(sync.stream_synchronizers) == 2
        
        # Remove stream
        await sync.remove_stream("stream1")
        assert "stream1" not in sync.stream_synchronizers
        assert len(sync.stream_synchronizers) == 1
    
    @pytest.mark.asyncio
    async def test_multi_stream_highlight_detection(self, multi_synchronizer):
        """Test multi-stream highlight correlation."""
        sync = multi_synchronizer
        
        # Mock stream synchronizers
        stream1_sync = Mock()
        stream1_sync.get_highlights = AsyncMock(return_value=[])
        sync.stream_synchronizers["stream1"] = stream1_sync
        
        stream2_sync = Mock()
        stream2_sync.get_highlights = AsyncMock(return_value=[])
        sync.stream_synchronizers["stream2"] = stream2_sync
        
        # Create overlapping highlights
        highlight1 = HighlightCandidate(
            start_timestamp=100.0,
            end_timestamp=120.0,
            confidence=0.8,
            sentiment_score=0.7,
            intensity=0.8,
            category=SentimentCategory.HYPE,
            chat_events=[],
            velocity_metrics=Mock()
        )
        
        highlight2 = HighlightCandidate(
            start_timestamp=105.0,
            end_timestamp=125.0,
            confidence=0.85,
            sentiment_score=0.75,
            intensity=0.85,
            category=SentimentCategory.EXCITEMENT,
            chat_events=[],
            velocity_metrics=Mock()
        )
        
        stream2_sync.get_highlights.return_value = [highlight2]
        
        # Process highlight from stream1
        await sync._on_stream_highlight("stream1", highlight1)
        
        # Should create multi-stream highlight
        assert len(sync.multi_stream_highlights) == 1
        
        multi_highlight = sync.multi_stream_highlights[0]
        assert multi_highlight["primary_stream"] == "stream1"
        assert len(multi_highlight["streams"]) == 2
        assert multi_highlight["start_timestamp"] == 100.0
        assert multi_highlight["end_timestamp"] == 125.0
        assert multi_highlight["confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_correlation_detection(self, multi_synchronizer):
        """Test highlight correlation detection."""
        sync = multi_synchronizer
        
        # Create test highlights with various overlaps
        source_highlight = HighlightCandidate(
            start_timestamp=100.0,
            end_timestamp=120.0,
            confidence=0.8,
            sentiment_score=0.7,
            intensity=0.8,
            category=SentimentCategory.HYPE,
            chat_events=[],
            velocity_metrics=Mock()
        )
        
        # High overlap highlight
        high_overlap = HighlightCandidate(
            start_timestamp=102.0,
            end_timestamp=118.0,
            confidence=0.75,
            sentiment_score=0.65,
            intensity=0.75,
            category=SentimentCategory.EXCITEMENT,
            chat_events=[],
            velocity_metrics=Mock()
        )
        
        # Low overlap highlight
        low_overlap = HighlightCandidate(
            start_timestamp=115.0,
            end_timestamp=130.0,
            confidence=0.7,
            sentiment_score=0.6,
            intensity=0.7,
            category=SentimentCategory.POSITIVE,
            chat_events=[],
            velocity_metrics=Mock()
        )
        
        # No overlap highlight
        no_overlap = HighlightCandidate(
            start_timestamp=125.0,
            end_timestamp=140.0,
            confidence=0.65,
            sentiment_score=0.55,
            intensity=0.65,
            category=SentimentCategory.NEUTRAL,
            chat_events=[],
            velocity_metrics=Mock()
        )
        
        # Set up mock synchronizers
        other_sync = Mock()
        other_sync.get_highlights = AsyncMock(
            return_value=[high_overlap, low_overlap, no_overlap]
        )
        sync.stream_synchronizers["other_stream"] = other_sync
        
        # Find correlating highlights
        correlating = await sync._find_correlating_highlights("source_stream", source_highlight)
        
        # Should only find high overlap highlight
        assert len(correlating) == 1
        assert "other_stream" in correlating
        assert correlating["other_stream"] == high_overlap


@pytest.mark.asyncio
async def test_integration_scenario():
    """Test a complete integration scenario."""
    # Create synchronizer
    sync = TimelineSynchronizer(
        stream_id="integration_test",
        chat_source=ChatSourceType.TWITCH_EVENTSUB,
        video_type=StreamType.TWITCH_HLS,
        sync_strategy=SyncStrategy.OFFSET_BASED
    )
    
    # Mock dependencies with more realistic behavior
    sync.sentiment_analyzer = Mock()
    sync.sentiment_analyzer.process_event = AsyncMock()
    sync.sentiment_analyzer.temporal_analyzer = Mock()
    sync.sentiment_analyzer.event_tracker = Mock()
    sync.frame_synchronizer = Mock()
    sync.frame_synchronizer.normalize_timestamp = Mock(side_effect=lambda _, ts, __: ts)
    sync.frame_synchronizer.register_stream = Mock()
    
    # Track highlights
    detected_highlights = []
    sync.add_highlight_callback(lambda h: detected_highlights.append(h))
    
    try:
        # Start synchronizer
        await sync.start()
        
        # Simulate chat activity spike
        base_time = datetime.now(timezone.utc)
        
        # Add regular messages
        for i in range(20):
            user = ChatUser(id=f"user{i}", username=f"user{i}", display_name=f"User{i}")
            message = ChatMessage(
                id=f"msg{i}",
                user=user,
                text="Normal chat message",
                timestamp=base_time + timedelta(seconds=i*0.5)
            )
            event = ChatEvent(
                id=f"event{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message}
            )
            await sync.add_chat_event(event)
        
        # Simulate highlight-worthy activity
        hype_messages = [
            "POGGERS!!!",
            "NO WAY!!!",
            "INSANE PLAY!!!",
            "HOLY SHIT!!!",
            "LETS GOOOO!!!",
            "THAT WAS CRAZY!!!",
            "UNREAL!!!",
            "OMG OMG OMG!!!",
            "BEST PLAY EVER!!!",
            "GODLIKE!!!"
        ]
        
        hype_time = base_time + timedelta(seconds=15)
        for i, text in enumerate(hype_messages):
            user = ChatUser(id=f"hype_user{i}", username=f"hype{i}", display_name=f"HypeUser{i}")
            message = ChatMessage(
                id=f"hype_msg{i}",
                user=user,
                text=text,
                timestamp=hype_time + timedelta(seconds=i*0.2)
            )
            event = ChatEvent(
                id=f"hype_event{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message}
            )
            await sync.add_chat_event(event)
        
        # Add a special event
        raid_event = ChatEvent(
            id="raid_event",
            type=ChatEventType.RAID,
            timestamp=hype_time + timedelta(seconds=3),
            data={"viewers": 500}
        )
        await sync.add_chat_event(raid_event)
        
        # Mock sentiment analysis returning high values during hype
        window_sentiment = WindowSentiment(
            start_time=hype_time,
            end_time=hype_time + timedelta(seconds=5),
            message_count=len(hype_messages),
            unique_users=len(hype_messages),
            avg_sentiment=0.9,
            sentiment_variance=0.05,
            dominant_category=SentimentCategory.HYPE,
            intensity=0.95,
            momentum=0.8,
            confidence=0.9,
            emote_density=2.0,
            keyword_density=3.0,
            spam_ratio=0.0
        )
        
        velocity_metrics = VelocityMetrics(
            timestamp=hype_time + timedelta(seconds=2),
            messages_per_second=5.0,
            acceleration=3.0,
            jerk=0.5,
            unique_users_per_second=4.0,
            emotes_per_second=10.0,
            spike_detected=True,
            spike_intensity=0.9
        )
        
        sync.sentiment_analyzer.analyze_window = AsyncMock(return_value=window_sentiment)
        sync.sentiment_analyzer.temporal_analyzer.calculate_velocity_metrics = Mock(return_value=velocity_metrics)
        sync.sentiment_analyzer.event_tracker.get_current_impact = Mock(return_value=0.8)
        sync.sentiment_analyzer.get_highlight_confidence = Mock(return_value=0.95)
        sync.sentiment_analyzer.window_history = [window_sentiment]
        
        # Trigger analysis
        await sync._analyze_chat_window()
        
        # Wait for highlight to end
        await asyncio.sleep(0.1)
        
        # Mock low activity to end highlight
        sync.sentiment_analyzer.get_highlight_confidence = Mock(return_value=0.2)
        await sync._check_highlight_end()
        
        # Verify results
        assert sync.is_synchronized
        assert len(sync.event_buffer) > 30
        assert len(detected_highlights) == 1
        
        highlight = detected_highlights[0]
        assert highlight.confidence >= 0.9
        assert highlight.category == SentimentCategory.HYPE
        assert highlight.duration >= 5.0
        
    finally:
        await sync.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])