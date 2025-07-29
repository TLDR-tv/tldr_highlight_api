"""
Integration tests for the chat-video timeline synchronization system.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.services.chat_adapters.timeline_sync import (
    TimelineSynchronizer,
    MultiStreamTimelineSynchronizer,
    SyncStrategy,
    ChatSourceType,
    HighlightCandidate
)
from src.services.chat_adapters.base import ChatEvent, ChatEventType, ChatMessage, ChatUser
from src.services.chat_adapters.twitch_eventsub import TwitchEventSubAdapter
from src.services.chat_adapters.youtube import YouTubeChatAdapter
from src.services.content_processing.stream_buffer_manager import StreamBufferManager, StreamType
from src.utils.frame_synchronizer import FrameSynchronizer
from src.utils.video_buffer import VideoFrame, BufferFormat, FrameType


class MockStreamAdapter:
    """Mock stream adapter for testing."""
    
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.frame_count = 0
        self.is_running = False
    
    async def get_stream_data(self):
        """Generate mock stream data."""
        self.is_running = True
        while self.is_running:
            # Generate HLS segment data
            segment_data = {
                "type": "hls_segment",
                "sequence": self.frame_count,
                "timestamp": datetime.now(timezone.utc).timestamp(),
                "duration": 2.0,
                "data": b"mock_segment_data"
            }
            
            self.frame_count += 1
            yield json.dumps(segment_data).encode()
            
            await asyncio.sleep(2.0)  # Simulate 2-second segments
    
    def stop(self):
        """Stop generating data."""
        self.is_running = False


@pytest.fixture
async def stream_buffer_manager():
    """Create a stream buffer manager for testing."""
    manager = StreamBufferManager()
    yield manager
    await manager.close()


@pytest.fixture
async def frame_synchronizer():
    """Create a frame synchronizer for testing."""
    sync = FrameSynchronizer()
    yield sync
    await sync.close()


class TestTimelineSyncIntegration:
    """Integration tests for timeline synchronization."""
    
    @pytest.mark.asyncio
    async def test_twitch_stream_synchronization(self, stream_buffer_manager, frame_synchronizer):
        """Test synchronization with a Twitch stream."""
        stream_id = "twitch_test_stream"
        
        # Create mock Twitch adapter
        mock_adapter = MockStreamAdapter(stream_id)
        
        # Add stream to buffer manager
        await stream_buffer_manager.add_stream(
            stream_id,
            mock_adapter,
            StreamType.TWITCH_HLS
        )
        
        # Create timeline synchronizer
        timeline_sync = TimelineSynchronizer(
            stream_id=stream_id,
            chat_source=ChatSourceType.TWITCH_EVENTSUB,
            video_type=StreamType.TWITCH_HLS,
            buffer_manager=stream_buffer_manager,
            frame_synchronizer=frame_synchronizer,
            sync_strategy=SyncStrategy.OFFSET_BASED
        )
        
        # Track highlights
        detected_highlights = []
        timeline_sync.add_highlight_callback(
            lambda h: detected_highlights.append(h)
        )
        
        try:
            # Start synchronizer
            await timeline_sync.start()
            
            # Simulate chat messages
            base_time = datetime.now(timezone.utc)
            
            # Regular chat flow
            for i in range(10):
                user = ChatUser(
                    id=f"user_{i}",
                    username=f"user{i}",
                    display_name=f"User {i}"
                )
                
                message = ChatMessage(
                    id=f"msg_{i}",
                    user=user,
                    text=f"Regular message {i}",
                    timestamp=base_time + timedelta(seconds=i)
                )
                
                event = ChatEvent(
                    id=f"event_{i}",
                    type=ChatEventType.MESSAGE,
                    timestamp=message.timestamp,
                    user=user,
                    data={"message": message}
                )
                
                await timeline_sync.add_chat_event(event)
                await asyncio.sleep(0.1)
            
            # Verify synchronization
            assert timeline_sync.is_synchronized
            assert timeline_sync.timestamp_offset.offset_seconds == -2.0  # Twitch offset
            
            # Simulate highlight-worthy moment
            hype_start = datetime.now(timezone.utc)
            
            # Raid event
            raid_event = ChatEvent(
                id="raid_1",
                type=ChatEventType.RAID,
                timestamp=hype_start,
                data={"viewers": 1000, "from_channel": "big_streamer"}
            )
            await timeline_sync.add_chat_event(raid_event)
            
            # Hype messages
            hype_messages = [
                ("RAID HYPE!!!", "raider_1"),
                ("POGGERS RAID!!!", "raider_2"),
                ("1000 VIEWERS LETS GO!!!", "viewer_1"),
                ("THIS IS INSANE!!!", "viewer_2"),
                ("WELCOME RAIDERS!!!", "mod_1"),
                ("HYPE TRAIN!!!", "sub_1"),
                ("NO WAY 1K VIEWERS!!!", "viewer_3"),
                ("BEST RAID EVER!!!", "raider_3"),
            ]
            
            for i, (text, username) in enumerate(hype_messages):
                user = ChatUser(
                    id=f"{username}_id",
                    username=username,
                    display_name=username.replace("_", " ").title()
                )
                
                message = ChatMessage(
                    id=f"hype_msg_{i}",
                    user=user,
                    text=text,
                    timestamp=hype_start + timedelta(seconds=i * 0.5),
                    emotes=[{"name": "PogChamp"}, {"name": "Kreygasm"}] if i % 2 == 0 else []
                )
                
                event = ChatEvent(
                    id=f"hype_event_{i}",
                    type=ChatEventType.MESSAGE,
                    timestamp=message.timestamp,
                    user=user,
                    data={"message": message}
                )
                
                await timeline_sync.add_chat_event(event)
                await asyncio.sleep(0.05)
            
            # Wait for processing
            await asyncio.sleep(2.0)
            
            # Get synchronized events
            video_start = hype_start.timestamp() - 2.0  # Apply Twitch offset
            video_end = video_start + 10.0
            
            sync_events = await timeline_sync.get_synchronized_events(video_start, video_end)
            assert len(sync_events) > 0
            
            # Check sync status
            status = timeline_sync.get_sync_status()
            assert status["is_synchronized"]
            assert status["buffered_events"] > 15
            assert status["timestamp_offset"]["offset_seconds"] == -2.0
            
        finally:
            # Cleanup
            mock_adapter.stop()
            await timeline_sync.stop()
            await stream_buffer_manager.remove_stream(stream_id)
    
    @pytest.mark.asyncio
    async def test_youtube_vod_synchronization(self, stream_buffer_manager, frame_synchronizer):
        """Test synchronization with YouTube VOD."""
        stream_id = "youtube_vod_test"
        
        # Create timeline synchronizer for VOD
        timeline_sync = TimelineSynchronizer(
            stream_id=stream_id,
            chat_source=ChatSourceType.YOUTUBE_VOD,
            video_type=StreamType.YOUTUBE_HLS,
            buffer_manager=stream_buffer_manager,
            frame_synchronizer=frame_synchronizer,
            sync_strategy=SyncStrategy.OFFSET_BASED
        )
        
        try:
            await timeline_sync.start()
            
            # VOD should have zero offset
            assert timeline_sync.timestamp_offset.offset_seconds == 0.0
            
            # Simulate VOD replay with relative timestamps
            vod_start = datetime.now(timezone.utc)
            
            # Add chat replay data
            replay_messages = [
                (0.0, "First message in VOD"),
                (5.0, "Something happening!"),
                (10.0, "POGGERS!!!"),
                (10.5, "NO WAY!!!"),
                (11.0, "INSANE PLAY!!!"),
                (11.5, "CLIP IT!!!"),
                (12.0, "BEST MOMENT!!!"),
                (15.0, "That was amazing"),
                (20.0, "GG"),
            ]
            
            for relative_time, text in replay_messages:
                user = ChatUser(
                    id=f"vod_user_{int(relative_time)}",
                    username=f"viewer{int(relative_time)}",
                    display_name=f"Viewer {int(relative_time)}"
                )
                
                # For VOD, timestamp is relative to start
                timestamp = vod_start + timedelta(seconds=relative_time)
                
                message = ChatMessage(
                    id=f"vod_msg_{relative_time}",
                    user=user,
                    text=text,
                    timestamp=timestamp
                )
                
                event = ChatEvent(
                    id=f"vod_event_{relative_time}",
                    type=ChatEventType.MESSAGE,
                    timestamp=timestamp,
                    user=user,
                    data={"message": message}
                )
                
                await timeline_sync.add_chat_event(event)
            
            # Check buffered events
            assert len(timeline_sync.event_buffer) == len(replay_messages)
            
            # All events should be synchronized with zero offset
            for buffered_event in timeline_sync.event_buffer:
                if buffered_event.is_synchronized:
                    # Video timestamp should match chat timestamp for VOD
                    chat_ts = buffered_event.event.timestamp.timestamp()
                    assert abs(buffered_event.video_timestamp - chat_ts) < 0.1
            
        finally:
            await timeline_sync.stop()
    
    @pytest.mark.asyncio
    async def test_adaptive_synchronization(self, stream_buffer_manager, frame_synchronizer):
        """Test adaptive synchronization strategy."""
        stream_id = "adaptive_test_stream"
        
        # Create timeline synchronizer with adaptive strategy
        timeline_sync = TimelineSynchronizer(
            stream_id=stream_id,
            chat_source=ChatSourceType.GENERIC_WEBSOCKET,
            video_type=StreamType.GENERIC_HLS,
            buffer_manager=stream_buffer_manager,
            frame_synchronizer=frame_synchronizer,
            sync_strategy=SyncStrategy.ADAPTIVE
        )
        
        # Mock correlation detection
        with patch.object(timeline_sync, '_get_chat_velocity_signal') as mock_chat_signal, \
             patch.object(timeline_sync, '_get_video_activity_signal') as mock_video_signal:
            
            # Create signals with known offset
            import numpy as np
            
            # Chat signal (leads by 1.5 seconds)
            chat_signal = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0, 0])
            mock_chat_signal.return_value = chat_signal
            
            # Video signal (delayed)
            video_signal = np.array([0, 0, 0, 1, 2, 3, 2, 1, 0, 0])
            mock_video_signal.return_value = video_signal
            
            try:
                await timeline_sync.start()
                
                # Wait for adaptive sync
                await asyncio.sleep(0.5)
                
                # Should detect offset
                # Note: The actual offset calculation depends on correlation
                # For testing, we'll just verify it attempted detection
                assert mock_chat_signal.called
                assert mock_video_signal.called
                
            finally:
                await timeline_sync.stop()


class TestMultiStreamIntegration:
    """Integration tests for multi-stream synchronization."""
    
    @pytest.mark.asyncio
    async def test_multi_perspective_highlights(self, stream_buffer_manager, frame_synchronizer):
        """Test highlight detection across multiple streams."""
        multi_sync = MultiStreamTimelineSynchronizer(
            buffer_manager=stream_buffer_manager,
            frame_synchronizer=frame_synchronizer
        )
        
        try:
            # Add multiple streams
            stream1 = await multi_sync.add_stream(
                "streamer1",
                ChatSourceType.TWITCH_EVENTSUB,
                StreamType.TWITCH_HLS
            )
            
            stream2 = await multi_sync.add_stream(
                "streamer2",
                ChatSourceType.TWITCH_EVENTSUB,
                StreamType.TWITCH_HLS
            )
            
            # Create overlapping highlights in both streams
            base_time = datetime.now(timezone.utc)
            
            # Stream 1 highlight
            highlight1 = HighlightCandidate(
                start_timestamp=100.0,
                end_timestamp=120.0,
                confidence=0.85,
                sentiment_score=0.8,
                intensity=0.9,
                category="hype",
                chat_events=[],
                velocity_metrics=Mock(),
                metadata={"stream": "streamer1"}
            )
            
            # Stream 2 highlight (overlapping)
            highlight2 = HighlightCandidate(
                start_timestamp=105.0,
                end_timestamp=125.0,
                confidence=0.90,
                sentiment_score=0.85,
                intensity=0.95,
                category="excitement",
                chat_events=[],
                velocity_metrics=Mock(),
                metadata={"stream": "streamer2"}
            )
            
            # Mock the synchronizers to return highlights
            stream1.get_highlights = AsyncMock(return_value=[highlight1])
            stream2.get_highlights = AsyncMock(return_value=[highlight2])
            
            # Trigger multi-stream detection
            await multi_sync._on_stream_highlight("streamer1", highlight1)
            
            # Check multi-stream highlights
            multi_highlights = await multi_sync.get_multi_stream_highlights()
            assert len(multi_highlights) == 1
            
            multi_highlight = multi_highlights[0]
            assert len(multi_highlight["streams"]) == 2
            assert "streamer1" in multi_highlight["streams"]
            assert "streamer2" in multi_highlight["streams"]
            assert multi_highlight["confidence"] == 0.90  # Max confidence
            
            # Check status
            status = multi_sync.get_status()
            assert status["active_streams"] == 2
            assert status["multi_stream_highlights"] == 1
            
        finally:
            # Cleanup
            await multi_sync.remove_stream("streamer1")
            await multi_sync.remove_stream("streamer2")
    
    @pytest.mark.asyncio
    async def test_tournament_synchronization(self, stream_buffer_manager, frame_synchronizer):
        """Test synchronization for tournament-style multi-stream setup."""
        multi_sync = MultiStreamTimelineSynchronizer(
            buffer_manager=stream_buffer_manager,
            frame_synchronizer=frame_synchronizer
        )
        
        # Simulate tournament with 4 player perspectives
        players = ["player1", "player2", "player3", "player4"]
        synchronizers = {}
        
        try:
            # Add all player streams
            for player in players:
                sync = await multi_sync.add_stream(
                    player,
                    ChatSourceType.TWITCH_EVENTSUB,
                    StreamType.TWITCH_HLS,
                    SyncStrategy.OFFSET_BASED  # Use consistent offset
                )
                synchronizers[player] = sync
            
            # Simulate a tournament highlight moment
            # All players react to the same event at slightly different times
            event_time = datetime.now(timezone.utc)
            
            reactions = [
                ("player1", 0.0, ["INSANE PLAY!!!", "NO WAY!!!", "CLIP THAT!!!"]),
                ("player2", 0.5, ["WHAT!!!", "HOW DID HE DO THAT?!", "UNREAL!!!"]),
                ("player3", 1.0, ["GG", "That was sick!", "Best play of the tournament!"]),
                ("player4", 1.5, ["I can't believe it!", "LEGENDARY!!!", "GOAT!!!"]),
            ]
            
            # Add reactions to each stream
            for player, delay, messages in reactions:
                sync = synchronizers[player]
                reaction_time = event_time + timedelta(seconds=delay)
                
                for i, text in enumerate(messages):
                    user = ChatUser(
                        id=f"{player}_fan_{i}",
                        username=f"{player}_fan{i}",
                        display_name=f"{player.title()} Fan {i}"
                    )
                    
                    message = ChatMessage(
                        id=f"{player}_msg_{i}",
                        user=user,
                        text=text,
                        timestamp=reaction_time + timedelta(seconds=i * 0.3)
                    )
                    
                    event = ChatEvent(
                        id=f"{player}_event_{i}",
                        type=ChatEventType.MESSAGE,
                        timestamp=message.timestamp,
                        user=user,
                        data={"message": message}
                    )
                    
                    await sync.add_chat_event(event)
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            # Check that all streams have events
            for player in players:
                sync = synchronizers[player]
                assert len(sync.event_buffer) >= 3
                assert sync.is_synchronized
            
            # Get overall status
            status = multi_sync.get_status()
            assert status["active_streams"] == 4
            
            # Each stream should have similar sync settings
            for player in players:
                stream_status = status["stream_status"][player]
                assert stream_status["is_synchronized"]
                assert stream_status["timestamp_offset"]["offset_seconds"] == -2.0  # Twitch offset
            
        finally:
            # Cleanup all streams
            for player in players:
                await multi_sync.remove_stream(player)


@pytest.mark.asyncio
async def test_real_time_highlight_generation():
    """Test real-time highlight generation with simulated stream."""
    # This test simulates a more realistic scenario
    stream_id = "realtime_test"
    
    # Create components
    buffer_manager = StreamBufferManager()
    frame_sync = FrameSynchronizer()
    
    timeline_sync = TimelineSynchronizer(
        stream_id=stream_id,
        chat_source=ChatSourceType.TWITCH_EVENTSUB,
        video_type=StreamType.TWITCH_HLS,
        buffer_manager=buffer_manager,
        frame_synchronizer=frame_sync,
        sync_strategy=SyncStrategy.HYBRID
    )
    
    # Track generated highlights
    highlights = []
    timeline_sync.add_highlight_callback(lambda h: highlights.append(h))
    
    try:
        await timeline_sync.start()
        
        # Simulate 60 seconds of stream with varying activity
        start_time = datetime.now(timezone.utc)
        
        # Activity pattern: low -> spike -> high -> decline -> low
        activity_pattern = [
            (0, 10, 1, "low"),      # 0-10s: low activity (1 msg/s)
            (10, 15, 10, "spike"),  # 10-15s: activity spike (10 msg/s)
            (15, 25, 8, "high"),    # 15-25s: sustained high (8 msg/s)
            (25, 30, 4, "decline"), # 25-30s: declining (4 msg/s)
            (30, 40, 1, "low"),     # 30-40s: back to low (1 msg/s)
            (40, 45, 12, "spike"),  # 40-45s: another spike (12 msg/s)
            (45, 50, 6, "medium"),  # 45-50s: medium activity (6 msg/s)
            (50, 60, 1, "low"),     # 50-60s: low again (1 msg/s)
        ]
        
        # Generate messages according to pattern
        message_count = 0
        for start_sec, end_sec, rate, activity_type in activity_pattern:
            num_messages = int((end_sec - start_sec) * rate)
            
            for i in range(num_messages):
                # Time within the window
                offset = (i / num_messages) * (end_sec - start_sec)
                timestamp = start_time + timedelta(seconds=start_sec + offset)
                
                # Generate appropriate message based on activity
                if activity_type in ["spike", "high"]:
                    texts = [
                        "POGGERS!!!", "INSANE!!!", "NO WAY!!!", "LETS GO!!!",
                        "HOLY SHIT!!!", "UNREAL!!!", "CLIP IT!!!", "HYPE!!!"
                    ]
                    text = texts[i % len(texts)]
                    emotes = [{"name": "PogChamp"}, {"name": "Kreygasm"}]
                elif activity_type == "medium":
                    texts = ["Nice play!", "Good job!", "Well done!", "GG!"]
                    text = texts[i % len(texts)]
                    emotes = []
                else:
                    text = f"Message {message_count}"
                    emotes = []
                
                user = ChatUser(
                    id=f"user_{message_count % 50}",  # 50 unique users
                    username=f"user{message_count % 50}",
                    display_name=f"User {message_count % 50}"
                )
                
                message = ChatMessage(
                    id=f"msg_{message_count}",
                    user=user,
                    text=text,
                    timestamp=timestamp,
                    emotes=emotes
                )
                
                event = ChatEvent(
                    id=f"event_{message_count}",
                    type=ChatEventType.MESSAGE,
                    timestamp=timestamp,
                    user=user,
                    data={"message": message}
                )
                
                await timeline_sync.add_chat_event(event)
                message_count += 1
            
            # Small delay between activity changes
            await asyncio.sleep(0.1)
        
        # Add a special event during high activity
        special_event = ChatEvent(
            id="special_1",
            type=ChatEventType.CHEER,
            timestamp=start_time + timedelta(seconds=20),
            data={"bits": 1000, "message": "TAKE MY BITS!!!"}
        )
        await timeline_sync.add_chat_event(special_event)
        
        # Wait for processing to complete
        await asyncio.sleep(3.0)
        
        # Get final highlights
        final_highlights = await timeline_sync.get_highlights(min_confidence=0.6)
        
        # Verify highlights were detected
        # Should detect at least the two spike periods
        assert len(final_highlights) >= 2
        
        # Check highlight properties
        for highlight in final_highlights:
            assert highlight.duration >= 5.0  # Minimum duration
            assert highlight.confidence >= 0.6
            assert highlight.sentiment_score > 0.5  # Positive sentiment
            assert len(highlight.chat_events) > 0
            
            # The spike periods should have high intensity
            if 10 <= highlight.start_timestamp <= 25 or 40 <= highlight.start_timestamp <= 50:
                assert highlight.intensity > 0.7
        
        # Check synchronization maintained throughout
        status = timeline_sync.get_sync_status()
        assert status["is_synchronized"]
        assert status["buffered_events"] > 0
        
    finally:
        await timeline_sync.stop()
        await buffer_manager.close()
        await frame_sync.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])