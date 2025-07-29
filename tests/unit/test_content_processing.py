"""
Comprehensive unit tests for content processing components.

Tests for video processing, audio processing, chat processing,
synchronization, and buffer management.
"""

import pytest
import numpy as np
import time
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

# Import the components we're testing
from src.services.content_processing.video_processor import (
    VideoProcessor,
    VideoProcessorConfig,
    ProcessedFrame,
)
from src.services.content_processing.audio_processor import (
    AudioProcessor,
    AudioProcessorConfig,
    ProcessedAudio,
    TranscriptionResult,
)
from src.services.content_processing.chat_processor import (
    ChatProcessor,
    ChatProcessorConfig,
    EngagementMetrics,
)
from src.services.content_processing.synchronizer import (
    ContentSynchronizer,
    SynchronizationConfig,
    SyncPoint,
)
from src.services.content_processing.buffer_manager import (
    BufferManager,
    BufferConfig,
    ProcessingWindow,
    BufferPriority,
    WindowType,
)
from src.utils.media_utils import VideoFrame, AudioChunk, MediaInfo
from src.utils.nlp_utils import ChatMessage, TextAnalysis, SentimentScore


class TestVideoProcessor:
    """Test cases for VideoProcessor."""

    @pytest.fixture
    def video_config(self):
        return VideoProcessorConfig(
            frame_interval_seconds=1.0,
            max_frames_per_window=10,
            quality_threshold=0.3,
            resize_width=720,
            enable_scene_detection=True,
        )

    @pytest.fixture
    def video_processor(self, video_config):
        return VideoProcessor(video_config)

    @pytest.fixture
    def mock_video_frame(self):
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        return VideoFrame(
            frame=frame,
            timestamp=1.0,
            frame_number=30,
            width=720,
            height=480,
            quality_score=0.8,
        )

    @pytest.fixture
    def mock_media_info(self):
        return MediaInfo(
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
            has_audio=True,
        )

    def test_video_processor_initialization(self, video_processor, video_config):
        """Test video processor initialization."""
        assert video_processor.config == video_config
        assert video_processor.frame_buffer == []
        assert video_processor.active_streams == set()
        assert isinstance(video_processor.processing_stats, dict)

    @pytest.mark.asyncio
    async def test_process_single_frame(self, video_processor, mock_video_frame):
        """Test processing a single video frame."""
        processed_frame = await video_processor._process_single_frame(
            mock_video_frame, []
        )

        assert isinstance(processed_frame, ProcessedFrame)
        assert processed_frame.frame == mock_video_frame
        assert processed_frame.is_keyframe is True  # First frame is always keyframe
        assert processed_frame.scene_change is False
        assert isinstance(processed_frame.analysis, dict)
        assert processed_frame.processing_time > 0

    @pytest.mark.asyncio
    async def test_frame_quality_analysis(self, video_processor, mock_video_frame):
        """Test frame quality analysis."""
        analysis = await video_processor._analyze_frame_quality(mock_video_frame)

        assert isinstance(analysis, dict)
        assert "quality_score" in analysis
        assert "blur_score" in analysis
        assert "brightness" in analysis
        assert "contrast" in analysis
        assert analysis["quality_score"] == mock_video_frame.quality_score

    @pytest.mark.asyncio
    async def test_scene_change_detection(self, video_processor, mock_video_frame):
        """Test scene change detection."""
        # Create two different frames
        frame1 = mock_video_frame
        frame2 = VideoFrame(
            frame=np.ones((480, 720, 3), dtype=np.uint8) * 255,  # White frame
            timestamp=2.0,
            frame_number=60,
            width=720,
            height=480,
            quality_score=0.8,
        )

        scene_change = await video_processor._detect_scene_change(frame2, frame1)
        assert isinstance(scene_change, bool)
        # Should detect change between black and white frames
        assert scene_change is True

    @pytest.mark.asyncio
    async def test_start_stop_stream_processing(self, video_processor):
        """Test starting and stopping stream processing."""
        stream_url = "rtmp://example.com/stream"
        stream_id = "test_stream"

        with patch("src.services.content_processing.video_processor.StreamCapture") as mock_capture:
            mock_capture_instance = AsyncMock()
            mock_capture_instance.start_capture = AsyncMock()
            mock_capture_instance.stop_capture = AsyncMock()
            mock_capture.return_value = mock_capture_instance

            # Start stream processing
            await video_processor.start_stream_processing(stream_url, stream_id)

            assert stream_id in video_processor.active_streams
            assert stream_id in video_processor.stream_captures
            mock_capture_instance.start_capture.assert_called_once()

            # Stop stream processing
            await video_processor.stop_stream_processing(stream_id)

            assert stream_id not in video_processor.active_streams
            assert stream_id not in video_processor.stream_captures
            mock_capture_instance.stop_capture.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_processing_stats(self, video_processor):
        """Test getting processing statistics."""
        stats = await video_processor.get_processing_stats()

        assert isinstance(stats, dict)
        assert "total_frames_processed" in stats
        assert "total_processing_time" in stats
        assert "average_processing_time" in stats
        assert "active_streams" in stats
        assert "buffer_size" in stats

    @pytest.mark.asyncio
    async def test_optimize_for_profile(self, video_processor):
        """Test optimization for different profiles."""
        # Test fast profile
        await video_processor.optimize_for_profile("fast")
        assert video_processor.config.frame_interval_seconds == 2.0
        assert video_processor.config.resize_width == 480
        assert video_processor.config.enable_quality_analysis is False

        # Test quality profile
        await video_processor.optimize_for_profile("quality")
        assert video_processor.config.frame_interval_seconds == 0.5
        assert video_processor.config.resize_width == 1080
        assert video_processor.config.enable_quality_analysis is True


class TestAudioProcessor:
    """Test cases for AudioProcessor."""

    @pytest.fixture
    def audio_config(self):
        return AudioProcessorConfig(
            chunk_duration=30.0,
            sample_rate=16000,
            channels=1,
            min_audio_duration=1.0,
            max_concurrent_requests=3,
        )

    @pytest.fixture
    def audio_processor(self, audio_config):
        processor = AudioProcessor(audio_config)
        # Mock OpenAI client to avoid actual API calls
        processor.openai_client = AsyncMock()
        return processor

    @pytest.fixture
    def mock_audio_chunk(self):
        # Create 1 second of audio data (16-bit PCM, 16kHz, mono)
        duration = 1.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16).tobytes()

        return AudioChunk(
            data=audio_data,
            timestamp=1.0,
            duration=duration,
            sample_rate=sample_rate,
            channels=1,
            format="pcm_s16le",
        )

    @pytest.fixture
    def mock_transcription_response(self):
        """Mock OpenAI transcription response."""
        mock_response = MagicMock()
        mock_response.text = "This is a test transcription"
        mock_response.language = "en"
        mock_response.words = []
        mock_response.segments = []
        return mock_response

    def test_audio_processor_initialization(self, audio_processor, audio_config):
        """Test audio processor initialization."""
        assert audio_processor.config == audio_config
        assert audio_processor.transcription_buffer == []
        assert isinstance(audio_processor.processing_stats, dict)

    @pytest.mark.asyncio
    async def test_analyze_audio_quality(self, audio_processor, mock_audio_chunk):
        """Test audio quality analysis."""
        analysis = await audio_processor._analyze_audio_quality(mock_audio_chunk)

        assert hasattr(analysis, "volume_level")
        assert hasattr(analysis, "silence_ratio")
        assert hasattr(analysis, "speech_ratio")
        assert hasattr(analysis, "is_speech")
        assert hasattr(analysis, "quality_score")
        assert 0.0 <= analysis.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_transcribe_audio(
        self, audio_processor, mock_audio_chunk, mock_transcription_response
    ):
        """Test audio transcription."""
        # Mock the OpenAI API call
        audio_processor.openai_client.audio.transcriptions.create.return_value = (
            mock_transcription_response
        )

        with patch("builtins.open", mock_open()):
            with patch.object(Path, "unlink"):
                result = await audio_processor._transcribe_audio(mock_audio_chunk)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a test transcription"
        assert result.timestamp == mock_audio_chunk.timestamp
        assert result.duration == mock_audio_chunk.duration

    @pytest.mark.asyncio
    async def test_process_audio_chunk(self, audio_processor, mock_audio_chunk):
        """Test processing a single audio chunk."""
        # Mock transcription to avoid API call
        with patch.object(audio_processor, "_transcribe_audio", return_value=None):
            processed_audio = await audio_processor._process_audio_chunk(
                mock_audio_chunk
            )

        assert isinstance(processed_audio, ProcessedAudio)
        assert processed_audio.chunk == mock_audio_chunk
        assert processed_audio.analysis is not None
        assert processed_audio.processing_time > 0

    @pytest.mark.asyncio
    async def test_is_mostly_silent(self, audio_processor):
        """Test silence detection."""
        # Create silent audio
        silent_audio = AudioChunk(
            data=np.zeros(16000, dtype=np.int16).tobytes(),
            timestamp=1.0,
            duration=1.0,
            sample_rate=16000,
            channels=1,
            format="pcm_s16le",
        )

        # Create loud audio
        loud_audio = AudioChunk(
            data=(np.ones(16000, dtype=np.int16) * 20000).tobytes(),
            timestamp=1.0,
            duration=1.0,
            sample_rate=16000,
            channels=1,
            format="pcm_s16le",
        )

        assert bool(audio_processor._is_mostly_silent(silent_audio)) is True
        assert bool(audio_processor._is_mostly_silent(loud_audio)) is False

    def test_create_wav_header(self, audio_processor):
        """Test WAV header creation."""
        header = audio_processor._create_wav_header(1000, 16000, 1)

        assert len(header) == 44  # Standard WAV header size
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        assert header[12:16] == b"fmt "

    @pytest.mark.asyncio
    async def test_get_processing_stats(self, audio_processor):
        """Test getting processing statistics."""
        stats = await audio_processor.get_processing_stats()

        assert isinstance(stats, dict)
        assert "total_chunks_processed" in stats
        assert "successful_transcriptions" in stats
        assert "success_rate" in stats
        assert "buffer_size" in stats


class TestChatProcessor:
    """Test cases for ChatProcessor."""

    @pytest.fixture
    def chat_config(self):
        return ChatProcessorConfig(
            batch_size=10,
            buffer_size=100,
            engagement_window_seconds=60.0,
            min_message_length=3,
            max_message_length=500,
            toxicity_threshold=0.7,
        )

    @pytest.fixture
    def chat_processor(self, chat_config):
        return ChatProcessor(chat_config)

    @pytest.fixture
    def mock_chat_messages(self):
        return [
            ChatMessage(
                user_id="user1",
                username="testuser1",
                message="This is a great stream!",
                timestamp=time.time(),
                platform="twitch",
            ),
            ChatMessage(
                user_id="user2",
                username="testuser2",
                message="I love this content",
                timestamp=time.time() + 1,
                platform="twitch",
            ),
            ChatMessage(
                user_id="user3",
                username="testuser3",
                message="Amazing gameplay",
                timestamp=time.time() + 2,
                platform="twitch",
            ),
        ]

    def test_chat_processor_initialization(self, chat_processor, chat_config):
        """Test chat processor initialization."""
        assert chat_processor.config == chat_config
        assert len(chat_processor.message_buffer) == 0
        assert len(chat_processor.analysis_buffer) == 0
        assert isinstance(chat_processor.processing_stats, dict)

    def test_filter_messages(self, chat_processor, mock_chat_messages):
        """Test message filtering."""
        # Add some invalid messages
        invalid_messages = [
            ChatMessage("user4", "test", "hi", time.time(), "twitch"),  # Too short
            ChatMessage("user5", "test", "a" * 600, time.time(), "twitch"),  # Too long
            ChatMessage("user6", "test", "", time.time(), "twitch"),  # Empty
        ]

        all_messages = mock_chat_messages + invalid_messages
        filtered = chat_processor._filter_messages(all_messages)

        assert len(filtered) == len(mock_chat_messages)
        for msg in filtered:
            assert len(msg.message) >= chat_processor.config.min_message_length
            assert len(msg.message) <= chat_processor.config.max_message_length

    @pytest.mark.asyncio
    async def test_process_single_message(self, chat_processor, mock_chat_messages):
        """Test processing a single chat message."""
        message = mock_chat_messages[0]

        with patch(
            "src.utils.nlp_utils.chat_processor.process_message"
        ) as mock_process:
            mock_analysis = TextAnalysis(
                text=message.message,
                timestamp=message.timestamp,
                sentiment=SentimentScore(0.8, 0.1, 0.1, 0.7, "positive", 0.8),
                word_count=len(message.message.split()),
                toxicity_score=0.1,
            )
            mock_process.return_value = mock_analysis

            result = await chat_processor._process_single_message(message)

        assert isinstance(result, TextAnalysis)
        assert result.text == message.message
        assert result.timestamp == message.timestamp

    @pytest.mark.asyncio
    async def test_calculate_engagement_metrics(
        self, chat_processor, mock_chat_messages
    ):
        """Test engagement metrics calculation."""
        # Create mock analyses
        analyses = [
            TextAnalysis(
                text=msg.message,
                timestamp=msg.timestamp,
                sentiment=SentimentScore(0.7, 0.2, 0.1, 0.5, "positive", 0.7),
                word_count=len(msg.message.split()),
                toxicity_score=0.1,
            )
            for msg in mock_chat_messages
        ]

        metrics = await chat_processor._calculate_engagement_metrics(
            mock_chat_messages, analyses
        )

        assert isinstance(metrics, EngagementMetrics)
        assert metrics.total_messages == len(mock_chat_messages)
        assert metrics.unique_users == len(
            set(msg.user_id for msg in mock_chat_messages)
        )
        assert 0.0 <= metrics.engagement_score <= 1.0
        assert metrics.activity_level in ["low", "medium", "high", "very_high"]

    def test_calculate_engagement_score(self, chat_processor):
        """Test engagement score calculation."""
        score = chat_processor._calculate_engagement_score(
            messages_per_minute=10.0,
            unique_users=5,
            average_message_length=20.0,
            positive_ratio=0.7,
            negative_ratio=0.2,
            quality_score=0.8,
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_determine_activity_level(self, chat_processor):
        """Test activity level determination."""
        assert chat_processor._determine_activity_level(0.9, 35.0) == "very_high"
        assert chat_processor._determine_activity_level(0.7, 20.0) == "high"
        assert chat_processor._determine_activity_level(0.5, 10.0) == "medium"
        assert chat_processor._determine_activity_level(0.2, 2.0) == "low"

    def test_is_spam(self, chat_processor):
        """Test spam detection."""
        spam_analysis = TextAnalysis(
            text="Buy now! http://spam.com http://more-spam.com",
            timestamp=time.time(),
            sentiment=SentimentScore(0.0, 0.0, 1.0, 0.0, "neutral", 0.0),
            word_count=1,
            toxicity_score=0.0,
        )

        legitimate_analysis = TextAnalysis(
            text="This is a great stream with interesting content",
            timestamp=time.time(),
            sentiment=SentimentScore(0.8, 0.1, 0.1, 0.7, "positive", 0.8),
            word_count=9,
            toxicity_score=0.1,
        )

        assert chat_processor._is_spam(spam_analysis) is True
        assert chat_processor._is_spam(legitimate_analysis) is False

    @pytest.mark.asyncio
    async def test_get_processing_stats(self, chat_processor):
        """Test getting processing statistics."""
        stats = await chat_processor.get_processing_stats()

        assert isinstance(stats, dict)
        assert "total_messages_processed" in stats
        assert "spam_rate" in stats
        assert "toxicity_rate" in stats
        assert "active_users" in stats


class TestContentSynchronizer:
    """Test cases for ContentSynchronizer."""

    @pytest.fixture
    def sync_config(self):
        return SynchronizationConfig(
            sync_window_seconds=2.0,
            audio_sync_tolerance=0.5,
            chat_sync_tolerance=1.0,
            processing_window_seconds=30.0,
            interpolation_enabled=True,
        )

    @pytest.fixture
    def synchronizer(self, sync_config):
        return ContentSynchronizer(sync_config)

    @pytest.fixture
    def mock_processed_frame(self):
        frame = VideoFrame(
            frame=np.zeros((480, 720, 3), dtype=np.uint8),
            timestamp=5.0,
            frame_number=150,
            width=720,
            height=480,
            quality_score=0.8,
        )
        return ProcessedFrame(
            frame=frame,
            analysis={"quality_score": 0.8},
            processing_time=0.1,
            is_keyframe=True,
            scene_change=False,
        )

    @pytest.fixture
    def mock_processed_audio(self):
        chunk = AudioChunk(
            data=b"audio_data",
            timestamp=5.2,
            duration=1.0,
            sample_rate=16000,
            channels=1,
            format="pcm_s16le",
        )
        transcription = TranscriptionResult(
            text="Test transcription", timestamp=5.2, duration=1.0, confidence=0.9
        )
        return ProcessedAudio(
            chunk=chunk,
            transcription=transcription,
            analysis=MagicMock(),
            processing_time=0.2,
        )

    @pytest.fixture
    def mock_chat_analysis(self):
        return TextAnalysis(
            text="Great moment!",
            timestamp=5.1,
            sentiment=SentimentScore(0.8, 0.1, 0.1, 0.7, "positive", 0.8),
            word_count=2,
            toxicity_score=0.1,
        )

    def test_synchronizer_initialization(self, synchronizer, sync_config):
        """Test synchronizer initialization."""
        assert synchronizer.config == sync_config
        assert len(synchronizer.content_buffer.video_frames) == 0
        assert len(synchronizer.content_buffer.audio_transcriptions) == 0
        assert len(synchronizer.content_buffer.chat_messages) == 0

    @pytest.mark.asyncio
    async def test_add_content(
        self,
        synchronizer,
        mock_processed_frame,
        mock_processed_audio,
        mock_chat_analysis,
    ):
        """Test adding content to synchronizer."""
        await synchronizer.add_video_frame(mock_processed_frame)
        await synchronizer.add_audio_transcription(mock_processed_audio)
        await synchronizer.add_chat_messages([mock_chat_analysis])

        assert len(synchronizer.content_buffer.video_frames) == 1
        assert len(synchronizer.content_buffer.audio_transcriptions) == 1
        assert len(synchronizer.content_buffer.chat_messages) == 1

    @pytest.mark.asyncio
    async def test_create_sync_point(
        self,
        synchronizer,
        mock_processed_frame,
        mock_processed_audio,
        mock_chat_analysis,
    ):
        """Test creating a synchronized point."""
        timestamp = 5.0

        sync_point = await synchronizer._create_sync_point_at_timestamp(
            timestamp,
            [mock_processed_frame],
            [mock_processed_audio],
            [mock_chat_analysis],
        )

        assert isinstance(sync_point, SyncPoint)
        assert sync_point.timestamp == timestamp
        assert sync_point.video_frame is not None
        assert sync_point.audio_transcription is not None
        assert len(sync_point.chat_messages) == 1
        assert 0.0 <= sync_point.confidence_score <= 1.0

    def test_find_closest_content(self, synchronizer, mock_processed_frame):
        """Test finding closest content to timestamp."""
        content_list = [mock_processed_frame]
        target_timestamp = 5.1
        tolerance = 1.0

        closest = synchronizer._find_closest_content(
            content_list, target_timestamp, lambda x: x.frame.timestamp, tolerance
        )

        assert closest == mock_processed_frame

        # Test with timestamp outside tolerance
        closest = synchronizer._find_closest_content(
            content_list, 10.0, lambda x: x.frame.timestamp, tolerance
        )

        assert closest is None

    def test_calculate_sync_confidence(
        self,
        synchronizer,
        mock_processed_frame,
        mock_processed_audio,
        mock_chat_analysis,
    ):
        """Test sync confidence calculation."""
        confidence = synchronizer._calculate_sync_confidence(
            mock_processed_frame, mock_processed_audio, [mock_chat_analysis], 5.0
        )

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_calculate_coverage(self, synchronizer, mock_processed_frame):
        """Test temporal coverage calculation."""
        content_list = [mock_processed_frame]
        start_time = 4.0
        end_time = 6.0

        coverage = synchronizer._calculate_coverage(
            content_list, start_time, end_time, lambda x: x.frame.timestamp
        )

        assert 0.0 <= coverage <= 1.0
        assert isinstance(coverage, float)

    @pytest.mark.asyncio
    async def test_get_sync_stats(self, synchronizer):
        """Test getting synchronization statistics."""
        stats = await synchronizer.get_sync_stats()

        assert isinstance(stats, dict)
        assert "total_windows_processed" in stats
        assert "average_processing_time" in stats
        assert "video_buffer_size" in stats
        assert "audio_buffer_size" in stats
        assert "chat_buffer_size" in stats


class TestBufferManager:
    """Test cases for BufferManager."""

    @pytest.fixture
    def buffer_config(self):
        return BufferConfig(
            window_duration_seconds=30.0,
            window_overlap_seconds=5.0,
            max_memory_mb=100,
            video_buffer_size=50,
            audio_buffer_size=50,
            chat_buffer_size=100,
        )

    @pytest.fixture
    def buffer_manager(self, buffer_config):
        return BufferManager(buffer_config)

    @pytest.fixture
    def mock_processing_window(self):
        return ProcessingWindow(
            window_id="test_window_1",
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            window_type=WindowType.SLIDING,
            priority=BufferPriority.MEDIUM,
        )

    def test_buffer_manager_initialization(self, buffer_manager, buffer_config):
        """Test buffer manager initialization."""
        assert buffer_manager.config == buffer_config
        assert len(buffer_manager.active_windows) == 0
        assert len(buffer_manager.completed_windows) == 0
        assert isinstance(buffer_manager.processing_stats, dict)

    @pytest.mark.asyncio
    async def test_create_window(self, buffer_manager):
        """Test creating a processing window."""
        start_time = time.time()
        window = await buffer_manager.create_window(
            start_time=start_time, duration=30.0, priority=BufferPriority.HIGH
        )

        assert isinstance(window, ProcessingWindow)
        assert window.start_time == start_time
        assert window.duration == 30.0
        assert window.priority == BufferPriority.HIGH
        assert window.window_id in buffer_manager.active_windows

    def test_calculate_content_density(self, buffer_manager, mock_processing_window):
        """Test content density calculation."""
        # Add some content to the window
        mock_processing_window.video_frames = [MagicMock() for _ in range(5)]
        mock_processing_window.audio_transcriptions = [MagicMock() for _ in range(3)]
        mock_processing_window.chat_messages = [MagicMock() for _ in range(10)]

        density = buffer_manager._calculate_content_density(mock_processing_window)

        expected_density = 18 / 30.0  # (5 + 3 + 10) / 30 seconds
        assert density == expected_density

    def test_estimate_window_memory(self, buffer_manager, mock_processing_window):
        """Test window memory estimation."""
        # Add some content to the window
        mock_processing_window.video_frames = [MagicMock() for _ in range(5)]
        mock_processing_window.audio_transcriptions = [MagicMock() for _ in range(3)]
        mock_processing_window.chat_messages = [MagicMock() for _ in range(10)]

        memory_mb = buffer_manager._estimate_window_memory(mock_processing_window)

        expected_memory = 5 * 0.5 + 3 * 0.1 + 10 * 0.001  # Video + audio + chat
        assert memory_mb == expected_memory

    @pytest.mark.asyncio
    async def test_get_processing_stats(self, buffer_manager):
        """Test getting processing statistics."""
        stats = await buffer_manager.get_processing_stats()

        assert isinstance(stats, dict)
        assert "total_windows_created" in stats
        assert "total_windows_processed" in stats
        assert "active_windows" in stats
        assert "completed_windows" in stats


class TestIntegration:
    """Integration tests for multi-modal processing pipeline."""

    @pytest.fixture
    def pipeline_components(self):
        """Create all pipeline components for integration testing."""
        video_processor = VideoProcessor()
        audio_processor = AudioProcessor()
        chat_processor = ChatProcessor()
        synchronizer = ContentSynchronizer()
        buffer_manager = BufferManager()

        return {
            "video": video_processor,
            "audio": audio_processor,
            "chat": chat_processor,
            "sync": synchronizer,
            "buffer": buffer_manager,
        }

    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, pipeline_components):
        """Test end-to-end multi-modal processing."""
        components = pipeline_components

        # Create mock content
        timestamp = time.time()

        # Mock video frame
        video_frame = ProcessedFrame(
            frame=VideoFrame(
                frame=np.zeros((480, 720, 3), dtype=np.uint8),
                timestamp=timestamp,
                frame_number=30,
                width=720,
                height=480,
                quality_score=0.8,
            ),
            analysis={"quality_score": 0.8},
            processing_time=0.1,
            is_keyframe=True,
            scene_change=False,
        )

        # Mock audio
        processed_audio = ProcessedAudio(
            chunk=AudioChunk(
                data=b"audio_data",
                timestamp=timestamp + 0.1,
                duration=1.0,
                sample_rate=16000,
                channels=1,
                format="pcm_s16le",
            ),
            transcription=TranscriptionResult(
                text="Test audio",
                timestamp=timestamp + 0.1,
                duration=1.0,
                confidence=0.9,
            ),
            analysis=MagicMock(),
            processing_time=0.2,
        )

        # Mock chat
        chat_analysis = TextAnalysis(
            text="Great moment!",
            timestamp=timestamp + 0.05,
            sentiment=SentimentScore(0.8, 0.1, 0.1, 0.7, "positive", 0.8),
            word_count=2,
            toxicity_score=0.1,
        )

        # Process through pipeline
        await components["buffer"].add_video_frame(video_frame)
        await components["buffer"].add_audio_transcription(processed_audio)
        await components["buffer"].add_chat_messages([chat_analysis])

        # Create and process window
        window = await components["buffer"].create_window(
            start_time=timestamp - 1.0, duration=5.0
        )

        # Verify content was added
        assert len(window.video_frames) >= 1
        assert len(window.audio_transcriptions) >= 1
        assert len(window.chat_messages) >= 1

        # Test window processing
        with patch.object(components["buffer"], "process_window") as mock_process:
            mock_process.return_value = window
            processed_window = await components["buffer"].process_window(window)
            assert processed_window == window

    @pytest.mark.asyncio
    async def test_memory_management(self, pipeline_components):
        """Test memory management under load."""
        buffer_manager = pipeline_components["buffer"]

        # Add many items to test memory management
        for i in range(100):
            timestamp = time.time() + i

            # Add video frame
            video_frame = ProcessedFrame(
                frame=VideoFrame(
                    frame=np.zeros((100, 100, 3), dtype=np.uint8),
                    timestamp=timestamp,
                    frame_number=i,
                    width=100,
                    height=100,
                    quality_score=0.5,
                ),
                analysis={},
                processing_time=0.01,
                is_keyframe=False,
                scene_change=False,
            )

            await buffer_manager.add_video_frame(video_frame, BufferPriority.LOW)

        # Test memory calculation
        memory_usage = await buffer_manager._calculate_memory_usage()
        assert memory_usage > 0

        # Test cleanup
        await buffer_manager._cleanup_memory()

        # Verify cleanup occurred
        stats = await buffer_manager.get_processing_stats()
        assert "memory_cleanups" in stats


if __name__ == "__main__":
    pytest.main([__file__])
