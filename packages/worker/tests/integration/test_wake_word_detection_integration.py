"""Integration tests for wake word detection pipeline."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
import tempfile
import wave
import numpy as np
from datetime import datetime, timezone

from worker.services.ffmpeg_processor import FFmpegProcessor, FFmpegConfig
from worker.tasks.wake_word_detection import (
    detect_wake_words_task,
    _check_wake_word_in_transcript,
)
from worker.tasks.stream_processing import process_stream_task
from shared.domain.models.wake_word import WakeWord
from shared.domain.models.stream import Stream, StreamStatus


@pytest.mark.integration
class TestWakeWordDetectionIntegration:
    """Integration tests for complete wake word detection flow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def create_test_audio(self, temp_dir):
        """Create test audio files."""
        def _create_audio(filename, duration_seconds=30, sample_rate=16000):
            """Create a test WAV file with silence."""
            filepath = temp_dir / filename
            
            # Generate silence
            num_samples = int(duration_seconds * sample_rate)
            samples = np.zeros(num_samples, dtype=np.int16)
            
            # Write WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            
            return filepath
        
        return _create_audio
    
    @pytest.fixture
    def mock_stream(self):
        """Create mock stream for testing."""
        return Stream(
            id=uuid4(),
            organization_id=uuid4(),
            url="https://example.com/stream.m3u8",
            status=StreamStatus.PENDING,
        )
    
    @pytest.fixture
    def mock_wake_words(self):
        """Create mock wake words for testing."""
        org_id = uuid4()
        return [
            WakeWord(
                id=uuid4(),
                organization_id=org_id,
                phrase="hey assistant",
                max_edit_distance=2,
                similarity_threshold=0.8,
                pre_roll_seconds=5,
                post_roll_seconds=10,
            ),
            WakeWord(
                id=uuid4(),
                organization_id=org_id,
                phrase="start recording",
                max_edit_distance=1,
                similarity_threshold=0.9,
                cooldown_seconds=60,
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_full_pipeline_with_wake_word_detection(
        self, temp_dir, create_test_audio, mock_stream, mock_wake_words
    ):
        """Test complete pipeline from stream to wake word detection."""
        # Create test video with audio
        video_path = create_test_audio("test_video.mp4", duration_seconds=150)
        
        # Configure processor
        config = FFmpegConfig(
            segment_duration=60,  # 1-minute video segments
            audio_segment_duration=30,  # 30-second audio chunks
            audio_overlap=5,
            video_ring_buffer_size=3,
            audio_ring_buffer_size=10,
        )
        
        processor = FFmpegProcessor(
            stream_url=str(video_path),
            output_dir=temp_dir,
            config=config,
        )
        
        # Track processed segments
        processed_segments = []
        detected_wake_words = []
        
        # Mock handlers
        class TestHandler:
            async def handle_segment(self, segment):
                processed_segments.append(segment)
        
        handler = TestHandler()
        
        # Mock database and repositories
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Mock wake word repository
            with patch("worker.tasks.wake_word_detection.WakeWordRepository") as mock_repo:
                mock_repo.return_value.get_active_by_organization.return_value = mock_wake_words
                mock_repo.return_value.get_by_id.side_effect = lambda id: next(
                    (w for w in mock_wake_words if w.id == id), None
                )
                
                # Mock Whisper transcription
                with patch("worker.tasks.wake_word_detection.WhisperModel") as mock_whisper:
                    # Simulate transcription results
                    mock_model = Mock()
                    mock_segments = [
                        Mock(
                            text=" hey assistant please help me ",
                            words=[
                                Mock(word="hey", start=5.0, end=5.3),
                                Mock(word="assistant", start=5.4, end=6.0),
                                Mock(word="please", start=6.1, end=6.5),
                                Mock(word="help", start=6.6, end=6.9),
                                Mock(word="me", start=7.0, end=7.2),
                            ]
                        ),
                        Mock(
                            text=" okay start recording now ",
                            words=[
                                Mock(word="okay", start=15.0, end=15.3),
                                Mock(word="start", start=15.4, end=15.8),
                                Mock(word="recording", start=15.9, end=16.5),
                                Mock(word="now", start=16.6, end=16.9),
                            ]
                        )
                    ]
                    mock_info = Mock(language="en", language_probability=0.99)
                    mock_model.transcribe.return_value = (mock_segments, mock_info)
                    mock_whisper.return_value = mock_model
                    
                    # Process stream with wake word detection
                    async with processor:
                        segment_count = 0
                        async for segment in processor.process_stream(handler):
                            segment_count += 1
                            
                            # Simulate wake word detection for audio chunks
                            for audio_chunk in segment.audio_chunks:
                                # Import the async function directly
                                from worker.tasks.wake_word_detection import _detect_wake_words_async, WakeWordDetectionTask
                                
                                # Create a mock task instance
                                mock_task = WakeWordDetectionTask()
                                mock_task._whisper_model = mock_model
                                
                                result = await _detect_wake_words_async(
                                    mock_task,
                                    str(mock_stream.id),
                                    {
                                        "id": str(audio_chunk.chunk_id),
                                        "path": str(audio_chunk.path),
                                        "start_time": audio_chunk.start_time,
                                        "end_time": audio_chunk.end_time,
                                        "video_segment_number": segment.segment_number,
                                    },
                                    str(mock_wake_words[0].organization_id),
                                )
                                
                                if result["detected"]:
                                    detected_wake_words.extend(result["detected"])
                            
                            # Process first 2 segments only for test speed
                            if segment_count >= 2:
                                break
        
        # Verify results
        assert len(processed_segments) >= 2
        assert len(detected_wake_words) >= 2  # Should detect both wake words
        
        # Check wake word detections
        detected_phrases = {d["wake_word_phrase"] for d in detected_wake_words}
        assert "hey assistant" in detected_phrases
        assert "start recording" in detected_phrases
    
    @pytest.mark.asyncio
    async def test_wake_word_at_chunk_boundary(self, temp_dir, create_test_audio):
        """Test wake word detection at audio chunk boundaries."""
        # This tests the overlap functionality
        audio_path = create_test_audio("boundary_test.wav", duration_seconds=35)
        
        config = FFmpegConfig(
            audio_segment_duration=30,
            audio_overlap=5,
        )
        
        # Mock transcription where wake word spans chunks
        with patch("worker.tasks.wake_word_detection.WhisperModel") as mock_whisper:
            mock_model = Mock()
            
            # First chunk ends with "hey"
            chunk1_segments = [Mock(
                text=" ... conversation hey ",
                words=[
                    Mock(word="conversation", start=27.0, end=28.0),
                    Mock(word="hey", start=29.5, end=29.8),
                ]
            )]
            
            # Second chunk starts with "assistant" (in overlap region)
            chunk2_segments = [Mock(
                text=" hey assistant please ",
                words=[
                    Mock(word="hey", start=29.5, end=29.8),
                    Mock(word="assistant", start=30.0, end=30.6),
                    Mock(word="please", start=30.7, end=31.0),
                ]
            )]
            
            mock_model.transcribe.side_effect = [
                (chunk1_segments, Mock(language="en", language_probability=0.99)),
                (chunk2_segments, Mock(language="en", language_probability=0.99)),
            ]
            
            # Should detect wake word in second chunk due to overlap
            # Test implementation would verify this
    
    @pytest.mark.asyncio
    async def test_cooldown_period_enforcement(self, mock_wake_words):
        """Test that cooldown periods are properly enforced."""
        wake_word = mock_wake_words[1]  # "start recording" with 60s cooldown
        wake_word.last_triggered_at = datetime.now(timezone.utc)
        
        # Mock rapid detections
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            with patch("worker.tasks.wake_word_detection.WakeWordRepository") as mock_repo:
                mock_repo.return_value.get_active_by_organization.return_value = [wake_word]
                mock_repo.return_value.get_by_id.return_value = wake_word
                
                # First detection should update
                assert wake_word.can_trigger is False  # Just triggered
                
                # Simulate time passing
                wake_word.last_triggered_at = datetime.now(timezone.utc) - timedelta(seconds=61)
                assert wake_word.can_trigger is True  # Cooldown expired
    
    @pytest.mark.asyncio
    async def test_memory_management_with_ring_buffers(self, temp_dir, create_test_audio):
        """Test that ring buffers prevent memory issues with long streams."""
        # Create a long stream
        video_path = create_test_audio("long_stream.mp4", duration_seconds=600)  # 10 minutes
        
        config = FFmpegConfig(
            segment_duration=60,
            audio_segment_duration=30,
            video_ring_buffer_size=5,  # Keep only 5 video segments
            audio_ring_buffer_size=20,  # Keep only 20 audio chunks
        )
        
        processor = FFmpegProcessor(
            stream_url=str(video_path),
            output_dir=temp_dir,
            config=config,
        )
        
        # Track memory usage
        max_video_segments = 0
        max_audio_chunks = 0
        
        class MemoryTrackingHandler:
            async def handle_segment(self, segment):
                nonlocal max_video_segments, max_audio_chunks
                max_video_segments = max(max_video_segments, len(processor._video_ring_buffer))
                max_audio_chunks = max(max_audio_chunks, len(processor._audio_ring_buffer))
        
        handler = MemoryTrackingHandler()
        
        # Process stream
        async with processor:
            segment_count = 0
            async for segment in processor.process_stream(handler):
                segment_count += 1
                if segment_count >= 10:  # Process 10 segments
                    break
        
        # Verify ring buffers maintained size limits
        assert max_video_segments <= config.video_ring_buffer_size
        assert max_audio_chunks <= config.audio_ring_buffer_size
    
    @pytest.mark.asyncio
    async def test_error_recovery_in_pipeline(self, temp_dir, mock_stream):
        """Test error recovery in the detection pipeline."""
        # Test FFmpeg failure recovery
        config = FFmpegConfig(reconnect_attempts=3)
        processor = FFmpegProcessor(
            stream_url="invalid://stream",
            output_dir=temp_dir,
            config=config,
        )
        
        # Should handle invalid stream gracefully
        with pytest.raises(Exception):
            async with processor:
                pass
        
        # Test transcription failure recovery
        with patch("worker.tasks.wake_word_detection.WhisperModel") as mock_whisper:
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Transcription failed")
            
            # Task should retry with backoff
            task = detect_wake_words_task
            task._whisper_model = mock_model
            task.request = Mock(retries=0)
            task.retry = Mock(side_effect=Exception("Retry"))
            
            with pytest.raises(Exception):
                # Simulate task execution with retry
                try:
                    result = detect_wake_words_task(
                        task,
                        stream_id=str(mock_stream.id),
                        audio_chunk={"id": "test", "path": "/fake/path"},
                        organization_id=str(uuid4()),
                    )
                except Exception:
                    raise
            
            task.retry.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_stream_processing(self, temp_dir, create_test_audio):
        """Test processing multiple streams concurrently."""
        # Create multiple test streams
        streams = []
        for i in range(3):
            video_path = create_test_audio(f"stream_{i}.mp4", duration_seconds=60)
            processor = FFmpegProcessor(
                stream_url=str(video_path),
                output_dir=temp_dir / f"stream_{i}",
                config=FFmpegConfig(
                    segment_duration=30,
                    audio_segment_duration=15,
                ),
            )
            streams.append(processor)
        
        # Process all streams concurrently
        async def process_single_stream(processor, stream_id):
            segments = []
            handler = Mock()
            handler.handle_segment = AsyncMock()
            
            async with processor:
                async for segment in processor.process_stream(handler):
                    segments.append(segment)
                    if len(segments) >= 2:
                        break
            
            return stream_id, segments
        
        # Run all streams concurrently
        results = await asyncio.gather(*[
            process_single_stream(proc, i) for i, proc in enumerate(streams)
        ])
        
        # Verify all streams processed
        assert len(results) == 3
        for stream_id, segments in results:
            assert len(segments) >= 2
            # Each segment should have audio chunks
            for segment in segments:
                assert len(segment.audio_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_wake_word_variations(self):
        """Test detection of wake word variations."""
        test_cases = [
            # (wake_phrase, transcript, should_match, reason)
            ("hey assistant", "hey assistant", True, "Exact match"),
            ("hey assistant", "HEY ASSISTANT", True, "Case insensitive"),
            ("hey assistant", "hey assistent", True, "Fuzzy match"),
            ("hey assistant", "hey there assistant", False, "Extra word"),
            ("activate system", "activate the system", False, "Extra word in phrase"),
            ("ok google", "okay google", True, "Common variation"),
        ]
        
        for wake_phrase, transcript, should_match, reason in test_cases:
            wake_word = WakeWord(
                phrase=wake_phrase,
                max_edit_distance=2,
                similarity_threshold=0.8,
            )
            
            result = _check_wake_word_in_transcript(
                wake_word, transcript, [], 0.0
            )
            
            if should_match:
                assert result is not None, f"Failed: {reason}"
            else:
                assert result is None, f"Failed: {reason}"


@pytest.mark.integration
class TestWakeWordDetectionPerformance:
    """Performance tests for wake word detection."""
    
    @pytest.mark.asyncio
    async def test_large_transcript_performance(self):
        """Test performance with large transcripts."""
        # Create a very long transcript
        long_transcript = " ".join(["random word"] * 1000)  # 2000 words
        long_transcript += " hey assistant "  # Wake word in the middle
        long_transcript += " ".join(["more words"] * 1000)
        
        wake_word = WakeWord(
            phrase="hey assistant",
            max_edit_distance=1,
            similarity_threshold=0.9,
        )
        
        # Should find wake word efficiently
        import time
        start_time = time.time()
        
        result = _check_wake_word_in_transcript(
            wake_word, long_transcript, [], 0.0
        )
        
        end_time = time.time()
        
        assert result is not None
        assert end_time - start_time < 1.0  # Should complete in under 1 second
    
    @pytest.mark.asyncio
    async def test_many_wake_words_performance(self):
        """Test performance with many wake words."""
        # Create many wake words
        wake_words = []
        for i in range(100):
            wake_words.append(WakeWord(
                phrase=f"wake phrase {i}",
                organization_id=uuid4(),
            ))
        
        transcript = "testing with wake phrase 50 in the middle"
        
        # Check all wake words
        matches = []
        for wake_word in wake_words:
            result = _check_wake_word_in_transcript(
                wake_word, transcript, [], 0.0
            )
            if result:
                matches.append(result)
        
        assert len(matches) == 1
        assert matches[0]["wake_word_phrase"] == "wake phrase 50"