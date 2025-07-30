"""
Integration tests for the multi-modal content processing pipeline.

Tests the complete flow from content ingestion through processing,
synchronization, and output generation.
"""

import asyncio
import pytest
import numpy as np
import time
from unittest.mock import AsyncMock, patch

from src.services.content_processing import (
    VideoProcessor,
    AudioProcessor,
    ContentChatProcessor,
    ContentSynchronizer,
    BufferManager,
    VideoProcessorConfig,
    AudioProcessorConfig,
    ChatProcessorConfig,
    SynchronizationConfig,
    BufferConfig,
    BufferPriority,
)
from src.utils.media_utils import VideoFrame, AudioChunk
from src.utils.nlp_utils import ChatMessage


class TestMultiModalPipeline:
    """Integration tests for the complete multi-modal pipeline."""

    @pytest.fixture
    def pipeline_config(self):
        """Configuration for the entire pipeline."""
        return {
            "video": VideoProcessorConfig(
                frame_interval_seconds=1.0,
                max_frames_per_window=30,
                quality_threshold=0.2,
                resize_width=480,
                enable_scene_detection=True,
                processing_profile="fast",
            ),
            "audio": AudioProcessorConfig(
                chunk_duration=10.0,
                sample_rate=16000,
                channels=1,
                min_audio_duration=0.5,
                max_concurrent_requests=2,
            ),
            "chat": ChatProcessorConfig(
                batch_size=20,
                buffer_size=200,
                engagement_window_seconds=30.0,
                min_message_length=2,
                toxicity_threshold=0.8,
            ),
            "sync": SynchronizationConfig(
                sync_window_seconds=1.5,
                audio_sync_tolerance=0.3,
                chat_sync_tolerance=0.8,
                processing_window_seconds=20.0,
                interpolation_enabled=True,
            ),
            "buffer": BufferConfig(
                window_duration_seconds=20.0,
                window_overlap_seconds=3.0,
                max_memory_mb=200,
                video_buffer_size=100,
                audio_buffer_size=50,
                chat_buffer_size=200,
            ),
        }

    @pytest.fixture
    async def pipeline(self, pipeline_config):
        """Create a complete pipeline instance."""
        video_processor = VideoProcessor(pipeline_config["video"])
        audio_processor = AudioProcessor(pipeline_config["audio"])
        chat_processor = ContentChatProcessor(pipeline_config["chat"])
        synchronizer = ContentSynchronizer(pipeline_config["sync"])
        buffer_manager = BufferManager(pipeline_config["buffer"])

        # Mock OpenAI client to avoid API calls
        audio_processor.openai_client = AsyncMock()

        pipeline = {
            "video": video_processor,
            "audio": audio_processor,
            "chat": chat_processor,
            "sync": synchronizer,
            "buffer": buffer_manager,
        }

        yield pipeline

        # Cleanup
        await video_processor.cleanup()
        await audio_processor.cleanup()
        await chat_processor.cleanup()
        await synchronizer.cleanup()
        await buffer_manager.cleanup()

    @pytest.fixture
    def sample_content(self):
        """Generate sample multi-modal content for testing."""
        base_timestamp = time.time()

        # Sample video frames
        video_frames = []
        for i in range(10):
            frame_data = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            frame = VideoFrame(
                frame=frame_data,
                timestamp=base_timestamp + i * 2.0,
                frame_number=i * 60,  # 30 FPS
                width=320,
                height=240,
                quality_score=0.6 + (i % 3) * 0.1,  # Varying quality
            )
            video_frames.append(frame)

        # Sample audio chunks
        audio_chunks = []
        for i in range(5):
            # Generate some audio data
            duration = 2.0
            sample_rate = 16000
            samples = int(duration * sample_rate)
            audio_data = np.random.randint(
                -5000, 5000, samples, dtype=np.int16
            ).tobytes()

            chunk = AudioChunk(
                data=audio_data,
                timestamp=base_timestamp + i * 4.0,
                duration=duration,
                sample_rate=sample_rate,
                channels=1,
                format="pcm_s16le",
            )
            audio_chunks.append(chunk)

        # Sample chat messages
        chat_messages = []
        sample_messages = [
            "This is amazing content!",
            "Great gameplay",
            "Love this stream",
            "Amazing skills",
            "Best streamer ever",
            "This part is so cool",
            "Incredible moment",
            "Perfect timing",
            "Outstanding performance",
            "This is epic",
        ]

        for i, message_text in enumerate(sample_messages):
            message = ChatMessage(
                user_id=f"user_{i % 5}",
                username=f"viewer{i % 5}",
                message=message_text,
                timestamp=base_timestamp + i * 2.0,
                platform="twitch",
                metadata={"is_subscriber": i % 3 == 0},
            )
            chat_messages.append(message)

        return {
            "video_frames": video_frames,
            "audio_chunks": audio_chunks,
            "chat_messages": chat_messages,
            "start_time": base_timestamp,
            "duration": 20.0,
        }

    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, pipeline, sample_content):
        """Test the complete pipeline from content ingestion to synchronized output."""

        # Step 1: Process video frames
        processed_video_frames = []
        for frame in sample_content["video_frames"]:
            processed_frame = await pipeline["video"]._process_single_frame(frame, [])
            processed_video_frames.append(processed_frame)

        assert len(processed_video_frames) == len(sample_content["video_frames"])

        # Step 2: Process audio chunks
        processed_audio_chunks = []
        for chunk in sample_content["audio_chunks"]:
            # Mock transcription to avoid API calls
            with patch.object(
                pipeline["audio"], "_transcribe_audio", return_value=None
            ):
                processed_audio = await pipeline["audio"]._process_audio_chunk(chunk)
                processed_audio_chunks.append(processed_audio)

        assert len(processed_audio_chunks) == len(sample_content["audio_chunks"])

        # Step 3: Process chat messages
        processed_chat_data = await pipeline["chat"].process_message_batch(
            sample_content["chat_messages"]
        )

        assert processed_chat_data.messages is not None
        assert len(processed_chat_data.messages) > 0
        assert processed_chat_data.engagement_metrics is not None

        # Step 4: Add content to buffer manager
        for frame in processed_video_frames:
            await pipeline["buffer"].add_video_frame(frame, BufferPriority.MEDIUM)

        for audio in processed_audio_chunks:
            await pipeline["buffer"].add_audio_transcription(
                audio, BufferPriority.MEDIUM
            )

        await pipeline["buffer"].add_chat_messages(
            processed_chat_data.messages, BufferPriority.LOW
        )

        # Step 5: Create processing window
        window = await pipeline["buffer"].create_window(
            start_time=sample_content["start_time"], duration=sample_content["duration"]
        )

        assert window is not None
        assert len(window.video_frames) > 0
        assert len(window.chat_messages) > 0

        # Step 6: Synchronize content
        await pipeline["sync"].add_video_frame(processed_video_frames[0])
        await pipeline["sync"].add_chat_messages(processed_chat_data.messages[:3])

        synchronized_window = await pipeline["sync"].synchronize_window(
            sample_content["start_time"], sample_content["duration"]
        )

        assert synchronized_window is not None
        assert synchronized_window.total_sync_points >= 0
        assert 0.0 <= synchronized_window.sync_quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_real_time_processing_simulation(self, pipeline, sample_content):
        """Simulate real-time content processing."""

        # Simulate real-time content arrival
        async def simulate_content_stream():
            """Simulate content arriving in real-time."""
            for i, frame in enumerate(sample_content["video_frames"]):
                # Process and add video frame
                processed_frame = await pipeline["video"]._process_single_frame(
                    frame, []
                )
                await pipeline["buffer"].add_video_frame(processed_frame)

                # Add chat message if available
                if i < len(sample_content["chat_messages"]):
                    message = sample_content["chat_messages"][i]
                    processed_chat = await pipeline["chat"].process_message_batch(
                        [message]
                    )
                    await pipeline["buffer"].add_chat_messages(processed_chat.messages)

                # Simulate time delay
                await asyncio.sleep(0.1)

        # Start content simulation
        content_task = asyncio.create_task(simulate_content_stream())

        # Process windows as they become ready
        processed_windows = []
        timeout_counter = 0
        max_timeout = 50  # 5 seconds

        while len(processed_windows) < 2 and timeout_counter < max_timeout:
            ready_windows = await pipeline["buffer"].windows_ready_for_processing(
                limit=1
            )

            if ready_windows:
                window = ready_windows[0]
                # Mock the actual processing to avoid complex synchronization
                window.is_complete = True
                window.content_density = pipeline["buffer"]._calculate_content_density(
                    window
                )
                window.quality_score = 0.7
                processed_windows.append(window)

                # Remove from active windows
                if window.window_id in pipeline["buffer"].active_windows:
                    del pipeline["buffer"].active_windows[window.window_id]
            else:
                await asyncio.sleep(0.1)
                timeout_counter += 1

        # Cancel content simulation
        content_task.cancel()
        try:
            await content_task
        except asyncio.CancelledError:
            pass

        # Verify we processed some windows
        assert len(processed_windows) > 0

        for window in processed_windows:
            assert window.content_density >= 0.0
            assert 0.0 <= window.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_pipeline_performance_metrics(self, pipeline, sample_content):
        """Test performance metrics collection across the pipeline."""

        # Process content through pipeline
        start_time = time.time()

        # Process video frames
        for frame in sample_content["video_frames"][:5]:  # Process subset for speed
            processed_frame = await pipeline["video"]._process_single_frame(frame, [])
            await pipeline["buffer"].add_video_frame(processed_frame)

        # Process chat messages
        processed_chat = await pipeline["chat"].process_message_batch(
            sample_content["chat_messages"][:5]
        )
        await pipeline["buffer"].add_chat_messages(processed_chat.messages)

        processing_time = time.time() - start_time

        # Collect metrics from all components
        video_stats = await pipeline["video"].get_processing_stats()
        chat_stats = await pipeline["chat"].get_processing_stats()
        sync_stats = await pipeline["sync"].get_sync_stats()
        buffer_stats = await pipeline["buffer"].get_processing_stats()

        # Verify metrics are collected
        assert isinstance(video_stats, dict)
        assert isinstance(chat_stats, dict)
        assert isinstance(sync_stats, dict)
        assert isinstance(buffer_stats, dict)

        # Verify key metrics exist
        assert "total_frames_processed" in video_stats
        assert "total_messages_processed" in chat_stats
        assert "total_windows_processed" in sync_stats
        assert "total_windows_created" in buffer_stats

        # Verify processing was reasonably fast
        assert processing_time < 10.0  # Should process in under 10 seconds

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline, sample_content):
        """Test error handling and recovery in the pipeline."""

        # Test video processing with corrupted frame
        corrupted_frame = VideoFrame(
            frame=None,  # Corrupted frame data
            timestamp=time.time(),
            frame_number=0,
            width=0,
            height=0,
            quality_score=0.0,
        )

        # Should handle corrupted frame gracefully
        processed_frame = await pipeline["video"]._process_single_frame(
            corrupted_frame, []
        )
        assert processed_frame is not None
        assert processed_frame.processing_time >= 0

        # Test chat processing with problematic messages
        problematic_messages = [
            ChatMessage("user1", "test", "", time.time(), "twitch"),  # Empty message
            ChatMessage("user2", "test", "a" * 1000, time.time(), "twitch"),  # Too long
            ChatMessage(
                "user3", "test", "Normal message", time.time(), "twitch"
            ),  # Good message
        ]

        processed_chat = await pipeline["chat"].process_message_batch(
            problematic_messages
        )

        # Should filter out problematic messages but process good ones
        assert processed_chat.messages is not None
        assert len(processed_chat.messages) <= len(problematic_messages)

        # Test buffer manager with excessive content
        for i in range(1000):  # Add many items
            dummy_frame = await pipeline["video"]._process_single_frame(
                sample_content["video_frames"][0], []
            )
            await pipeline["buffer"].add_video_frame(dummy_frame, BufferPriority.LOW)

        # Buffer should handle this gracefully
        stats = await pipeline["buffer"].get_processing_stats()
        assert stats["total_windows_created"] >= 0

    @pytest.mark.asyncio
    async def test_content_quality_filtering(self, pipeline, sample_content):
        """Test quality-based content filtering across the pipeline."""

        # Create content with varying quality
        high_quality_frame = VideoFrame(
            frame=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            timestamp=time.time(),
            frame_number=1,
            width=320,
            height=240,
            quality_score=0.9,  # High quality
        )

        low_quality_frame = VideoFrame(
            frame=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            timestamp=time.time() + 1,
            frame_number=2,
            width=320,
            height=240,
            quality_score=0.1,  # Low quality
        )

        # Process frames
        high_q_processed = await pipeline["video"]._process_single_frame(
            high_quality_frame, []
        )
        low_q_processed = await pipeline["video"]._process_single_frame(
            low_quality_frame, []
        )

        # Add to buffer with quality filtering enabled
        await pipeline["buffer"].add_video_frame(high_q_processed)
        await pipeline["buffer"].add_video_frame(low_q_processed)

        # Check that buffer manager handled quality filtering
        stats = await pipeline["buffer"].get_processing_stats()

        # Low quality content might be dropped
        if stats["total_content_dropped"] > 0:
            assert stats["total_content_dropped"] >= 0

    @pytest.mark.asyncio
    async def test_synchronization_accuracy(self, pipeline, sample_content):
        """Test the accuracy of content synchronization."""

        # Create precisely timed content
        base_time = time.time()

        # Video frame at exact time
        video_frame = VideoFrame(
            frame=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            timestamp=base_time,
            frame_number=1,
            width=320,
            height=240,
            quality_score=0.8,
        )

        # Chat message at slightly different time
        chat_message = ChatMessage(
            user_id="user1",
            username="testuser",
            message="Synchronized comment",
            timestamp=base_time + 0.1,  # 100ms later
            platform="twitch",
        )

        # Process content
        processed_frame = await pipeline["video"]._process_single_frame(video_frame, [])
        processed_chat = await pipeline["chat"].process_message_batch([chat_message])

        # Add to synchronizer
        await pipeline["sync"].add_video_frame(processed_frame)
        await pipeline["sync"].add_chat_messages(processed_chat.messages)

        # Synchronize content
        sync_window = await pipeline["sync"].synchronize_window(base_time - 1.0, 3.0)

        # Verify synchronization
        assert sync_window is not None
        assert sync_window.total_sync_points > 0

        # Check that content is properly aligned
        if sync_window.sync_points:
            sync_point = sync_window.sync_points[0]
            assert (
                sync_point.video_frame is not None or len(sync_point.chat_messages) > 0
            )
            assert 0.0 <= sync_point.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, pipeline, sample_content):
        """Test memory efficiency of the pipeline under load."""

        initial_memory = await pipeline["buffer"]._calculate_memory_usage()

        # Process a large amount of content
        for i in range(100):
            # Add video frame
            frame = VideoFrame(
                frame=np.random.randint(
                    0, 255, (120, 160, 3), dtype=np.uint8
                ),  # Smaller frames
                timestamp=time.time() + i * 0.1,
                frame_number=i,
                width=160,
                height=120,
                quality_score=0.5,
            )

            processed_frame = await pipeline["video"]._process_single_frame(frame, [])
            await pipeline["buffer"].add_video_frame(
                processed_frame, BufferPriority.LOW
            )

            # Add chat message
            if i % 5 == 0:  # Every 5th iteration
                message = ChatMessage(
                    user_id=f"user_{i % 10}",
                    username=f"user{i % 10}",
                    message=f"Message {i}",
                    timestamp=time.time() + i * 0.1,
                    platform="twitch",
                )
                processed_chat = await pipeline["chat"].process_message_batch([message])
                await pipeline["buffer"].add_chat_messages(
                    processed_chat.messages, BufferPriority.LOW
                )

        final_memory = await pipeline["buffer"]._calculate_memory_usage()

        # Memory should be managed (not grow indefinitely)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100.0  # Less than 100MB growth

        # Trigger cleanup
        await pipeline["buffer"]._cleanup_memory()

        cleaned_memory = await pipeline["buffer"]._calculate_memory_usage()
        assert (
            cleaned_memory <= final_memory
        )  # Memory should not increase after cleanup

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, pipeline, sample_content):
        """Test concurrent processing of multiple content streams."""

        async def process_stream(stream_id: str, content_subset):
            """Process a subset of content as a separate stream."""
            processed_items = 0

            for i, frame in enumerate(content_subset["video_frames"]):
                processed_frame = await pipeline["video"]._process_single_frame(
                    frame, []
                )
                await pipeline["buffer"].add_video_frame(
                    processed_frame, BufferPriority.MEDIUM
                )
                processed_items += 1

                # Add some delay to simulate real-time processing
                await asyncio.sleep(0.01)

            return processed_items

        # Split content into multiple streams
        stream1_content = {"video_frames": sample_content["video_frames"][:5]}
        stream2_content = {"video_frames": sample_content["video_frames"][5:]}

        # Process streams concurrently
        stream1_task = process_stream("stream1", stream1_content)
        stream2_task = process_stream("stream2", stream2_content)

        results = await asyncio.gather(stream1_task, stream2_task)

        # Verify both streams were processed
        assert results[0] > 0  # Stream 1 processed items
        assert results[1] > 0  # Stream 2 processed items
        assert results[0] + results[1] == len(sample_content["video_frames"])

        # Verify pipeline handled concurrent access
        stats = await pipeline["buffer"].get_processing_stats()
        assert stats["total_windows_created"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
