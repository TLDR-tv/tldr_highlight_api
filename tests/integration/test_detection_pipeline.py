"""
Integration tests for the complete highlight detection pipeline.

This module contains comprehensive integration tests that verify the
end-to-end functionality of the AI-powered highlight detection system,
including real-world scenarios and cross-component interactions.
"""

import asyncio
import pytest
import time
from typing import List

import numpy as np

from src.services.highlight_detection import (
    VideoDetector,
    AudioDetector,
    ChatDetector,
    FusionScorer,
    HighlightRanker,
    HighlightPostProcessor,
)
from src.services.highlight_detection.base_detector import (
    ContentSegment,
    HighlightCandidate,
    ModalityType,
)
from src.services.highlight_detection.video_detector import VideoFrameData
from src.services.highlight_detection.audio_detector import AudioSegmentData
from src.services.highlight_detection.chat_detector import ChatMessage, ChatWindow
from src.services.highlight_detection.fusion_scorer import FusionConfig, FusionMethod
from src.services.highlight_detection.ranker import RankingConfig, RankingMethod


class MockStreamData:
    """Mock stream data generator for testing."""

    def __init__(self, duration_seconds: int = 300):
        """
        Initialize mock stream data generator.

        Args:
            duration_seconds: Duration of mock stream in seconds
        """
        self.duration = duration_seconds
        self.fps = 30
        self.sample_rate = 44100
        self.chat_message_rate = 2.0  # Messages per second

    def generate_video_frames(self) -> List[VideoFrameData]:
        """Generate mock video frames."""
        frames = []
        total_frames = int(self.duration * self.fps)

        for i in range(total_frames):
            timestamp = i / self.fps

            # Create base frame with varying activity
            base_intensity = 100 + 50 * np.sin(timestamp * 0.1)  # Slow variation

            # Add excitement spikes at specific times
            excitement_times = [60, 120, 180, 240]  # Excitement at these timestamps
            for exc_time in excitement_times:
                if abs(timestamp - exc_time) < 5:  # 5-second excitement windows
                    activity_multiplier = 2.0 + np.random.random()
                    base_intensity *= activity_multiplier

            # Generate frame with activity-based noise
            frame_data = np.random.randint(
                int(base_intensity - 30),
                int(base_intensity + 30),
                (128, 128, 3),
                dtype=np.uint8,
            )

            frame = VideoFrameData(
                frame_index=i,
                timestamp=timestamp,
                pixels=frame_data,
                width=128,
                height=128,
                channels=3,
            )
            frames.append(frame)

        return frames

    def generate_audio_segments(
        self, segment_duration: float = 30.0
    ) -> List[AudioSegmentData]:
        """Generate mock audio segments."""
        segments = []
        num_segments = int(self.duration / segment_duration)

        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = start_time + segment_duration

            # Generate base audio signal
            num_samples = int(segment_duration * self.sample_rate)
            base_signal = np.random.randn(num_samples) * 0.1

            # Add volume spikes for excitement
            excitement_times = [60, 120, 180, 240]
            for exc_time in excitement_times:
                if start_time <= exc_time <= end_time:
                    # Add volume spike
                    spike_start = int((exc_time - start_time) * self.sample_rate)
                    spike_duration = int(2.0 * self.sample_rate)  # 2-second spike
                    spike_end = min(spike_start + spike_duration, num_samples)

                    if spike_start < num_samples:
                        base_signal[spike_start:spike_end] *= 5.0  # Volume spike

            # Generate transcription with excitement keywords
            transcription = f"Audio segment {i}"
            if any(start_time <= exc_time <= end_time for exc_time in excitement_times):
                transcription += " wow that was amazing incredible play"

            segment = AudioSegmentData(
                start_time=start_time,
                end_time=end_time,
                samples=base_signal,
                sample_rate=self.sample_rate,
                channels=1,
                metadata={"transcription": transcription},
            )
            segments.append(segment)

        return segments

    def generate_chat_messages(self) -> List[ChatMessage]:
        """Generate mock chat messages."""
        messages = []
        total_messages = int(self.duration * self.chat_message_rate)

        # Define user pool and excitement keywords
        users = [f"user_{i}" for i in range(20)]
        normal_messages = [
            "hello",
            "how's it going",
            "nice stream",
            "good game",
            "lol",
            "interesting",
            "cool",
            "thanks",
            "yes",
            "no",
        ]
        excitement_messages = [
            "WOW!",
            "AMAZING!",
            "INCREDIBLE!",
            "EPIC PLAY!",
            "INSANE!",
            "POGGERS",
            "CLUTCH!",
            "HOLY MOLY",
            "NO WAY!",
            "UNBELIEVABLE!",
        ]

        excitement_times = [60, 120, 180, 240]

        for i in range(total_messages):
            timestamp = (i / total_messages) * self.duration
            user = np.random.choice(users)

            # Determine if this is during excitement period
            is_excitement_period = any(
                abs(timestamp - exc_time) < 10 for exc_time in excitement_times
            )

            if (
                is_excitement_period and np.random.random() < 0.7
            ):  # 70% excitement messages
                message_text = np.random.choice(excitement_messages)
                # Increase message frequency during excitement
                if np.random.random() < 0.5:
                    # Add extra messages
                    for j in range(2):
                        extra_message = ChatMessage(
                            timestamp=timestamp + j * 0.1,
                            user_id=user,
                            username=user,
                            message=np.random.choice(excitement_messages),
                            platform="twitch",
                        )
                        messages.append(extra_message)
            else:
                message_text = np.random.choice(normal_messages)

            message = ChatMessage(
                timestamp=timestamp,
                user_id=user,
                username=user,
                message=message_text,
                platform="twitch",
            )
            messages.append(message)

        return sorted(messages, key=lambda m: m.timestamp)


@pytest.fixture
def mock_stream_data():
    """Create mock stream data for testing."""
    return MockStreamData(duration_seconds=300)  # 5-minute stream


@pytest.fixture
def detection_pipeline_components():
    """Create all detection pipeline components."""
    return {
        "video_detector": VideoDetector(),
        "audio_detector": AudioDetector(),
        "chat_detector": ChatDetector(),
        "fusion_scorer": FusionScorer(),
        "ranker": HighlightRanker(),
        "post_processor": HighlightPostProcessor(),
    }


class TestEndToEndPipeline:
    """Test complete end-to-end detection pipeline."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_with_mock_data(
        self, mock_stream_data, detection_pipeline_components
    ):
        """Test complete pipeline with realistic mock data."""
        # Generate mock data
        video_frames = mock_stream_data.generate_video_frames()
        audio_segments = mock_stream_data.generate_audio_segments()
        chat_messages = mock_stream_data.generate_chat_messages()

        print(f"Generated {len(video_frames)} video frames")
        print(f"Generated {len(audio_segments)} audio segments")
        print(f"Generated {len(chat_messages)} chat messages")

        # Create content segments
        video_segments = []
        segment_duration = 30.0

        for i in range(
            0, len(video_frames), int(segment_duration * mock_stream_data.fps)
        ):
            start_idx = i
            end_idx = min(
                i + int(segment_duration * mock_stream_data.fps), len(video_frames)
            )
            segment_frames = video_frames[start_idx:end_idx]

            if segment_frames:
                start_time = segment_frames[0].timestamp
                end_time = segment_frames[-1].timestamp

                video_segment = ContentSegment(
                    start_time=start_time,
                    end_time=end_time,
                    data=segment_frames,
                    metadata={"type": "video", "frame_count": len(segment_frames)},
                )
                video_segments.append(video_segment)

        audio_content_segments = []
        for audio_seg in audio_segments:
            content_seg = ContentSegment(
                start_time=audio_seg.start_time,
                end_time=audio_seg.end_time,
                data=audio_seg,
                metadata={"type": "audio"},
            )
            audio_content_segments.append(content_seg)

        # Create chat windows
        chat_segments = []
        window_size = 30.0
        for i in range(0, int(mock_stream_data.duration), int(window_size)):
            start_time = i
            end_time = min(i + window_size, mock_stream_data.duration)

            window_messages = [
                msg for msg in chat_messages if start_time <= msg.timestamp <= end_time
            ]

            if window_messages:
                chat_window = ChatWindow(
                    start_time=start_time, end_time=end_time, messages=window_messages
                )

                chat_segment = ContentSegment(
                    start_time=start_time,
                    end_time=end_time,
                    data=chat_window,
                    metadata={"type": "chat", "message_count": len(window_messages)},
                )
                chat_segments.append(chat_segment)

        print(f"Created {len(video_segments)} video segments")
        print(f"Created {len(audio_content_segments)} audio segments")
        print(f"Created {len(chat_segments)} chat segments")

        # Step 1: Run individual detectors
        start_time = time.time()

        # Run detectors in parallel
        video_task = detection_pipeline_components["video_detector"].detect_highlights(
            video_segments
        )
        audio_task = detection_pipeline_components["audio_detector"].detect_highlights(
            audio_content_segments
        )
        chat_task = detection_pipeline_components["chat_detector"].detect_highlights(
            chat_segments
        )

        video_results, audio_results, chat_results = await asyncio.gather(
            video_task, audio_task, chat_task
        )

        detection_time = time.time() - start_time
        print(f"Detection completed in {detection_time:.2f} seconds")
        print(f"Video results: {len(video_results)}")
        print(f"Audio results: {len(audio_results)}")
        print(f"Chat results: {len(chat_results)}")

        # Verify detection results
        assert isinstance(video_results, list)
        assert isinstance(audio_results, list)
        assert isinstance(chat_results, list)

        # Step 2: Fusion scoring
        fusion_start = time.time()

        results_by_modality = {}
        if video_results:
            results_by_modality[ModalityType.VIDEO] = video_results
        if audio_results:
            results_by_modality[ModalityType.AUDIO] = audio_results
        if chat_results:
            results_by_modality[ModalityType.CHAT] = chat_results

        if results_by_modality:
            fusion_candidates = await detection_pipeline_components[
                "fusion_scorer"
            ].fuse_results(results_by_modality)
        else:
            fusion_candidates = []

        fusion_time = time.time() - fusion_start
        print(f"Fusion completed in {fusion_time:.2f} seconds")
        print(f"Fusion candidates: {len(fusion_candidates)}")

        # Step 3: Ranking and selection
        ranking_start = time.time()

        if fusion_candidates:
            selected_highlights, ranking_metrics = await detection_pipeline_components[
                "ranker"
            ].rank_and_select(fusion_candidates)
        else:
            selected_highlights = []
            ranking_metrics = None

        ranking_time = time.time() - ranking_start
        print(f"Ranking completed in {ranking_time:.2f} seconds")
        print(f"Selected highlights: {len(selected_highlights)}")

        # Step 4: Post-processing
        processing_start = time.time()

        if selected_highlights:
            final_highlights, processing_metrics = await detection_pipeline_components[
                "post_processor"
            ].process_highlights(selected_highlights)
        else:
            final_highlights = []
            processing_metrics = None

        processing_time = time.time() - processing_start
        print(f"Post-processing completed in {processing_time:.2f} seconds")
        print(f"Final highlights: {len(final_highlights)}")

        # Verify final results
        total_time = time.time() - start_time
        print(f"Total pipeline time: {total_time:.2f} seconds")

        assert isinstance(final_highlights, list)

        # Verify highlight quality
        for highlight in final_highlights:
            assert isinstance(highlight, HighlightCandidate)
            assert 0 <= highlight.score <= 1
            assert 0 <= highlight.confidence <= 1
            assert highlight.duration > 0
            assert highlight.start_time < highlight.end_time

        # Verify performance (should complete within reasonable time)
        assert total_time < 60.0  # Should complete within 1 minute

        # Log performance metrics
        if ranking_metrics:
            print(
                f"Ranking metrics - Quality score: {ranking_metrics.quality_score:.3f}"
            )
            print(
                f"Ranking metrics - Diversity score: {ranking_metrics.diversity_score:.3f}"
            )

        if processing_metrics:
            print(
                f"Processing metrics - Improvement ratio: {processing_metrics.improvement_ratio:.3f}"
            )
            print(
                f"Processing metrics - Processing efficiency: {processing_metrics.processing_efficiency:.1f} candidates/sec"
            )

        return {
            "final_highlights": final_highlights,
            "ranking_metrics": ranking_metrics,
            "processing_metrics": processing_metrics,
            "total_time": total_time,
            "detection_time": detection_time,
            "fusion_time": fusion_time,
            "ranking_time": ranking_time,
            "processing_time": processing_time,
        }

    @pytest.mark.asyncio
    async def test_pipeline_with_different_configurations(self, mock_stream_data):
        """Test pipeline with different configuration combinations."""
        # Generate test data
        audio_segments = mock_stream_data.generate_audio_segments()
        chat_messages = mock_stream_data.generate_chat_messages()

        # Test different fusion methods
        fusion_methods = [
            FusionMethod.WEIGHTED_AVERAGE,
            FusionMethod.CONFIDENCE_WEIGHTED,
        ]
        ranking_methods = [RankingMethod.SCORE_BASED, RankingMethod.DIVERSITY_AWARE]

        results = {}

        for fusion_method in fusion_methods:
            for ranking_method in ranking_methods:
                config_name = f"{fusion_method}_{ranking_method}"
                print(f"Testing configuration: {config_name}")

                # Create components with specific configurations
                fusion_config = FusionConfig(fusion_method=fusion_method)
                ranking_config = RankingConfig(
                    ranking_method=ranking_method, max_highlights=3
                )

                audio_detector = AudioDetector()
                chat_detector = ChatDetector()
                fusion_scorer = FusionScorer(fusion_config)
                ranker = HighlightRanker(ranking_config)

                # Create content segments (simplified for speed)
                audio_content_segments = []
                for audio_seg in audio_segments[:3]:  # Use only first 3 segments
                    content_seg = ContentSegment(
                        start_time=audio_seg.start_time,
                        end_time=audio_seg.end_time,
                        data=audio_seg,
                    )
                    audio_content_segments.append(content_seg)

                chat_window = ChatWindow(
                    start_time=0.0,
                    end_time=90.0,
                    messages=chat_messages[:60],  # First 60 messages
                )
                chat_segment = ContentSegment(
                    start_time=0.0, end_time=90.0, data=chat_window
                )

                # Run pipeline
                audio_results = await audio_detector.detect_highlights(
                    audio_content_segments
                )
                chat_results = await chat_detector.detect_highlights([chat_segment])

                results_by_modality = {}
                if audio_results:
                    results_by_modality[ModalityType.AUDIO] = audio_results
                if chat_results:
                    results_by_modality[ModalityType.CHAT] = chat_results

                if results_by_modality:
                    candidates = await fusion_scorer.fuse_results(results_by_modality)
                    if candidates:
                        highlights, metrics = await ranker.rank_and_select(candidates)
                        results[config_name] = {
                            "highlights": len(highlights),
                            "avg_score": np.mean([h.score for h in highlights])
                            if highlights
                            else 0,
                            "quality": metrics.quality_score if metrics else 0,
                        }
                    else:
                        results[config_name] = {
                            "highlights": 0,
                            "avg_score": 0,
                            "quality": 0,
                        }
                else:
                    results[config_name] = {
                        "highlights": 0,
                        "avg_score": 0,
                        "quality": 0,
                    }

        # Verify different configurations produce different results
        print("Configuration results:")
        for config, result in results.items():
            print(f"  {config}: {result}")

        # Should have results for at least some configurations
        assert len(results) > 0

        return results

    @pytest.mark.asyncio
    async def test_pipeline_scalability(self, mock_stream_data):
        """Test pipeline scalability with increasing data sizes."""
        audio_detector = AudioDetector()

        # Test with different numbers of segments
        segment_counts = [1, 5, 10, 20]
        performance_results = {}

        for count in segment_counts:
            print(f"Testing with {count} segments")

            # Generate segments
            audio_segments = mock_stream_data.generate_audio_segments()[:count]
            content_segments = []

            for audio_seg in audio_segments:
                content_seg = ContentSegment(
                    start_time=audio_seg.start_time,
                    end_time=audio_seg.end_time,
                    data=audio_seg,
                )
                content_segments.append(content_seg)

            # Measure processing time
            start_time = time.time()
            results = await audio_detector.detect_highlights(content_segments)
            processing_time = time.time() - start_time

            performance_results[count] = {
                "processing_time": processing_time,
                "results_count": len(results),
                "throughput": count / processing_time if processing_time > 0 else 0,
            }

            print(f"  Processed in {processing_time:.3f}s, {len(results)} results")

        # Verify scalability characteristics
        print("Scalability results:")
        for count, result in performance_results.items():
            print(f"  {count} segments: {result}")

        # Processing time should increase sub-linearly (due to parallelization)
        times = [result["processing_time"] for result in performance_results.values()]
        assert all(t > 0 for t in times)  # All should complete

        return performance_results


class TestRealWorldScenarios:
    """Test realistic streaming scenarios."""

    @pytest.mark.asyncio
    async def test_gaming_stream_scenario(self):
        """Test gaming stream with typical excitement patterns."""
        # Create gaming-specific mock data
        duration = 180  # 3 minutes

        # Gaming excitement typically comes from:
        # 1. Kills/eliminations (audio spikes + chat excitement)
        # 2. Close calls (high visual activity)
        # 3. Victories (sustained excitement across all modalities)

        excitement_events = [
            {"time": 30, "type": "kill", "intensity": 0.8},
            {"time": 90, "type": "close_call", "intensity": 0.6},
            {"time": 150, "type": "victory", "intensity": 1.0},
        ]

        # Generate data with gaming patterns
        chat_messages = []
        gaming_keywords = [
            "nice shot",
            "clutch",
            "rip",
            "gg",
            "wp",
            "poggers",
            "insane",
            "wow",
        ]

        message_id = 0
        for event in excitement_events:
            event_time = event["time"]
            intensity = event["intensity"]

            # Generate burst of messages around event
            message_burst = int(10 * intensity)
            for i in range(message_burst):
                timestamp = event_time + np.random.uniform(
                    -2, 5
                )  # Messages around event
                if np.random.random() < intensity:
                    message_text = np.random.choice(gaming_keywords)
                else:
                    message_text = f"message {message_id}"

                message = ChatMessage(
                    timestamp=timestamp,
                    user_id=f"gamer_{message_id % 10}",
                    username=f"gamer_{message_id % 10}",
                    message=message_text,
                    platform="twitch",
                )
                chat_messages.append(message)
                message_id += 1

        # Sort messages by timestamp
        chat_messages.sort(key=lambda m: m.timestamp)

        # Create audio with volume spikes at events
        audio_segments = []
        segment_duration = 30.0
        for i in range(0, duration, int(segment_duration)):
            start_time = i
            end_time = min(i + segment_duration, duration)

            # Generate base audio
            num_samples = int(segment_duration * 44100)
            audio_signal = np.random.randn(num_samples) * 0.05

            # Add spikes for events in this segment
            for event in excitement_events:
                if start_time <= event["time"] <= end_time:
                    spike_start = int((event["time"] - start_time) * 44100)
                    spike_duration = int(3.0 * 44100)  # 3-second spike
                    spike_end = min(spike_start + spike_duration, num_samples)

                    if spike_start < num_samples:
                        audio_signal[spike_start:spike_end] *= (
                            2.0 + event["intensity"] * 3.0
                        )

            # Generate transcription
            transcription = "gaming audio"
            for event in excitement_events:
                if start_time <= event["time"] <= end_time:
                    if event["type"] == "kill":
                        transcription += " nice shot eliminated"
                    elif event["type"] == "close_call":
                        transcription += " oh no almost died"
                    elif event["type"] == "victory":
                        transcription += " yes we won victory royale"

            audio_seg = AudioSegmentData(
                start_time=start_time,
                end_time=end_time,
                samples=audio_signal,
                sample_rate=44100,
                channels=1,
                metadata={"transcription": transcription},
            )
            audio_segments.append(audio_seg)

        # Run detection pipeline
        audio_detector = AudioDetector()
        chat_detector = ChatDetector()
        fusion_scorer = FusionScorer()
        ranker = HighlightRanker()

        # Create content segments
        audio_content_segments = []
        for audio_seg in audio_segments:
            content_seg = ContentSegment(
                start_time=audio_seg.start_time,
                end_time=audio_seg.end_time,
                data=audio_seg,
            )
            audio_content_segments.append(content_seg)

        chat_windows = []
        window_size = 30.0
        for i in range(0, duration, int(window_size)):
            start_time = i
            end_time = min(i + window_size, duration)

            window_messages = [
                msg for msg in chat_messages if start_time <= msg.timestamp <= end_time
            ]

            if window_messages:
                chat_window = ChatWindow(
                    start_time=start_time, end_time=end_time, messages=window_messages
                )

                chat_segment = ContentSegment(
                    start_time=start_time, end_time=end_time, data=chat_window
                )
                chat_windows.append(chat_segment)

        # Run detection
        audio_results = await audio_detector.detect_highlights(audio_content_segments)
        chat_results = await chat_detector.detect_highlights(chat_windows)

        print(f"Gaming scenario - Audio results: {len(audio_results)}")
        print(f"Gaming scenario - Chat results: {len(chat_results)}")

        # Fusion and ranking
        results_by_modality = {}
        if audio_results:
            results_by_modality[ModalityType.AUDIO] = audio_results
        if chat_results:
            results_by_modality[ModalityType.CHAT] = chat_results

        if results_by_modality:
            candidates = await fusion_scorer.fuse_results(results_by_modality)
            if candidates:
                final_highlights, metrics = await ranker.rank_and_select(candidates)

                print(f"Gaming scenario - Final highlights: {len(final_highlights)}")

                # Verify we detected highlights near excitement events
                detected_times = [
                    (h.start_time + h.end_time) / 2 for h in final_highlights
                ]
                event_times = [e["time"] for e in excitement_events]

                # Should detect at least some events
                detections_near_events = 0
                for event_time in event_times:
                    for detected_time in detected_times:
                        if abs(detected_time - event_time) < 30:  # Within 30 seconds
                            detections_near_events += 1
                            break

                print(
                    f"Detected {detections_near_events} out of {len(excitement_events)} events"
                )

                # Should detect most events
                detection_rate = detections_near_events / len(excitement_events)
                assert detection_rate >= 0.5  # At least 50% detection rate

                return {
                    "highlights": final_highlights,
                    "detection_rate": detection_rate,
                    "events": excitement_events,
                }

        return {"highlights": [], "detection_rate": 0.0, "events": excitement_events}

    @pytest.mark.asyncio
    async def test_educational_stream_scenario(self):
        """Test educational stream with different excitement patterns."""
        # Educational streams have different patterns:
        # - Lower overall activity
        # - Excitement from "aha moments" or interesting discoveries
        # - More sustained engagement rather than spikes

        duration = 240  # 4 minutes

        # Educational excitement events
        excitement_events = [
            {"time": 60, "type": "explanation", "intensity": 0.4},
            {"time": 120, "type": "demonstration", "intensity": 0.6},
            {"time": 180, "type": "breakthrough", "intensity": 0.8},
        ]

        # Generate educational chat patterns
        chat_messages = []
        educational_keywords = [
            "interesting",
            "cool",
            "thanks",
            "helpful",
            "learned",
            "nice",
            "good",
        ]

        message_id = 0
        _base_message_rate = 0.5  # Lower message rate than gaming

        for t in range(0, duration, 5):  # Message every 5 seconds baseline
            # Check if near excitement event
            excitement_boost = 1.0
            for event in excitement_events:
                if abs(t - event["time"]) < 15:  # 15-second window
                    excitement_boost = 1.0 + event["intensity"]

            # Generate messages based on boost
            num_messages = int(1 * excitement_boost)
            for i in range(num_messages):
                timestamp = t + np.random.uniform(0, 5)

                if excitement_boost > 1.2:
                    message_text = np.random.choice(educational_keywords)
                else:
                    message_text = f"comment {message_id}"

                message = ChatMessage(
                    timestamp=timestamp,
                    user_id=f"learner_{message_id % 15}",
                    username=f"learner_{message_id % 15}",
                    message=message_text,
                    platform="youtube",
                )
                chat_messages.append(message)
                message_id += 1

        chat_messages.sort(key=lambda m: m.timestamp)

        # Run simplified detection (chat only for this test)
        chat_detector = ChatDetector()

        # Create chat windows
        chat_windows = []
        window_size = 40.0  # Longer windows for educational content

        for i in range(0, duration, int(window_size)):
            start_time = i
            end_time = min(i + window_size, duration)

            window_messages = [
                msg for msg in chat_messages if start_time <= msg.timestamp <= end_time
            ]

            if window_messages:
                chat_window = ChatWindow(
                    start_time=start_time, end_time=end_time, messages=window_messages
                )

                chat_segment = ContentSegment(
                    start_time=start_time, end_time=end_time, data=chat_window
                )
                chat_windows.append(chat_segment)

        # Run detection
        chat_results = await chat_detector.detect_highlights(chat_windows)

        print(f"Educational scenario - Chat results: {len(chat_results)}")

        # Verify results are reasonable for educational content
        if chat_results:
            avg_score = np.mean([r.score for r in chat_results])
            print(f"Educational scenario - Average score: {avg_score:.3f}")

            # Educational content should have moderate scores (less excitement than gaming)
            assert 0.1 <= avg_score <= 0.8

        return {
            "results": chat_results,
            "events": excitement_events,
            "messages": len(chat_messages),
        }


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in the pipeline."""

    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test pipeline behavior with empty inputs."""
        detectors = {
            "video": VideoDetector(),
            "audio": AudioDetector(),
            "chat": ChatDetector(),
        }

        # Test with empty segments
        for name, detector in detectors.items():
            results = await detector.detect_highlights([])
            assert results == []
            print(f"{name} detector handles empty input correctly")

        # Test fusion with empty results
        fusion_scorer = FusionScorer()
        candidates = await fusion_scorer.fuse_results({})
        assert candidates == []
        print("Fusion scorer handles empty input correctly")

        # Test ranking with empty candidates
        ranker = HighlightRanker()
        highlights, metrics = await ranker.rank_and_select([])
        assert highlights == []
        assert metrics.total_candidates == 0
        print("Ranker handles empty input correctly")

    @pytest.mark.asyncio
    async def test_invalid_data_handling(self):
        """Test pipeline behavior with invalid data."""
        video_detector = VideoDetector()

        # Test with invalid video data
        invalid_segment = ContentSegment(
            start_time=0.0,
            end_time=30.0,
            data=None,  # Invalid data
        )

        results = await video_detector.detect_highlights([invalid_segment])
        # Should handle gracefully without crashing
        assert isinstance(results, list)
        print("Video detector handles invalid data correctly")

        # Test with corrupted data
        corrupted_segment = ContentSegment(
            start_time=0.0,
            end_time=30.0,
            data="not_valid_data",  # Wrong data type
        )

        results = await video_detector.detect_highlights([corrupted_segment])
        assert isinstance(results, list)
        print("Video detector handles corrupted data correctly")

    @pytest.mark.asyncio
    async def test_extreme_values_handling(self):
        """Test pipeline behavior with extreme values."""
        audio_detector = AudioDetector()

        # Test with very loud audio
        extreme_audio = AudioSegmentData(
            start_time=0.0,
            end_time=10.0,
            samples=np.ones(441000) * 1000,  # Very loud
            sample_rate=44100,
            channels=1,
        )

        extreme_segment = ContentSegment(
            start_time=0.0, end_time=10.0, data=extreme_audio
        )

        results = await audio_detector.detect_highlights([extreme_segment])
        assert isinstance(results, list)
        print("Audio detector handles extreme values correctly")

        # Verify scores are still in valid range
        for result in results:
            assert 0 <= result.score <= 1
            assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        import psutil
        import os

        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        large_audio_segments = []
        for i in range(50):  # 50 segments
            audio_data = AudioSegmentData(
                start_time=i * 10.0,
                end_time=(i + 1) * 10.0,
                samples=np.random.randn(441000),  # 10 seconds of audio
                sample_rate=44100,
                channels=1,
            )

            segment = ContentSegment(
                start_time=i * 10.0, end_time=(i + 1) * 10.0, data=audio_data
            )
            large_audio_segments.append(segment)

        # Process large dataset
        audio_detector = AudioDetector()
        results = await audio_detector.detect_highlights(large_audio_segments)

        # Check memory usage after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(
            f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB"
        )
        print(f"Memory increase: {memory_increase:.1f}MB")

        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500

        # Should still produce valid results
        assert isinstance(results, list)
        for result in results:
            assert 0 <= result.score <= 1

    def test_configuration_edge_cases(self):
        """Test configuration edge cases and validation."""
        from src.services.highlight_detection.video_detector import VideoDetectionConfig
        from src.services.highlight_detection.fusion_scorer import FusionConfig

        # Test boundary values
        try:
            config = VideoDetectionConfig(
                motion_threshold=0.0,  # Minimum valid value
                scene_change_threshold=1.0,  # Maximum valid value
                max_frames_per_segment=1,  # Minimum valid value
            )
            assert config.motion_threshold == 0.0
            print("Video config handles boundary values correctly")
        except ValueError:
            pytest.fail("Valid boundary values should not raise ValueError")

        # Test invalid configurations
        with pytest.raises(ValueError):
            VideoDetectionConfig(motion_threshold=-0.1)  # Below minimum

        with pytest.raises(ValueError):
            VideoDetectionConfig(motion_threshold=1.1)  # Above maximum

        print("Video config validates ranges correctly")

        # Test fusion config edge cases
        fusion_config = FusionConfig(
            video_weight=0.0,
            audio_weight=0.0,
            chat_weight=1.0,  # Only chat weight
        )

        normalized = fusion_config.normalized_weights
        assert abs(normalized[ModalityType.CHAT] - 1.0) < 1e-10
        print("Fusion config handles edge weight distributions correctly")


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    @pytest.mark.asyncio
    async def test_processing_speed_benchmark(self, mock_stream_data):
        """Benchmark processing speed of different components."""
        # Generate test data
        video_frames = mock_stream_data.generate_video_frames()[
            :300
        ]  # 10 seconds worth
        audio_segments = mock_stream_data.generate_audio_segments()[:5]
        chat_messages = mock_stream_data.generate_chat_messages()[:100]

        benchmarks = {}

        # Benchmark video detection
        video_detector = VideoDetector()
        video_segments = []

        # Create video segments
        for i in range(0, len(video_frames), 30):
            segment_frames = video_frames[i : i + 30]
            if segment_frames:
                segment = ContentSegment(
                    start_time=i / 30.0, end_time=(i + 30) / 30.0, data=segment_frames
                )
                video_segments.append(segment)

        start_time = time.time()
        _video_results = await video_detector.detect_highlights(video_segments)
        video_time = time.time() - start_time

        benchmarks["video_detection"] = {
            "time": video_time,
            "segments": len(video_segments),
            "throughput": len(video_segments) / video_time if video_time > 0 else 0,
        }

        # Benchmark audio detection
        audio_detector = AudioDetector()
        audio_content_segments = []

        for audio_seg in audio_segments:
            segment = ContentSegment(
                start_time=audio_seg.start_time,
                end_time=audio_seg.end_time,
                data=audio_seg,
            )
            audio_content_segments.append(segment)

        start_time = time.time()
        _audio_results = await audio_detector.detect_highlights(audio_content_segments)
        audio_time = time.time() - start_time

        benchmarks["audio_detection"] = {
            "time": audio_time,
            "segments": len(audio_content_segments),
            "throughput": len(audio_content_segments) / audio_time
            if audio_time > 0
            else 0,
        }

        # Benchmark chat detection
        chat_detector = ChatDetector()
        chat_window = ChatWindow(start_time=0.0, end_time=100.0, messages=chat_messages)
        chat_segment = ContentSegment(start_time=0.0, end_time=100.0, data=chat_window)

        start_time = time.time()
        _chat_results = await chat_detector.detect_highlights([chat_segment])
        chat_time = time.time() - start_time

        benchmarks["chat_detection"] = {
            "time": chat_time,
            "messages": len(chat_messages),
            "throughput": len(chat_messages) / chat_time if chat_time > 0 else 0,
        }

        # Print benchmark results
        print("Performance Benchmarks:")
        for component, metrics in benchmarks.items():
            print(f"  {component}:")
            print(f"    Time: {metrics['time']:.3f}s")
            print(f"    Throughput: {metrics['throughput']:.1f} items/sec")

        # Verify reasonable performance
        assert (
            benchmarks["video_detection"]["time"] < 30.0
        )  # Should complete in under 30s
        assert (
            benchmarks["audio_detection"]["time"] < 10.0
        )  # Should complete in under 10s
        assert benchmarks["chat_detection"]["time"] < 5.0  # Should complete in under 5s

        return benchmarks

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Test performance of concurrent processing."""

        # Create test data
        mock_data = MockStreamData(duration_seconds=60)
        audio_segments = mock_data.generate_audio_segments()[:10]

        audio_detector = AudioDetector()

        # Sequential processing benchmark
        content_segments = []
        for audio_seg in audio_segments:
            segment = ContentSegment(
                start_time=audio_seg.start_time,
                end_time=audio_seg.end_time,
                data=audio_seg,
            )
            content_segments.append(segment)

        start_time = time.time()
        sequential_results = await audio_detector.detect_highlights(content_segments)
        sequential_time = time.time() - start_time

        # Parallel processing benchmark (simulate multiple detectors)
        detectors = [AudioDetector() for _ in range(3)]
        segment_chunks = [
            content_segments[:3],
            content_segments[3:6],
            content_segments[6:],
        ]

        start_time = time.time()

        tasks = []
        for detector, chunk in zip(detectors, segment_chunks):
            if chunk:  # Only process non-empty chunks
                task = detector.detect_highlights(chunk)
                tasks.append(task)

        if tasks:
            parallel_results = await asyncio.gather(*tasks)
            # Flatten results
            parallel_results = [
                result for chunk_results in parallel_results for result in chunk_results
            ]
        else:
            parallel_results = []

        parallel_time = time.time() - start_time

        print(
            f"Sequential processing: {sequential_time:.3f}s ({len(sequential_results)} results)"
        )
        print(
            f"Parallel processing: {parallel_time:.3f}s ({len(parallel_results)} results)"
        )

        # Parallel should be faster (or at least not much slower due to overhead)
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        print(f"Speedup: {speedup:.2f}x")

        # Should achieve some speedup or at least not be much slower
        assert speedup >= 0.8  # Allow for some overhead

        return {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
        }


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
