"""Example of using the unified buffering system with stream adapters.

This example demonstrates how to:
1. Set up stream adapters with buffering enabled
2. Process multiple streams simultaneously
3. Get synchronized segments for analysis
4. Handle different stream formats (HLS, RTMP)
"""

import asyncio
import logging
from datetime import datetime, timezone

from src.services.stream_adapters.youtube import YouTubeAdapter
from src.services.stream_adapters.twitch import TwitchAdapter
from src.services.stream_adapters.rtmp import EnhancedRTMPAdapter
from src.services.content_processing.stream_buffer_manager import (
    StreamBufferManager,
    StreamBufferConfig,
    StreamType,
)
from src.utils.video_buffer import BufferConfig
from src.utils.segment_processor import SegmentConfig, SegmentStrategy
from src.utils.frame_synchronizer import SyncConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_single_stream_example():
    """Example of processing a single stream with local buffering."""
    logger.info("=== Single Stream Processing Example ===")
    
    # Create YouTube adapter with buffering
    adapter = YouTubeAdapter(
        url="https://www.youtube.com/watch?v=example",
        buffer_config=BufferConfig(
            max_memory_mb=200,
            retention_seconds=300,  # Keep 5 minutes
            enable_keyframe_priority=True,
        ),
        enable_buffering=True,
        segment_duration=10.0,
        segment_overlap=2.0,
    )
    
    try:
        # Start the adapter
        await adapter.start()
        
        # Process stream data for a while
        processing_time = 30  # Process for 30 seconds
        start_time = datetime.now(timezone.utc).timestamp()
        
        logger.info("Processing stream data...")
        
        # Get segments as they become available
        segment_count = 0
        async for segment in adapter.get_segments(duration=10.0, overlap=2.0):
            segment_count += 1
            logger.info(
                f"Received segment {segment_count}: "
                f"{segment['frame_count']} frames, "
                f"duration: {segment['duration']:.2f}s"
            )
            
            # Process segment frames
            frames = segment['frames']
            keyframes = [f for f in frames if f.is_keyframe]
            logger.info(f"  - Keyframes: {len(keyframes)}")
            logger.info(f"  - Average quality: {sum(f.quality_score for f in frames) / len(frames):.2f}")
            
            # Stop after processing time
            if datetime.now(timezone.utc).timestamp() - start_time > processing_time:
                break
        
        # Get buffer statistics
        buffer = await adapter.get_buffer()
        if buffer:
            stats = buffer.get_stats()
            logger.info(f"\nBuffer Statistics:")
            logger.info(f"  - Frames buffered: {stats['current_frames']}")
            logger.info(f"  - Memory usage: {stats['memory_usage_mb']:.2f} MB")
            logger.info(f"  - Buffer duration: {stats['buffer_duration_seconds']:.2f} seconds")
            logger.info(f"  - Frames dropped: {stats['frames_dropped']}")
            
    finally:
        await adapter.stop()


async def process_multiple_streams_example():
    """Example of processing multiple streams with synchronized buffering."""
    logger.info("\n=== Multiple Stream Processing Example ===")
    
    # Create stream buffer manager with configuration
    buffer_config = StreamBufferConfig(
        buffer_config=BufferConfig(
            max_memory_mb=500,
            retention_seconds=300,
            enable_keyframe_priority=True,
        ),
        sync_config=SyncConfig(
            max_clock_drift_ms=100,
            sync_window_ms=50,
            enable_drift_correction=True,
        ),
        segment_config=SegmentConfig(
            strategy=SegmentStrategy.SLIDING_WINDOW,
            segment_duration_seconds=10.0,
            overlap_seconds=2.0,
            enable_quality_filtering=True,
            min_segment_quality=0.5,
        ),
        enable_multi_stream_sync=True,
        max_streams=5,
        max_total_memory_mb=1000,
    )
    
    manager = StreamBufferManager(buffer_config)
    
    # Create adapters
    youtube_adapter = YouTubeAdapter(
        url="https://www.youtube.com/watch?v=example1",
        buffer_manager=manager,
        enable_buffering=True,
    )
    
    twitch_adapter = TwitchAdapter(
        url="https://www.twitch.tv/example_channel",
        buffer_manager=manager,
        enable_buffering=True,
    )
    
    rtmp_adapter = EnhancedRTMPAdapter(
        url="rtmp://example.com/live/stream",
        buffer_manager=manager,
        enable_buffering=True,
    )
    
    # Start adapters
    adapters = [youtube_adapter, twitch_adapter, rtmp_adapter]
    stream_ids = []
    
    try:
        # Start all adapters
        for i, adapter in enumerate(adapters):
            stream_id = f"stream_{i}"
            stream_ids.append(stream_id)
            
            # Initialize adapter
            await adapter.start()
            
            # Add to buffer manager
            stream_type = {
                0: StreamType.YOUTUBE_HLS,
                1: StreamType.TWITCH_HLS,
                2: StreamType.RTMP_FLV,
            }[i]
            
            await manager.add_stream(
                stream_id=stream_id,
                adapter=adapter,
                stream_type=stream_type,
                priority=5,
            )
            
            logger.info(f"Added {stream_type.value} stream: {stream_id}")
        
        # Add callbacks for monitoring
        def on_frame(stream_id: str, frame):
            logger.debug(f"Frame from {stream_id}: timestamp={frame.timestamp:.3f}, keyframe={frame.is_keyframe}")
        
        def on_segment(stream_id: str, segment):
            logger.info(
                f"Segment from {stream_id}: "
                f"{segment.frame_count} frames, "
                f"quality={segment.quality_score:.2f}, "
                f"duration={segment.duration:.2f}s"
            )
        
        manager.add_frame_callback(on_frame)
        manager.add_segment_callback(on_segment)
        
        # Process synchronized segments
        logger.info("\nProcessing synchronized segments...")
        
        segment_count = 0
        async for sync_segments in manager.get_synchronized_segments(duration=10.0, overlap=2.0):
            segment_count += 1
            logger.info(f"\n--- Synchronized Segment {segment_count} ---")
            
            # Process each stream's segment
            for stream_id, segment in sync_segments.items():
                if segment:
                    logger.info(
                        f"{stream_id}: {segment.frame_count} frames, "
                        f"keyframes: {len(segment.keyframe_indices)}, "
                        f"time: {segment.start_time:.2f}-{segment.end_time:.2f}"
                    )
                    
                    # Example: Compare quality across streams
                    quality_score = segment.quality_score
                    logger.info(f"  Quality score: {quality_score:.2f}")
                else:
                    logger.warning(f"{stream_id}: No segment available")
            
            # Stop after a few segments for demo
            if segment_count >= 5:
                break
        
        # Get final statistics
        all_stats = manager.get_stream_stats()
        
        logger.info("\n=== Final Statistics ===")
        for stream_id, stats in all_stats.items():
            if stream_id == "global":
                logger.info("\nGlobal Stats:")
                logger.info(f"  Active streams: {stats['active_streams']}")
                logger.info(f"  Total memory: {stats['total_memory_usage_mb']:.2f} MB")
            else:
                logger.info(f"\n{stream_id}:")
                logger.info(f"  Type: {stats['stream_type']}")
                logger.info(f"  Frames processed: {stats['frames_processed']}")
                logger.info(f"  Segments processed: {stats['segments_processed']}")
                logger.info(f"  Keyframes extracted: {stats['keyframes_extracted']}")
                logger.info(f"  Dropped frames: {stats['dropped_frames']}")
                logger.info(f"  Average quality: {stats['avg_frame_quality']:.2f}")
                logger.info(f"  Sync accuracy: {stats['sync_accuracy']:.2f}")
                logger.info(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
        
    finally:
        # Clean up
        for adapter in adapters:
            await adapter.stop()
        
        await manager.close()


async def advanced_processing_example():
    """Example of advanced processing with custom segment strategies."""
    logger.info("\n=== Advanced Processing Example ===")
    
    # Create manager with adaptive segmentation
    config = StreamBufferConfig(
        segment_config=SegmentConfig(
            strategy=SegmentStrategy.ADAPTIVE,
            min_segment_duration=5.0,
            max_segment_duration=30.0,
            target_complexity=0.5,
            enable_quality_filtering=True,
        )
    )
    
    manager = StreamBufferManager(config)
    
    # Create adapter
    adapter = YouTubeAdapter(
        url="https://www.youtube.com/watch?v=example",
        buffer_manager=manager,
        enable_buffering=True,
    )
    
    try:
        await adapter.start()
        
        # Add stream with custom processing
        await manager.add_stream(
            stream_id="adaptive_stream",
            adapter=adapter,
            stream_type=StreamType.YOUTUBE_HLS,
            custom_config={
                'enable_keyframe_detection': True,
                'segment_duration': None,  # Let adaptive strategy decide
            }
        )
        
        # Process with callbacks
        segment_durations = []
        
        def analyze_segment(stream_id: str, segment):
            # Track segment durations
            segment_durations.append(segment.duration)
            
            # Analyze complexity
            if hasattr(segment, 'complexity_score'):
                logger.info(
                    f"Adaptive segment: duration={segment.duration:.2f}s, "
                    f"complexity={segment.complexity_score:.2f}, "
                    f"frames={segment.frame_count}"
                )
        
        manager.add_segment_callback(analyze_segment)
        
        # Process for a while
        await asyncio.sleep(60)
        
        # Analyze adaptive behavior
        if segment_durations:
            avg_duration = sum(segment_durations) / len(segment_durations)
            min_duration = min(segment_durations)
            max_duration = max(segment_durations)
            
            logger.info("\nAdaptive Segmentation Analysis:")
            logger.info(f"  Segments created: {len(segment_durations)}")
            logger.info(f"  Average duration: {avg_duration:.2f}s")
            logger.info(f"  Duration range: {min_duration:.2f}s - {max_duration:.2f}s")
        
    finally:
        await adapter.stop()
        await manager.close()


async def main():
    """Run all examples."""
    try:
        # Example 1: Single stream with local buffering
        await process_single_stream_example()
        
        # Example 2: Multiple streams with synchronization
        await process_multiple_streams_example()
        
        # Example 3: Advanced processing with adaptive segmentation
        await advanced_processing_example()
        
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())