# Unified Video Buffering Architecture

## Overview

The TL;DR Highlight API implements a sophisticated unified video buffering system that enables efficient real-time processing of video streams from multiple sources (YouTube HLS, Twitch HLS, RTMP). This system provides consistent frame management, timestamp synchronization, and intelligent segmentation across all stream types.

## Architecture Components

### 1. Video Buffer (`src/utils/video_buffer.py`)

The core circular buffer implementation that manages video frames in memory.

#### Key Features:
- **Multi-format Support**: Handles HLS segments, FLV frames, and raw video frames
- **Circular Buffer Design**: Efficient memory usage with configurable retention policies
- **Keyframe Prioritization**: Preserves important keyframes when memory is constrained
- **Thread-Safe Operations**: Supports concurrent read/write access
- **Automatic Garbage Collection**: Manages memory pressure and evicts old data

#### Configuration:
```python
BufferConfig(
    max_memory_mb=500,           # Maximum memory usage
    max_items=10000,            # Maximum number of frames
    retention_seconds=300.0,     # Keep 5 minutes of data
    enable_keyframe_priority=True, # Prioritize keyframes
    gc_interval_seconds=30.0     # Garbage collection interval
)
```

### 2. Frame Synchronizer (`src/utils/frame_synchronizer.py`)

Handles timestamp normalization and synchronization across multiple streams.

#### Key Features:
- **Timestamp Format Conversion**: Normalizes different timestamp formats to a common reference
- **Clock Drift Detection**: Automatically detects and corrects clock drift between streams
- **Multi-Stream Alignment**: Aligns frames from different streams for synchronized processing
- **Outlier Detection**: Identifies and handles timestamp anomalies
- **Interpolation Support**: Estimates missing timestamps when needed

#### Supported Timestamp Formats:
- `EPOCH_SECONDS`: Unix timestamp in seconds
- `EPOCH_MILLIS`: Unix timestamp in milliseconds
- `HLS_TIMESTAMP`: HLS segment timestamps
- `RTMP_TIMESTAMP`: RTMP message timestamps
- `RELATIVE_SECONDS`: Seconds from stream start

### 3. Segment Processor (`src/utils/segment_processor.py`)

Creates analysis windows from continuous video streams.

#### Key Features:
- **Multiple Segmentation Strategies**:
  - Fixed Duration: Regular fixed-size segments
  - Sliding Window: Overlapping segments for continuous analysis
  - Keyframe-Based: Segments aligned with keyframes
  - Adaptive: Dynamic sizing based on content complexity
  - Scene-Based: Segments at scene changes
  - Event-Driven: Triggered by specific events

- **Quality Filtering**: Filters out low-quality segments
- **Parallel Processing**: Process multiple segments concurrently
- **Flexible Configuration**: Customizable duration, overlap, and thresholds

### 4. Stream Buffer Manager (`src/services/content_processing/stream_buffer_manager.py`)

High-level orchestration layer that coordinates all buffering components.

#### Key Features:
- **Unified Interface**: Single API for all stream types
- **Multi-Stream Management**: Handle multiple concurrent streams
- **Automatic Format Detection**: Identifies and handles different stream formats
- **Resource Management**: Monitors and controls total memory usage
- **Event Callbacks**: Hooks for frame and segment processing

## Integration with Stream Adapters

### Base Adapter Enhancement

The base stream adapter (`src/services/stream_adapters/base.py`) now includes:

```python
class BaseStreamAdapter:
    def __init__(
        self,
        url: str,
        buffer_config: Optional[BufferConfig] = None,
        enable_buffering: bool = True,
        buffer_manager: Optional[StreamBufferManager] = None,
        **kwargs
    ):
        # Buffering configuration
        self.buffer_config = buffer_config or BufferConfig()
        self.enable_buffering = enable_buffering
        self._buffer_manager = buffer_manager
        
    async def initialize_buffer(self) -> bool:
        """Initialize buffering for this stream."""
        
    async def get_segments(
        self,
        duration: Optional[float] = None,
        overlap: Optional[float] = None
    ) -> AsyncGenerator[ProcessedSegment, None]:
        """Get buffered segments for analysis."""
```

### Stream Type Detection

Each adapter automatically identifies its stream type:
- YouTube → `YOUTUBE_HLS`
- Twitch → `TWITCH_HLS`
- RTMP → `RTMP_FLV`

## Usage Patterns

### 1. Single Stream with Local Buffer

```python
# Create adapter with local buffering
adapter = YouTubeAdapter(
    url="https://youtube.com/watch?v=...",
    buffer_config=BufferConfig(max_memory_mb=200),
    enable_buffering=True,
    segment_duration=10.0,
    segment_overlap=2.0
)

# Start adapter (buffering initializes automatically)
await adapter.start()

# Process segments
async for segment in adapter.get_segments():
    # Analyze segment frames
    process_segment(segment)
```

### 2. Multiple Streams with Synchronization

```python
# Create buffer manager
manager = StreamBufferManager(
    StreamBufferConfig(
        enable_multi_stream_sync=True,
        max_streams=5
    )
)

# Add multiple streams
await manager.add_stream("stream1", youtube_adapter, StreamType.YOUTUBE_HLS)
await manager.add_stream("stream2", twitch_adapter, StreamType.TWITCH_HLS)

# Get synchronized segments
async for sync_segments in manager.get_synchronized_segments():
    # sync_segments = {"stream1": segment1, "stream2": segment2}
    analyze_synchronized_data(sync_segments)
```

### 3. Custom Segmentation Strategy

```python
# Configure adaptive segmentation
config = StreamBufferConfig(
    segment_config=SegmentConfig(
        strategy=SegmentStrategy.ADAPTIVE,
        min_segment_duration=5.0,
        max_segment_duration=30.0,
        target_complexity=0.5
    )
)

manager = StreamBufferManager(config)
```

## Performance Optimization

### Memory Management

1. **Circular Buffer**: Automatically evicts old frames when reaching limits
2. **Keyframe Priority**: Keeps keyframes longer than regular frames
3. **Memory Pooling**: Reuses frame objects to reduce allocation overhead
4. **Weak References**: Allows garbage collection of unused objects

### Throughput Optimization

1. **Batch Processing**: Process frames in batches for efficiency
2. **Parallel Segmentation**: Process multiple segments concurrently
3. **Lazy Loading**: Load frame data only when needed
4. **Background Tasks**: GC and stats collection run asynchronously

### Best Practices

1. **Configure Memory Limits**: Set appropriate `max_memory_mb` based on available RAM
2. **Tune Retention**: Balance `retention_seconds` with memory constraints
3. **Enable Keyframe Priority**: Essential for quality highlight detection
4. **Use Appropriate Segmentation**: Choose strategy based on content type
5. **Monitor Statistics**: Track buffer performance and adjust configuration

## Monitoring and Debugging

### Buffer Statistics

```python
# Get buffer stats
stats = buffer.get_stats()
# Returns:
# {
#     "current_frames": 1500,
#     "memory_usage_mb": 245.3,
#     "frames_dropped": 12,
#     "keyframe_ratio": 0.15,
#     "buffer_duration_seconds": 300.0
# }
```

### Stream Manager Statistics

```python
# Get all stream stats
all_stats = manager.get_stream_stats()
# Returns stats for each stream plus global metrics
```

### Event Callbacks

```python
# Monitor frame processing
manager.add_frame_callback(
    lambda stream_id, frame: print(f"Frame: {frame.timestamp}")
)

# Monitor segment creation
manager.add_segment_callback(
    lambda stream_id, segment: print(f"Segment: {segment.duration}s")
)
```

## Error Handling

The buffering system includes comprehensive error handling:

1. **Memory Pressure**: Gracefully handles out-of-memory conditions
2. **Format Errors**: Safely handles malformed frame data
3. **Synchronization Errors**: Falls back to independent processing
4. **Stream Disconnects**: Preserves buffered data during reconnections

## Future Enhancements

1. **Persistent Storage**: Option to spill buffer to disk
2. **Distributed Buffering**: Share buffers across multiple processes
3. **Advanced Compression**: Reduce memory usage with video compression
4. **ML-Based Segmentation**: Use ML models for intelligent segmentation
5. **WebRTC Support**: Add support for WebRTC streams