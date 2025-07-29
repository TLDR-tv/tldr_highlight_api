# YouTube HLS Streaming Implementation

## Overview

This document describes the comprehensive HLS/DASH parsing implementation for YouTube Live streams in the TL;DR Highlight API. The implementation enables real video data access from YouTube livestreams through HLS manifest parsing and segment downloading.

## Architecture

### Core Components

1. **HLS Parser Module (`src/utils/hls_parser.py`)**
   - Comprehensive HLS manifest parsing
   - YouTube-specific optimizations
   - Quality selection utilities
   - Bandwidth estimation

2. **Enhanced YouTube Adapter (`src/services/stream_adapters/youtube.py`)**
   - Real video streaming capabilities
   - Adaptive quality selection
   - Error recovery and reconnection logic
   - Comprehensive metadata enhancement

## Features Implemented

### 1. HLS Manifest Parsing

#### Stream Quality Management
- **Quality Detection**: Automatic detection of available qualities (240p, 480p, 720p, 1080p, 4K)
- **Codec Information**: Full codec parsing (video: H.264/H.265, audio: AAC)
- **Bitrate Analysis**: Bandwidth requirements and optimization
- **Frame Rate Detection**: FPS information for quality selection

#### Manifest Structure
```python
StreamManifest
├── master_playlist_uri
├── playlists: List[HLSPlaylist]
│   ├── video_playlists (video tracks)
│   └── audio_playlists (audio tracks)
├── is_live: bool
└── quality metadata
```

#### Quality Selection Logic
```python
def select_optimal_quality(
    qualities: List[StreamQuality],
    target_height: Optional[int] = None,
    max_bandwidth: Optional[int] = None,
    prefer_quality: str = "best"
) -> Optional[StreamQuality]
```

### 2. Video Segment Access

#### Segment Download Pipeline
- **Concurrent Downloading**: Efficient segment retrieval
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Recovery**: Graceful handling of failed segments
- **Rate Limiting**: Respect for YouTube's API policies

#### Live Stream Handling
- **Manifest Refresh**: Automatic refresh for new segments
- **Sequence Tracking**: Proper segment ordering
- **Buffer Management**: Configurable segment buffering
- **Discontinuity Handling**: Support for ad breaks and stream changes

### 3. Enhanced YouTube Adapter

#### Configuration Options
```python
YouTubeAdapter(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    preferred_quality="720p",  # "best", "worst", "720p", etc.
    max_bandwidth=5000000,     # Max bandwidth in bps
    target_height=720,         # Target video height
    enable_adaptive_streaming=True,
    segment_buffer_size=5,
    manifest_refresh_interval=30.0,
    segment_retry_attempts=3,
    segment_timeout=10.0
)
```

#### Stream Data Access
```python
async def get_stream_data(self) -> AsyncGenerator[bytes, None]:
    """Yields video stream data chunks (TS segments)"""
```

#### Quality Management
```python
async def get_available_qualities(self) -> List[Dict[str, Any]]
async def switch_quality(self, target_quality: str) -> bool
```

## Implementation Details

### 1. YouTube Stream URL Resolution

#### URL Extraction Process
1. **API Metadata Analysis**: Extract URLs from `liveStreamingDetails`
2. **HLS Manifest Detection**: Look for `hlsManifestUrl`
3. **DASH Manifest Detection**: Look for `dashManifestUrl`
4. **Fallback Handling**: Graceful degradation to metadata-only mode

#### Example Response
```json
{
  "hls_url": "https://manifest.googlevideo.com/api/manifest/hls_playlist/...",
  "dash_url": "https://manifest.googlevideo.com/api/manifest/dash/...",
  "streaming_available": true
}
```

### 2. HLS Manifest Structure

#### Master Playlist Parsing
```
#EXTM3U
#EXT-X-VERSION:6
#EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720,CODECS="avc1.42001f,mp4a.40.2"
720p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080,CODECS="avc1.64001f,mp4a.40.2"
1080p.m3u8
```

#### Media Playlist Processing
```
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:12345
#EXTINF:6.0,
segment12345.ts
#EXTINF:6.0,
segment12346.ts
```

### 3. Error Handling and Recovery

#### Connection Recovery
- **Automatic Reconnection**: Configurable retry attempts
- **Exponential Backoff**: Progressive delay increase
- **Circuit Breaker**: Prevent cascading failures
- **Health Monitoring**: Continuous stream health checks

#### Segment Error Handling
- **Skip Corrupt Segments**: Continue streaming despite individual segment failures
- **Bandwidth Adaptation**: Automatic quality downgrade on network issues
- **Timeout Management**: Configurable timeouts for different operations

## Usage Examples

### Basic Usage
```python
# Initialize adapter with HLS streaming
adapter = YouTubeAdapter(
    "https://www.youtube.com/watch?v=LIVE_VIDEO_ID",
    api_key="your_youtube_api_key"
)

# Connect and start streaming
await adapter.connect()

# Stream video data
async for chunk in adapter.get_stream_data():
    # Process video chunk (TS segment data)
    process_video_chunk(chunk)
```

### Advanced Configuration
```python
# High-quality streaming with adaptive bitrate
adapter = YouTubeAdapter(
    url="https://www.youtube.com/watch?v=LIVE_VIDEO_ID",
    preferred_quality="1080p",
    max_bandwidth=8000000,  # 8 Mbps max
    enable_adaptive_streaming=True,
    segment_buffer_size=10,
    api_key="your_youtube_api_key"
)

await adapter.connect()

# Check available qualities
qualities = await adapter.get_available_qualities()
print(f"Available: {[q['name'] for q in qualities]}")

# Switch quality during streaming
await adapter.switch_quality("720p")
```

### Quality Selection
```python
# Target specific height
adapter = YouTubeAdapter(
    url="https://www.youtube.com/watch?v=LIVE_VIDEO_ID",
    target_height=720,  # Closest to 720p
    api_key="your_youtube_api_key"
)

# Bandwidth-constrained streaming
adapter = YouTubeAdapter(
    url="https://www.youtube.com/watch?v=LIVE_VIDEO_ID",
    max_bandwidth=2000000,  # 2 Mbps max
    prefer_quality="best",   # Best within bandwidth limit
    api_key="your_youtube_api_key"
)
```

## Testing

### Comprehensive Test Coverage

#### Unit Tests (`tests/unit/test_hls_parser.py`)
- **StreamQuality**: Quality detection and naming
- **StreamSegment**: Segment metadata and encryption detection
- **HLSPlaylist**: Playlist duration and segment management
- **StreamManifest**: Quality filtering and selection
- **HLSParser**: Manifest parsing and caching
- **Quality Selection**: Optimal quality algorithms

#### Integration Tests (`tests/unit/test_stream_adapters.py`)
- **HLS Streaming**: End-to-end streaming with mocked manifests
- **Fallback Behavior**: Graceful degradation without HLS
- **Quality Switching**: Dynamic quality changes
- **Error Recovery**: Network failure simulation

### Running Tests
```bash
# Run HLS parser tests
uv run pytest tests/unit/test_hls_parser.py -v

# Run YouTube adapter tests
uv run pytest tests/unit/test_stream_adapters.py::TestYouTubeAdapter -v

# Run all tests with coverage
uv run pytest --cov=src --cov-report=html
```

## Performance Considerations

### Memory Optimization
- **Streaming Processing**: No full video buffering
- **Segment Caching**: Limited segment buffer size
- **Manifest Caching**: 5-minute TTL for manifests
- **Async Processing**: Non-blocking I/O operations

### Network Efficiency
- **Concurrent Downloads**: Parallel segment fetching
- **HTTP Connection Reuse**: Persistent connections
- **Compression Support**: Gzip/deflate encoding
- **Rate Limiting**: Respect YouTube's API limits

### Bandwidth Management
```python
def estimate_bandwidth_requirement(
    quality: StreamQuality, 
    buffer_seconds: float = 10.0
) -> int:
    """Estimate required bandwidth with 20% overhead"""
    base_bandwidth = quality.bandwidth
    overhead = int(base_bandwidth * 0.2)
    buffer_factor = max(1.0, buffer_seconds / 10.0)
    return int((base_bandwidth + overhead) * buffer_factor)
```

## Limitations and Considerations

### YouTube API Restrictions
1. **Rate Limits**: YouTube enforces strict API quotas
2. **Manifest Availability**: Not all streams provide HLS manifests
3. **Geographic Restrictions**: Some streams may be region-locked
4. **Terms of Service**: Must comply with YouTube's developer policies

### Technical Limitations
1. **DASH Support**: Currently not implemented (HLS only)
2. **DRM Content**: Encrypted streams not supported
3. **Live-only**: Optimized for live streams (VOD support limited)
4. **Platform Specific**: YouTube-specific optimizations

### Fallback Behavior
When HLS manifests are not available:
- Automatic fallback to metadata-only mode
- JSON metadata streaming instead of video data
- Continued chat and analytics functionality
- Graceful error reporting

## Future Enhancements

### Planned Features
1. **DASH Support**: Implementation of DASH manifest parsing
2. **DRM Handling**: Support for encrypted content
3. **Multi-CDN**: Support for multiple content delivery networks
4. **Quality Analytics**: Advanced quality selection algorithms
5. **Caching Layer**: Persistent segment caching

### Performance Improvements
1. **Predictive Buffering**: Intelligent segment prefetching
2. **Network Adaptation**: Dynamic quality adjustment
3. **Edge Optimization**: CDN-aware routing
4. **Compression**: Video segment compression

## Dependencies

### Required Packages
- `m3u8>=6.0.0`: HLS manifest parsing
- `aiohttp>=0.28.0`: Async HTTP client (already included)
- `asyncio`: Async/await support (Python standard library)

### Optional Dependencies
- `ffmpeg-python>=0.2.0`: Video processing utilities (already included)
- `opencv-python>=4.8.0`: Video frame analysis (already included)

## Security Considerations

### Data Protection
- **No Credential Storage**: API keys handled securely
- **HTTPS Only**: All communication over secure channels
- **Input Validation**: URL and parameter sanitization
- **Error Sanitization**: No sensitive data in error messages

### Rate Limiting
- **API Quotas**: Respect YouTube's daily quotas
- **Request Throttling**: Configurable rate limiting
- **Circuit Breakers**: Prevent API abuse
- **Exponential Backoff**: Progressive retry delays

## Conclusion

This implementation provides a robust, production-ready solution for accessing YouTube Live stream video data through HLS parsing. It includes comprehensive error handling, quality management, and performance optimizations while respecting YouTube's API policies and rate limits.

The architecture is designed for enterprise use cases requiring real video data access for AI-powered highlight extraction, with fallback mechanisms ensuring reliable operation even when streaming manifests are not available.