# Google Gemini Integration Guide

This guide explains how to migrate from OpenAI-based processing to Google Gemini's unified multimodal approach in the TL;DR Highlight API.

## Overview

Google Gemini 2.0 Flash provides native video understanding capabilities that eliminate the need for separate frame extraction, audio transcription, and fusion scoring. This results in:

- **Better context understanding**: Gemini processes video holistically rather than analyzing separate modalities
- **Improved performance**: No need for frame extraction or separate audio processing
- **Real-time capabilities**: Live API supports streaming analysis
- **Cost efficiency**: Single API call instead of multiple service calls

## Key Advantages

### 1. Native Video Understanding
- Processes up to 2 hours of video (2M context) or 1 hour (1M context)
- Direct YouTube URL support
- Built-in audio transcription
- 1 FPS video sampling, 1Kbps audio processing

### 2. Processing Modes
- **File API**: Batch processing for uploaded videos (up to 2GB)
- **Live API**: Real-time streaming analysis
- **Direct URL**: Process YouTube videos without downloading

### 3. Unified Analysis
- Single model handles video, audio, and text understanding
- Better temporal correlation between modalities
- More accurate highlight detection with full context

## Migration Steps

### 1. Update Configuration

Add your Gemini API key to your environment:

```bash
GEMINI_API_KEY=your-gemini-api-key
```

### 2. Choose Processing Mode

#### Option A: Use Gemini as Primary Detector (Recommended)

```python
from src.services.highlight_detection import GeminiDetector, GeminiDetectionConfig

# Configure Gemini detector
config = GeminiDetectionConfig(
    highlight_score_threshold=0.6,
    highlight_confidence_threshold=0.7,
    merge_window_seconds=5.0,
    category_weights={
        "action": 1.2,
        "emotional": 1.1,
        "informative": 1.0
    }
)

# Initialize detector
detector = GeminiDetector(config)

# Process entire video
highlights = await detector.detect_highlights_unified("path/to/video.mp4")

# Or process YouTube URL directly
highlights = await detector.detect_highlights_unified("https://youtube.com/watch?v=...")
```

#### Option B: Use Gemini in Hybrid Mode

Keep existing detectors but use Gemini for enhanced analysis:

```python
from src.services.content_processing import GeminiProcessor
from src.services.highlight_detection import VideoDetector, AudioDetector, FusionScorer

# Use Gemini for preprocessing
gemini = GeminiProcessor()
result = await gemini.process_video_file("video.mp4")

# Convert Gemini results to segments for existing pipeline
segments = convert_gemini_to_segments(result)

# Continue with existing detectors if needed
video_results = await video_detector.detect_highlights(segments)
```

### 3. Stream Processing

For live streams, use Gemini's Live API:

```python
# Process live stream
async for highlight in detector.process_stream_with_gemini(video_stream, audio_stream):
    # Handle real-time highlights
    await send_highlight_notification(highlight)
```

### 4. Batch Processing

For existing content libraries:

```python
from src.services.content_processing import ProcessingMode

# Process with File API for large files
result = await gemini.process_video_file(
    "large_video.mp4",
    mode=ProcessingMode.FILE_API,
    start_time=0,
    duration=3600  # 1 hour
)
```

## Configuration Options

### Gemini Processor Configuration

```python
from src.services.content_processing import GeminiProcessorConfig, GeminiModel

config = GeminiProcessorConfig(
    model_name=GeminiModel.FLASH_2_0,  # Latest model
    chunk_duration_seconds=300,         # 5-minute chunks
    temperature=0.3,                    # Lower for consistent results
    max_retries=3,
    highlight_prompt_template="""
    Analyze this video for highlight moments.
    Focus on: action sequences, emotional peaks, key information.
    Provide timestamps and confidence scores.
    """
)
```

### Detection Configuration

```python
detection_config = GeminiDetectionConfig(
    # Scoring thresholds
    highlight_score_threshold=0.6,
    highlight_confidence_threshold=0.7,
    
    # Temporal grouping
    merge_window_seconds=5.0,
    min_highlight_duration=5.0,
    max_highlight_duration=60.0,
    
    # Quality boosting
    enable_quality_boost=True,
    quality_boost_factor=0.2,
    
    # Output options
    include_transcriptions=True,
    include_visual_descriptions=True
)
```

## Performance Considerations

### 1. Context Window Management
- 1M context: ~1 hour of video
- 2M context: ~2 hours of video
- Chunk longer videos appropriately

### 2. Rate Limiting
- Implement appropriate retry logic
- Use exponential backoff for API errors
- Monitor quota usage

### 3. Caching
- Gemini detector includes built-in caching
- Results are cached by segment ID
- Configure cache size based on your needs

## Testing

Run the comprehensive test suite:

```bash
# Run Gemini integration tests
uv run pytest tests/unit/test_gemini_integration.py -v

# Test with real video (requires API key)
uv run python -m tests.integration.test_gemini_live
```

## Rollback Plan

If you need to rollback to OpenAI-based processing:

1. Keep both implementations available
2. Use feature flags to switch between processors:

```python
if settings.ai_provider == "gemini":
    detector = GeminiDetector(config)
else:
    # Use traditional pipeline
    video_detector = VideoDetector()
    audio_detector = AudioDetector()
    fusion_scorer = FusionScorer()
```

## Cost Comparison

### OpenAI Approach
- Whisper API: $0.006/minute for audio
- GPT-4 Vision: $0.01-0.03 per image (multiple frames)
- Total: Higher cost due to multiple API calls

### Gemini Approach
- Single API call for entire video
- Pricing based on token usage
- Generally more cost-effective for video content

## Best Practices

1. **Start with small tests**: Test with short videos first
2. **Monitor quality**: Compare results with existing pipeline
3. **Adjust prompts**: Customize prompts for your content type
4. **Use appropriate models**: 
   - Gemini 2.0 Flash for speed
   - Gemini 1.5 Pro for quality
5. **Handle errors gracefully**: Implement fallback to traditional pipeline

## Support

For issues or questions:
1. Check Gemini API documentation
2. Review error logs for API responses
3. Test with different video formats
4. Verify API key and quotas

## Conclusion

Migrating to Gemini provides significant advantages for video highlight detection. The unified approach offers better accuracy, lower latency, and reduced complexity compared to the multi-service pipeline.