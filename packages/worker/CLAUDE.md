# CLAUDE.md - Worker Package

This file provides guidance to Claude Code when working with the worker package, which handles all asynchronous processing tasks for the TLDR Highlight API.

## Package Overview

The `worker` package contains **Celery workers** that handle CPU-intensive tasks like stream processing, AI-powered highlight detection, wake word detection, and webhook delivery. It's the processing engine of the system. Following recent refactoring, this package now also contains all worker-specific services including AI processing and video analysis.

## Working with This Package

### Running Commands
```bash
# Always use uv from the package directory
cd packages/worker
uv run pytest  # Run tests
uv run celery -A worker.app worker --loglevel=info  # Run worker

# Or from root with --directory
uv --directory packages/worker run pytest
```

### Adding Dependencies
```bash
cd packages/worker
uv add opencv-python  # Add production dependency
uv add --dev pytest-celery  # Add dev dependency
```

## Package Structure

### Services (`src/worker/services/`)
Worker-specific processing services:

1. **AI Processing Services**:
   - `highlight_detector.py` - AI-powered highlight detection logic
   - `gemini_scoring.py` - Gemini AI integration for video scoring
   - `dimension_framework.py` - Flexible scoring system framework
   - `scoring_factory.py` - Factory for creating scoring strategies

2. **Media Processing Services**:
   - `ffmpeg_processor.py` - Video/audio stream processing
   - `video_utils.py` - Video manipulation utilities
   - `audio_utils.py` - Audio processing utilities

3. **Detection Services**:
   - `wake_word_detector.py` - Wake word detection in audio
   - `rule_scorer.py` - Rule-based scoring strategies

### Tasks (`src/worker/tasks/`)
Celery task implementations:
- `stream_processing.py` - Main stream processing orchestration
- `highlight_detection.py` - Highlight detection tasks
- `wake_word_detection.py` - Wake word detection tasks
- `webhook_delivery.py` - Webhook notification delivery

## Celery Configuration

### Application Setup (`app.py`)
```python
app = Celery("highlight_worker")
app.config_from_object({
    "broker_url": settings.redis_url,
    "result_backend": settings.redis_url,
    "task_time_limit": 3600,  # 1 hour
    "task_routes": {
        "worker.tasks.stream_processing.*": {"queue": "stream_processing"},
        "worker.tasks.highlight_detection.*": {"queue": "highlight_detection"},
        "worker.tasks.webhook_delivery.*": {"queue": "webhooks"},
    }
})
```

### Running Workers
```bash
# Run all queues
uv run celery -A worker.app worker --loglevel=info

# Run specific queue
uv run celery -A worker.app worker -Q stream_processing --loglevel=info

# Run with concurrency
uv run celery -A worker.app worker --concurrency=4
```

## Task Implementation Guidelines

### Task Structure (`tasks/`)
```python
@app.task(
    bind=True,
    name="worker.tasks.stream_processing.process_stream",
    max_retries=3,
    retry_backoff=True,
)
async def process_stream(self, stream_id: str) -> dict:
    """Process a stream and extract segments."""
    try:
        # Task implementation
        async with get_async_session() as session:
            # Database operations
            pass
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
```

### Task Best Practices
1. **Always use task names**: Explicit names prevent issues during refactoring
2. **Implement retries**: Use exponential backoff for transient failures
3. **Update status**: Keep database records updated with progress
4. **Send webhooks**: Notify clients of important events
5. **Clean up resources**: Use context managers for file/connection cleanup

## Core Processing Services

### Highlight Detector (`services/highlight_detector.py`)
```python
# Now a worker-specific service
from worker.services.highlight_detector import HighlightDetector
from worker.services.dimension_framework import ScoringRubric

detector = HighlightDetector(
    scoring_strategy=gemini_strategy,
    min_highlight_duration=10.0,
    max_highlight_duration=120.0
)

candidates = await detector.detect_highlights(
    stream=stream,
    segments=video_segments,
    rubric=scoring_rubric
)
```

### Gemini Scoring (`services/gemini_scoring.py`)
```python
# AI-powered video analysis (now in worker package)
from worker.services.gemini_scoring import GeminiScoringStrategy

strategy = GeminiScoringStrategy(settings)
scores = await strategy.score(
    content=video_path,
    rubric=scoring_rubric,
    context=previous_segments
)
```

### FFmpeg Processor (`services/ffmpeg_processor.py`)
```python
# Stream processing with ring buffer
async with FFmpegStreamProcessor(
    stream_url=stream.url,
    output_dir=temp_dir,
    segment_duration=120,  # 2 minutes
    max_segments=5,  # Ring buffer size
) as processor:
    async for segment in processor.process():
        # Queue highlight detection for segment
        app.send_task(
            "worker.tasks.highlight_detection.detect_highlights",
            args=[str(stream.id), segment.path]
        )
```

### Key FFmpeg Features
- **Universal format support**: RTMP, HLS, DASH, HTTP, local files
- **Robust reconnection**: Exponential backoff for network issues
- **Ring buffer management**: Automatic cleanup of old segments
- **Audio extraction**: Optimized for WhisperX transcription
- **CSV tracking**: Reliable segment metadata storage

### Dimension Framework (`services/dimension_framework.py`)
```python
# Flexible scoring system
dimension = DimensionDefinition(
    name="action_intensity",
    type=DimensionType.NUMERIC,
    range=(0, 10),
    weight=1.0,
    prompt="Rate the intensity of action/excitement",
    examples=[
        {"context": "Fast-paced combat", "score": 9},
        {"context": "Calm dialogue", "score": 2}
    ]
)

# Create scoring rubric
rubric = ScoringRubric(
    name="Gaming Highlights",
    dimensions=[dimension],
    highlight_threshold=7.0,
    highlight_confidence_threshold=0.8
)
```

## Task Types

### 1. Stream Processing (`tasks/stream_processing.py`)
**Purpose**: Segment streams and coordinate processing
```python
# Main processing flow
1. Update stream status to PROCESSING
2. Initialize FFmpeg processor
3. Process segments asynchronously
4. Extract audio chunks for wake words
5. Queue downstream tasks
6. Update stream status to COMPLETED/FAILED
```

### 2. Highlight Detection (`tasks/highlight_detection.py`)
**Purpose**: AI analysis and highlight creation
```python
# Detection flow using local services
from worker.services.highlight_detector import HighlightDetector
from worker.services.gemini_scoring import GeminiScoringStrategy

# Initialize with worker-specific services
scoring_strategy = GeminiScoringStrategy(settings)
highlight_detector = HighlightDetector(
    scoring_strategy=scoring_strategy,
    min_highlight_duration=processing_options.get("min_duration", 10.0)
)

# Process segment
detected_highlights = await highlight_detector.process_segment(
    segment=segment,
    stream=stream,
    rubric=scoring_rubric
)
```

### 3. Wake Word Detection (`tasks/wake_word_detection.py`)
**Purpose**: Detect keywords in audio
```python
# Detection flow
1. Transcribe audio with faster-whisper
2. Extract word-level timestamps
3. Fuzzy match against wake words
4. Calculate similarity scores
5. Trigger clip generation if match
6. Update wake word usage stats
```

### 4. Webhook Delivery (`tasks/webhook_delivery.py`)
**Purpose**: Reliable event notifications
```python
# Delivery flow
1. Validate webhook URL
2. Prepare payload with event data
3. Generate HMAC signature
4. Send POST request
5. Retry with backoff on failure
6. Log delivery status
```

## Testing Strategies

### Unit Tests
```python
# Mock external services
@patch("worker.services.gemini_scoring.genai")
async def test_gemini_scorer(mock_genai):
    mock_genai.upload_file.return_value = Mock(name="test.mp4")
    # Test implementation
```

### Integration Tests
```python
# Test complete task flow
async def test_stream_processing_integration():
    # Create test stream
    stream = await create_test_stream()
    
    # Execute task
    result = await process_stream.apply_async(args=[str(stream.id)])
    
    # Verify results
    assert result.successful()
```

## Performance Optimization

### Memory Management
```python
# Use ring buffers for segments
processor = FFmpegStreamProcessor(max_segments=5)

# Clean up temporary files
with tempfile.TemporaryDirectory() as temp_dir:
    # Process files
    pass  # Auto cleanup
```

### Concurrent Processing
```python
# Process multiple segments concurrently
tasks = []
async for segment in processor.process():
    task = detect_highlights.apply_async(args=[segment])
    tasks.append(task)

# Wait for completion with timeout
await asyncio.wait_for(
    asyncio.gather(*tasks),
    timeout=300
)
```

### Resource Limits
```python
# Set task time limits
@app.task(task_time_limit=300)  # 5 minutes

# Limit file sizes
if os.path.getsize(video_path) > 2 * 1024**3:  # 2GB
    raise ValueError("Video file too large")
```

## Error Handling

### Retry Strategies
```python
# Exponential backoff
@app.task(
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=600,  # Max 10 minutes
)

# Custom retry logic
except TemporaryError as e:
    raise self.retry(exc=e, countdown=60)  # Retry in 1 minute
```

### Graceful Degradation
```python
try:
    # Try AI analysis
    scores = await gemini_scorer.score_video(video_path)
except Exception as e:
    logger.warning(f"AI scoring failed: {e}")
    # Fallback to rule-based scoring
    scores = await rule_based_scorer.score_video(video_path)
```

## Multi-Tenant Considerations

### Organization Isolation
```python
# Always filter by organization
wake_words = await session.execute(
    select(WakeWord).where(
        WakeWord.organization_id == stream.organization_id
    )
)

# S3 path isolation
s3_key = f"organizations/{organization_id}/highlights/{highlight_id}/clip.mp4"
```

### Resource Tracking
```python
# Update usage statistics
organization.usage_stats["minutes_processed"] += segment_duration
organization.usage_stats["highlights_detected"] += 1
await session.commit()
```

## Common Tasks

### Adding a New Task Type
1. Create task module in `tasks/`
2. Define task with proper decorators
3. Add to task routes in `app.py`
4. Write unit and integration tests
5. Document webhook events if applicable

### Adding a New Service
1. Create service in `src/worker/services/`
2. Implement required interfaces (e.g., ScoringStrategy)
3. Write unit tests for the service
4. Integrate into relevant tasks
5. Document usage patterns

### Debugging Tasks
```bash
# Run worker with debug logging
uv run celery -A worker.app worker --loglevel=debug

# Monitor task execution
uv run celery -A worker.app events

# Inspect queue contents
uv run celery -A worker.app inspect active
```

### Local Development
```bash
# Run Redis for broker/backend
docker run -p 6379:6379 redis:alpine

# Run worker with auto-reload
uv run watchmedo auto-restart -d . -p '*.py' -- celery -A worker.app worker
```

## Common Pitfalls to Avoid

1. **Don't block the event loop** - Use async operations throughout
2. **Don't forget cleanup** - Always clean temporary files
3. **Don't ignore retries** - Implement proper retry logic
4. **Don't skip organization checks** - Maintain multi-tenant isolation
5. **Don't hardcode limits** - Use configuration for thresholds
6. **Don't mix concerns** - Keep AI logic in services, not tasks

## Integration Points

### With Shared Package
- Import domain models (`Stream`, `Highlight`, `Organization`)
- Use repository implementations
- Access shared infrastructure (database, config)

### With API Package
- Receive tasks via Celery
- Update shared database records
- Send webhooks for notifications

## Import Guidelines

Following the refactoring, use these import patterns:
```python
# Worker-specific services (now in worker package)
from worker.services.highlight_detector import HighlightDetector
from worker.services.gemini_scoring import GeminiScoringStrategy
from worker.services.dimension_framework import ScoringRubric, DimensionDefinition

# Shared imports (unchanged)
from shared.domain.models.stream import Stream
from shared.infrastructure.storage.repositories import StreamRepository
from shared.infrastructure.config.config import Settings
```

## Monitoring and Observability

```python
# Structured logging
logger.info(
    "Processing stream segment",
    stream_id=stream_id,
    segment_number=segment.number,
    duration=segment.duration,
)

# Performance tracking
with timer("highlight_detection"):
    scores = await detect_highlights(segment)
```