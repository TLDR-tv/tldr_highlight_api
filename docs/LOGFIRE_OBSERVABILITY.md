# Logfire Observability Integration

This document describes the Pydantic Logfire observability integration for the TL;DR Highlight API, providing comprehensive monitoring, tracing, and metrics collection.

## Overview

Logfire provides a complete observability solution built on OpenTelemetry standards, offering:
- Distributed tracing across all services
- Structured logging with rich context
- Custom business metrics
- Performance monitoring
- Error tracking and alerting

## Configuration

### Environment Variables

Configure Logfire through environment variables:

```bash
# Basic Configuration
LOGFIRE_ENABLED=true                    # Enable/disable Logfire
LOGFIRE_PROJECT_NAME=tldr-highlight-api  # Project name in Logfire
LOGFIRE_API_KEY=your-api-key-here       # API key (optional for local dev)
LOGFIRE_ENVIRONMENT=production           # Environment (development/staging/production)

# Service Configuration
LOGFIRE_SERVICE_NAME=tldr-api           # Service identifier
LOGFIRE_SERVICE_VERSION=1.0.0           # Service version
LOGFIRE_CONSOLE_ENABLED=true            # Enable console output
LOGFIRE_LOG_LEVEL=INFO                  # Log level

# Feature Toggles
LOGFIRE_CAPTURE_HEADERS=true            # Capture HTTP headers
LOGFIRE_CAPTURE_BODY=false              # Capture request/response bodies
LOGFIRE_SQL_ENABLED=true                # Enable SQL query logging
LOGFIRE_REDIS_ENABLED=true              # Enable Redis command logging
LOGFIRE_CELERY_ENABLED=true             # Enable Celery task logging
LOGFIRE_SYSTEM_METRICS_ENABLED=true     # Enable system metrics
LOGFIRE_CUSTOM_METRICS_ENABLED=true     # Enable custom business metrics
```

## Usage

### 1. Automatic Instrumentation

The application automatically instruments:
- **FastAPI**: All HTTP requests/responses
- **SQLAlchemy**: Database queries
- **Redis**: Cache operations
- **Celery**: Background tasks
- **HTTPX**: External API calls

### 2. Using Decorators

#### Trace Functions
```python
from src.infrastructure.observability import traced, traced_service_method

@traced(name="custom_operation", capture_args=True)
async def process_data(data: dict) -> dict:
    # Function is automatically traced
    return processed_data

@traced_service_method()
async def business_logic(param: str) -> str:
    # Service method with standard attributes
    return result
```

#### Time Operations
```python
from src.infrastructure.observability import timed

@timed(metric_name="processing.duration")
async def expensive_operation() -> None:
    # Execution time is automatically recorded
    await do_work()
```

### 3. Creating Custom Spans

```python
from src.infrastructure.observability import with_span
import logfire

# As context manager
async def process_stream(stream_id: str):
    with with_span("process_stream", stream_id=stream_id):
        # Add attributes to current span
        logfire.set_attribute("stream.platform", "twitch")
        
        # Nested spans
        with with_span("extract_frames"):
            frames = await extract_frames()
        
        with with_span("analyze_content"):
            highlights = await analyze(frames)
```

### 4. Recording Metrics

#### Business Metrics
```python
from src.infrastructure.observability import metrics

# Stream processing metrics
metrics.increment_stream_started(
    platform="youtube",
    organization_id="org_123",
    stream_type="live"
)

metrics.record_stream_duration(
    duration_seconds=125.3,
    platform="youtube", 
    organization_id="org_123"
)

# Highlight detection metrics
metrics.increment_highlights_detected(
    count=5,
    platform="twitch",
    organization_id="org_123",
    detection_method="ai_multimodal"
)

metrics.record_highlight_confidence(
    confidence=0.92,
    detection_method="video_analysis",
    platform="twitch"
)

# API metrics
metrics.increment_api_key_usage(
    api_key_id="key_abc",
    organization_id="org_123",
    endpoint="/api/v1/streams"
)

# AI/ML metrics
metrics.record_ai_tokens_used(
    tokens=1500,
    provider="openai",
    model="gpt-4",
    organization_id="org_123"
)
```

#### Custom Metrics
```python
from src.infrastructure.observability import gauge, histogram

# Set a gauge value
gauge("active_connections", 42, service="websocket")

# Record a histogram value
histogram("request_size_bytes", 1024, endpoint="/upload")
```

### 5. Structured Logging

```python
import logfire

# Log with structured data
logfire.info(
    "Processing started",
    stream_id="stream_123",
    platform="youtube",
    organization_id="org_123"
)

# Log errors with context
try:
    await process()
except Exception as e:
    logfire.error(
        "Processing failed",
        stream_id="stream_123",
        error_type=type(e).__name__,
        exc_info=True  # Include stack trace
    )
```

### 6. Adding Context

```python
from src.infrastructure.observability.logfire_setup import (
    add_user_context,
    add_processing_context,
    set_correlation_id
)

# Add user context for all operations
add_user_context(
    organization_id="org_123",
    user_id="user_456",
    api_key_id="key_abc"
)

# Add processing context
add_processing_context(
    stream_id="stream_123",
    platform="twitch",
    processing_stage="frame_extraction"
)

# Set correlation ID for request tracking
set_correlation_id("req_789xyz")
```

## Best Practices

### 1. Use Semantic Naming
```python
# Good: Descriptive, hierarchical names
with with_span("stream.process.extract_frames"):
    pass

# Bad: Generic names
with with_span("process"):
    pass
```

### 2. Add Relevant Attributes
```python
# Add context that helps debugging
logfire.set_attribute("user.organization_id", org_id)
logfire.set_attribute("stream.platform", platform)
logfire.set_attribute("processing.frame_count", frame_count)
```

### 3. Track Business KPIs
```python
# Track metrics that matter to the business
metrics.increment_highlights_detected(...)
metrics.record_api_latency(...)
metrics.increment_webhook_sent(...)
```

### 4. Handle Sensitive Data
```python
# Don't log sensitive information
logfire.info("User login", user_email=mask_email(email))

# Use LOGFIRE_CAPTURE_BODY=false in production
```

### 5. Use Appropriate Log Levels
```python
logfire.debug("Detailed debugging info")
logfire.info("Normal operational messages")
logfire.warning("Warning conditions")
logfire.error("Error conditions", exc_info=True)
```

## Dashboards and Alerts

### Available Dashboards

1. **System Overview**
   - Request rate and latency
   - Error rates
   - Active streams
   - Resource utilization

2. **Stream Processing**
   - Streams by platform
   - Processing duration
   - Highlight detection rates
   - Success/failure rates

3. **API Performance**
   - Endpoint latency
   - Request volume by organization
   - Rate limit violations
   - Error breakdown

4. **Business Metrics**
   - Highlights detected per hour
   - Webhook delivery success
   - AI token usage
   - Storage consumption

### Setting Up Alerts

Configure alerts in Logfire dashboard for:
- High error rates (> 1%)
- Slow response times (> 1s)
- Failed stream processing
- Webhook delivery failures
- Resource exhaustion

## Troubleshooting

### Debug Logfire Configuration
```python
# Check if Logfire is enabled
if settings.logfire_enabled:
    print("Logfire is enabled")

# Test span creation
with logfire.span("test_span") as span:
    logfire.info("Test message")
```

### Common Issues

1. **No data in Logfire**
   - Check LOGFIRE_ENABLED=true
   - Verify LOGFIRE_API_KEY is set
   - Check network connectivity

2. **Missing traces**
   - Ensure decorators are applied
   - Check middleware is added
   - Verify instrumentation is enabled

3. **Performance impact**
   - Use sampling for high-volume endpoints
   - Disable body capture in production
   - Reduce span attributes

## Examples

See the following examples for implementation patterns:
- `/examples/logfire_usage_example.py` - Comprehensive usage examples
- `/examples/logfire_service_instrumentation.py` - Service instrumentation patterns

## Resources

- [Pydantic Logfire Documentation](https://logfire.pydantic.dev/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Logfire Dashboard](https://logfire.pydantic.dev/dashboard)