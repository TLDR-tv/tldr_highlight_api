# Streams API

The Streams API provides endpoints for processing livestreams and video content with AI-powered highlight detection.

## Overview

Streams are the core processing unit in the TL;DR Highlight API. When you create a stream:

- The system ingests video content from the provided URL
- AI models analyze video, audio, and chat (if available)
- Highlights are automatically detected based on configurable criteria
- Results are delivered via webhooks or polling

## Supported Platforms

The API supports multiple streaming platforms and formats:

| Platform | URL Format | Features |
|----------|------------|----------|
| **Twitch** | `https://twitch.tv/channel` | Live streams, VODs, clips, chat analysis |
| **YouTube** | `https://youtube.com/watch?v=...` | Live streams, videos, premiere events |
| **Custom RTMP** | `rtmp://server/app/stream` | Direct RTMP ingestion |
| **HLS** | `https://example.com/stream.m3u8` | HTTP Live Streaming |
| **Direct Files** | `https://example.com/video.mp4` | MP4, WebM, MOV, AVI files |

## Endpoints

### Get Stream Service Status

Check if the stream processing service is operational.

```http
GET /api/v1/streams/status
```

#### Response

```json
{
  "status": "Stream processing service operational",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Create Stream

Start processing a livestream or video with AI-powered highlight detection.

```http
POST /api/v1/streams
```

#### Request Body

```json
{
  "source_url": "https://twitch.tv/shroud",
  "platform": "twitch",
  "options": {
    "highlight_threshold": 0.8,
    "min_duration": 15,
    "max_duration": 60,
    "processing_modules": {
      "video_analysis": true,
      "audio_analysis": true,
      "chat_analysis": true,
      "multimodal_fusion": true
    },
    "dimension_set_id": "gaming_default",
    "custom_dimensions": {
      "action_intensity": {
        "weight": 0.3,
        "min_threshold": 0.7
      },
      "audience_reaction": {
        "weight": 0.2,
        "min_threshold": 0.6
      }
    }
  },
  "metadata": {
    "title": "Shroud VALORANT Ranked",
    "tags": ["valorant", "fps", "competitive"],
    "language": "en",
    "streamer_name": "shroud"
  },
  "webhooks": {
    "on_highlight": "https://api.example.com/webhooks/highlight",
    "on_complete": "https://api.example.com/webhooks/complete",
    "on_error": "https://api.example.com/webhooks/error"
  }
}
```

#### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_url` | string | Yes | URL of the stream or video to process |
| `platform` | string | No | Platform hint: `twitch`, `youtube`, `custom` (auto-detected if not provided) |
| `options` | object | No | Processing configuration |
| `options.highlight_threshold` | float | No | Confidence threshold (0.0-1.0, default: 0.8) |
| `options.min_duration` | integer | No | Minimum highlight duration in seconds (default: 15) |
| `options.max_duration` | integer | No | Maximum highlight duration in seconds (default: 60) |
| `options.processing_modules` | object | No | Enable/disable specific analysis modules |
| `options.dimension_set_id` | string | No | Predefined dimension set: `gaming_default`, `education_default`, etc. |
| `options.custom_dimensions` | object | No | Custom scoring dimensions with weights |
| `metadata` | object | No | Additional metadata for the stream |
| `webhooks` | object | No | Custom webhook URLs for this stream |

#### Response

```json
{
  "id": 12345,
  "source_url": "https://twitch.tv/shroud",
  "platform": "twitch",
  "status": "pending",
  "options": {
    "highlight_threshold": 0.8,
    "min_duration": 15,
    "max_duration": 60
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "is_active": true,
  "highlight_count": 0,
  "processing_progress": {
    "percentage": 0,
    "current_timestamp": 0,
    "total_duration": null,
    "estimated_completion": null
  },
  "stream_info": {
    "title": "Shroud VALORANT Ranked",
    "thumbnail_url": "https://static-cdn.jtvnw.net/previews-ttv/live_user_shroud.jpg",
    "viewer_count": 45678,
    "is_live": true,
    "started_at": "2024-01-15T09:00:00Z"
  }
}
```

#### Status Values

| Status | Description |
|--------|-------------|
| `pending` | Stream created, waiting to start processing |
| `processing` | Actively processing the stream |
| `completed` | Processing finished successfully |
| `failed` | Processing failed with errors |
| `stopped` | Processing was manually stopped |

#### Error Responses

- `400 Bad Request` - Invalid URL or parameters
- `401 Unauthorized` - Invalid or missing API key
- `403 Forbidden` - Insufficient permissions
- `409 Conflict` - Stream URL already being processed
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit or quota exceeded

### List Streams

Get a list of streams for the authenticated user.

```http
GET /api/v1/streams
```

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |
| `status` | string | No | Filter by status: `pending`, `processing`, `completed`, `failed` |
| `platform` | string | No | Filter by platform: `twitch`, `youtube`, `custom` |
| `created_after` | datetime | No | Filter streams created after this date |
| `created_before` | datetime | No | Filter streams created before this date |
| `sort` | string | No | Sort by: `created_at`, `updated_at`, `highlight_count` |
| `order` | string | No | Sort order: `asc`, `desc` (default: `desc`) |

#### Response

```json
{
  "items": [
    {
      "id": 12345,
      "source_url": "https://twitch.tv/shroud",
      "platform": "twitch",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T12:45:00Z",
      "is_active": false,
      "highlight_count": 23,
      "duration_seconds": 8100,
      "metadata": {
        "title": "Shroud VALORANT Ranked",
        "streamer_name": "shroud"
      }
    },
    {
      "id": 12344,
      "source_url": "https://youtube.com/watch?v=abc123",
      "platform": "youtube",
      "status": "processing",
      "created_at": "2024-01-15T09:00:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "is_active": true,
      "highlight_count": 5,
      "processing_progress": {
        "percentage": 45,
        "current_timestamp": 3600,
        "total_duration": 8000
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 145,
    "pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

### Get Stream Details

Get detailed information about a specific stream.

```http
GET /api/v1/streams/{stream_id}
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `stream_id` | integer | path | Yes | Stream ID |

#### Response

```json
{
  "id": 12345,
  "source_url": "https://twitch.tv/shroud",
  "platform": "twitch",
  "status": "completed",
  "options": {
    "highlight_threshold": 0.8,
    "min_duration": 15,
    "max_duration": 60,
    "processing_modules": {
      "video_analysis": true,
      "audio_analysis": true,
      "chat_analysis": true,
      "multimodal_fusion": true
    }
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T12:45:00Z",
  "started_at": "2024-01-15T10:31:00Z",
  "completed_at": "2024-01-15T12:45:00Z",
  "is_active": false,
  "highlight_count": 23,
  "duration_seconds": 8100,
  "processing_time_seconds": 8040,
  "stream_info": {
    "title": "Shroud VALORANT Ranked",
    "description": "Climbing to Radiant",
    "thumbnail_url": "https://static-cdn.jtvnw.net/previews-ttv/live_user_shroud.jpg",
    "language": "en",
    "tags": ["valorant", "fps", "competitive"],
    "category": "VALORANT",
    "streamer": {
      "id": "37402112",
      "username": "shroud",
      "display_name": "shroud",
      "profile_image_url": "https://static-cdn.jtvnw.net/jtv_user_pictures/shroud.png"
    }
  },
  "statistics": {
    "total_frames_processed": 486000,
    "total_audio_segments": 8100,
    "chat_messages_analyzed": 45678,
    "average_highlight_confidence": 0.87,
    "processing_speed_multiplier": 1.01,
    "dimension_scores": {
      "action_intensity": {
        "average": 0.72,
        "max": 0.98,
        "distribution": [0.1, 0.2, 0.3, 0.25, 0.15]
      },
      "audience_reaction": {
        "average": 0.65,
        "max": 0.95,
        "distribution": [0.15, 0.3, 0.35, 0.15, 0.05]
      }
    }
  },
  "errors": [],
  "metadata": {
    "title": "Shroud VALORANT Ranked",
    "tags": ["valorant", "fps", "competitive"],
    "custom_data": {
      "tournament": "Ranked Grind",
      "session": 3
    }
  }
}
```

#### Error Responses

- `404 Not Found` - Stream not found
- `403 Forbidden` - User doesn't own this stream

### Stop Stream Processing

Stop processing a stream and finalize any extracted highlights.

```http
DELETE /api/v1/streams/{stream_id}
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `stream_id` | integer | path | Yes | Stream ID |
| `force` | boolean | query | No | Force stop even if processing incomplete (default: false) |

#### Response

```json
{
  "status": "Stream 12345 stopped successfully. 18 highlights extracted.",
  "timestamp": "2024-01-15T11:30:00Z"
}
```

#### Error Responses

- `404 Not Found` - Stream not found
- `403 Forbidden` - User doesn't own this stream
- `422 Unprocessable Entity` - Stream not in a stoppable state

### Update Stream Configuration

Update processing options for a stream (only allowed while status is `pending`).

```http
PATCH /api/v1/streams/{stream_id}
```

#### Request Body

```json
{
  "options": {
    "highlight_threshold": 0.9,
    "max_duration": 90
  },
  "metadata": {
    "tags": ["valorant", "fps", "competitive", "tournament"]
  }
}
```

#### Response

Returns the updated stream object.

#### Error Responses

- `404 Not Found` - Stream not found
- `403 Forbidden` - User doesn't own this stream
- `422 Unprocessable Entity` - Stream already processing
- `501 Not Implemented` - Feature not yet available

## Processing Options

### Highlight Detection Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `highlight_threshold` | float | 0.8 | Minimum confidence score (0.0-1.0) |
| `min_duration` | integer | 15 | Minimum highlight duration in seconds |
| `max_duration` | integer | 60 | Maximum highlight duration in seconds |
| `merge_threshold` | integer | 5 | Merge highlights within N seconds |
| `buffer_before` | integer | 2 | Seconds to include before highlight |
| `buffer_after` | integer | 3 | Seconds to include after highlight |

### Processing Modules

Enable or disable specific analysis modules:

| Module | Description | Default |
|--------|-------------|---------|
| `video_analysis` | Analyze video frames for visual events | true |
| `audio_analysis` | Analyze audio for peaks, speech, music | true |
| `chat_analysis` | Analyze chat for reactions (Twitch/YouTube) | true |
| `multimodal_fusion` | Combine signals from multiple modules | true |

### Dimension Sets

Pre-configured dimension sets for different content types:

| Set ID | Description | Best For |
|--------|-------------|----------|
| `gaming_default` | Action, reactions, achievements | Gaming streams |
| `education_default` | Key concepts, Q&A, summaries | Educational content |
| `sports_default` | Goals, plays, crowd reactions | Sports broadcasts |
| `music_default` | Performances, crowd energy | Music streams |
| `corporate_default` | Key points, Q&A, decisions | Corporate meetings |

## Webhooks

Configure webhooks to receive real-time notifications:

### Stream-Level Webhooks

Set custom webhooks for individual streams:

```json
{
  "webhooks": {
    "on_highlight": "https://api.example.com/webhooks/highlight",
    "on_complete": "https://api.example.com/webhooks/complete",
    "on_error": "https://api.example.com/webhooks/error",
    "on_progress": "https://api.example.com/webhooks/progress"
  }
}
```

### Webhook Payloads

#### Highlight Detected

```json
{
  "event": "stream.highlight_detected",
  "stream_id": 12345,
  "highlight": {
    "id": 67890,
    "start_time": 3600,
    "end_time": 3645,
    "duration": 45,
    "confidence": 0.92,
    "type": "gameplay_highlight",
    "dimensions": {
      "action_intensity": 0.95,
      "audience_reaction": 0.88
    }
  },
  "timestamp": "2024-01-15T11:00:00Z"
}
```

#### Stream Completed

```json
{
  "event": "stream.completed",
  "stream_id": 12345,
  "status": "completed",
  "duration_seconds": 8100,
  "highlight_count": 23,
  "processing_time_seconds": 8040,
  "timestamp": "2024-01-15T12:45:00Z"
}
```

## Stream Quotas and Limits

Limits vary by subscription plan:

| Limit | STARTER | PROFESSIONAL | ENTERPRISE |
|-------|---------|--------------|------------|
| Concurrent streams | 2 | 10 | Unlimited |
| Monthly stream hours | 100 | 1,000 | Unlimited |
| Max stream duration | 4 hours | 12 hours | Unlimited |
| Processing priority | Standard | Priority | Dedicated |

## Examples

### Process a Twitch Stream

```python
from tldr_highlight_api import TLDRClient

client = TLDRClient(api_key="tldr_sk_your_api_key")

# Start processing a Twitch stream
stream = client.streams.create(
    source_url="https://twitch.tv/pokimane",
    options={
        "highlight_threshold": 0.85,
        "min_duration": 20,
        "max_duration": 90,
        "dimension_set_id": "gaming_default"
    },
    metadata={
        "title": "Pokimane Valorant Stream",
        "tags": ["valorant", "gaming", "variety"]
    }
)

print(f"Stream {stream.id} created with status: {stream.status}")

# Poll for completion
import time
while True:
    stream = client.streams.get(stream.id)
    if stream.status in ["completed", "failed"]:
        break
    print(f"Progress: {stream.processing_progress.percentage}%")
    time.sleep(30)

print(f"Stream completed with {stream.highlight_count} highlights")
```

### Process YouTube Video with Custom Dimensions

```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({ apiKey: 'tldr_sk_your_api_key' });

// Process educational content with custom dimensions
const stream = await client.streams.create({
  sourceUrl: 'https://youtube.com/watch?v=dQw4w9WgXcQ',
  options: {
    highlightThreshold: 0.75,
    customDimensions: {
      educational_value: {
        weight: 0.4,
        minThreshold: 0.7,
        prompt: "Score how educational or informative this segment is"
      },
      entertainment_value: {
        weight: 0.3,
        minThreshold: 0.6,
        prompt: "Score how entertaining or engaging this segment is"
      },
      memorable_moment: {
        weight: 0.3,
        minThreshold: 0.8,
        prompt: "Score how memorable or quotable this moment is"
      }
    }
  },
  webhooks: {
    onComplete: 'https://api.myapp.com/webhooks/stream-complete'
  }
});

console.log(`Processing stream ${stream.id}`);
```

### Batch Process Multiple Streams

```bash
#!/bin/bash

# List of URLs to process
urls=(
  "https://twitch.tv/ninja"
  "https://youtube.com/watch?v=abc123"
  "https://example.com/stream.m3u8"
)

# Process each URL
for url in "${urls[@]}"; do
  response=$(curl -s -X POST "https://api.tldr.tv/api/v1/streams" \
    -H "X-API-Key: tldr_sk_your_api_key" \
    -H "Content-Type: application/json" \
    -d "{
      \"source_url\": \"$url\",
      \"options\": {
        \"highlight_threshold\": 0.8,
        \"dimension_set_id\": \"gaming_default\"
      }
    }")
  
  stream_id=$(echo $response | jq -r '.id')
  echo "Created stream $stream_id for $url"
done
```

## Best Practices

### URL Validation

1. **Ensure URLs are accessible** - Test URLs before submitting
2. **Use direct URLs when possible** - Avoid redirects
3. **Check platform status** - Ensure the platform is online
4. **Respect rate limits** - Don't overwhelm source servers

### Processing Optimization

1. **Set appropriate thresholds** - Balance quality vs quantity
2. **Use dimension sets** - Leverage pre-configured settings
3. **Enable only needed modules** - Save processing time
4. **Set reasonable durations** - Avoid very short or long highlights

### Monitoring and Management

1. **Use webhooks** - Get real-time updates
2. **Implement exponential backoff** - For polling status
3. **Handle errors gracefully** - Implement retry logic
4. **Monitor quotas** - Track usage against limits

## Troubleshooting

### Common Issues

1. **Stream URL Not Accessible**
   - Verify the URL is correct and public
   - Check if authentication is required
   - Ensure the stream is live (for livestreams)

2. **Processing Takes Too Long**
   - Check stream duration and complexity
   - Verify processing modules configuration
   - Monitor system status page

3. **No Highlights Detected**
   - Lower the highlight threshold
   - Check if content type matches dimension set
   - Verify stream has sufficient activity

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `STREAM_URL_INVALID` | URL format not recognized | Check URL format |
| `STREAM_NOT_ACCESSIBLE` | Cannot access stream | Verify stream is public |
| `QUOTA_EXCEEDED` | Monthly limit reached | Upgrade plan or wait |
| `CONCURRENT_LIMIT` | Too many active streams | Wait or stop other streams |

---

*See also: [Highlights API](./highlights.md) | [Webhooks API](./webhooks.md) | [API Overview](./overview.md)*