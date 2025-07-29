# TL;DR Highlight API - Quick Reference

## 🚀 Quick Start

### 1. Get API Key
```bash
curl -X POST https://api.tldr-highlight.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "your@email.com", "company": "Your Company"}'
```

### 2. Process a Stream
```bash
curl -X POST https://api.tldr-highlight.com/v1/streams \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "https://twitch.tv/streamername",
    "options": {
      "analysis_quality": "high",
      "sensitivity": "medium"
    }
  }'
```

### 3. Get Highlights
```bash
curl https://api.tldr-highlight.com/v1/streams/{stream_id}/highlights \
  -H "X-API-Key: your_api_key"
```

## 📚 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/auth/register` | Register new account |
| POST | `/v1/auth/login` | Login to get API key |
| GET | `/v1/api-keys` | List your API keys |
| POST | `/v1/api-keys` | Create new API key |
| DELETE | `/v1/api-keys/{key_id}` | Revoke API key |

### Stream Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/streams` | Start stream processing |
| GET | `/v1/streams/{id}` | Get stream status |
| GET | `/v1/streams/{id}/highlights` | Get stream highlights |
| POST | `/v1/streams/{id}/stop` | Stop stream processing |
| DELETE | `/v1/streams/{id}` | Delete stream and highlights |

### Batch Processing
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/batch` | Upload videos for processing |
| GET | `/v1/batch/{id}` | Get batch status |
| GET | `/v1/batch/{id}/highlights` | Get batch highlights |
| DELETE | `/v1/batch/{id}` | Cancel batch processing |

### Highlights
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/highlights` | List all your highlights |
| GET | `/v1/highlights/{id}` | Get highlight details |
| PUT | `/v1/highlights/{id}` | Update highlight metadata |
| DELETE | `/v1/highlights/{id}` | Delete highlight |

### Webhooks
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/webhooks` | List webhook configurations |
| POST | `/v1/webhooks` | Create webhook |
| PUT | `/v1/webhooks/{id}` | Update webhook |
| DELETE | `/v1/webhooks/{id}` | Delete webhook |
| POST | `/v1/webhooks/{id}/test` | Test webhook |

### Usage & Billing
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/usage` | Get current usage stats |
| GET | `/v1/usage/history` | Get usage history |
| GET | `/v1/billing` | Get billing information |
| GET | `/v1/billing/invoices` | List invoices |

## 🔧 Request Examples

### Stream Processing with All Options
```json
POST /v1/streams
{
  "source_url": "https://twitch.tv/streamername",
  "platform": "twitch",
  "options": {
    "analysis_quality": "high",
    "sensitivity": "high",
    "clip_duration": 30,
    "include_chat": true,
    "include_audio": true,
    "custom_tags": ["gaming", "fps", "tournament"],
    "webhook_url": "https://your-domain.com/webhook",
    "frame_interval": 1.0,
    "max_highlights": 20
  },
  "metadata": {
    "tournament": "Summer Championship",
    "round": "semifinals"
  }
}
```

### Batch Upload
```json
POST /v1/batch
{
  "video_urls": [
    "https://example.com/video1.mp4",
    "https://example.com/video2.mp4"
  ],
  "options": {
    "analysis_quality": "premium",
    "sensitivity": "medium",
    "parallel_processing": true,
    "max_highlights_per_video": 5
  }
}
```

### Webhook Configuration
```json
POST /v1/webhooks
{
  "url": "https://your-domain.com/webhook",
  "events": [
    "highlight.created",
    "stream.completed",
    "batch.completed"
  ],
  "secret": "your_webhook_secret",
  "active": true
}
```

## 📊 Response Examples

### Stream Response
```json
{
  "id": "str_1234567890ab",
  "status": "processing",
  "source_url": "https://twitch.tv/streamername",
  "platform": "twitch",
  "options": {
    "analysis_quality": "high",
    "sensitivity": "high",
    "clip_duration": 30
  },
  "progress": {
    "processed_duration": 1800,
    "total_duration": null,
    "highlights_found": 12
  },
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Highlight Response
```json
{
  "id": "hl_abc123def456",
  "stream_id": "str_1234567890ab",
  "title": "Epic Triple Kill",
  "description": "Player achieves triple elimination with sniper rifle",
  "video_url": "https://cdn.tldr-highlight.com/highlights/hl_abc123def456.mp4",
  "thumbnail_url": "https://cdn.tldr-highlight.com/thumbnails/hl_abc123def456.jpg",
  "duration": 30,
  "timestamp": 1234,
  "confidence_score": 0.95,
  "tags": ["action", "elimination", "sniper"],
  "metadata": {
    "game": "Valorant",
    "chat_activity": "high",
    "audio_peaks": true,
    "analysis_quality": "high"
  },
  "created_at": "2024-01-15T10:20:34Z"
}
```

## 🛠️ SDK Usage

### Python
```python
from tldr_highlight import TLDRClient

client = TLDRClient(api_key="your_api_key")

# Process stream
stream = client.streams.create(
    url="https://twitch.tv/streamername",
    options={"analysis_quality": "high"}
)

# Get highlights
highlights = client.streams.get_highlights(stream.id)
```

### Node.js
```javascript
const { TLDRClient } = require('@tldr/highlight-sdk');

const client = new TLDRClient({ apiKey: 'your_api_key' });

// Process stream
const stream = await client.streams.create({
  url: 'https://twitch.tv/streamername',
  options: { analysisQuality: 'high' }
});

// Get highlights
const highlights = await client.streams.getHighlights(stream.id);
```

### Go
```go
import "github.com/tldr/highlight-sdk-go"

client := tldr.NewClient("your_api_key")

// Process stream
stream, err := client.Streams.Create(&tldr.StreamCreateRequest{
    URL: "https://twitch.tv/streamername",
    Options: tldr.StreamOptions{
        AnalysisQuality: "high",
    },
})

// Get highlights
highlights, err := client.Streams.GetHighlights(stream.ID)
```

## 🔑 Authentication

### Headers
```
X-API-Key: your_api_key_here
```

### API Key Scopes
- `streams.read` - Read stream data
- `streams.write` - Create/manage streams
- `batch.read` - Read batch data
- `batch.write` - Create/manage batches
- `webhooks.manage` - Manage webhooks
- `billing.read` - View billing info

## ⚡ Rate Limits

| Plan | Requests/Hour | Concurrent Streams | Storage |
|------|---------------|-------------------|---------|
| Free | 100 | 1 | 1 GB |
| Starter | 1,000 | 5 | 10 GB |
| Pro | 10,000 | 20 | 100 GB |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Rate Limit Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 998
X-RateLimit-Reset: 1642512000
```

## 🎯 Analysis Quality Levels

| Level | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `premium` | Medium | Excellent | High-quality highlights |
| `high` | Medium | Very Good | Professional content |
| `standard` | Fast | Good | Balanced performance |
| `fast` | Very Fast | Acceptable | Real-time processing |

## 📡 Webhook Events

| Event | Description |
|-------|-------------|
| `highlight.created` | New highlight detected |
| `stream.started` | Stream processing started |
| `stream.completed` | Stream processing finished |
| `stream.error` | Stream processing error |
| `batch.completed` | Batch processing finished |
| `batch.error` | Batch processing error |

### Webhook Payload Example
```json
{
  "id": "evt_123456",
  "type": "highlight.created",
  "created": 1642512000,
  "data": {
    "highlight": {
      "id": "hl_abc123",
      "title": "Amazing Play",
      "video_url": "https://cdn.tldr-highlight.com/hl_abc123.mp4"
    }
  }
}
```

### Webhook Signature Verification
```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## 🚨 Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 429 | Too Many Requests - Rate limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "The analysis_quality field must be one of: premium, high, standard, fast",
    "details": {
      "field": "options.analysis_quality",
      "value": "invalid-quality"
    }
  }
}
```

## 📈 Status Codes

### Stream Status
- `pending` - Waiting to start
- `processing` - Currently processing
- `completed` - Successfully completed
- `error` - Processing failed
- `cancelled` - Cancelled by user

### Batch Status
- `uploading` - Files uploading
- `queued` - In processing queue
- `processing` - Currently processing
- `completed` - Successfully completed
- `error` - Processing failed

## 🔗 Useful Links

- **Documentation**: https://docs.tldr-highlight.com
- **API Reference**: https://api.tldr-highlight.com/docs
- **Status Page**: https://status.tldr-highlight.com
- **Support**: support@tldr-highlight.com
- **SDK Examples**: https://github.com/tldr-highlight/examples