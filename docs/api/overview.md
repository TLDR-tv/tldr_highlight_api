# API Overview

The TL;DR Highlight API is an enterprise B2B service that provides AI-powered highlight extraction from livestreams and video content. It's designed for business clients who need to integrate highlight detection into their own platforms and workflows.

## 🚀 Key Features

- **Real-time livestream processing** with multi-modal analysis
- **AI-powered highlight detection** using Google Gemini 2.0 Flash
- **Flexible dimension-based scoring** for custom highlight types
- **Enterprise-grade security** with API key authentication and scoped permissions
- **Comprehensive webhook system** for event-driven architectures
- **Multi-tenant architecture** with isolated processing per organization
- **Rate limiting and quota management** for fair usage
- **Detailed observability** with Logfire integration

## 🌐 Base URLs

```
Production: https://api.tldr.tv/api/v1
Development: http://localhost:8000/api/v1
```

## 🔐 Authentication

The API uses API key authentication with scoped permissions. All requests (except health checks) require a valid API key.

### API Key Format
```
X-API-Key: tldr_sk_[32-character-string]
```

### Authentication Flow
1. **Register** for an account to receive your first API key
2. **Include the API key** in the `X-API-Key` header for all requests
3. **Manage additional keys** through the API key management endpoints

### Scopes
API keys support granular permission scopes:

| Scope | Description |
|-------|-------------|
| `streams:read` | Read stream information and status |
| `streams:write` | Create and manage streams |
| `streams:delete` | Delete streams |
| `highlights:read` | Read highlight information |
| `highlights:write` | Update highlight metadata |
| `highlights:delete` | Delete highlights |
| `webhooks:read` | Read webhook configurations |
| `webhooks:write` | Create and manage webhooks |
| `webhooks:delete` | Delete webhooks |
| `organizations:read` | Read organization information |
| `organizations:write` | Update organization settings |
| `users:read` | Read user information |
| `users:write` | Update user information |
| `admin` | Full administrative access |

## ⚡ Rate Limiting

### Rate Limits
The API implements sliding window rate limiting:

- **Per-minute limit**: 100 requests per minute (default)
- **Per-hour limit**: 1000 requests per hour (default)
- **Burst limit**: 20 requests per 10 seconds

Rate limits may vary based on your API key tier and organization plan.

### Rate Limit Headers
```http
X-RateLimit-Limit-Minute: 100
X-RateLimit-Limit-Hour: 1000
X-RateLimit-Remaining-Minute: 95
X-RateLimit-Remaining-Hour: 987
```

### Rate Limit Exceeded Response
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded: 100 requests per minute",
    "status_code": 429,
    "details": {
      "remaining_minute": 0,
      "remaining_hour": 845,
      "limit_minute": 100,
      "limit_hour": 1000,
      "reset_minute": "2024-01-15T10:45:00Z",
      "reset_hour": "2024-01-15T11:00:00Z"
    }
  }
}
```

## 📨 Request Format

### Content Type
All requests should use `application/json` content type:
```http
Content-Type: application/json
```

### Request Headers
```http
X-API-Key: your-api-key-here
Content-Type: application/json
User-Agent: YourApp/1.0
```

### Example Request
```bash
curl -X POST "https://api.tldr.tv/api/v1/streams" \
  -H "X-API-Key: tldr_sk_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://twitch.tv/example_stream",
    "processing_options": {
      "enable_video_analysis": true,
      "enable_audio_analysis": true,
      "enable_chat_analysis": false
    }
  }'
```

## 📄 Response Format

### Success Response
```json
{
  "success": true,
  "data": {
    "id": "123",
    "status": "processing",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "status_code": 400,
    "details": {
      "field": "url",
      "reason": "Invalid URL format"
    }
  }
}
```

## 🚦 HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| `200` | OK - Request successful |
| `201` | Created - Resource created successfully |
| `204` | No Content - Request successful, no response body |
| `400` | Bad Request - Invalid request parameters |
| `401` | Unauthorized - Invalid or missing API key |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource not found |
| `409` | Conflict - Resource already exists |
| `422` | Unprocessable Entity - Invalid data format |
| `429` | Too Many Requests - Rate limit exceeded |
| `500` | Internal Server Error - Server error |
| `502` | Bad Gateway - Upstream service error |
| `503` | Service Unavailable - Service temporarily unavailable |

## 📋 Common Error Codes

| Error Code | Description |
|------------|-------------|
| `AUTHENTICATION_REQUIRED` | Missing or invalid API key |
| `INSUFFICIENT_PERMISSIONS` | API key lacks required scopes |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `VALIDATION_ERROR` | Invalid request parameters |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist |
| `DUPLICATE_RESOURCE` | Resource already exists |
| `PROCESSING_ERROR` | Error during stream processing |
| `QUOTA_EXCEEDED` | Organization quota exceeded |
| `SERVICE_UNAVAILABLE` | Temporary service outage |

## 🔄 Pagination

List endpoints support cursor-based pagination:

### Request Parameters
```http
GET /api/v1/streams?page=2&per_page=50&sort=created_at&order=desc
```

### Response Format
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 2,
    "per_page": 50,
    "total": 250,
    "total_pages": 5,
    "has_next": true,
    "has_previous": true,
    "next_cursor": "eyJ0aW1lc3RhbXAiOiIyMDI0LTAxLTE1VDEwOjMwOjAwWiJ9",
    "previous_cursor": "eyJ0aW1lc3RhbXAiOiIyMDI0LTAxLTE0VDEwOjMwOjAwWiJ9"
  }
}
```

## 🔍 Filtering and Sorting

Most list endpoints support filtering and sorting:

### Filtering
```http
GET /api/v1/streams?status=completed&created_after=2024-01-01&platform=twitch
```

### Sorting
```http
GET /api/v1/highlights?sort=confidence_score&order=desc
```

### Common Filter Parameters
- `created_after` / `created_before` - Date range filtering
- `updated_after` / `updated_before` - Update date filtering
- `status` - Resource status filtering
- `platform` - Platform-specific filtering
- `search` - Text search in relevant fields

## 📦 SDKs and Libraries

Official SDKs are available for popular programming languages:

- **Python**: `pip install tldr-highlight-api`
- **Node.js**: `npm install tldr-highlight-api`
- **Go**: `go get github.com/tldr-tv/go-sdk`
- **Java**: Maven/Gradle integration available
- **C#**: NuGet package available

### Python SDK Example
```python
from tldr_highlight_api import TLDRClient

client = TLDRClient(api_key="tldr_sk_your_api_key_here")

# Create a stream
stream = client.streams.create(
    url="https://twitch.tv/example_stream",
    processing_options={
        "enable_video_analysis": True,
        "enable_audio_analysis": True
    }
)

# Get highlights
highlights = client.highlights.list(stream_id=stream.id)
```

### Node.js SDK Example
```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({
  apiKey: 'tldr_sk_your_api_key_here'
});

// Create a stream
const stream = await client.streams.create({
  url: 'https://twitch.tv/example_stream',
  processingOptions: {
    enableVideoAnalysis: true,
    enableAudioAnalysis: true
  }
});

// Get highlights
const highlights = await client.highlights.list({
  streamId: stream.id
});
```

## 🔐 Content Delivery

The API provides secure content delivery through signed URLs, enabling external users (like streamers) to access their content without requiring API authentication.

### Signed URL Access

```http
GET /api/v1/content/{highlight_id}?token={jwt}    # Access specific highlight
GET /api/v1/content/stream/{stream_id}?token={jwt} # List stream highlights
```

### Key Features
- **No Authentication Required**: Access via signed JWT tokens
- **Per-Organization Keys**: Each organization has unique signing keys  
- **Advanced Security**: Token revocation, IP restrictions, usage limits
- **Flexible Expiration**: Configure token lifetime (hours to days)

See the [Content Security documentation](../content_delivery/content_security.md) for implementation details.

## 🔔 Webhooks

The API supports webhooks for event-driven integrations. See the [Webhooks API documentation](./webhooks.md) for detailed information.

### Supported Events
- `stream.created` - New stream processing started
- `stream.completed` - Stream processing finished
- `stream.failed` - Stream processing failed
- `highlight.detected` - New highlight detected
- `highlight.processed` - Highlight processing completed

## 🏥 Health Checks

Monitor API availability with health check endpoints:

```http
GET /health/                # Comprehensive health check
GET /health/live           # Simple liveness check
GET /health/ready          # Readiness check
GET /health/database       # Database health
GET /health/redis          # Redis health  
GET /health/storage        # S3 storage health
```

## 📚 Next Steps

- [Authentication Guide](./authentication.md) - Detailed authentication setup
- [Streams API](./streams.md) - Stream processing endpoints
- [Highlights API](./highlights.md) - Highlight management endpoints
- [Webhooks API](./webhooks.md) - Event notification system
- [Error Handling](./errors.md) - Comprehensive error documentation

---

*Need help? Contact our support team or check the [troubleshooting guide](../troubleshooting/common-issues.md).*