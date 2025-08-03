# Rate Limiting

The TLDR Highlight API implements comprehensive rate limiting to protect against abuse and ensure fair usage across all clients.

## Overview

Rate limiting is enforced at multiple levels:
- **Global Rate Limit**: Protects the entire API from DoS attacks
- **Endpoint-Specific Limits**: Stricter limits on sensitive operations
- **Organization-Based Limits**: Different limits based on billing tier

## Rate Limit Headers

All API responses include rate limit information in headers:

```
X-RateLimit-Limit: 100        # Maximum requests allowed
X-RateLimit-Remaining: 45     # Requests remaining in current window
X-RateLimit-Reset: 1648839600  # Unix timestamp when limit resets
Retry-After: 60               # Seconds to wait (only on 429 responses)
```

## Default Limits

### Global Limits
- **Unauthenticated requests**: 1000 requests/minute per IP
- **Authenticated requests**: Based on organization tier

### Endpoint-Specific Limits
| Endpoint | Limit | Purpose |
|----------|-------|---------|
| `/auth/login` | 5/minute | Prevent brute force attacks |
| `/auth/register` | 5/hour | Prevent spam registrations |
| `/auth/forgot-password` | 3/hour | Prevent email bombing |
| `/auth/reset-password` | 5/hour | Prevent token guessing |
| `/streams` (POST) | 20/minute | Control resource usage |
| `/streams/{id}/process` | 10/minute | Limit expensive operations |
| `/webhooks` | 5/hour | Prevent configuration spam |

### Organization Tier Limits
| Tier | Limit | Use Case |
|------|-------|----------|
| Free | 100/minute | Development and testing |
| Pro | 1000/minute | Production applications |
| Enterprise | 10000/minute | High-volume applications |

## Rate Limit Exceeded Response

When rate limit is exceeded, the API returns:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 60
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1648839600

{
  "detail": "Rate limit exceeded",
  "retry_after": 60,
  "error_code": "RATE_LIMIT_EXCEEDED"
}
```

## Best Practices

### 1. Handle 429 Responses
```python
import time
import httpx

async def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = await client.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        return response
    
    raise Exception("Max retries exceeded")
```

### 2. Monitor Rate Limit Headers
```python
def check_rate_limit_status(response):
    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
    limit = int(response.headers.get("X-RateLimit-Limit", 0))
    
    if remaining < limit * 0.2:  # Less than 20% remaining
        print(f"Warning: Only {remaining}/{limit} requests remaining")
```

### 3. Implement Exponential Backoff
```python
import asyncio
import random

async def exponential_backoff_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitExceeded:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
```

### 4. Use Webhooks for Status Updates
Instead of polling endpoints repeatedly, use webhooks to receive real-time updates:

```python
# Bad: Polling every second
while stream.status != "completed":
    stream = await client.get(f"/streams/{stream_id}")
    time.sleep(1)  # This will hit rate limits!

# Good: Use webhooks
await client.post("/webhooks", json={
    "url": "https://your-app.com/webhook",
    "events": ["stream.completed", "highlight.detected"]
})
```

## Configuration

Rate limits can be configured via environment variables:

```bash
# Enable/disable rate limiting
RATE_LIMIT_ENABLED=true

# Redis backend for distributed rate limiting
RATE_LIMIT_STORAGE_URL=redis://localhost:6379/3

# Global limits
RATE_LIMIT_GLOBAL=1000/minute
RATE_LIMIT_BURST=20

# Endpoint-specific limits
RATE_LIMIT_AUTH=5/minute
RATE_LIMIT_STREAM_CREATE=20/minute
RATE_LIMIT_STREAM_PROCESS=10/minute

# Organization tiers
RATE_LIMIT_TIER_FREE=100/minute
RATE_LIMIT_TIER_PRO=1000/minute
RATE_LIMIT_TIER_ENTERPRISE=10000/minute
```

## Rate Limit Bypass

Internal services and monitoring systems can bypass rate limits using special headers or API keys. Contact support for bypass credentials if you have a valid use case.

## Monitoring

Rate limit violations are logged and monitored. Repeated violations may result in temporary IP bans or account suspension. Use the rate limit headers to stay within limits.

## Future Enhancements

- **Cost-based limiting**: Different weights for expensive operations
- **Burst allowances**: Allow short bursts above the sustained rate
- **Per-endpoint quotas**: Daily/monthly quotas for specific operations
- **Geographic rate limiting**: Different limits by region