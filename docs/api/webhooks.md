# Webhooks API

The Webhooks API enables real-time event notifications for stream processing, highlight detection, and system events in the TL;DR Highlight API.

## Overview

Webhooks provide a powerful way to receive real-time notifications about events in your TL;DR account:

- **Real-time Updates**: Get notified instantly when events occur
- **Event-Driven Architecture**: Build reactive applications
- **Reduced Polling**: Eliminate the need for constant API polling
- **Reliable Delivery**: Automatic retries with exponential backoff
- **Secure Communication**: HMAC signature verification

## Webhook Configuration Endpoints

### Get Webhooks Service Status

Check if the webhooks service is operational.

```http
GET /api/v1/webhooks/status
```

#### Response

```json
{
  "status": "Webhooks service operational",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Create Webhook Configuration

Create a new webhook endpoint configuration.

```http
POST /api/v1/webhooks
```

#### Request Body

```json
{
  "url": "https://api.example.com/webhooks/tldr",
  "events": [
    "stream.started",
    "stream.completed",
    "highlight.detected",
    "usage.limit_reached"
  ],
  "description": "Production webhook endpoint",
  "is_active": true,
  "headers": {
    "X-Custom-Header": "custom-value"
  },
  "retry_config": {
    "max_attempts": 5,
    "initial_delay_seconds": 1,
    "max_delay_seconds": 300
  }
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | HTTPS URL to receive webhook events |
| `events` | array | Yes | List of event types to subscribe to |
| `description` | string | No | Description of the webhook |
| `is_active` | boolean | No | Whether webhook is active (default: true) |
| `headers` | object | No | Custom headers to include in requests |
| `retry_config` | object | No | Custom retry configuration |

#### Response

```json
{
  "id": "whk_1a2b3c4d",
  "url": "https://api.example.com/webhooks/tldr",
  "events": [
    "stream.started",
    "stream.completed",
    "highlight.detected",
    "usage.limit_reached"
  ],
  "description": "Production webhook endpoint",
  "is_active": true,
  "secret": "whsec_5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "last_triggered_at": null,
  "failure_count": 0,
  "success_count": 0
}
```

**Important**: Save the `secret` value - it's only shown once and is required for signature verification.

### List Webhook Configurations

Get all webhook configurations for your organization.

```http
GET /api/v1/webhooks
```

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |
| `is_active` | boolean | No | Filter by active status |

#### Response

```json
{
  "items": [
    {
      "id": "whk_1a2b3c4d",
      "url": "https://api.example.com/webhooks/tldr",
      "events": ["stream.completed", "highlight.detected"],
      "description": "Production webhook",
      "is_active": true,
      "created_at": "2024-01-15T10:30:00Z",
      "last_triggered_at": "2024-01-15T11:45:00Z",
      "failure_count": 0,
      "success_count": 145,
      "health_status": "healthy"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 3,
    "pages": 1
  }
}
```

### Get Webhook Configuration

Get details of a specific webhook configuration.

```http
GET /api/v1/webhooks/{webhook_id}
```

#### Response

```json
{
  "id": "whk_1a2b3c4d",
  "url": "https://api.example.com/webhooks/tldr",
  "events": [
    "stream.started",
    "stream.completed",
    "highlight.detected",
    "usage.limit_reached"
  ],
  "description": "Production webhook endpoint",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "last_triggered_at": "2024-01-15T11:45:00Z",
  "headers": {
    "X-Custom-Header": "custom-value"
  },
  "retry_config": {
    "max_attempts": 5,
    "initial_delay_seconds": 1,
    "max_delay_seconds": 300
  },
  "statistics": {
    "total_attempts": 1567,
    "success_count": 1456,
    "failure_count": 111,
    "average_response_time_ms": 234,
    "success_rate": 0.929
  },
  "health_status": "healthy",
  "last_error": null
}
```

### Update Webhook Configuration

Update an existing webhook configuration.

```http
PUT /api/v1/webhooks/{webhook_id}
```

#### Request Body

```json
{
  "url": "https://api.example.com/webhooks/tldr-v2",
  "events": [
    "stream.completed",
    "highlight.detected"
  ],
  "is_active": true,
  "headers": {
    "X-Custom-Header": "new-value"
  }
}
```

#### Response

Returns the updated webhook object.

### Delete Webhook Configuration

Delete a webhook configuration.

```http
DELETE /api/v1/webhooks/{webhook_id}
```

#### Response

```http
204 No Content
```

### Test Webhook

Send a test event to verify webhook configuration.

```http
POST /api/v1/webhooks/{webhook_id}/test
```

#### Request Body

```json
{
  "event_type": "test.ping"
}
```

#### Response

```json
{
  "success": true,
  "response": {
    "status_code": 200,
    "headers": {
      "content-type": "application/json"
    },
    "body": "{\"received\": true}",
    "response_time_ms": 145
  },
  "signature_valid": true
}
```

### Get Webhook Delivery History

Get the delivery history for a webhook.

```http
GET /api/v1/webhooks/{webhook_id}/deliveries
```

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |
| `status` | string | No | Filter by status: `success`, `failed` |
| `event_type` | string | No | Filter by event type |

#### Response

```json
{
  "items": [
    {
      "id": "del_9z8y7x6w",
      "webhook_id": "whk_1a2b3c4d",
      "event_id": "evt_5v4u3t2s",
      "event_type": "highlight.detected",
      "attempted_at": "2024-01-15T11:45:00Z",
      "status": "success",
      "status_code": 200,
      "response_time_ms": 145,
      "attempts": 1,
      "request": {
        "url": "https://api.example.com/webhooks/tldr",
        "headers": {
          "Content-Type": "application/json",
          "X-TlDR-Event": "highlight.detected",
          "X-TlDR-Signature": "sha256=abc123..."
        },
        "body_preview": "{\"event\":\"highlight.detected\",\"highlight\":{\"id\":67890..."
      },
      "response": {
        "status_code": 200,
        "headers": {
          "content-type": "application/json"
        },
        "body_preview": "{\"received\":true}"
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 234,
    "pages": 12
  }
}
```

## Webhook Events

### Stream Events

| Event | Description | Payload |
|-------|-------------|---------|
| `stream.started` | Stream processing has started | Stream object |
| `stream.processing` | Stream is actively being processed | Stream object with progress |
| `stream.completed` | Stream processing completed successfully | Stream object with highlights count |
| `stream.failed` | Stream processing failed | Stream object with error details |
| `stream.stopped` | Stream processing was manually stopped | Stream object |

### Highlight Events

| Event | Description | Payload |
|-------|-------------|---------|
| `highlight.detected` | New highlight detected | Highlight object |
| `highlight.processed` | Highlight video processing completed | Highlight object with URLs |
| `highlight.updated` | Highlight metadata was updated | Highlight object |
| `highlight.deleted` | Highlight was deleted | Highlight ID and metadata |

### Organization Events

| Event | Description | Payload |
|-------|-------------|---------|
| `organization.usage.warning` | Usage approaching limit (80%) | Usage statistics |
| `organization.usage.limit_reached` | Usage limit reached | Usage statistics |
| `organization.plan.upgraded` | Plan was upgraded | Organization object |
| `organization.user.added` | User added to organization | User object |
| `organization.user.removed` | User removed from organization | User ID |

### System Events

| Event | Description | Payload |
|-------|-------------|---------|
| `test.ping` | Test webhook connectivity | Test message |
| `webhook.disabled` | Webhook auto-disabled due to failures | Webhook object |

## Webhook Payload Structure

### Standard Payload Format

All webhook events follow this structure:

```json
{
  "id": "evt_1a2b3c4d5e6f",
  "type": "highlight.detected",
  "created_at": "2024-01-15T11:45:00Z",
  "data": {
    // Event-specific data
  },
  "metadata": {
    "organization_id": 123,
    "api_version": "v1",
    "retry_count": 0
  }
}
```

### Example Event Payloads

#### Stream Completed Event

```json
{
  "id": "evt_1a2b3c4d5e6f",
  "type": "stream.completed",
  "created_at": "2024-01-15T12:45:00Z",
  "data": {
    "stream": {
      "id": 12345,
      "source_url": "https://twitch.tv/shroud",
      "platform": "twitch",
      "status": "completed",
      "duration_seconds": 8100,
      "highlight_count": 23,
      "processing_time_seconds": 8040,
      "created_at": "2024-01-15T10:30:00Z",
      "completed_at": "2024-01-15T12:45:00Z"
    },
    "statistics": {
      "total_frames": 486000,
      "total_highlights": 23,
      "average_confidence": 0.87
    }
  }
}
```

#### Highlight Detected Event

```json
{
  "id": "evt_2b3c4d5e6f7g",
  "type": "highlight.detected",
  "created_at": "2024-01-15T11:00:00Z",
  "data": {
    "highlight": {
      "id": 67890,
      "stream_id": 12345,
      "title": "Epic clutch 1v5 ace",
      "start_time": 3600,
      "end_time": 3645,
      "duration": 45,
      "confidence_score": 0.95,
      "type": "gameplay_highlight",
      "video_url": "https://cdn.tldr.tv/highlights/67890/video.mp4",
      "thumbnail_url": "https://cdn.tldr.tv/highlights/67890/thumbnail.jpg"
    },
    "stream": {
      "id": 12345,
      "title": "Shroud VALORANT Ranked",
      "platform": "twitch"
    }
  }
}
```

## Webhook Security

### Signature Verification

All webhook requests include a signature header for verification:

```
X-TlDR-Signature: t=1234567890,v1=abc123def456...
```

#### Verification Process

1. Extract timestamp and signature from header
2. Construct signed payload: `timestamp.payload_body`
3. Compute HMAC with SHA256 using webhook secret
4. Compare computed signature with received signature

#### Example Verification (Python)

```python
import hmac
import hashlib
import time

def verify_webhook_signature(payload: bytes, signature_header: str, secret: str) -> bool:
    # Parse signature header
    elements = {}
    for element in signature_header.split(','):
        key, value = element.split('=')
        elements[key] = value
    
    # Extract timestamp and signature
    timestamp = elements.get('t')
    signature = elements.get('v1')
    
    if not timestamp or not signature:
        return False
    
    # Check timestamp is recent (5 minutes)
    current_time = int(time.time())
    if abs(current_time - int(timestamp)) > 300:
        return False
    
    # Compute expected signature
    signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
    expected_sig = hmac.new(
        secret.encode('utf-8'),
        signed_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures
    return hmac.compare_digest(signature, expected_sig)
```

### Security Best Practices

1. **Always verify signatures** - Never process webhooks without verification
2. **Use HTTPS only** - Webhooks are only sent to HTTPS endpoints
3. **Validate timestamps** - Reject old webhooks to prevent replay attacks
4. **Implement idempotency** - Handle duplicate deliveries gracefully
5. **Secure webhook secrets** - Store secrets in secure vaults
6. **Monitor failures** - Track and investigate delivery failures

## Webhook Receiver Endpoints

These endpoints are for receiving webhooks FROM external platforms:

### Receive Stream Events

Endpoint for receiving stream-related events.

```http
POST /webhooks/receive/stream
```

Requires API key authentication.

### Receive Platform Webhooks

Platform-specific webhook receivers:

```http
POST /webhooks/receive/100ms     # 100ms platform webhooks
POST /webhooks/receive/twitch    # Twitch EventSub webhooks
```

These endpoints verify platform-specific signatures.

## Retry Logic

Failed webhook deliveries are retried with exponential backoff:

| Attempt | Delay | Total Time |
|---------|-------|------------|
| 1 | Immediate | 0s |
| 2 | 1 second | 1s |
| 3 | 4 seconds | 5s |
| 4 | 16 seconds | 21s |
| 5 | 64 seconds | 85s |
| 6 | 256 seconds | 341s (~5.7 min) |

### Automatic Disabling

Webhooks are automatically disabled after:
- 100 consecutive failures
- 7 days of continuous failures
- Manual intervention required to re-enable

## Examples

### Configure Webhook Endpoint

```python
from tldr_highlight_api import TLDRClient

client = TLDRClient(api_key="tldr_sk_your_api_key")

# Create webhook configuration
webhook = client.webhooks.create(
    url="https://api.myapp.com/webhooks/tldr",
    events=[
        "stream.completed",
        "highlight.detected",
        "usage.limit_reached"
    ],
    description="Production webhook",
    headers={
        "X-API-Key": "my-internal-api-key"
    }
)

print(f"Webhook created: {webhook.id}")
print(f"Secret (save this!): {webhook.secret}")
```

### Handle Webhook Events (Express.js)

```javascript
const express = require('express');
const crypto = require('crypto');

const app = express();
app.use(express.raw({ type: 'application/json' }));

// Webhook secret from configuration
const WEBHOOK_SECRET = process.env.TLDR_WEBHOOK_SECRET;

// Verify signature middleware
function verifySignature(req, res, next) {
  const signature = req.headers['x-tldr-signature'];
  if (!signature) {
    return res.status(401).send('No signature');
  }

  // Parse signature header
  const elements = signature.split(',').reduce((acc, element) => {
    const [key, value] = element.split('=');
    acc[key] = value;
    return acc;
  }, {});

  const timestamp = elements.t;
  const receivedSig = elements.v1;

  // Compute expected signature
  const payload = `${timestamp}.${req.body}`;
  const expectedSig = crypto
    .createHmac('sha256', WEBHOOK_SECRET)
    .update(payload)
    .digest('hex');

  // Verify signature
  if (receivedSig !== expectedSig) {
    return res.status(401).send('Invalid signature');
  }

  // Parse body for handler
  req.body = JSON.parse(req.body);
  next();
}

// Webhook handler
app.post('/webhooks/tldr', verifySignature, async (req, res) => {
  const { type, data } = req.body;

  try {
    switch (type) {
      case 'stream.completed':
        await handleStreamCompleted(data.stream);
        break;
      
      case 'highlight.detected':
        await handleHighlightDetected(data.highlight);
        break;
      
      case 'usage.limit_reached':
        await handleUsageLimitReached(data);
        break;
      
      default:
        console.log(`Unhandled event type: ${type}`);
    }

    res.json({ received: true });
  } catch (error) {
    console.error('Webhook processing error:', error);
    res.status(500).json({ error: 'Processing failed' });
  }
});

async function handleStreamCompleted(stream) {
  console.log(`Stream ${stream.id} completed with ${stream.highlight_count} highlights`);
  // Process completed stream
}

async function handleHighlightDetected(highlight) {
  console.log(`New highlight detected: ${highlight.title}`);
  // Process new highlight
}

async function handleUsageLimitReached(usage) {
  console.log('Usage limit reached, sending alert...');
  // Send alerts, pause processing, etc.
}
```

### Monitor Webhook Health

```bash
#!/bin/bash

# Check webhook health
webhook_id="whk_1a2b3c4d"
api_key="tldr_sk_your_api_key"

# Get webhook status
webhook=$(curl -s -H "X-API-Key: $api_key" \
  "https://api.tldr.tv/api/v1/webhooks/$webhook_id")

health_status=$(echo $webhook | jq -r '.health_status')
failure_count=$(echo $webhook | jq -r '.failure_count')
success_rate=$(echo $webhook | jq -r '.statistics.success_rate')

echo "Webhook Health Report"
echo "===================="
echo "Status: $health_status"
echo "Failures: $failure_count"
echo "Success Rate: $(printf "%.1f" $(echo "$success_rate * 100" | bc))%"

# Check recent failures
if [ "$health_status" != "healthy" ]; then
  echo -e "\nRecent Failures:"
  
  failures=$(curl -s -H "X-API-Key: $api_key" \
    "https://api.tldr.tv/api/v1/webhooks/$webhook_id/deliveries?status=failed&per_page=5")
  
  echo $failures | jq -r '.items[] | "  - \(.attempted_at): \(.response.status_code) - \(.event_type)"'
fi
```

## Best Practices

### Webhook Implementation

1. **Respond quickly** - Return 200 OK immediately, process async
2. **Implement idempotency** - Use event IDs to prevent duplicates
3. **Handle out-of-order delivery** - Don't assume event order
4. **Validate payloads** - Verify expected fields exist
5. **Log everything** - Keep detailed logs for debugging

### Error Handling

1. **Return appropriate status codes**:
   - `200 OK` - Successfully processed
   - `400 Bad Request` - Invalid payload
   - `401 Unauthorized` - Signature verification failed
   - `500 Internal Server Error` - Processing failed

2. **Implement retry logic** for transient failures
3. **Alert on repeated failures**
4. **Have a manual replay mechanism**

### Scaling Considerations

1. **Process asynchronously** - Queue events for processing
2. **Implement rate limiting** - Protect your endpoints
3. **Use connection pooling** - Reuse HTTP connections
4. **Monitor performance** - Track processing times

## Troubleshooting

### Common Issues

1. **Webhook not receiving events**
   - Verify URL is publicly accessible
   - Check webhook is active
   - Ensure events are subscribed
   - Verify no firewall blocking

2. **Signature verification fails**
   - Ensure using raw request body
   - Check secret is correct
   - Verify timestamp handling
   - Compare signature algorithms

3. **High failure rate**
   - Check endpoint availability
   - Monitor response times
   - Verify payload processing
   - Check for memory leaks

### Debug Checklist

- [ ] Webhook URL is HTTPS
- [ ] Endpoint returns 200 OK
- [ ] Signature verification implemented
- [ ] Events are being generated
- [ ] No IP whitelisting blocking TL;DR
- [ ] Proper error handling in place
- [ ] Monitoring and alerting configured

---

*See also: [API Overview](./overview.md) | [Streams API](./streams.md) | [Highlights API](./highlights.md)*