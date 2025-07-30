# Authentication Guide

This guide covers authentication methods and security practices for the TL;DR Highlight API.

## 🔐 Authentication Overview

The TL;DR Highlight API uses **API key authentication** with scoped permissions for secure access control. This approach provides:

- **Simplicity**: Easy integration without complex OAuth flows
- **Security**: Cryptographically secure keys with limited scopes
- **Flexibility**: Multiple keys per organization with different permissions
- **Auditability**: Complete access logging and usage tracking

## 🗝️ API Key Format

API keys follow a structured format for easy identification:

```
tldr_sk_[32-character-string]
```

**Example**: `tldr_sk_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p`

- **Prefix**: `tldr_sk_` identifies the service and key type
- **Body**: 32 characters of cryptographically secure random data
- **Encoding**: URL-safe base64 characters (A-Z, a-z, 0-9, -, _)

## 🚀 Getting Started

### 1. Create Your Account

Sign up at [https://app.tldr.tv](https://app.tldr.tv) to create your organization and receive your first API key.

### 2. Generate API Keys

```bash
# Using the dashboard
# 1. Log in to https://app.tldr.tv
# 2. Navigate to Settings → API Keys
# 3. Click "Generate New Key"
# 4. Select scopes and name your key
# 5. Copy the key (shown only once!)

# Using the API (requires existing key with admin scope)
curl -X POST "https://api.tldr.tv/api/v1/auth/api-keys" \
  -H "X-API-Key: tldr_sk_your_existing_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Integration",
    "scopes": ["streams:read", "streams:write", "highlights:read"],
    "expires_at": "2025-12-31T23:59:59Z"
  }'
```

### 3. Make Your First Request

```bash
curl -H "X-API-Key: tldr_sk_your_api_key_here" \
  "https://api.tldr.tv/api/v1/streams"
```

## 🎯 Permission Scopes

API keys support granular permission scopes for security and access control:

### Core Resource Scopes

| Scope | Description | Operations |
|-------|-------------|------------|
| `streams:read` | Read stream information | GET /streams, GET /streams/{id} |
| `streams:write` | Create and manage streams | POST /streams, PATCH /streams/{id} |
| `streams:delete` | Delete streams | DELETE /streams/{id} |
| `highlights:read` | Read highlight information | GET /highlights, GET /highlights/{id} |
| `highlights:write` | Update highlight metadata | PATCH /highlights/{id} |
| `highlights:delete` | Delete highlights | DELETE /highlights/{id} |
| `webhooks:read` | Read webhook configurations | GET /webhooks |
| `webhooks:write` | Manage webhooks | POST /webhooks, PATCH /webhooks/{id} |
| `webhooks:delete` | Delete webhooks | DELETE /webhooks/{id} |

### Organization Scopes

| Scope | Description | Operations |
|-------|-------------|------------|
| `organizations:read` | Read organization info | GET /organizations/{id} |
| `organizations:write` | Update organization settings | PATCH /organizations/{id} |
| `users:read` | Read user information | GET /users, GET /users/{id} |
| `users:write` | Manage organization users | POST /users, PATCH /users/{id} |

### Administrative Scopes

| Scope | Description | Operations |
|-------|-------------|------------|
| `admin` | Full administrative access | All operations |
| `api-keys:write` | Manage API keys | POST /auth/api-keys, DELETE /auth/api-keys/{id} |
| `usage:read` | Read usage analytics | GET /usage, GET /analytics |

### Scope Combinations

```json
{
  "name": "Stream Processing Bot",
  "scopes": ["streams:read", "streams:write", "highlights:read"],
  "description": "Bot for automated stream processing"
}
```

```json
{
  "name": "Analytics Dashboard", 
  "scopes": ["streams:read", "highlights:read", "usage:read"],
  "description": "Read-only access for analytics dashboard"
}
```

```json
{
  "name": "Full Integration",
  "scopes": ["streams:*", "highlights:*", "webhooks:*"],
  "description": "Complete access for main application"
}
```

## 🔒 Security Best Practices

### Key Management

#### Secure Storage
```bash
# Environment variables (recommended)
export TLDR_API_KEY="tldr_sk_your_api_key_here"

# Configuration files (ensure proper permissions)
chmod 600 config/api_keys.env

# Secret management services
# - AWS Secrets Manager
# - HashiCorp Vault  
# - Azure Key Vault
# - Google Secret Manager
```

#### Key Rotation
```bash
# Regular rotation schedule
# - Development keys: Every 90 days
# - Production keys: Every 30 days
# - Compromised keys: Immediately

# Rotation process:
# 1. Generate new key with same scopes
# 2. Update applications to use new key
# 3. Test functionality with new key
# 4. Revoke old key
# 5. Monitor for any failures
```

#### Key Scoping
```python
# Principle of least privilege
production_key_scopes = [
    "streams:read",
    "streams:write", 
    "highlights:read",
    "webhooks:read"
]

# Avoid admin scope unless absolutely necessary
avoid_scopes = ["admin", "users:write", "organizations:write"]
```

### Network Security

#### HTTPS Only
```bash
# Always use HTTPS in production
API_BASE_URL="https://api.tldr.tv"

# Verify SSL certificates
curl --ssl-reqd --cert-status "https://api.tldr.tv/health"
```

#### IP Allowlisting (Enterprise)
```json
{
  "api_key_config": {
    "allowed_ips": [
      "192.168.1.0/24",
      "10.0.0.0/8"
    ],
    "allowed_regions": ["us-east-1", "us-west-2"]
  }
}
```

### Error Handling

#### Secure Error Messages
```python
# Good: Generic error without sensitive info
{
  "success": false,
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Invalid or expired API key",
    "status_code": 401
  }
}

# Bad: Exposes key details
{
  "error": "API key tldr_sk_abc123... is expired since 2024-01-15"
}
```

## 📋 API Key Management

### Creating API Keys

#### Via Dashboard
1. Log in to [https://app.tldr.tv](https://app.tldr.tv)
2. Navigate to **Settings** → **API Keys**
3. Click **"Generate New Key"**
4. Configure key settings:
   - **Name**: Descriptive name for the key
   - **Scopes**: Select required permissions
   - **Expiration**: Set expiration date (optional)
   - **Description**: Purpose and usage notes
5. **Copy the key immediately** (shown only once!)

#### Via API
```bash
# Create new API key
curl -X POST "https://api.tldr.tv/api/v1/auth/api-keys" \
  -H "X-API-Key: tldr_sk_admin_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Mobile App Integration",
    "scopes": ["streams:read", "highlights:read"],
    "expires_at": "2025-06-30T23:59:59Z",
    "description": "API key for mobile app backend"
  }'
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "key_1a2b3c4d",
    "name": "Mobile App Integration",
    "key": "tldr_sk_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p",
    "scopes": ["streams:read", "highlights:read"],
    "created_at": "2024-01-15T10:30:00Z",
    "expires_at": "2025-06-30T23:59:59Z",
    "last_used_at": null,
    "is_active": true
  }
}
```

### Listing API Keys

```bash
curl -H "X-API-Key: tldr_sk_your_key_here" \
  "https://api.tldr.tv/api/v1/auth/api-keys"
```

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "id": "key_1a2b3c4d",
      "name": "Production API",
      "key_preview": "tldr_sk_1a2b****",
      "scopes": ["streams:*", "highlights:*"],
      "created_at": "2024-01-01T00:00:00Z",
      "expires_at": null,
      "last_used_at": "2024-01-15T09:45:00Z",
      "usage_count": 1247,
      "is_active": true
    },
    {
      "id": "key_5e6f7g8h", 
      "name": "Development Testing",
      "key_preview": "tldr_sk_5e6f****",
      "scopes": ["streams:read"],
      "created_at": "2024-01-10T15:30:00Z",
      "expires_at": "2024-04-10T15:30:00Z",
      "last_used_at": "2024-01-14T16:20:00Z",
      "usage_count": 89,
      "is_active": true
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 2,
    "total_pages": 1
  }
}
```

### Updating API Keys

```bash
# Update key name and scopes
curl -X PATCH "https://api.tldr.tv/api/v1/auth/api-keys/key_1a2b3c4d" \
  -H "X-API-Key: tldr_sk_admin_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Production API",
    "scopes": ["streams:*", "highlights:*", "webhooks:read"],
    "is_active": false
  }'
```

### Revoking API Keys

```bash
# Revoke specific key
curl -X DELETE "https://api.tldr.tv/api/v1/auth/api-keys/key_1a2b3c4d" \
  -H "X-API-Key: tldr_sk_admin_key_here"

# Revoke all keys (emergency)
curl -X DELETE "https://api.tldr.tv/api/v1/auth/api-keys" \
  -H "X-API-Key: tldr_sk_admin_key_here" \
  -H "Content-Type: application/json" \
  -d '{"revoke_all": true, "reason": "Security incident"}'
```

## 🔄 Authentication Flow Examples

### Python SDK
```python
from tldr_highlight_api import TLDRClient

# Initialize client
client = TLDRClient(api_key="tldr_sk_your_api_key_here")

# Create stream
stream = await client.streams.create(
    url="https://twitch.tv/example_stream",
    processing_options={
        "enable_video_analysis": True,
        "enable_audio_analysis": True
    }
)

# List highlights
highlights = await client.highlights.list(stream_id=stream.id)
```

### Node.js SDK
```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({
  apiKey: 'tldr_sk_your_api_key_here'
});

// Create stream
const stream = await client.streams.create({
  url: 'https://twitch.tv/example_stream',
  processingOptions: {
    enableVideoAnalysis: true,
    enableAudioAnalysis: true
  }
});

// List highlights  
const highlights = await client.highlights.list({
  streamId: stream.id
});
```

### cURL Examples
```bash
# Create stream
curl -X POST "https://api.tldr.tv/api/v1/streams" \
  -H "X-API-Key: tldr_sk_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://twitch.tv/example_stream",
    "processing_options": {
      "enable_video_analysis": true,
      "enable_audio_analysis": true
    }
  }'

# Get stream status
curl -H "X-API-Key: tldr_sk_your_api_key_here" \
  "https://api.tldr.tv/api/v1/streams/123"

# List highlights
curl -H "X-API-Key: tldr_sk_your_api_key_here" \
  "https://api.tldr.tv/api/v1/highlights?stream_id=123"
```

## ⚠️ Error Handling

### Authentication Errors

#### Invalid API Key
```json
{
  "success": false,
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid or malformed",
    "status_code": 401,
    "details": {
      "header": "X-API-Key",
      "expected_format": "tldr_sk_[32-chars]"
    }
  }
}
```

#### Expired API Key
```json
{
  "success": false,
  "error": {
    "code": "API_KEY_EXPIRED", 
    "message": "The API key has expired",
    "status_code": 401,
    "details": {
      "expired_at": "2024-01-15T00:00:00Z",
      "action": "Generate a new API key in your dashboard"
    }
  }
}
```

#### Insufficient Permissions
```json
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "message": "API key lacks required scope for this operation",
    "status_code": 403,
    "details": {
      "required_scope": "streams:write",
      "current_scopes": ["streams:read", "highlights:read"],
      "action": "Update API key scopes or use a different key"
    }
  }
}
```

#### Rate Limit Exceeded
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "API rate limit exceeded",
    "status_code": 429,
    "details": {
      "limit": 100,
      "window": "1 minute",
      "retry_after": 45,
      "reset_at": "2024-01-15T10:31:00Z"
    }
  }
}
```

### Error Handling Best Practices

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TLDRAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tldr.tv/api/v1"
        self.session = self._create_session()
    
    def _create_session(self):
        session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        })
        
        return session
    
    def _handle_response(self, response):
        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired API key")
        elif response.status_code == 403:
            raise PermissionError("Insufficient API key permissions")
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after}s")
        
        response.raise_for_status()
        return response.json()
```

## 🔍 Monitoring and Analytics

### Usage Tracking
- **Request Counts**: Track API calls per key
- **Error Rates**: Monitor authentication failures
- **Usage Patterns**: Identify unusual access patterns
- **Scope Usage**: Track which permissions are used

### Security Monitoring
- **Failed Attempts**: Monitor invalid key usage
- **Geographic Analysis**: Unusual location access
- **Rate Limiting**: Track rate limit violations
- **Key Rotation**: Monitor key age and usage

### Alerts and Notifications
- **Expired Keys**: Notification before expiration
- **High Usage**: Alerts for unusual usage spikes
- **Security Events**: Notifications for potential security issues
- **Quota Limits**: Warnings before reaching limits

---

This authentication system provides secure, flexible, and scalable access control for the TL;DR Highlight API while maintaining simplicity for developers.