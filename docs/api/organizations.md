# Organizations API

The Organizations API provides endpoints for managing enterprise organizations, user memberships, and usage tracking in the TL;DR Highlight API.

## Overview

Organizations are the primary billing and access control entity in the TL;DR Highlight API. Each organization:

- Has a subscription plan (STARTER, PROFESSIONAL, ENTERPRISE)
- Contains multiple users with different roles
- Has usage quotas and rate limits based on their plan
- Owns streams, highlights, and API keys
- Receives aggregated billing and usage reports

## Endpoints

### Get Organization Details

Retrieve detailed information about an organization including subscription plan and usage limits.

```http
GET /api/v1/organizations/{org_id}
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `org_id` | integer | path | Yes | Organization ID |

#### Response

```json
{
  "id": 123,
  "name": "Acme Streaming Platform",
  "slug": "acme-streaming",
  "plan": {
    "type": "PROFESSIONAL",
    "name": "Professional Plan",
    "limits": {
      "monthly_stream_hours": 1000,
      "concurrent_streams": 10,
      "api_calls_per_minute": 1000,
      "storage_gb": 500,
      "users": 50
    }
  },
  "usage": {
    "current_month": {
      "stream_hours": 245.5,
      "storage_gb": 123.4,
      "api_calls": 45678,
      "active_users": 12
    }
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "settings": {
    "webhook_url": "https://api.acme.com/webhooks/tldr",
    "default_highlight_duration": 30,
    "auto_process_streams": true
  }
}
```

#### Error Responses

- `404 Not Found` - Organization not found
- `403 Forbidden` - User doesn't have access to this organization

### Update Organization

Update organization details such as name and settings.

```http
PUT /api/v1/organizations/{org_id}
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `org_id` | integer | path | Yes | Organization ID |

#### Request Body

```json
{
  "name": "Updated Organization Name",
  "settings": {
    "webhook_url": "https://api.acme.com/webhooks/tldr-v2",
    "default_highlight_duration": 60,
    "auto_process_streams": false
  }
}
```

#### Response

Returns the updated organization object (same format as GET).

#### Error Responses

- `404 Not Found` - Organization not found
- `403 Forbidden` - User doesn't have admin access
- `422 Unprocessable Entity` - Invalid data provided

### List Organization Users

Get a list of all users in an organization with their roles and permissions.

```http
GET /api/v1/organizations/{org_id}/users
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `org_id` | integer | path | Yes | Organization ID |
| `page` | integer | query | No | Page number (default: 1) |
| `per_page` | integer | query | No | Items per page (default: 20, max: 100) |
| `role` | string | query | No | Filter by role (OWNER, ADMIN, MEMBER, VIEWER) |

#### Response

```json
{
  "users": [
    {
      "id": 456,
      "email": "admin@acme.com",
      "full_name": "John Admin",
      "role": "ADMIN",
      "permissions": [
        "streams:*",
        "highlights:*",
        "webhooks:*",
        "users:read"
      ],
      "joined_at": "2024-01-01T00:00:00Z",
      "last_active": "2024-01-15T09:30:00Z",
      "is_active": true
    },
    {
      "id": 457,
      "email": "viewer@acme.com",
      "full_name": "Jane Viewer",
      "role": "VIEWER",
      "permissions": [
        "streams:read",
        "highlights:read"
      ],
      "joined_at": "2024-01-05T10:00:00Z",
      "last_active": "2024-01-14T15:20:00Z",
      "is_active": true
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 12,
    "pages": 1
  }
}
```

#### Error Responses

- `404 Not Found` - Organization not found
- `403 Forbidden` - User doesn't have access to view users

### Add User to Organization

Add a new user to an organization with a specific role.

```http
POST /api/v1/organizations/{org_id}/users
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `org_id` | integer | path | Yes | Organization ID |

#### Request Body

```json
{
  "email": "newuser@acme.com",
  "role": "MEMBER",
  "send_invitation": true
}
```

#### Roles

| Role | Description | Permissions |
|------|-------------|-------------|
| `OWNER` | Organization owner | Full access to everything |
| `ADMIN` | Administrator | Manage users, settings, and all resources |
| `MEMBER` | Regular member | Create and manage own streams/highlights |
| `VIEWER` | Read-only viewer | View streams and highlights only |

#### Response

```json
{
  "id": 458,
  "email": "newuser@acme.com",
  "role": "MEMBER",
  "invitation_sent": true,
  "invitation_expires_at": "2024-01-22T10:30:00Z",
  "joined_at": null,
  "is_active": false
}
```

#### Error Responses

- `404 Not Found` - Organization not found
- `403 Forbidden` - User doesn't have admin access
- `409 Conflict` - User already exists in organization
- `422 Unprocessable Entity` - Invalid email or role

### Remove User from Organization

Remove a user from an organization.

```http
DELETE /api/v1/organizations/{org_id}/users/{user_id}
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `org_id` | integer | path | Yes | Organization ID |
| `user_id` | integer | path | Yes | User ID to remove |

#### Response

```http
204 No Content
```

#### Error Responses

- `404 Not Found` - Organization or user not found
- `403 Forbidden` - User doesn't have admin access
- `400 Bad Request` - Cannot remove organization owner

### Get Organization Usage

Retrieve detailed usage statistics for an organization.

```http
GET /api/v1/organizations/{org_id}/usage
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `org_id` | integer | path | Yes | Organization ID |
| `period` | string | query | No | Time period: `current_month` (default), `last_month`, `last_30_days`, `last_90_days`, `year_to_date` |
| `breakdown` | string | query | No | Breakdown type: `daily`, `weekly`, `monthly` |

#### Response

```json
{
  "organization_id": 123,
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  },
  "usage": {
    "streams": {
      "total_count": 145,
      "total_hours": 245.5,
      "by_status": {
        "completed": 140,
        "failed": 3,
        "processing": 2
      },
      "by_platform": {
        "twitch": 89,
        "youtube": 45,
        "custom": 11
      }
    },
    "highlights": {
      "total_count": 1234,
      "total_duration_minutes": 6170,
      "average_per_stream": 8.5,
      "by_type": {
        "gameplay": 456,
        "reaction": 345,
        "funny": 234,
        "emotional": 199
      }
    },
    "storage": {
      "current_gb": 123.4,
      "files_count": 2468,
      "average_file_size_mb": 52.3
    },
    "api": {
      "total_calls": 45678,
      "by_endpoint": {
        "/streams": 12345,
        "/highlights": 23456,
        "/webhooks": 9877
      },
      "error_rate": 0.02
    },
    "users": {
      "active_count": 12,
      "total_count": 15,
      "by_role": {
        "OWNER": 1,
        "ADMIN": 2,
        "MEMBER": 8,
        "VIEWER": 4
      }
    }
  },
  "limits": {
    "monthly_stream_hours": {
      "used": 245.5,
      "limit": 1000,
      "percentage": 24.55
    },
    "concurrent_streams": {
      "current": 2,
      "limit": 10,
      "percentage": 20
    },
    "storage_gb": {
      "used": 123.4,
      "limit": 500,
      "percentage": 24.68
    },
    "users": {
      "used": 15,
      "limit": 50,
      "percentage": 30
    }
  },
  "overage": {
    "stream_hours": 0,
    "storage_gb": 0,
    "api_calls": 0
  }
}
```

#### Breakdown Response (when `breakdown=daily`)

```json
{
  "organization_id": 123,
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  },
  "breakdown": [
    {
      "date": "2024-01-15",
      "streams": {
        "count": 12,
        "hours": 18.5
      },
      "highlights": {
        "count": 98,
        "duration_minutes": 490
      },
      "api_calls": 3456,
      "storage_gb": 123.4
    },
    {
      "date": "2024-01-14",
      "streams": {
        "count": 10,
        "hours": 15.2
      },
      "highlights": {
        "count": 85,
        "duration_minutes": 425
      },
      "api_calls": 3012,
      "storage_gb": 122.8
    }
    // ... more daily entries
  ]
}
```

#### Error Responses

- `404 Not Found` - Organization not found
- `403 Forbidden` - User doesn't have access to view usage

## Subscription Plans

Organizations can have one of the following subscription plans:

| Plan | Monthly Stream Hours | Concurrent Streams | Storage | API Rate Limit | Users |
|------|---------------------|-------------------|---------|----------------|-------|
| **STARTER** | 100 | 2 | 50 GB | 100/min | 5 |
| **PROFESSIONAL** | 1,000 | 10 | 500 GB | 1,000/min | 50 |
| **ENTERPRISE** | Unlimited | Unlimited | Custom | Custom | Unlimited |

### Plan Features

#### STARTER
- Basic highlight detection
- Standard AI models
- Email support
- 7-day data retention

#### PROFESSIONAL
- Advanced highlight detection
- Premium AI models
- Priority support
- 30-day data retention
- Custom webhooks
- API analytics

#### ENTERPRISE
- Custom AI models
- Dedicated support
- Unlimited retention
- SLA guarantees
- Custom integrations
- On-premise deployment options

## Permissions

Organization-related permissions:

| Permission | Description |
|------------|-------------|
| `organizations:read` | View organization details and settings |
| `organizations:write` | Update organization settings |
| `organizations:admin` | Manage users and billing |
| `usage:read` | View usage statistics and reports |

## Rate Limiting

Organization endpoints are subject to the following rate limits:

- **Read operations**: 100 requests per minute
- **Write operations**: 20 requests per minute
- **Usage reports**: 10 requests per minute

Rate limits are applied per API key, not per organization.

## Webhooks

Organizations can configure webhooks to receive real-time notifications about:

- User added/removed
- Plan upgraded/downgraded
- Usage threshold reached (80%, 90%, 100%)
- Monthly usage report available

See the [Webhooks API documentation](./webhooks.md) for details on configuring webhooks.

## Examples

### Check Organization Usage Limits

```bash
# Get current usage vs limits
curl -H "X-API-Key: tldr_sk_your_api_key" \
  "https://api.tldr.tv/api/v1/organizations/123/usage?period=current_month"

# Check if approaching limits
response=$(curl -s -H "X-API-Key: tldr_sk_your_api_key" \
  "https://api.tldr.tv/api/v1/organizations/123/usage")

stream_hours_pct=$(echo $response | jq '.limits.monthly_stream_hours.percentage')
if (( $(echo "$stream_hours_pct > 80" | bc -l) )); then
  echo "WARNING: Stream hours usage at ${stream_hours_pct}%"
fi
```

### Manage Organization Users

```python
from tldr_highlight_api import TLDRClient

client = TLDRClient(api_key="tldr_sk_your_api_key")

# List all users
users = client.organizations.list_users(org_id=123)
for user in users:
    print(f"{user.email} - {user.role}")

# Add a new admin
new_admin = client.organizations.add_user(
    org_id=123,
    email="newadmin@company.com",
    role="ADMIN",
    send_invitation=True
)

# Remove a user
client.organizations.remove_user(org_id=123, user_id=456)
```

### Monitor Usage Trends

```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({ apiKey: 'tldr_sk_your_api_key' });

// Get daily breakdown for the last 30 days
const usage = await client.organizations.getUsage(123, {
  period: 'last_30_days',
  breakdown: 'daily'
});

// Calculate average daily stream hours
const avgDailyHours = usage.breakdown.reduce((sum, day) => 
  sum + day.streams.hours, 0) / usage.breakdown.length;

console.log(`Average daily stream hours: ${avgDailyHours.toFixed(2)}`);

// Check for usage spikes
usage.breakdown.forEach(day => {
  if (day.streams.hours > avgDailyHours * 2) {
    console.log(`Usage spike on ${day.date}: ${day.streams.hours} hours`);
  }
});
```

## Migration Guide

If you're migrating from single-user API keys to organizations:

1. **Create an Organization**: Contact support to create your organization
2. **Migrate API Keys**: New keys will be organization-scoped
3. **Update Permissions**: Review and update user roles
4. **Monitor Usage**: Set up usage alerts and monitoring
5. **Configure Webhooks**: Set up organization-level webhooks

## Support

For organization management support:

- **Billing questions**: billing@tldr.tv
- **Technical support**: support@tldr.tv
- **Enterprise plans**: enterprise@tldr.tv

---

*See also: [Authentication Guide](./authentication.md) | [Users API](./users.md) | [Webhooks API](./webhooks.md)*