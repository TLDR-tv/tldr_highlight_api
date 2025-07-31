# Users API

The Users API provides endpoints for managing user profiles, preferences, and account settings in the TL;DR Highlight API.

## Overview

Users are individuals who have access to the TL;DR Highlight API through an organization. Each user:

- Belongs to one or more organizations
- Has a role within each organization (OWNER, ADMIN, MEMBER, VIEWER)
- Can create and manage API keys
- Has personal preferences and settings
- Owns streams and highlights they create

## Endpoints

### Get Current User Profile

Retrieve the authenticated user's profile information.

```http
GET /api/v1/users/me
```

#### Response

```json
{
  "id": 456,
  "email": "john.doe@acme.com",
  "full_name": "John Doe",
  "avatar_url": "https://api.tldr.tv/avatars/456.jpg",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "email_verified": true,
  "is_active": true,
  "organizations": [
    {
      "id": 123,
      "name": "Acme Streaming Platform",
      "role": "ADMIN",
      "joined_at": "2024-01-01T00:00:00Z"
    },
    {
      "id": 124,
      "name": "Personal Projects",
      "role": "OWNER",
      "joined_at": "2024-01-05T12:00:00Z"
    }
  ],
  "preferences": {
    "email_notifications": {
      "stream_completed": true,
      "highlight_detected": false,
      "weekly_summary": true,
      "usage_alerts": true
    },
    "default_stream_settings": {
      "platform": "twitch",
      "highlight_threshold": 0.8,
      "min_duration": 15,
      "max_duration": 60
    },
    "timezone": "America/New_York",
    "language": "en"
  },
  "stats": {
    "total_streams": 145,
    "total_highlights": 1234,
    "total_watch_time_hours": 245.5,
    "last_active": "2024-01-15T09:30:00Z"
  }
}
```

### Update User Profile

Update the authenticated user's profile information.

```http
PUT /api/v1/users/me
```

#### Request Body

```json
{
  "full_name": "John Updated Doe",
  "preferences": {
    "email_notifications": {
      "stream_completed": false,
      "highlight_detected": true,
      "weekly_summary": true,
      "usage_alerts": true
    },
    "default_stream_settings": {
      "platform": "youtube",
      "highlight_threshold": 0.9,
      "min_duration": 20,
      "max_duration": 90
    },
    "timezone": "Europe/London",
    "language": "en-GB"
  }
}
```

#### Response

Returns the updated user object (same format as GET).

#### Error Responses

- `400 Bad Request` - Invalid data provided
- `422 Unprocessable Entity` - Validation error

### Change Password

Change the authenticated user's password.

```http
POST /api/v1/users/me/change-password
```

#### Request Body

```json
{
  "current_password": "oldPassword123!",
  "new_password": "newSecurePassword456!",
  "confirm_password": "newSecurePassword456!"
}
```

#### Password Requirements

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

#### Response

```json
{
  "message": "Password changed successfully",
  "password_changed_at": "2024-01-15T10:45:00Z"
}
```

#### Error Responses

- `400 Bad Request` - Passwords don't match
- `401 Unauthorized` - Current password incorrect
- `422 Unprocessable Entity` - New password doesn't meet requirements

## User Preferences

### Email Notification Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `stream_completed` | Notify when stream processing completes | true |
| `highlight_detected` | Notify for each new highlight (can be noisy) | false |
| `weekly_summary` | Weekly summary of activity and highlights | true |
| `usage_alerts` | Alerts when approaching usage limits | true |
| `security_alerts` | Notifications for security events | true |

### Default Stream Settings

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| `platform` | Default streaming platform | "twitch" | twitch, youtube, custom |
| `highlight_threshold` | Confidence threshold for highlights | 0.8 | 0.0 - 1.0 |
| `min_duration` | Minimum highlight duration (seconds) | 15 | 5 - 300 |
| `max_duration` | Maximum highlight duration (seconds) | 60 | 10 - 600 |
| `auto_process` | Automatically start processing | false | true/false |

### Localization Settings

| Setting | Description | Options |
|---------|-------------|---------|
| `timezone` | User's timezone for timestamps | IANA timezone (e.g., "America/New_York") |
| `language` | Preferred language | en, es, fr, de, ja, ko, zh |
| `date_format` | Date display format | "MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD" |
| `time_format` | Time display format | "12h", "24h" |

## User Roles and Permissions

Users have different roles within each organization they belong to:

### Role Hierarchy

| Role | Description | Key Permissions |
|------|-------------|-----------------|
| **OWNER** | Organization owner | • Full control over organization<br>• Billing management<br>• Cannot be removed |
| **ADMIN** | Administrator | • Manage users and settings<br>• View all streams/highlights<br>• Configure webhooks |
| **MEMBER** | Regular member | • Create and manage own content<br>• View organization content<br>• Use API keys |
| **VIEWER** | Read-only access | • View streams and highlights<br>• Cannot create content<br>• Limited API access |

### Permission Matrix

| Action | OWNER | ADMIN | MEMBER | VIEWER |
|--------|-------|-------|---------|---------|
| View organization details | ✓ | ✓ | ✓ | ✓ |
| View all streams | ✓ | ✓ | ✓ | ✓ |
| Create streams | ✓ | ✓ | ✓ | ✗ |
| Delete any stream | ✓ | ✓ | ✗ | ✗ |
| Manage users | ✓ | ✓ | ✗ | ✗ |
| Update organization | ✓ | ✓ | ✗ | ✗ |
| Manage billing | ✓ | ✗ | ✗ | ✗ |
| Create API keys | ✓ | ✓ | ✓ | ✗ |

## User Statistics

User statistics are automatically tracked and include:

| Metric | Description | Update Frequency |
|--------|-------------|------------------|
| `total_streams` | Total streams created | Real-time |
| `total_highlights` | Total highlights extracted | Real-time |
| `total_watch_time_hours` | Sum of all stream durations | Daily |
| `average_highlights_per_stream` | Calculated average | Daily |
| `last_active` | Last API activity | Real-time |
| `storage_used_gb` | Personal storage usage | Hourly |

## Account Management

### Email Verification

New users must verify their email address:

```http
POST /api/v1/users/verify-email
```

```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Password Reset

Request a password reset:

```http
POST /api/v1/users/reset-password
```

```json
{
  "email": "john.doe@acme.com"
}
```

Complete password reset:

```http
POST /api/v1/users/reset-password/confirm
```

```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "new_password": "newSecurePassword789!",
  "confirm_password": "newSecurePassword789!"
}
```

### Account Deactivation

Users can request account deactivation:

```http
POST /api/v1/users/me/deactivate
```

```json
{
  "reason": "No longer using the service",
  "feedback": "Optional feedback message"
}
```

**Note**: Account deactivation:
- Preserves data for 30 days
- Revokes all API keys
- Removes from all organizations
- Can be reversed by contacting support

## Multi-Organization Access

Users can belong to multiple organizations and switch between them:

### Get User Organizations

```http
GET /api/v1/users/me/organizations
```

Response:
```json
{
  "organizations": [
    {
      "id": 123,
      "name": "Acme Streaming Platform",
      "slug": "acme-streaming",
      "role": "ADMIN",
      "joined_at": "2024-01-01T00:00:00Z",
      "is_default": true
    },
    {
      "id": 124,
      "name": "Personal Projects",
      "slug": "personal-projects",
      "role": "OWNER",
      "joined_at": "2024-01-05T12:00:00Z",
      "is_default": false
    }
  ]
}
```

### Switch Default Organization

```http
POST /api/v1/users/me/organizations/default
```

```json
{
  "organization_id": 124
}
```

## Examples

### Update User Preferences

```python
from tldr_highlight_api import TLDRClient

client = TLDRClient(api_key="tldr_sk_your_api_key")

# Get current user
user = client.users.get_me()

# Update preferences
updated_user = client.users.update_me(
    preferences={
        "email_notifications": {
            "stream_completed": False,
            "weekly_summary": True
        },
        "default_stream_settings": {
            "highlight_threshold": 0.9,
            "max_duration": 120
        },
        "timezone": "Asia/Tokyo"
    }
)

print(f"Updated timezone to: {updated_user.preferences.timezone}")
```

### Manage Password Security

```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({ apiKey: 'tldr_sk_your_api_key' });

// Change password
try {
  await client.users.changePassword({
    currentPassword: 'oldPassword123!',
    newPassword: 'newSecurePassword456!',
    confirmPassword: 'newSecurePassword456!'
  });
  
  console.log('Password changed successfully');
  
  // Optionally revoke all existing API keys for security
  const apiKeys = await client.auth.listApiKeys();
  for (const key of apiKeys) {
    await client.auth.revokeApiKey(key.id);
  }
  
  // Create new API key with new password
  const newKey = await client.auth.createApiKey({
    name: 'Post password change key',
    scopes: ['streams:*', 'highlights:*']
  });
  
} catch (error) {
  console.error('Password change failed:', error.message);
}
```

### Monitor User Activity

```bash
#!/bin/bash

# Get user stats
user_stats=$(curl -s -H "X-API-Key: tldr_sk_your_api_key" \
  "https://api.tldr.tv/api/v1/users/me" | jq '.stats')

# Extract metrics
total_streams=$(echo $user_stats | jq '.total_streams')
total_highlights=$(echo $user_stats | jq '.total_highlights')
last_active=$(echo $user_stats | jq -r '.last_active')

# Check if user is active
last_active_ts=$(date -d "$last_active" +%s)
current_ts=$(date +%s)
days_inactive=$(( ($current_ts - $last_active_ts) / 86400 ))

echo "User Activity Report"
echo "==================="
echo "Total Streams: $total_streams"
echo "Total Highlights: $total_highlights"
echo "Days Since Last Active: $days_inactive"

if [ $days_inactive -gt 30 ]; then
  echo "WARNING: User has been inactive for over 30 days"
fi
```

## Best Practices

### Security

1. **Password Management**
   - Enforce strong password requirements
   - Encourage regular password changes
   - Use password managers
   - Enable two-factor authentication (coming soon)

2. **API Key Hygiene**
   - Regularly rotate API keys
   - Use minimal required scopes
   - Revoke unused keys
   - Monitor key usage

3. **Account Security**
   - Verify email addresses
   - Monitor login attempts
   - Review active sessions
   - Set up security alerts

### Profile Management

1. **Keep Information Updated**
   - Maintain current email address
   - Update timezone when traveling
   - Set appropriate notification preferences

2. **Optimize Settings**
   - Configure default stream settings for efficiency
   - Set appropriate highlight thresholds
   - Choose relevant notification preferences

3. **Multi-Organization Management**
   - Set appropriate default organization
   - Use organization-specific API keys
   - Maintain consistent settings across organizations

## Troubleshooting

### Common Issues

1. **Cannot Update Profile**
   - Ensure you're using a valid API key
   - Check that email is not already in use
   - Verify request body format

2. **Password Change Fails**
   - Verify current password is correct
   - Ensure new password meets requirements
   - Check that passwords match

3. **Missing Organizations**
   - Verify organization membership
   - Check if invitation was accepted
   - Ensure organization is active

### Support

For user account issues:

- **Account recovery**: recovery@tldr.tv
- **Security concerns**: security@tldr.tv
- **General support**: support@tldr.tv

---

*See also: [Authentication Guide](./authentication.md) | [Organizations API](./organizations.md) | [API Overview](./overview.md)*