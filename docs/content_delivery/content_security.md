# Content Security

This document outlines the security mechanisms for content delivery in the TL;DR Highlight API, with a focus on signed URLs for secure content access.

## Overview

The TL;DR Highlight API provides secure access to video highlights through signed URLs. This approach ensures that:

1. Only authorized users can access content
2. Access can be time-limited
3. Content can be shared securely with external users
4. Access patterns can be monitored and controlled

## Signed URL Implementation

### What are Signed URLs?

Signed URLs are temporary URLs that provide controlled access to private content. They include authentication information as part of the URL, allowing access without requiring the user to have API credentials.

### Key Components

- **Base URL**: The location of the content
- **Resource Identifier**: The specific highlight or clip being accessed
- **Signature**: A cryptographic signature that validates the URL
- **Expiration Time**: When the URL becomes invalid
- **Access Parameters**: Optional parameters controlling access rights

### Security Mechanisms

1. **JWT-Based Signatures**: We use JWT (JSON Web Tokens) to create secure, tamper-proof signatures
2. **Time-Limited Access**: All signed URLs have an expiration time
3. **Resource-Specific Tokens**: Each URL is valid only for a specific resource
4. **Revocation Capability**: URLs can be revoked before expiration if needed

## Use Cases

### B2B Multi-Tenancy Access

For our B2B customers (streaming platforms), we provide two levels of access:

1. **Platform Representative Access**: Full access to all highlights via API keys
2. **External Streamer Access**: Limited access to specific highlights via signed URLs

### External Streamer Access

A key feature is providing streamers with access to their content without requiring them to be managed users in our system:

- Streamers can access all clips they have created
- Streamers can access all clips from their streams
- Access is provided through signed URLs that don't require authentication
- URLs can be embedded in external applications or shared directly

## Implementation Details

### Generating Signed URLs

Signed URLs are generated using the following process:

1. Create a JWT payload containing:
   - `highlight_id`: The specific highlight being accessed
   - `stream_id`: The associated stream
   - `exp`: Expiration timestamp
   - `org_id`: The organization (streaming platform) ID

2. Sign the JWT using the organization's secret key

3. Construct the URL with the JWT as a query parameter:
   ```
   https://api.tldr.tv/api/v1/content/{highlight_id}?token={jwt}
   ```

### Verifying Signed URLs

When a signed URL is used to access content:

1. The JWT is extracted and verified using the organization's secret key
2. The expiration time is checked
3. The highlight ID in the URL is matched against the JWT payload
4. Access is granted only if all checks pass

## Best Practices

1. **Set Appropriate Expiration Times**:
   - Short-lived for one-time access (1-24 hours)
   - Longer for persistent access (up to 7 days)

2. **Limit URL Scope**:
   - Generate URLs for specific resources, not broad collections
   - Include only necessary access rights

3. **Monitor Usage**:
   - Track URL generation and usage
   - Watch for unusual patterns that might indicate abuse

4. **Secure Key Management**:
   - Protect the signing keys used to generate URLs
   - Rotate keys periodically

## Example Implementation

```python
# Generate a signed URL for a highlight
def generate_signed_url(highlight_id, stream_id, org_id, expiry_hours=24):
    payload = {
        "highlight_id": highlight_id,
        "stream_id": stream_id,
        "org_id": org_id,
        "exp": datetime.utcnow() + timedelta(hours=expiry_hours)
    }
    
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")
    
    return f"https://api.tldr.tv/api/v1/content/{highlight_id}?token={token}"
```

## Conclusion

Signed URLs provide a secure, flexible way to share content with external users without requiring them to have accounts in our system. This is particularly valuable for B2B streaming platforms whose streamers need access to their content without being directly managed by our system.