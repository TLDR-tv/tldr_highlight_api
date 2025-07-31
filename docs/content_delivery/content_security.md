# Content Security

This document outlines the security mechanisms for content delivery in the TL;DR Highlight API, with a focus on signed URLs for secure content access.

## Overview

The TL;DR Highlight API provides secure access to video highlights through signed URLs with enterprise-grade security features:

1. **Per-Organization Signing Keys**: Each organization has unique cryptographic keys
2. **Key Rotation**: Automatic and manual key rotation for enhanced security
3. **Token Revocation**: Ability to revoke access before expiration
4. **IP Restrictions**: Optional IP-based access control
5. **Usage Limits**: Control how many times a URL can be used
6. **Comprehensive Audit Trail**: Track all access attempts and key usage

## Enhanced Security Architecture

### Per-Organization Signing Keys

Each organization (streaming platform) has its own set of cryptographic signing keys:

- **Primary Key**: The active key used for signing new URLs
- **Rotating Keys**: Keys being phased out but still valid for verification
- **Key Versioning**: Track key evolution and rotation history
- **Encrypted Storage**: Keys are encrypted at rest using Fernet encryption

### Key Lifecycle Management

Keys follow a defined lifecycle:

1. **ACTIVE**: Currently used for signing and verification
2. **ROTATING**: Being phased out, valid for verification only
3. **DEACTIVATED**: No longer valid but kept for audit
4. **EXPIRED**: Past expiration date
5. **COMPROMISED**: Marked as compromised and immediately invalid

### Automatic Key Rotation

Keys are automatically rotated based on:

- **Age**: Keys older than 180 days
- **Usage**: Keys that have signed over 1 million tokens
- **Expiration**: Keys approaching expiration (30 days)
- **Security Events**: Immediate rotation on compromise detection

## Signed URL Implementation

### JWT Token Structure

Signed URLs now include comprehensive JWT claims:

```json
{
  // Standard JWT Claims
  "jti": "unique-token-id",        // Unique ID for revocation
  "iss": "tldr-api",               // Issuer identifier
  "aud": "org-123",                // Organization audience
  "sub": "highlight-456",          // Subject (resource)
  "iat": 1234567890,               // Issued at timestamp
  "exp": 1234654290,               // Expiration timestamp
  
  // Custom Security Claims
  "organization_id": 123,          // Organization ID
  "scope": "view",                 // Access scope (view/list/download)
  "version": "2.0",                // Token format version
  "kid": "org123_key_20240115",    // Key identifier
  
  // Resource Claims
  "highlight_id": 456,             // Highlight being accessed
  "stream_id": 789,                // Associated stream
  
  // Optional Security Claims
  "ip": "192.168.1.1",            // IP restriction
  "usage_limit": 10,              // Maximum uses
  "usage_count": 0                // Current usage
}
```

### Access Scopes

Tokens can be issued with different access scopes:

- **VIEW**: Read access to specific content
- **LIST**: List highlights for a stream
- **DOWNLOAD**: Download original content
- **SHARE**: Create derivative signed URLs

### Token Types

The system supports multiple token types:

1. **Single Highlight Access**: Access one specific highlight
2. **Stream Access**: List all highlights for a stream
3. **Batch Access**: Access multiple highlights with one token
4. **Time-boxed Access**: Access during specific time windows

## Security Features

### 1. Token Revocation

Tokens can be revoked before expiration:

```python
# Revoke a token
await url_signer.revoke_token(
    token="eyJ...",
    reason="User request"
)
```

Revoked tokens are stored in Redis with TTL matching original expiration.

### 2. IP Restrictions

Tokens can be restricted to specific IP addresses:

```python
# Generate IP-restricted URL
signed_url = await url_signer.generate_highlight_url(
    base_url="https://api.tldr.tv",
    highlight_id=123,
    ip_restriction="192.168.1.100"
)
```

### 3. Usage Limits

Control how many times a URL can be used:

```python
# Generate single-use URL
signed_url = await url_signer.generate_highlight_url(
    base_url="https://api.tldr.tv",
    highlight_id=123,
    usage_limit=1  # Single use only
)
```

### 4. Audit Logging

All token operations are logged:

- Token generation with JTI and claims
- Verification attempts (success/failure)
- Revocation events
- Key usage statistics

## API Endpoints

### Content Access Endpoints

1. **Single Highlight Access**
   ```
   GET /api/v1/content/{highlight_id}?token={jwt}
   ```
   Redirects to actual content URL after validation

2. **Stream Highlights Listing**
   ```
   GET /api/v1/content/stream/{stream_id}?token={jwt}
   ```
   Returns paginated list of highlights with signed URLs

3. **Batch Content Access**
   ```
   GET /api/v1/content/batch?token={jwt}&ids=1,2,3
   ```
   Access multiple highlights with one token

### Security Management Endpoints

1. **Key Information**
   ```python
   GET /api/v1/security/keys/{org_id}
   ```
   Returns current key status and rotation recommendations

2. **Manual Key Rotation**
   ```python
   POST /api/v1/security/keys/{org_id}/rotate
   ```
   Manually trigger key rotation

3. **Token Revocation**
   ```python
   POST /api/v1/security/tokens/revoke
   ```
   Revoke specific tokens

## Best Practices

### 1. Key Management

- **Automatic Rotation**: Enable automatic key rotation policies
- **Monitor Key Health**: Track key age, usage, and rotation needs
- **Secure Storage**: Use encrypted storage for all keys
- **Backup Keys**: Maintain secure backups of encryption keys

### 2. Token Configuration

- **Short Expiration**: Use shortest practical expiration times
- **Specific Scopes**: Grant minimal necessary permissions
- **JTI Required**: Always include JTI for revocation capability
- **IP Restrictions**: Use for sensitive content or high-value assets

### 3. Monitoring and Alerts

- **Usage Patterns**: Monitor for unusual access patterns
- **Failed Verifications**: Track and investigate failures
- **Key Rotation Events**: Alert on rotation needs
- **Revocation Activity**: Monitor revocation patterns

### 4. Integration Guidelines

- **Cache Verification Results**: Cache successful verifications briefly
- **Handle Rotation Gracefully**: Support multiple active keys
- **Implement Retry Logic**: Handle temporary verification failures
- **Log Security Events**: Maintain audit trail

## Example Implementation

### Generating Enhanced Signed URLs

```python
from src.api.dependencies.security import get_url_signer
from src.domain.services.url_signer_interface import TokenScope

# Generate a signed URL with full security features
async def create_secure_highlight_url(
    highlight_id: int,
    stream_id: int,
    organization_id: int,
    client_ip: str = None
):
    url_signer = await get_url_signer()
    
    signed_url = await url_signer.generate_highlight_url(
        base_url="https://api.tldr.tv",
        highlight_id=highlight_id,
        stream_id=stream_id,
        organization_id=organization_id,
        scope=TokenScope.VIEW,
        expiry_hours=24,
        ip_restriction=client_ip,  # Optional IP restriction
        usage_limit=100,           # Max 100 uses
        additional_claims={
            "content_quality": "1080p",
            "allow_download": False
        }
    )
    
    return signed_url
```

### Verifying Enhanced Tokens

```python
# Verify a token with all security checks
is_valid, payload, error = await url_signer.verify_token(
    token=token,
    required_claims={
        "highlight_id": highlight_id,
        "scope": "view"
    },
    verify_ip=request.client.host
)

if not is_valid:
    logger.warning(f"Token verification failed: {error}")
    raise HTTPException(401, detail=error)
```

### Managing Key Rotation

```python
# Check if rotation is needed
key_info = await url_signer.get_organization_key_info(org_id)
if key_info["should_rotate"]:
    # Trigger rotation
    success = await url_signer.rotate_signing_key(
        organization_id=org_id,
        reason=key_info["rotate_reason"],
        created_by="system"
    )
```

## Migration Guide

### Upgrading from Global Keys

1. **Generate Organization Keys**: Create unique keys per organization
2. **Dual Verification**: Support both old and new tokens temporarily
3. **Gradual Migration**: Issue new tokens with organization keys
4. **Monitor Usage**: Track old vs new token usage
5. **Deprecate Global Key**: Remove after migration complete

### Configuration Changes

Update your security configuration:

```python
# Old configuration
JWT_SECRET_KEY = "global-secret"

# New configuration
SECURITY_CONFIG = {
    "jwt_default_algorithm": "HS256",
    "master_signing_key": None,  # Deprecated
    "min_key_length": 32,
    "require_jti": True,
    "enable_token_revocation": True,
    "require_ip_validation": False,
    "max_token_lifetime_hours": 168,  # 7 days
    "jwt_issuer": "tldr-api"
}
```

## Security Considerations

### Threat Model

The enhanced system protects against:

1. **Key Compromise**: Per-org keys limit blast radius
2. **Token Replay**: JTI and usage limits prevent replay attacks
3. **Unauthorized Access**: IP restrictions and scopes limit access
4. **Long-lived Tokens**: Forced expiration and revocation
5. **Audit Gaps**: Comprehensive logging of all operations

### Compliance

The implementation supports:

- **GDPR**: Token revocation for right to erasure
- **SOC 2**: Audit trails and key rotation
- **PCI DSS**: Encrypted key storage
- **ISO 27001**: Security controls and monitoring

## Conclusion

The enhanced signed URL system provides enterprise-grade security for content delivery with:

- Per-organization cryptographic isolation
- Comprehensive token lifecycle management
- Advanced security features (revocation, IP restrictions, usage limits)
- Full audit trail and monitoring capabilities
- Seamless migration path from legacy systems

This ensures secure, scalable content delivery for B2B streaming platforms while maintaining flexibility for various use cases.