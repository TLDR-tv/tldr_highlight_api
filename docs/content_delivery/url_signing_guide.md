# URL Signing Implementation Guide

This guide provides detailed implementation instructions for the enhanced URL signing system in the TL;DR Highlight API.

## Quick Start

### Basic URL Generation

```python
from src.api.dependencies.security import get_url_signer

# Get the URL signer instance
url_signer = await get_url_signer()

# Generate a simple signed URL
signed_url = await url_signer.generate_highlight_url(
    base_url="https://api.tldr.tv",
    highlight_id=123,
    stream_id=456,
    organization_id=789,
    expiry_hours=24
)
```

### URL Verification

```python
# In your FastAPI endpoint
from fastapi import Depends
from src.api.dependencies.security import get_url_signer

@router.get("/content/{highlight_id}")
async def access_content(
    highlight_id: int,
    token: str,
    request: Request,
    url_signer: URLSigner = Depends(get_url_signer)
):
    # Verify the token
    is_valid, payload, error = await url_signer.verify_token(
        token=token,
        required_claims={"highlight_id": highlight_id},
        verify_ip=request.client.host
    )
    
    if not is_valid:
        raise HTTPException(401, detail=error)
    
    # Process the request...
```

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
├─────────────────────────┬───────────────────────────────────┤
│    URLSigner Service    │      Content Router               │
├─────────────────────────┴───────────────────────────────────┤
│                    Security Layer                            │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ KeyService   │ SecurityConfig│ TokenRevocation│ KeyEncryption│
├──────────────┴──────────────┴──────────────┴───────────────┤
│                  Persistence Layer                           │
├─────────────────────────┬───────────────────────────────────┤
│ OrganizationKeyRepo     │        Redis Cache                │
├─────────────────────────┴───────────────────────────────────┤
│                    Database                                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **URL Generation Request** → URLSigner
2. **Key Retrieval** → OrganizationKeyRepository
3. **Key Decryption** → KeyEncryption Service
4. **JWT Creation** → JWT Library with Claims
5. **URL Construction** → Return to Client

## Key Management

### Creating Organization Keys

```python
from src.infrastructure.security.key_service import KeyGenerationService
from src.domain.entities.organization_key import KeyAlgorithm

# Initialize key service
key_service = KeyGenerationService(
   security_config=config,
   master_encryption_key=master_key
)

# Create a new organization key
org_key = key_service.create_organization_key(
   organization_id=123,
   algorithm=KeyAlgorithm.HS256,
   expires_in_days=365,
   created_by="admin",
   description="Primary signing key"
)

# Save to repository
saved_key = await key_repository.create(org_key)
```

### Key Rotation

```python
# Automatic rotation check
should_rotate, reason = key_service.should_rotate_key(current_key)
if should_rotate:
    new_key = key_service.rotate_key(
        current_key=current_key,
        reason=reason,
        created_by="system"
    )
    
    # Save and activate new key
    await key_repository.create(new_key)
    await key_repository.set_primary(org_id, new_key.key_id)
```

### Key Lifecycle States

```python
from src.domain.entities.organization_key import KeyStatus

# Key status transitions
ACTIVE → ROTATING → DEACTIVATED
ACTIVE → EXPIRED(automatic
on
expiry)
ACTIVE → COMPROMISED(security
event)
```

## Advanced Features

### IP-Restricted URLs

```python
# Generate URL restricted to specific IP
signed_url = await url_signer.generate_highlight_url(
    base_url="https://api.tldr.tv",
    highlight_id=123,
    stream_id=456,
    organization_id=789,
    ip_restriction="192.168.1.100",  # Only accessible from this IP
    expiry_hours=24
)

# Verification will check IP automatically
is_valid, payload, error = await url_signer.verify_token(
    token=token,
    verify_ip=request.client.host  # Pass actual client IP
)
```

### Usage-Limited URLs

```python
# Single-use URL
single_use_url = await url_signer.generate_highlight_url(
    base_url="https://api.tldr.tv",
    highlight_id=123,
    stream_id=456,
    organization_id=789,
    usage_limit=1,  # Can only be used once
    expiry_hours=24
)

# Limited-use URL
limited_url = await url_signer.generate_highlight_url(
    base_url="https://api.tldr.tv",
    highlight_id=123,
    stream_id=456,
    organization_id=789,
    usage_limit=10,  # Can be used 10 times
    expiry_hours=168  # 7 days
)
```

### Batch Access URLs

```python
# Generate URL for multiple highlights
batch_url = await url_signer.generate_batch_url(
    base_url="https://api.tldr.tv",
    highlight_ids=[123, 124, 125, 126],
    organization_id=789,
    scope=TokenScope.VIEW,
    expiry_hours=24
)

# URL format: /api/v1/content/batch?token={jwt}&ids=123,124,125,126
```

### Custom Claims

```python
# Add custom claims to tokens
signed_url = await url_signer.generate_highlight_url(
    base_url="https://api.tldr.tv",
    highlight_id=123,
    stream_id=456,
    organization_id=789,
    additional_claims={
        "content_quality": "1080p",
        "allow_download": True,
        "watermark": "org-logo",
        "analytics_id": "campaign-123"
    }
)

# Access custom claims during verification
is_valid, payload, error = await url_signer.verify_token(token)
if is_valid:
    quality = payload.get("content_quality")  # "1080p"
    can_download = payload.get("allow_download")  # True
```

## Token Management

### Token Revocation

```python
# Revoke a token immediately
success = await url_signer.revoke_token(
    token="eyJ...",
    reason="Content removed"
)

# Check if a token is revoked
jti = "token-id-123"
is_revoked = await url_signer.check_token_revoked(jti)
```

### Token Information

```python
# Get token claims without verification
import jose.jwt as jwt

unverified_claims = jwt.get_unverified_claims(token)
jti = unverified_claims.get("jti")
exp = unverified_claims.get("exp")
org_id = unverified_claims.get("organization_id")
```

## Security Configuration

### Environment Variables

```bash
# Required
MASTER_ENCRYPTION_KEY=your-256-bit-encryption-key
JWT_ISSUER=tldr-api

# Optional
MIN_KEY_LENGTH=32
MAX_TOKEN_LIFETIME_HOURS=168
REQUIRE_JTI=true
ENABLE_TOKEN_REVOCATION=true
REQUIRE_IP_VALIDATION=false
```

### Configuration Object

```python
from src.infrastructure.security.config import SecurityConfig

config = SecurityConfig(
    jwt_default_algorithm=JWTAlgorithm.HS256,
    master_signing_key=None,  # Deprecated
    min_key_length=32,
    require_jti=True,
    enable_token_revocation=True,
    require_ip_validation=False,
    max_token_lifetime_hours=168,
    jwt_issuer="tldr-api"
)
```

## Integration Examples

### FastAPI Dependency

```python
from fastapi import Depends
from src.api.dependencies.security import get_url_signer

@router.post("/highlights/{highlight_id}/share")
async def create_share_url(
    highlight_id: int,
    request: ShareRequest,
    current_user: User = Depends(get_current_user),
    url_signer: URLSigner = Depends(get_url_signer)
):
    # Generate shareable URL
    share_url = await url_signer.generate_highlight_url(
        base_url=str(request.base_url),
        highlight_id=highlight_id,
        stream_id=request.stream_id,
        organization_id=current_user.organization_id,
        scope=TokenScope.VIEW,
        expiry_hours=request.expiry_hours or 24,
        ip_restriction=request.restrict_to_ip,
        usage_limit=request.max_views
    )
    
    return {"share_url": share_url}
```

### Webhook Integration

```python
# Send signed URLs in webhooks
async def send_highlight_webhook(highlight: Highlight, webhook_url: str):
    # Generate signed URL for webhook recipient
    signed_url = await url_signer.generate_highlight_url(
        base_url="https://api.tldr.tv",
        highlight_id=highlight.id,
        stream_id=highlight.stream_id,
        organization_id=highlight.organization_id,
        expiry_hours=72,  # 3 days for webhook processing
        additional_claims={
            "webhook_id": webhook.id,
            "event_type": "highlight.created"
        }
    )
    
    # Send webhook with signed URL
    await send_webhook(webhook_url, {
        "event": "highlight.created",
        "highlight": highlight.to_dict(),
        "access_url": signed_url
    })
```

### SDK Integration

```python
# Python SDK example
class TLDRClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tldr.tv"
    
    async def get_highlight_url(self, highlight_id: int, **options):
        """Get a signed URL for highlight access."""
        response = await self._post(
            f"/highlights/{highlight_id}/access-url",
            json={
                "expiry_hours": options.get("expiry_hours", 24),
                "ip_restriction": options.get("ip_restriction"),
                "usage_limit": options.get("usage_limit")
            }
        )
        return response["signed_url"]
```

## Monitoring and Debugging

### Key Health Check

```python
# Check key health for an organization
key_info = await url_signer.get_organization_key_info(org_id)
print(f"Key ID: {key_info['key_id']}")
print(f"Algorithm: {key_info['algorithm']}")
print(f"Version: {key_info['version']}")
print(f"Created: {key_info['created_at']}")
print(f"Expires: {key_info['expires_at']}")
print(f"Usage Count: {key_info['usage_count']}")
print(f"Should Rotate: {key_info['should_rotate']}")
print(f"Rotation Reason: {key_info['rotate_reason']}")
```

### Token Verification Logging

```python
import logging

logger = logging.getLogger(__name__)

# Enhanced logging for debugging
is_valid, payload, error = await url_signer.verify_token(token)

if not is_valid:
    logger.warning(
        "Token verification failed",
        extra={
            "error": error,
            "token_prefix": token[:20],
            "organization_id": payload.get("organization_id") if payload else None,
            "jti": payload.get("jti") if payload else None
        }
    )
else:
    logger.info(
        "Token verified successfully",
        extra={
            "organization_id": payload["organization_id"],
            "jti": payload.get("jti"),
            "scope": payload.get("scope"),
            "remaining_uses": payload.get("usage_limit", 0) - payload.get("usage_count", 0)
        }
    )
```

### Performance Monitoring

```python
import time
from prometheus_client import Histogram, Counter

# Metrics
token_generation_time = Histogram(
    "url_signing_generation_seconds",
    "Time to generate signed URL"
)
token_verification_time = Histogram(
    "url_signing_verification_seconds", 
    "Time to verify token"
)
token_verification_failures = Counter(
    "url_signing_verification_failures_total",
    "Total token verification failures",
    ["reason"]
)

# Usage
with token_generation_time.time():
    signed_url = await url_signer.generate_highlight_url(...)

with token_verification_time.time():
    is_valid, payload, error = await url_signer.verify_token(...)
    
if not is_valid:
    token_verification_failures.labels(reason=error).inc()
```

## Troubleshooting

### Common Issues

1. **"No signing key found for organization"**
   - Ensure organization has a primary key created
   - Check key status is ACTIVE
   - Verify organization_id is correct

2. **"Invalid token signature"**
   - Key might have been rotated
   - Token might be corrupted
   - Wrong organization_id in verification

3. **"Token has been revoked"**
   - Check Redis connectivity
   - Verify JTI exists in token
   - Check revocation reason in logs

4. **"IP mismatch"**
   - Ensure correct client IP is passed
   - Check for proxy/load balancer headers
   - Verify IP restriction is intended

### Debug Mode

```python
# Enable debug logging
import logging

logging.getLogger("src.infrastructure.security").setLevel(logging.DEBUG)

# Test key operations
async def debug_keys(org_id: int):
    # List all keys
    keys = await key_repository.get_active_keys(org_id)
    for key in keys:
        print(f"Key: {key.key_id}, Status: {key.status}, Primary: {key.is_primary}")
    
    # Test verification with each key
    for key in keys:
        try:
            # Create test token
            test_payload = {"test": True, "org_id": org_id}
            token = jwt.encode(
                test_payload,
                key_service.decrypt_key_value(key),
                algorithm=key.algorithm.value
            )
            
            # Try to verify
            is_valid, _, error = await url_signer.verify_token(token)
            print(f"Key {key.key_id}: Valid={is_valid}, Error={error}")
        except Exception as e:
            print(f"Key {key.key_id}: Error={str(e)}")
```

## Best Practices Summary

1. **Always use organization-specific keys** - Never fall back to global keys in production
2. **Set appropriate expiration times** - Balance security with user experience
3. **Monitor key health** - Set up alerts for rotation needs
4. **Use JTI for critical content** - Enable revocation capability
5. **Log security events** - Maintain audit trail for compliance
6. **Cache verification results** - Improve performance for repeated access
7. **Handle key rotation gracefully** - Support multiple active keys during transition
8. **Validate all inputs** - Sanitize organization_id, highlight_id, etc.
9. **Use strong encryption keys** - Minimum 256-bit keys for master encryption
10. **Regular security audits** - Review key usage patterns and access logs