# AWS Cognito Multi-Tenant Architecture Research for B2B SaaS

## Executive Summary

This document presents comprehensive research on AWS Cognito multi-tenant architecture patterns for B2B SaaS applications, conducted in January 2025. The research covers best practices, implementation strategies, and integration patterns with FastAPI.

## 1. Multi-Tenant Architecture Approaches

### 1.1 User Pool-Based Multi-Tenancy (Silo Model)

**Description**: Create a separate user pool for each tenant in your application.

**Pros**:
- Maximum isolation between tenants
- Different configurations per tenant (MFA, password policies, etc.)
- Complete control over tenant-specific authentication flows
- Ideal for Platinum/Enterprise tier customers

**Cons**:
- High development and operational overhead
- Hard limit of 4 Custom Domains per AWS account
- Requires automation tools for consistency
- Service quota shared across all pools in account/region

**Use When**:
- Strong isolation requirements exist
- Different authentication configurations needed per tenant
- Enterprise customers require dedicated infrastructure

### 1.2 Single User Pool with Custom Attributes (Pool Model)

**Description**: Store all tenants in one user pool with custom attributes like `tenantId`.

**Pros**:
- Easiest to implement and maintain
- Unified sign-up/sign-in experience
- Single standard for security and threat protection
- Cost-effective for large number of tenants

**Cons**:
- Less isolation between tenants
- Shared configuration for all tenants
- Cannot support tenant-specific IdP configurations

**Implementation**:
```json
{
  "custom:tenantId": "tenant-123",
  "custom:tenantName": "Acme Corp",
  "custom:tenantTier": "standard",
  "custom:organizationId": "org-456"
}
```

### 1.3 Group-Based Multi-Tenancy

**Description**: Use Cognito groups to represent tenants and manage permissions.

**Pros**:
- Built-in RBAC support
- Groups included in JWT tokens (`cognito:groups` claim)
- Can map to IAM roles through identity pools
- Supports hierarchical permission structures

**Cons**:
- Limited to 500 groups per user pool
- Group management can become complex
- Not ideal for large number of tenants

**Implementation Pattern**:
- Create groups like `tenant-123-admin`, `tenant-123-user`
- Use Lambda Authorizer to validate group membership
- Map groups to IAM roles for AWS resource access

### 1.4 Hybrid Approach (Recommended for B2B SaaS)

**Description**: Combine approaches based on customer tiers.

**Architecture**:
- **Platinum Tier**: Dedicated user pools (silo model)
- **Standard/Basic Tiers**: Shared pool with custom attributes
- **Group-based permissions**: Within each pool for role management

## 2. Tenant Isolation Best Practices

### 2.1 Data Isolation

**Key Requirements**:
- Each tenant accesses only their own data
- Tenant data invisible to other tenants
- Implement at multiple layers (API, database, storage)

**Implementation**:
1. **Lambda Authorizer Pattern**:
   ```python
   # Extract tenantId from JWT
   tenant_id = jwt_claims.get('custom:tenantId')
   
   # Inject into request context
   context['tenantId'] = tenant_id
   
   # Validate access to requested resources
   if resource_tenant_id != tenant_id:
       raise UnauthorizedError()
   ```

2. **Database Row-Level Security**:
   - Add `tenant_id` column to all tables
   - Enforce through application logic
   - Consider database-level RLS if supported

### 2.2 Configuration Management

**Custom Attribute Configuration**:
- Make `tenantId` read-only at user level
- Manage through admin APIs only
- Store additional tenant metadata in DynamoDB

**Tenant Mapping Table**:
```json
{
  "emailDomain": "acme.com",
  "tenantId": "tenant-123",
  "tenantName": "Acme Corp",
  "idpId": "saml-provider-123",
  "tier": "platinum"
}
```

## 3. Organization/Multi-User Management

### 3.1 B2B User Hierarchy

**Common Pattern**:
```
Organization (Tenant)
├── Admin Users
├── Regular Users
├── Read-Only Users
└── API Service Accounts
```

**Implementation Strategies**:

1. **Email Domain Mapping**:
   - Map email domains to organizations
   - Auto-assign users based on email domain
   - Support multiple domains per organization

2. **Invitation-Based System**:
   - Organization admins invite users
   - Pre-populate tenant attributes on signup
   - Validate invitation tokens

3. **SAML/OIDC Federation**:
   - Each organization uses their IdP
   - Map SAML attributes to Cognito attributes
   - Support multiple IdPs per tenant (for large orgs)

### 3.2 User Across Multiple Tenants

**Scenarios**:
- Consultants working with multiple clients
- Users switching between test/prod environments
- Partner organizations with shared users

**Solutions**:
1. **Separate Accounts**: User creates account per tenant
2. **Account Linking**: Link multiple Cognito accounts
3. **Master Account**: Central account with tenant switching

## 4. API Key Management Strategy

### 4.1 Hybrid Authentication Model

**Recommended Architecture**:
```
User Authentication: AWS Cognito (JWT)
API Authentication: API Keys (for programmatic access)
Combined: Both methods supported based on use case
```

### 4.2 API Key Implementation

**Storage Pattern**:
```json
{
  "apiKeyId": "ak_123456",
  "apiKeyHash": "sha256_hash",
  "tenantId": "tenant-123",
  "name": "Production API Key",
  "scopes": ["read", "write"],
  "createdBy": "user-456",
  "expiresAt": "2025-12-31",
  "lastUsed": "2025-01-07"
}
```

**Security Considerations**:
- Hash API keys before storage
- Implement key rotation
- Track usage for billing
- Scope keys to specific operations

### 4.3 Developer Portal Pattern

1. **User Registration**: Through Cognito
2. **API Client Creation**: Generate long-lived refresh token
3. **Token Storage**: Encrypted in DynamoDB
4. **Access Pattern**: 
   - Exchange refresh token for temporary ID token
   - Use ID token for API calls
   - Refresh automatically before expiration

## 5. FastAPI Integration

### 5.1 Authentication Middleware

**Using fastapi-cognito Library**:
```python
from fastapi_cognito import CognitoAuth, CognitoSettings
from fastapi import FastAPI, Depends

# Configure Cognito
cognito_settings = CognitoSettings(
    userpools={
        "default": {
            "region": "us-east-1",
            "userpool_id": "us-east-1_xxxxx",
            "app_client_id": "xxxxx"
        },
        "eu_pool": {  # Multi-region support
            "region": "eu-west-1",
            "userpool_id": "eu-west-1_xxxxx",
            "app_client_id": "xxxxx"
        }
    }
)

cognito_auth = CognitoAuth(settings=cognito_settings)
app = FastAPI()

@app.get("/protected")
async def protected_route(
    current_user = Depends(cognito_auth.cognito_jwt_bearer)
):
    tenant_id = current_user.get("custom:tenantId")
    return {"tenant": tenant_id, "user": current_user}
```

### 5.2 Multi-Tenant Middleware

**Custom Implementation**:
```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract tenant from JWT or API key
        if "Authorization" in request.headers:
            # JWT path
            token = request.headers["Authorization"]
            claims = validate_jwt(token)
            tenant_id = claims.get("custom:tenantId")
        elif "X-API-Key" in request.headers:
            # API key path
            api_key = request.headers["X-API-Key"]
            tenant_id = get_tenant_from_api_key(api_key)
        else:
            raise HTTPException(401, "No authentication provided")
        
        # Inject tenant context
        request.state.tenant_id = tenant_id
        
        response = await call_next(request)
        return response
```

### 5.3 Dependency Injection Pattern

```python
async def get_current_tenant(request: Request) -> str:
    return request.state.tenant_id

async def get_tenant_db(tenant_id: str = Depends(get_current_tenant)):
    # Return tenant-specific database connection
    return get_db_for_tenant(tenant_id)

@app.get("/api/resources")
async def list_resources(
    db = Depends(get_tenant_db),
    tenant_id: str = Depends(get_current_tenant)
):
    # Automatically scoped to tenant
    return db.query(Resource).filter_by(tenant_id=tenant_id).all()
```

## 6. Service Quotas and Scaling

### 6.1 AWS Cognito Limits (2024)

**Key Quotas**:
- 50,000 users per pool (without limit increase)
- 25 app clients per user pool
- 500 groups per user pool
- 40 calls/second for admin APIs
- 4 custom domains per AWS account

**Scaling Strategies**:
1. **Request Limit Increases**: For user pool size
2. **Multiple AWS Accounts**: For domain limits
3. **Rate Limiting**: Implement at application level
4. **Caching**: Cache JWT validation results

### 6.2 Regional Considerations

**Requirements**:
- Lambda functions must be in same region as user pool
- WAF ACLs must be in same region
- Consider multi-region for global applications

## 7. Cost Considerations (2024)

### 7.1 Pricing Structure

**New Tiers** (Late 2024):
- **Cognito Lite**: Basic features
- **Cognito Essentials**: ~$0.015/MAU after 10k free
- **Cognito Plus**: Advanced features

**B2B SaaS Considerations**:
- Costs typically negligible vs. revenue per user
- Consider alternatives if >15,000 MAU
- Factor in Lambda Authorizer costs

### 7.2 Cost Optimization

1. **Tier Appropriately**: Use Lite for dev/test
2. **Cache Aggressively**: Reduce token validation calls
3. **Batch Operations**: For user management
4. **Monitor Usage**: Set up billing alerts

## 8. Security Best Practices

### 8.1 Token Management

**JWT Best Practices**:
- Short-lived access tokens (1 hour default)
- Longer refresh tokens with rotation
- Store refresh tokens securely
- Implement token revocation

### 8.2 Multi-Factor Authentication

**B2B Requirements**:
- Enforce MFA for admin users
- Support TOTP and SMS
- Allow tenant-level MFA policies
- Consider adaptive MFA

### 8.3 Audit and Compliance

**Implementation**:
- Enable CloudTrail for all Cognito events
- Log all authentication attempts
- Track API key usage
- Implement data retention policies

## 9. Implementation Roadmap

### Phase 1: Foundation
1. Choose multi-tenancy approach based on requirements
2. Set up Cognito user pool(s)
3. Implement basic JWT validation in FastAPI
4. Create tenant isolation middleware

### Phase 2: Organization Management
1. Implement user invitation system
2. Add organization hierarchy
3. Set up SAML/OIDC for enterprise customers
4. Create admin portal for tenant management

### Phase 3: API Key System
1. Design API key storage schema
2. Implement key generation and validation
3. Add usage tracking
4. Create developer portal

### Phase 4: Advanced Features
1. Implement webhook system
2. Add advanced analytics
3. Set up multi-region support
4. Implement cost allocation per tenant

## 10. Key Takeaways

1. **No One-Size-Fits-All**: Choose approach based on customer requirements
2. **Start Simple**: Begin with custom attributes, evolve to dedicated pools
3. **Plan for Scale**: Consider limits and quotas early
4. **Layer Security**: Implement defense in depth
5. **Monitor Everything**: Track usage, costs, and performance
6. **Automate Operations**: Essential for multi-tenant management

## Resources and References

1. [AWS Cognito Multi-Tenant Best Practices](https://docs.aws.amazon.com/cognito/latest/developerguide/multi-tenant-application-best-practices.html)
2. [fastapi-cognito Library](https://pypi.org/project/fastapi-cognito/)
3. [AWS Multi-Tenant SaaS Example](https://github.com/aws-samples/amazon-cognito-example-for-multi-tenant)
4. [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)

## Appendix: Decision Matrix

| Approach | Isolation | Complexity | Cost | Scale | Flexibility |
|----------|-----------|------------|------|-------|-------------|
| User Pool per Tenant | High | High | High | Limited | High |
| Custom Attributes | Low | Low | Low | High | Medium |
| Groups | Medium | Medium | Low | Medium | Medium |
| Hybrid | Variable | High | Medium | High | High |

Choose based on your specific B2B SaaS requirements and customer expectations.