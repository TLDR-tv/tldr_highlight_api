# Multi-Tenant Architecture Plan for B2B Highlight API

## Executive Summary

This document outlines a comprehensive multi-tenant architecture for the B2B Highlight API, integrating AWS Cognito for user authentication while maintaining API key authentication for programmatic access. The design supports enterprise customers like CrowdCover with proper tenant isolation, scalability, and security.

## Current State Analysis

### Existing Implementation
- **User Model**: Basic user table with email/password authentication
- **Organization Model**: Simple organization ownership model
- **API Keys**: Custom implementation with SHA256 hashing
- **Authentication**: Local password hashing with bcrypt
- **tldrtv Integration**: Uses AWS Cognito for consumer authentication

### Gaps to Address
1. No true multi-tenancy with proper isolation
2. No SSO/SAML support for enterprise customers
3. Limited user-organization relationships (only ownership)
4. No tenant context injection in requests
5. No integration with AWS Cognito

## Proposed Architecture

### 1. Hybrid Authentication Model

We'll implement a hybrid approach combining AWS Cognito for user authentication and API keys for programmatic access:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Authentication Layer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                │
│  │   AWS Cognito    │    │   API Key Auth   │                │
│  ├──────────────────┤    ├──────────────────┤                │
│  │ • User Login     │    │ • Programmatic   │                │
│  │ • SSO/SAML       │    │ • Webhooks       │                │
│  │ • MFA            │    │ • CI/CD          │                │
│  │ • Password Reset │    │ • Rate Limiting  │                │
│  └─────────┬────────┘    └─────────┬────────┘                │
│            │                       │                          │
│  ┌─────────▼───────────────────────▼────────┐                │
│  │         Tenant Context Middleware         │                │
│  │    • Extract tenantId from token/key      │                │
│  │    • Inject into request context          │                │
│  │    • Validate permissions                 │                │
│  └──────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Multi-Tenant Data Model

#### Enhanced User Model
```python
class User(Base, TimestampMixin):
    """Enhanced user model with Cognito integration."""
    
    id: Mapped[int] = mapped_column(primary_key=True)
    cognito_sub: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    
    # Remove password_hash - handled by Cognito
    # password_hash: Mapped[str] = mapped_column(String(255))
    
    # User profile
    full_name: Mapped[str] = mapped_column(String(255))
    phone_number: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Tenant relationships
    default_organization_id: Mapped[Optional[int]] = mapped_column(ForeignKey("organizations.id"))
    
    # Many-to-many relationship with organizations
    organizations: Mapped[List["OrganizationMember"]] = relationship(
        "OrganizationMember", back_populates="user"
    )
```

#### Enhanced Organization Model
```python
class Organization(Base, TimestampMixin):
    """Multi-tenant organization with enhanced features."""
    
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)  # UUID
    name: Mapped[str] = mapped_column(String(255))
    
    # Billing and subscription
    plan_type: Mapped[str] = mapped_column(String(50))
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(255))
    
    # SSO Configuration
    sso_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    sso_provider: Mapped[Optional[str]] = mapped_column(String(50))  # "saml", "oidc"
    sso_metadata_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Domain verification for auto-join
    verified_domains: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Settings
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Relationships
    members: Mapped[List["OrganizationMember"]] = relationship(
        "OrganizationMember", back_populates="organization"
    )
```

#### Organization Membership Model
```python
class OrganizationMember(Base, TimestampMixin):
    """Many-to-many relationship between users and organizations."""
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    
    # Role-based access control
    role: Mapped[str] = mapped_column(String(50))  # "owner", "admin", "member", "viewer"
    
    # Invitation tracking
    invited_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    invited_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="organizations")
    organization: Mapped["Organization"] = relationship("Organization", back_populates="members")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", name="uq_user_organization"),
    )
```

### 3. AWS Cognito Configuration

#### User Pool Structure
```yaml
UserPool:
  Name: "tldr-highlight-b2b-${STAGE}"
  Schema:
    - Name: email
      Required: true
      Mutable: false
    - Name: tenant_id
      AttributeDataType: String
      Mutable: false  # Set during registration only
    - Name: organization_id
      AttributeDataType: Number
      Mutable: true   # Can change organizations
    - Name: default_role
      AttributeDataType: String
      Mutable: true
  
  Policies:
    PasswordPolicy:
      MinimumLength: 12
      RequireUppercase: true
      RequireLowercase: true
      RequireNumbers: true
      RequireSymbols: true
  
  MfaConfiguration: OPTIONAL
  EnabledMfas:
    - SOFTWARE_TOKEN_MFA
    - SMS_MFA
```

#### App Client Settings
```yaml
AppClient:
  Name: "tldr-highlight-b2b-client"
  ExplicitAuthFlows:
    - ALLOW_USER_PASSWORD_AUTH
    - ALLOW_REFRESH_TOKEN_AUTH
    - ALLOW_CUSTOM_AUTH  # For API key auth flow
  
  TokenValidity:
    AccessToken: 1 hour
    IdToken: 1 hour
    RefreshToken: 30 days
```

### 4. Tenant Isolation Strategy

#### Database Level
```sql
-- Row Level Security Policy Example
CREATE POLICY tenant_isolation ON highlights
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

-- Applied to all tenant-scoped tables:
-- streams, batches, highlights, webhooks, usage_records
```

#### Application Level
```python
class TenantContextMiddleware:
    """Middleware to inject tenant context into all requests."""
    
    async def __call__(self, request: Request, call_next):
        # Extract tenant from JWT or API key
        tenant_id = await self.extract_tenant_id(request)
        
        # Store in context for database queries
        request.state.tenant_id = tenant_id
        request.state.organization_id = organization_id
        
        # Set database session variable
        async with get_db() as db:
            await db.execute(
                text("SET LOCAL app.current_tenant = :tenant_id"),
                {"tenant_id": tenant_id}
            )
        
        response = await call_next(request)
        return response
```

### 5. API Key Management Enhancement

#### API Key Model Update
```python
class APIKey(Base, TimestampMixin):
    """Enhanced API key with tenant scoping."""
    
    id: Mapped[int] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    
    # Tenant scoping
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    tenant_id: Mapped[str] = mapped_column(String(36), index=True)
    
    # Permissions
    name: Mapped[str] = mapped_column(String(255))
    scopes: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Usage tracking
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Rate limiting
    rate_limit_override: Mapped[Optional[int]] = mapped_column(Integer)
```

### 6. Authentication Flow Implementation

#### User Authentication (Cognito)
```python
class CognitoAuthService:
    """Service for Cognito authentication operations."""
    
    async def authenticate_user(self, email: str, password: str) -> TokenResponse:
        """Authenticate user via Cognito."""
        try:
            # Initiate auth with Cognito
            result = self.cognito_client.initiate_auth(
                username=email,
                password=password
            )
            
            # Extract user attributes from ID token
            id_token = self._decode_token(result["IdToken"])
            tenant_id = id_token["custom:tenant_id"]
            org_id = id_token["custom:organization_id"]
            
            # Create session
            return TokenResponse(
                access_token=result["AccessToken"],
                id_token=result["IdToken"],
                refresh_token=result["RefreshToken"],
                tenant_id=tenant_id,
                organization_id=org_id
            )
        except CognitoClientError as e:
            raise AuthenticationError(str(e))
```

#### API Key Authentication
```python
async def get_current_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db)
) -> APIKey:
    """Validate API key and return key object with tenant context."""
    if not credentials.credentials.startswith("tldr_sk_"):
        raise AuthenticationError("Invalid API key format")
    
    # Hash the key
    key_hash = hash_api_key(credentials.credentials)
    
    # Look up key with organization
    result = await db.execute(
        select(APIKey)
        .options(selectinload(APIKey.organization))
        .where(APIKey.key_hash == key_hash)
        .where(APIKey.active == True)
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise AuthenticationError("Invalid API key")
    
    # Update usage
    api_key.last_used_at = datetime.now(timezone.utc)
    api_key.usage_count += 1
    await db.commit()
    
    return api_key
```

### 7. Organization Management Features

#### User Invitation Flow
```python
class OrganizationService:
    """Service for organization management."""
    
    async def invite_user(
        self,
        organization_id: int,
        email: str,
        role: str,
        invited_by: User,
        db: AsyncSession
    ) -> OrganizationInvite:
        """Invite a user to join an organization."""
        # Check if user exists in Cognito
        cognito_user = await self._find_cognito_user(email)
        
        if cognito_user:
            # Existing user - add to organization
            await self._add_user_to_organization(
                cognito_user, organization_id, role
            )
            # Send notification email
            await self._send_organization_added_email(email, organization)
        else:
            # New user - create Cognito account with pre-set attributes
            temp_password = self._generate_temp_password()
            await self.cognito_client.admin_create_user(
                username=email,
                temporary_password=temp_password,
                user_attributes={
                    "email": email,
                    "custom:tenant_id": organization.tenant_id,
                    "custom:organization_id": str(organization_id),
                    "custom:default_role": role
                }
            )
            # Send invitation email with temp password
            await self._send_invitation_email(email, organization, temp_password)
        
        # Record invitation
        return await self._create_invitation_record(
            organization_id, email, role, invited_by, db
        )
```

#### SSO Integration
```python
class SSOService:
    """Service for SSO integration."""
    
    async def configure_saml(
        self,
        organization: Organization,
        metadata_url: str,
        db: AsyncSession
    ) -> None:
        """Configure SAML SSO for an organization."""
        # Create identity provider in Cognito
        provider_name = f"saml-{organization.tenant_id}"
        
        await self.cognito_client.create_identity_provider(
            user_pool_id=self.user_pool_id,
            provider_name=provider_name,
            provider_type="SAML",
            provider_details={
                "MetadataURL": metadata_url
            },
            attribute_mapping={
                "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
                "custom:tenant_id": organization.tenant_id
            }
        )
        
        # Update organization
        organization.sso_enabled = True
        organization.sso_provider = "saml"
        organization.sso_metadata_url = metadata_url
        
        await db.commit()
```

### 8. Migration Strategy

#### Phase 1: Foundation (Week 1)
1. Set up AWS Cognito User Pool
2. Implement enhanced data models
3. Create database migrations
4. Build Cognito authentication service

#### Phase 2: Integration (Week 2)
1. Implement tenant context middleware
2. Update API key authentication
3. Build organization management endpoints
4. Add user invitation flow

#### Phase 3: Advanced Features (Post-Sprint)
1. SSO/SAML integration
2. Domain verification for auto-join
3. Advanced RBAC implementation
4. Usage analytics per tenant

### 9. Security Considerations

#### Data Isolation
- Enforce tenant_id in all queries
- Use database-level RLS policies
- Validate tenant context in middleware

#### Authentication
- Require MFA for admin roles
- Implement session management
- API key rotation policy

#### Authorization
- Role-based access control (RBAC)
- Resource-level permissions
- Audit logging for all actions

### 10. Implementation Checklist

- [ ] Create AWS Cognito User Pool with custom attributes
- [ ] Implement enhanced database models
- [ ] Build Cognito authentication service
- [ ] Create tenant context middleware
- [ ] Update API key authentication with tenant scoping
- [ ] Implement organization management endpoints
- [ ] Build user invitation system
- [ ] Add organization switching for multi-org users
- [ ] Create tenant-scoped database queries
- [ ] Implement usage tracking per organization
- [ ] Add comprehensive audit logging
- [ ] Create admin dashboard for tenant management
- [ ] Document API changes for clients
- [ ] Migrate existing users to Cognito
- [ ] Test with CrowdCover integration

## Conclusion

This multi-tenant architecture provides a robust foundation for the B2B Highlight API, supporting enterprise customers with proper isolation, scalability, and security. The hybrid authentication model allows both user-based and programmatic access while maintaining tenant boundaries throughout the system.