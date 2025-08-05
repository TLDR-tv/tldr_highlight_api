# CLAUDE.md - API Package

This file provides guidance to Claude Code when working with the API package, which serves as the FastAPI web application for the TLDR Highlight API.

## Package Overview

The `api` package provides the **REST API interface** for the enterprise B2B highlight detection service. It handles HTTP requests, authentication, and coordinates with Celery workers for asynchronous processing. Following recent refactoring, this package now also contains all API-specific services including authentication and user management.

## Working with This Package

### Running Commands
```bash
# Always use uv from the package directory
cd packages/api
uv run pytest  # Run tests
uv run uvicorn api.main:app --reload  # Run dev server

# Or from root with --directory
uv --directory packages/api run pytest
```

### Adding Dependencies
```bash
cd packages/api
uv add httpx  # Add production dependency
uv add --dev pytest-httpx  # Add dev dependency
```

## Package Structure

### Services (`src/api/services/`)
API-specific business logic:

1. **Authentication Services** (`auth/`):
   - `jwt_service.py` - JWT token creation and validation
   - `password_service.py` - Password hashing and verification

2. **Business Services**:
   - `user_service.py` - User management and authentication logic
   - `organization_service.py` - Organization management logic

### Routes (`src/api/routes/`)
FastAPI endpoints organized by resource:
- `auth.py` - Authentication endpoints (login, register, refresh)
- `users.py` - User management endpoints
- `organizations.py` - Organization management endpoints
- `streams.py` - Stream processing endpoints
- `highlights.py` - Highlight retrieval endpoints
- `webhooks.py` - Webhook configuration endpoints

## FastAPI Application Structure

### Main Application (`src/api/main.py`)
```python
# Application factory pattern
def create_app() -> FastAPI:
    app = FastAPI(
        title="TLDR Highlight API",
        version="1.0.0",
        docs_url="/docs" if settings.environment == "development" else None,
    )
    # Configure middleware, routes, etc.
    return app
```

### Route Organization
Each route module follows this pattern:
```python
router = APIRouter(prefix="/streams", tags=["streams"])

@router.post("/", response_model=StreamResponse)
async def create_stream(
    request: StreamCreateRequest,
    current_user: User = Depends(get_current_user),  # Auth dependency
    stream_service: StreamService = Depends(get_stream_service),  # Service injection
):
    # Route implementation
```

## Authentication & Security

### Dual Authentication System
1. **API Keys** for B2B clients:
   ```python
   api_key = Depends(get_api_key_from_header)  # X-API-Key header
   ```

2. **JWT Tokens** for web users:
   ```python
   current_user = Depends(get_current_user)  # Bearer token
   ```

### Service Dependencies
The API package now contains its own authentication services:
```python
# JWT Service (api/services/auth/jwt_service.py)
class JWTService:
    def create_access_token(...) -> str
    def verify_access_token(...) -> TokenPayload
    def create_refresh_token(...) -> str
    def verify_refresh_token(...) -> dict

# Password Service (api/services/auth/password_service.py)
class PasswordService:
    def hash_password(password: str) -> str
    def verify_password(plain: str, hashed: str) -> bool
    def validate_password_strength(password: str) -> tuple[bool, list[str]]
```

### Dependency Functions (`dependencies.py`)
```python
# Service initialization
def get_user_service(...) -> UserService:
    # Now imports from api.services.user_service
    
def get_organization_service(...) -> OrganizationService:
    # Now imports from api.services.organization_service

# Auth dependencies remain the same
async def get_current_user(...) -> User: ...
async def get_api_key_from_header(...) -> APIKey: ...
async def require_scope(scope: str): ...
```

### Multi-Tenant Isolation
```python
# Always filter by organization
highlights = await repository.get_by_organization(
    organization_id=current_user.organization_id
)
```

## Request/Response Schemas

### Schema Guidelines (`src/api/schemas/`)
```python
# Request schema
class StreamCreateRequest(BaseModel):
    url: HttpUrl
    processing_options: ProcessingOptions
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com/stream.m3u8",
                "processing_options": {...}
            }
        }
    )

# Response schema  
class StreamResponse(BaseModel):
    id: UUID
    status: StreamStatus
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
```

## Celery Integration

### Queuing Tasks (`celery_client.py`)
```python
# Queue async tasks
from api.celery_client import celery_app

task = celery_app.send_task(
    "worker.tasks.stream_processing.process_stream",
    args=[str(stream.id)],
    queue="stream_processing"
)
```

### Task Status Tracking
```python
# Update stream with task ID
stream.celery_task_id = task.id
await stream_repository.update(stream)
```

## Testing Guidelines

### Test Structure (`tests/`)
- `unit/` - Unit tests for services and utilities
  - `test_jwt_service.py` - JWT service tests
  - `test_password_service.py` - Password service tests
  - `test_rate_limiter.py` - Rate limiting tests
- `integration/` - API endpoint integration tests
- `factories.py` - Test data factories

### Common Test Patterns
```python
# Test authentication service
async def test_user_authentication():
    user_service = UserService(repo, password_service, jwt_service)
    user, access_token, refresh_token = await user_service.authenticate(
        email="test@example.com",
        password="secure_password"
    )
    assert user is not None
    assert access_token is not None

# Test API endpoint
def test_create_stream(client, api_key):
    response = client.post(
        "/api/v1/streams",
        headers={"X-API-Key": api_key.key},
        json={"url": "https://example.com/stream.m3u8"}
    )
    assert response.status_code == 201
```

## API Endpoint Patterns

### RESTful Design
```python
# Standard CRUD endpoints
@router.get("/")  # List with pagination
@router.post("/")  # Create new resource
@router.get("/{id}")  # Get single resource
@router.put("/{id}")  # Update resource
@router.delete("/{id}")  # Delete resource

# Custom actions
@router.post("/{id}/process")  # Trigger processing
```

### Error Handling
```python
# Use HTTPException for API errors
from fastapi import HTTPException

if not stream:
    raise HTTPException(
        status_code=404,
        detail="Stream not found"
    )

# Custom error responses
@router.exception_handler(OrganizationQuotaExceeded)
async def quota_exceeded_handler(request, exc):
    return JSONResponse(
        status_code=402,
        content={"detail": "Organization quota exceeded"}
    )
```

### Pagination
```python
# Standard pagination parameters
@router.get("/", response_model=PaginatedResponse[HighlightResponse])
async def list_highlights(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    # Implementation
```

## Common Tasks

### Adding a New Endpoint
1. Create route module in `src/api/routes/`
2. Define request/response schemas in `src/api/schemas/`
3. Add router to main app in `main.py`
4. Write integration tests
5. Update OpenAPI documentation
6. **IMPORTANT: Update the Postman collection** (`TLDR_Highlight_API.postman_collection.json`) with the new endpoint

### Adding a New Service
1. Create service in `src/api/services/`
2. Add dependency function in `dependencies.py`
3. Use dependency injection in routes
4. Write unit tests for the service
5. Update imports in existing code

### Adding Authentication to Endpoint
```python
@router.post("/protected")
async def protected_endpoint(
    # For API key auth
    api_key: APIKey = Depends(get_api_key_from_header),
    # For JWT auth
    current_user: User = Depends(get_current_user),
    # For scope checking
    _: None = Depends(require_scope("streams:write"))
):
    pass
```

### Webhook Integration
```python
# Queue webhook delivery
celery_app.send_task(
    "worker.tasks.webhook_delivery.deliver_webhook",
    args=[webhook_url, event_type, payload],
    queue="webhooks"
)
```

## Performance Considerations

### Async Best Practices
```python
# Use async dependencies
async def get_stream_service(
    session: AsyncSession = Depends(get_session)
) -> StreamService:
    return StreamService(session)

# Avoid blocking operations
# Bad: time.sleep(1)
# Good: await asyncio.sleep(1)
```

### Database Query Optimization
```python
# Use joinedload for relationships
query = select(Stream).options(
    joinedload(Stream.organization)
).where(Stream.id == stream_id)
```

## Middleware Configuration

### Request Logging
```python
# Automatic request ID and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    # Set context variables
    request_id_context_var.set(request_id)
```

### CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)
```

### Rate Limiting
```python
# Using slowapi for rate limiting
from api.middleware.rate_limit import RateLimiter

rate_limiter = RateLimiter(redis_url=settings.redis_url)
```

## Postman Collection Maintenance

**CRITICAL**: The Postman collection (`TLDR_Highlight_API.postman_collection.json`) must be kept in sync with API changes:

- **When adding endpoints**: Add corresponding requests to the collection
- **When modifying endpoints**: Update request URLs, methods, headers, and body examples
- **When deleting endpoints**: Remove from the collection
- **When changing authentication**: Update collection variables and auth settings

The Postman collection serves as both documentation and testing tool for API consumers.

## Common Pitfalls to Avoid

1. **Don't bypass authentication** - Always use dependency injection
2. **Don't forget multi-tenancy** - Filter by organization_id
3. **Don't block the event loop** - Use async operations
4. **Don't expose internal errors** - Use proper error handling
5. **Don't skip input validation** - Use Pydantic schemas
6. **Don't mix concerns** - Keep auth logic in services, not routes
7. **Don't forget to update Postman collection** - Keep it in sync with API changes

## Integration Points

### With Shared Package
- Import domain models (`User`, `Organization`, `Stream`, etc.)
- Use repository implementations
- Access shared infrastructure (database, config)
- Use shared security services (`api_key_service`, `url_signer`)

### With Worker Package  
- Queue Celery tasks for processing
- Track task status
- Handle async results via webhooks

## Debugging Tips

```bash
# Run with auto-reload
uv run uvicorn api.main:app --reload --log-level debug

# Test specific endpoint
uv run pytest tests/integration/test_streams.py::test_create_stream -v

# Check OpenAPI schema
curl http://localhost:8000/openapi.json | jq

# Test authentication
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure_password"}'
```

## Import Guidelines

Following the refactoring, use these import patterns:
```python
# Auth services (now in API package)
from api.services.auth.jwt_service import JWTService
from api.services.auth.password_service import PasswordService

# Business services (now in API package)
from api.services.user_service import UserService
from api.services.organization_service import OrganizationService

# Shared imports (unchanged)
from shared.domain.models.user import User
from shared.infrastructure.storage.repositories import UserRepository
from shared.infrastructure.config.config import Settings
```