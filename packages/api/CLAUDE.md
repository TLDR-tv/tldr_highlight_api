# CLAUDE.md - API Package

This file provides guidance to Claude Code when working with the API package, which serves as the FastAPI web application for the TLDR Highlight API.

## Package Overview

The `api` package provides the **REST API interface** for the enterprise B2B highlight detection service. It handles HTTP requests, authentication, and coordinates with Celery workers for asynchronous processing.

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

### Route Organization (`src/api/routes/`)
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

### Dependency Functions (`dependencies.py`)
```python
# Always use these for auth
async def get_current_user(...) -> User: ...
async def get_api_key_from_header(...) -> APIKey: ...
async def require_organization_member(...) -> Organization: ...

# Scope checking
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
```python
# Use TestClient for API testing
from fastapi.testclient import TestClient

@pytest.fixture
def client(app):
    return TestClient(app)

# Test with real database using testcontainers
@pytest.fixture
async def db_session():
    async with TestDatabase() as db:
        yield db.session
```

### Common Test Patterns
```python
# Test authentication
def test_requires_api_key(client):
    response = client.get("/api/v1/streams")
    assert response.status_code == 401

# Test with auth
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

## Common Pitfalls to Avoid

1. **Don't bypass authentication** - Always use dependency injection
2. **Don't forget multi-tenancy** - Filter by organization_id
3. **Don't block the event loop** - Use async operations
4. **Don't expose internal errors** - Use proper error handling
5. **Don't skip input validation** - Use Pydantic schemas

## Integration Points

### With Shared Package
- Import domain models and services
- Use repository protocols
- Apply business logic from application layer

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
```