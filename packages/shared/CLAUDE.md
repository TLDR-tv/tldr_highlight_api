# CLAUDE.md - Shared Package

This file provides guidance to Claude Code when working with the shared package, which serves as the core foundation of the TLDR Highlight API monorepo.

## Package Overview

The `shared` package is the **core foundation** containing domain models, infrastructure components, and only truly shared utilities used across both API and worker packages. It implements clean architecture principles with clear separation between domain, application, and infrastructure layers.

**Note**: Following recent refactoring, this package now contains only components that are genuinely shared between API and Worker packages. API-specific services (auth, user management) have been moved to the API package, and Worker-specific services (AI processing) have been moved to the Worker package.

## Working with This Package

### Running Commands
```bash
# Always use uv from the package directory
cd packages/shared
uv run pytest  # Run tests
uv run mypy .  # Type checking

# Or from root with --directory
uv --directory packages/shared run pytest
```

### Adding Dependencies
```bash
cd packages/shared
uv add pydantic  # Add production dependency
uv add --dev pytest-mock  # Add dev dependency
```

## Architecture Guidelines

### Domain Layer (`src/shared/domain/`)
When working with domain models and protocols:

1. **Domain Models** are pure dataclasses with business logic:
   ```python
   @dataclass
   class Highlight:
       id: UUID
       stream_id: UUID
       organization_id: UUID  # Always include for multi-tenancy
       dimension_scores: dict[str, float]
       # Business methods here, not getters/setters
   ```

2. **Protocols** define interfaces without implementation:
   ```python
   from typing import Protocol
   
   class HighlightRepository(Protocol):
       async def create(self, highlight: Highlight) -> Highlight: ...
       async def get_by_id(self, id: UUID, organization_id: UUID) -> Highlight | None: ...
   ```

3. **Multi-tenancy** is enforced at the domain level - always include `organization_id` in queries

### Application Layer (`src/shared/application/`)
Shared business logic only:

1. **Shared Services** (currently only `highlight_service.py`):
   ```python
   class HighlightService:
       def __init__(self, repository: HighlightRepository):
           self._repository = repository  # Dependency injection
   ```

2. **Schemas** use Pydantic for validation:
   ```python
   class HighlightCreate(BaseModel):
       model_config = ConfigDict(from_attributes=True)
       stream_id: UUID
       dimension_scores: dict[str, float]
   ```

**Note**: User and organization services have been moved to the API package where they belong.

### Infrastructure Layer (`src/shared/infrastructure/`)
Technical implementations:

1. **Repositories** implement domain protocols:
   ```python
   class SQLAlchemyHighlightRepository:
       async def create(self, highlight: Highlight) -> Highlight:
           # SQLAlchemy implementation
   ```

2. **Database Models** use SQLAlchemy 2.0 patterns:
   ```python
   class HighlightModel(Base):
       __tablename__ = "highlights"
       
       id: Mapped[UUID] = mapped_column(primary_key=True)
       organization_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id"))
       # Use indexes for query performance
       __table_args__ = (
           Index("idx_highlights_org_created", "organization_id", "created_at"),
       )
   ```

3. **Security Services** (only shared ones):
   - `api_key_service.py` - API key management (used by both packages)
   - `url_signer.py` - Secure URL signing (used by both packages)
   
**Note**: JWT and password services have been moved to the API package.

## Key Patterns to Follow

### 1. Clean Architecture
- Domain doesn't depend on infrastructure
- Use dependency injection via protocols
- Keep business logic in domain/application layers

### 2. Pythonic Code
```python
# Use properties, not getters
@property
def is_active(self) -> bool:
    return self.status == StreamStatus.ACTIVE

# Use f-strings
logger.info(f"Processing highlight {highlight_id}")

# Use comprehensions
active_streams = [s for s in streams if s.is_active]
```

### 3. Multi-Tenant Security
- Always filter by organization_id
- Never expose data across organizations
- Include organization_id in all queries and indexes

### 4. Testing
```python
# Write tests first (TDD)
async def test_highlight_service_creates_highlight():
    # Arrange
    repository = Mock(spec=HighlightRepository)
    service = HighlightService(repository)
    
    # Act
    result = await service.create_highlight(...)
    
    # Assert
    repository.create.assert_called_once()
```

## Common Tasks

### Adding a New Domain Model
1. Create model in `src/shared/domain/models/`
2. Define repository protocol in `src/shared/domain/protocols/`
3. Create SQLAlchemy model in `src/shared/infrastructure/database/models.py`
4. Implement repository in `src/shared/infrastructure/storage/repositories/`
5. Write tests for all components

### Adding a New Service
1. Define service interface as protocol if needed
2. Implement in `src/shared/application/services/`
3. Use dependency injection for repositories
4. Write integration tests with real database

### Database Migrations
```bash
# Generate migration
cd packages/shared
uv run alembic revision --autogenerate -m "Add new field"

# Apply migration
uv run alembic upgrade head
```

## Testing Guidelines

### Test Organization
- `tests/unit/` - Pure unit tests with mocks
- `tests/integration/` - Tests with real database
- `tests/domain/` - Domain model tests
- `tests/infrastructure/` - Infrastructure component tests

### Running Tests
```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/integration/test_highlight_repository.py

# With coverage
uv run pytest --cov=src/shared --cov-report=html
```

## Important Conventions

1. **No Versioned Files**: Never create `_v2.py` files - update existing files
2. **Type Everything**: Use type hints throughout
3. **Async First**: All I/O operations should be async
4. **Structured Logging**: Use structlog for all logging
5. **Configuration**: Use pydantic settings with environment variables
6. **Security**: Never log sensitive data (API keys, passwords)

## Common Pitfalls to Avoid

1. **Don't bypass multi-tenancy** - Always include organization_id
2. **Don't use raw SQL** - Use SQLAlchemy ORM for safety
3. **Don't hardcode configurations** - Use settings classes
4. **Don't create circular imports** - Keep clean layer separation
5. **Don't skip tests** - TDD is mandatory

## Integration with Other Packages

The shared package is imported by:
- **API Package**: Uses domain models, repositories, database config, and shared security services
- **Worker Package**: Uses domain models, repositories, database config, and infrastructure components

**What's Actually Shared**:
- Domain models: `user`, `organization`, `stream`, `highlight`, `api_key`, `wake_word`
- Infrastructure: Database configuration, repository implementations
- Security: `api_key_service`, `url_signer` (both used by API and Worker)
- Application: `highlight_service` (shared business logic)

Ensure changes maintain backward compatibility or coordinate updates across packages.