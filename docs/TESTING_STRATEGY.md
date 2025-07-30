# TL;DR Highlight API - Testing Strategy & Implementation Plan

## Current State Summary

**Overall Test Coverage: 2.8%** 🔴

- **Domain Layer**: 0% (49 modules untested)
- **Application Layer**: 0% (9 modules untested)  
- **Infrastructure Layer**: 4.6% (83 of 87 modules untested)
- **API Layer**: 2.8% (35 of 36 modules untested)

This represents a **CRITICAL** risk for an enterprise B2B API.

## Testing Philosophy

### Testing Pyramid
```
        /\
       /E2E\      (5%) - Critical user journeys
      /------\
     /  Integ  \   (15%) - Service integration
    /------------\
   /     Unit     \  (80%) - Business logic & components
  /----------------\
```

### Testing Principles
1. **Test Behavior, Not Implementation** - Focus on what, not how
2. **Fast Feedback** - Most tests should run in milliseconds
3. **Isolated Tests** - No test should depend on another
4. **Readable Tests** - Tests are documentation
5. **Test-First Development** - Write tests before code

## Priority-Based Implementation Plan

### 🚨 Week 1: Critical Security & Domain Foundation

#### Day 1-2: Setup & Security
```bash
# Create test structure
mkdir -p tests/unit/{domain,application,infrastructure,api}
mkdir -p tests/integration
mkdir -p tests/e2e
mkdir -p tests/factories
mkdir -p tests/fixtures
```

**Files to Create:**
1. `tests/factories/__init__.py` - Factory registry
2. `tests/factories/domain_factories.py` - Entity factories
3. `tests/fixtures/auth.py` - Authentication fixtures
4. `tests/unit/domain/test_security.py` - Security tests

**Critical Security Tests:**
- JWT token generation and validation
- API key authentication
- Permission scoping
- Webhook signature verification
- Rate limiting

#### Day 3-4: Domain Value Objects
Test all value objects (100% coverage target):
- `test_email.py` - Email validation
- `test_url.py` - URL validation  
- `test_timestamp.py` - Timestamp operations
- `test_duration.py` - Duration calculations
- `test_confidence_score.py` - Score boundaries

#### Day 5: Domain Entities - User & Organization
- `test_user.py` - User entity behavior
- `test_organization.py` - Organization management
- `test_api_key.py` - API key lifecycle

### 🎯 Week 2: Core Business Logic

#### Day 6-7: Stream Processing Domain
- `test_stream.py` - Stream entity state machine
- `test_stream_processing_service.py` - Processing logic
- `test_processing_options.py` - Configuration validation

#### Day 8-9: Highlight Detection Domain  
- `test_highlight.py` - Highlight entity
- `test_highlight_detection_service.py` - Detection logic
- `test_highlight_agent_config.py` - Agent configuration

#### Day 10: Webhook Domain
- `test_webhook.py` - Webhook entity
- `test_webhook_delivery_service.py` - Delivery logic
- `test_webhook_event.py` - Event handling

### 💼 Week 3: Application Layer (Use Cases)

#### Day 11-12: Authentication Use Cases
```python
# tests/unit/application/use_cases/test_authentication.py
class TestAuthenticationUseCase:
    async def test_login_success(self):
    async def test_login_invalid_credentials(self):
    async def test_register_new_user(self):
    async def test_register_duplicate_email(self):
```

#### Day 13-14: Stream Processing Use Cases
```python
# tests/unit/application/use_cases/test_stream_processing.py
class TestStreamProcessingUseCase:
    async def test_start_stream_success(self):
    async def test_start_stream_quota_exceeded(self):
    async def test_stop_stream(self):
    async def test_concurrent_stream_limit(self):
```

#### Day 15: Critical Integration Tests
```python
# tests/integration/test_stream_lifecycle.py
@pytest.mark.integration
class TestStreamLifecycle:
    async def test_complete_stream_workflow(self):
        """Test from stream start to highlight generation."""
```

### 🔧 Week 4: Infrastructure Layer

#### Day 16-17: Repository Tests
- Test all repository implementations with database
- Use test transactions for isolation
- Test complex queries and edge cases

#### Day 18-19: External Integration Tests
- S3 storage operations (using moto)
- Gemini AI client (with mocks)
- Streaming platform adapters

#### Day 20: Async Task Tests
- Celery task execution
- Task retry logic
- Error handling in tasks

### 🌐 Week 5: API Layer & E2E

#### Day 21-22: API Endpoint Tests
```python
# tests/unit/api/routers/test_streams_router.py
class TestStreamsRouter:
    async def test_create_stream_success(self):
    async def test_create_stream_unauthorized(self):
    async def test_create_stream_invalid_url(self):
```

#### Day 23-24: End-to-End Tests
```python
# tests/e2e/test_user_journeys.py
class TestUserJourneys:
    async def test_new_user_first_stream(self):
        """Complete journey from registration to first highlight."""
```

#### Day 25: Performance Tests
- Load testing critical endpoints
- Concurrent stream processing
- Database query performance

### 📊 Week 6-7: Advanced Testing

#### Observability Testing
- Verify Logfire integration
- Test metric collection
- Validate distributed tracing

#### Chaos Engineering
- Network failure scenarios
- Service degradation
- Recovery testing

#### Security Testing
- SQL injection attempts
- XSS prevention
- Authentication bypass attempts

## Test Implementation Templates

### 1. Domain Entity Test Template
```python
# tests/unit/domain/entities/test_stream.py
import pytest
from datetime import datetime
from src.domain.entities.stream import Stream, StreamStatus
from src.domain.exceptions import InvalidStateTransition
from tests.factories import StreamFactory

class TestStream:
    """Test Stream entity behavior and business rules."""
    
    def test_create_stream_with_valid_data(self):
        """Test creating a stream with all required fields."""
        stream = StreamFactory.build(
            title="Test Gaming Stream",
            url="https://twitch.tv/testuser"
        )
        
        assert stream.status == StreamStatus.PENDING
        assert stream.title == "Test Gaming Stream"
        assert stream.url.value == "https://twitch.tv/testuser"
        assert stream.created_at is not None
    
    def test_start_processing_changes_status(self):
        """Test transition from pending to processing status."""
        stream = StreamFactory.build(status=StreamStatus.PENDING)
        
        updated_stream = stream.start_processing()
        
        assert updated_stream.status == StreamStatus.PROCESSING
        assert updated_stream.started_at is not None
    
    def test_cannot_start_processing_from_completed(self):
        """Test that invalid state transitions raise exception."""
        stream = StreamFactory.build(status=StreamStatus.COMPLETED)
        
        with pytest.raises(InvalidStateTransition):
            stream.start_processing()
    
    @pytest.mark.parametrize("initial_status,target_status,should_succeed", [
        (StreamStatus.PENDING, StreamStatus.PROCESSING, True),
        (StreamStatus.PROCESSING, StreamStatus.COMPLETED, True),
        (StreamStatus.PROCESSING, StreamStatus.FAILED, True),
        (StreamStatus.COMPLETED, StreamStatus.PROCESSING, False),
        (StreamStatus.FAILED, StreamStatus.PROCESSING, False),
    ])
    def test_state_transitions(self, initial_status, target_status, should_succeed):
        """Test all possible state transitions."""
        # Implementation here
```

### 2. Use Case Test Template
```python
# tests/unit/application/use_cases/test_stream_processing.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.application.use_cases.stream_processing import (
    StreamProcessingUseCase,
    StreamStartRequest,
    ResultStatus
)
from src.domain.exceptions import QuotaExceededError
from tests.factories import UserFactory, OrganizationFactory

class TestStreamProcessingUseCase:
    """Test stream processing use case business logic."""
    
    @pytest.fixture
    def mock_repos(self):
        """Create mock repositories."""
        return {
            'user_repo': Mock(),
            'stream_repo': Mock(),
            'org_repo': Mock(),
            'highlight_repo': Mock(),
        }
    
    @pytest.fixture
    def use_case(self, mock_repos):
        """Create use case instance with mocked dependencies."""
        return StreamProcessingUseCase(**mock_repos)
    
    @pytest.mark.asyncio
    async def test_start_stream_success(self, use_case, mock_repos):
        """Test successful stream start."""
        # Arrange
        user = UserFactory.build(id=1)
        org = OrganizationFactory.build(id=1)
        mock_repos['user_repo'].get = AsyncMock(return_value=user)
        mock_repos['org_repo'].get_by_owner = AsyncMock(return_value=[org])
        mock_repos['stream_repo'].save = AsyncMock(return_value=Mock(id=123))
        
        request = StreamStartRequest(
            user_id=1,
            url="https://twitch.tv/test",
            title="Test Stream"
        )
        
        # Act
        result = await use_case.start_stream(request)
        
        # Assert
        assert result.status == ResultStatus.SUCCESS
        assert result.stream_id == 123
        mock_repos['stream_repo'].save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stream_quota_exceeded(self, use_case, mock_repos):
        """Test stream start when quota is exceeded."""
        # Setup to trigger quota exception
        # Assert QuotaExceededError is handled properly
```

### 3. Integration Test Template
```python
# tests/integration/test_webhook_delivery.py
import pytest
import httpx
from unittest.mock import patch
from tests.factories import WebhookFactory, StreamFactory
from src.infrastructure.async_processing.webhook_dispatcher import WebhookDispatcher

@pytest.mark.integration
class TestWebhookDelivery:
    """Test webhook delivery integration."""
    
    @pytest.mark.asyncio
    async def test_webhook_delivery_success(self, db_session, httpx_mock):
        """Test successful webhook delivery to endpoint."""
        # Arrange
        webhook = WebhookFactory.create(
            url="https://example.com/webhook",
            events=["stream.completed"]
        )
        stream = StreamFactory.create()
        
        httpx_mock.add_response(
            url="https://example.com/webhook",
            status_code=200
        )
        
        dispatcher = WebhookDispatcher()
        
        # Act
        result = await dispatcher.dispatch_webhook(
            webhook_id=webhook.id,
            event="stream.completed",
            data={"stream_id": stream.id}
        )
        
        # Assert
        assert result.success
        assert result.status_code == 200
        assert httpx_mock.call_count == 1
```

### 4. Factory Implementation
```python
# tests/factories/domain_factories.py
import factory
from factory import fuzzy
from datetime import datetime, timezone
from src.domain.entities.user import User
from src.domain.entities.stream import Stream, StreamStatus
from src.domain.value_objects.email import Email
from src.domain.value_objects.url import Url

class UserFactory(factory.Factory):
    """Factory for creating User entities."""
    
    class Meta:
        model = User
    
    id = factory.Sequence(lambda n: n)
    email = factory.LazyAttribute(lambda obj: Email(f"user{obj.id}@example.com"))
    username = factory.Faker('user_name')
    full_name = factory.Faker('name')
    is_active = True
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))

class StreamFactory(factory.Factory):
    """Factory for creating Stream entities."""
    
    class Meta:
        model = Stream
    
    id = factory.Sequence(lambda n: n)
    user_id = factory.Faker('pyint', min_value=1, max_value=100)
    title = factory.Faker('sentence', nb_words=4)
    url = factory.LazyAttribute(
        lambda obj: Url(f"https://twitch.tv/user_{obj.user_id}")
    )
    status = fuzzy.FuzzyChoice(StreamStatus)
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
```

## Testing Best Practices

### 1. Test Naming Convention
```python
def test_should_<expected_behavior>_when_<condition>():
    """Test method names should clearly describe the scenario."""
```

### 2. AAA Pattern
```python
def test_example():
    # Arrange - Set up test data
    user = UserFactory.build()
    
    # Act - Execute the behavior
    result = user.change_password("old", "new")
    
    # Assert - Verify the outcome
    assert result.success
```

### 3. Test Isolation
- Each test should be independent
- Use fixtures for shared setup
- Clean up after tests (database transactions)

### 4. Mock External Dependencies
```python
@patch('src.infrastructure.ai.gemini_client.GeminiClient')
def test_with_mocked_ai(mock_gemini):
    mock_gemini.analyze.return_value = {"score": 0.9}
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev]"
      - name: Run tests
        run: |
          uv run pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Success Metrics

### Coverage Goals by Phase
- **Phase 1 (Week 1)**: 20% overall, 90% security components
- **Phase 2 (Week 2)**: 40% overall, 80% domain layer
- **Phase 3 (Week 3)**: 60% overall, 80% application layer
- **Phase 4 (Week 4)**: 75% overall, 70% infrastructure
- **Phase 5 (Week 5)**: 85% overall, 80% API layer

### Quality Metrics
- All tests pass in < 30 seconds
- No flaky tests
- Clear test failure messages
- Tests serve as documentation

## Next Steps

1. **Create test directory structure** (30 minutes)
2. **Implement domain factories** (2 hours)
3. **Write first security tests** (4 hours)
4. **Set up CI pipeline** (1 hour)
5. **Begin domain entity tests** (ongoing)

The path from 2.8% to 85% coverage requires disciplined effort but is essential for enterprise reliability.