"""Test data factories."""

from datetime import datetime, UTC
from typing import Optional
from uuid import uuid4, UUID

import factory
from factory import fuzzy
from faker import Faker

from shared.domain.models.organization import Organization
from shared.domain.models.user import User, UserRole
from shared.domain.models.api_key import APIKey, APIScopes
from shared.domain.models.stream import Stream, StreamStatus, StreamType, StreamSource
from shared.domain.models.highlight import Highlight, DimensionScore
from shared.infrastructure.security.password_service import PasswordService

fake = Faker()
password_service = PasswordService()


class OrganizationFactory(factory.Factory):
    """Factory for creating test organizations."""

    class Meta:
        model = Organization

    id = factory.LazyFunction(uuid4)
    name = factory.LazyAttribute(lambda _: fake.company())
    slug = factory.LazyAttribute(lambda obj: Organization._generate_slug(obj.name))
    is_active = True
    webhook_url = factory.LazyAttribute(
        lambda _: fake.url(schemes=["https"])
        if fake.boolean(chance_of_getting_true=50)
        else None
    )
    webhook_secret = factory.LazyAttribute(lambda _: fake.sha256())
    wake_words = factory.LazyFunction(set)
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))

    # Usage tracking
    total_streams_processed = 0
    total_highlights_generated = 0
    total_processing_seconds = 0.0


class UserFactory(factory.Factory):
    """Factory for creating test users."""

    class Meta:
        model = User

    id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)
    email = factory.LazyAttribute(lambda _: fake.email())
    hashed_password = factory.LazyAttribute(
        lambda _: password_service.hash_password("test_password123")
    )
    name = factory.LazyAttribute(lambda _: fake.name())
    role = UserRole.MEMBER
    is_active = True
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))
    last_login_at = None


class APIKeyFactory(factory.Factory):
    """Factory for creating test API keys."""

    class Meta:
        model = APIKey

    id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)
    name = factory.LazyAttribute(lambda _: f"Test API Key {fake.word()}")
    key_hash = factory.LazyAttribute(lambda _: fake.sha256())
    prefix = factory.LazyAttribute(lambda _: fake.lexify("tldr_????"))
    scopes = factory.LazyFunction(
        lambda: {
            APIScopes.STREAMS_READ,
            APIScopes.STREAMS_WRITE,
            APIScopes.HIGHLIGHTS_READ,
        }
    )
    is_active = True
    expires_at = None
    last_used_at = None
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))


class StreamFactory(factory.Factory):
    """Factory for creating test streams."""

    class Meta:
        model = Stream

    id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)
    url = factory.LazyAttribute(lambda _: fake.url())
    name = factory.LazyAttribute(lambda _: f"Stream {fake.word()}")
    type = StreamType.LIVESTREAM
    status = StreamStatus.PENDING
    celery_task_id = None
    metadata = factory.LazyFunction(dict)
    stream_fingerprint = factory.LazyAttribute(lambda _: fake.sha256())
    source_type = StreamSource.DIRECT_URL
    started_at = None
    completed_at = None
    duration_seconds = 0.0
    segments_processed = 0
    highlights_generated = 0
    stats = None
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))
    error_message = None
    retry_count = 0


class HighlightFactory(factory.Factory):
    """Factory for creating test highlights."""

    class Meta:
        model = Highlight

    id = factory.LazyFunction(uuid4)
    stream_id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)
    start_time = factory.LazyAttribute(lambda _: fake.random_int(0, 1000))
    end_time = factory.LazyAttribute(lambda obj: obj.start_time + fake.random_int(10, 60))
    duration = factory.LazyAttribute(lambda obj: obj.end_time - obj.start_time)
    title = factory.LazyAttribute(lambda _: fake.sentence(nb_words=4))
    description = factory.LazyAttribute(lambda _: fake.paragraph(nb_sentences=2))
    tags = factory.LazyFunction(list)
    dimension_scores = factory.LazyFunction(list)
    overall_score = factory.LazyAttribute(lambda _: fake.random.uniform(0.7, 1.0))
    clip_path = factory.LazyAttribute(lambda _: f"s3://highlights/{fake.uuid4()}.mp4")
    thumbnail_path = factory.LazyAttribute(lambda _: f"s3://highlights/{fake.uuid4()}.jpg")
    transcript = None
    wake_word_triggered = False
    wake_word_detected = None
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))


# Helper functions for creating test objects

def create_test_organization(**kwargs) -> Organization:
    """Create a test organization with defaults."""
    return OrganizationFactory(**kwargs)


def create_test_user(
    organization_id: Optional[UUID] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    role: UserRole = UserRole.MEMBER,
    **kwargs,
) -> tuple[User, str]:
    """Create a test user with password.

    Returns:
        Tuple of (user, plain_password)
    """
    if not organization_id:
        organization_id = uuid4()

    if not email:
        email = fake.email().lower()

    if not password:
        password = fake.password(length=12, special_chars=True)

    user = UserFactory(
        organization_id=organization_id,
        email=email,
        hashed_password=password_service.hash_password(password),
        role=role,
        **kwargs,
    )

    return user, password


def create_test_api_key(
    organization_id: Optional[UUID] = None, scopes: Optional[set[str]] = None, **kwargs
) -> tuple[APIKey, str]:
    """Create a test API key.

    Returns:
        Tuple of (api_key, raw_key)
    """
    if not organization_id:
        organization_id = uuid4()

    if not scopes:
        scopes = {APIScopes.STREAMS_READ, APIScopes.HIGHLIGHTS_READ, APIScopes.ORG_READ}

    # Generate a properly formatted API key
    prefix = f"tldr_{fake.lexify('????????')}"
    key_part = fake.lexify("?" * 24)
    raw_key = f"{prefix}_{key_part}"

    api_key = APIKeyFactory(
        organization_id=organization_id,
        scopes=scopes,
        key_hash=password_service.hash_password(raw_key),
        prefix=prefix,
        **kwargs,
    )

    return api_key, raw_key


def create_test_stream(
    organization_id: Optional[UUID] = None,
    status: StreamStatus = StreamStatus.COMPLETED,
    **kwargs,
) -> Stream:
    """Create a test stream."""
    stream = StreamFactory(
        organization_id=organization_id or uuid4(), status=status, **kwargs
    )

    # Set appropriate fields based on status
    if status == StreamStatus.COMPLETED:
        stream.started_at = datetime.now(UTC)
        stream.completed_at = datetime.now(UTC)
        stream.duration_seconds = 300.0  # 5 minutes
        stream.segments_processed = 10
        stream.highlights_generated = 3
    elif status == StreamStatus.PROCESSING:
        stream.started_at = datetime.now(UTC)
        stream.segments_processed = 5
    elif status == StreamStatus.FAILED:
        stream.error_message = "Processing failed"
        stream.retry_count = 1

    return stream


def create_test_highlight(
    stream_id: Optional[UUID] = None,
    organization_id: Optional[UUID] = None,
    **kwargs,
) -> Highlight:
    """Create a test highlight."""
    return HighlightFactory(
        stream_id=stream_id or uuid4(),
        organization_id=organization_id or uuid4(),
        **kwargs,
    )