"""Test data factories."""

from datetime import datetime, UTC
from typing import Optional
from uuid import uuid4

import factory
from factory import fuzzy
from faker import Faker

from src.domain.models.organization import Organization
from src.domain.models.user import User, UserRole
from src.domain.models.api_key import APIKey, APIScopes
from src.domain.models.stream import Stream, StreamStatus, StreamSource
from src.domain.models.highlight import Highlight, DimensionScore
from src.infrastructure.security.password_service import PasswordService

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
    email = factory.LazyAttribute(lambda _: fake.email().lower())
    name = factory.LazyAttribute(lambda _: fake.name())
    hashed_password = factory.LazyAttribute(
        lambda _: password_service.hash_password(
            fake.password(length=12, special_chars=True)
        )
    )
    role = fuzzy.FuzzyChoice([UserRole.MEMBER, UserRole.ADMIN])
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
    name = factory.LazyAttribute(lambda _: f"Test Key - {fake.word()}")
    key_hash = factory.LazyAttribute(lambda _: fake.sha256())
    prefix = factory.LazyAttribute(lambda _: fake.lexify("????????"))
    scopes = factory.LazyFunction(
        lambda: {APIScopes.STREAMS_READ, APIScopes.HIGHLIGHTS_READ, APIScopes.ORG_READ}
    )

    # Usage tracking
    last_used_at = None
    usage_count = 0

    # Lifecycle
    is_active = True
    expires_at = None
    revoked_at = None
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))


def create_test_organization(**kwargs) -> Organization:
    """Create a test organization with defaults."""
    return OrganizationFactory(**kwargs)


def create_test_user(
    organization_id: Optional[uuid4] = None,
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
    organization_id: Optional[uuid4] = None, scopes: Optional[set[str]] = None, **kwargs
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


class StreamFactory(factory.Factory):
    """Factory for creating test streams."""

    class Meta:
        model = Stream

    id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)
    stream_url = factory.Faker("url")
    stream_fingerprint = factory.LazyFunction(lambda: fake.lexify("stream_????_????"))
    source_type = StreamSource.DIRECT_URL
    status = StreamStatus.PENDING

    # Metadata
    title = factory.Faker("sentence", nb_words=4)
    description = factory.Faker("paragraph")
    platform_user_id = factory.Faker("uuid4")

    # Processing info
    started_at = None
    completed_at = None
    duration_seconds = 0.0
    segments_processed = 0
    highlights_generated = 0

    # Timestamps
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))

    # Error tracking
    error_message = None
    retry_count = 0


def create_test_stream(
    organization_id: Optional[uuid4] = None,
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


class HighlightFactory(factory.Factory):
    """Factory for creating test highlights."""

    class Meta:
        model = Highlight

    id = factory.LazyFunction(uuid4)
    stream_id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)

    # Timing
    start_time = factory.Faker("pyfloat", min_value=0, max_value=300)
    end_time = factory.LazyAttribute(
        lambda o: o.start_time + 30.0
    )  # 30 second highlight
    duration = factory.LazyAttribute(lambda o: o.end_time - o.start_time)

    # Content
    title = factory.Faker("sentence", nb_words=5)
    description = factory.Faker("paragraph", nb_sentences=2)
    tags = factory.LazyFunction(lambda: [fake.word() for _ in range(3)])

    # Scoring
    dimension_scores = factory.LazyFunction(list)
    overall_score = factory.Faker("pyfloat", min_value=0.5, max_value=1.0)

    # Media files
    clip_path = factory.LazyFunction(lambda: f"s3://bucket/clips/{fake.uuid4()}.mp4")
    thumbnail_path = factory.LazyFunction(
        lambda: f"s3://bucket/thumbnails/{fake.uuid4()}.jpg"
    )

    # Metadata
    transcript = factory.Faker("paragraph")
    wake_word_triggered = False
    wake_word_detected = None

    # Timestamps
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))


def create_test_highlight(
    stream_id: Optional[uuid4] = None,
    organization_id: Optional[uuid4] = None,
    dimension_scores: Optional[list[DimensionScore]] = None,
    **kwargs,
) -> Highlight:
    """Create a test highlight with dimension scores."""
    # Default dimension scores if not provided
    if dimension_scores is None:
        dimension_scores = [
            DimensionScore(name="action_intensity", score=0.8, confidence=0.9),
            DimensionScore(name="excitement_level", score=0.7, confidence=0.85),
            DimensionScore(name="audience_engagement", score=0.75, confidence=0.88),
        ]

    highlight = HighlightFactory(
        stream_id=stream_id or uuid4(),
        organization_id=organization_id or uuid4(),
        dimension_scores=dimension_scores,
        **kwargs,
    )

    # Only calculate overall score from dimension scores if not explicitly set
    if dimension_scores and "overall_score" not in kwargs:
        total_weight = sum(d.confidence for d in dimension_scores)
        if total_weight > 0:
            highlight.overall_score = (
                sum(d.score * d.confidence for d in dimension_scores) / total_weight
            )

    return highlight
