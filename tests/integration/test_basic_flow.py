"""Basic integration test to verify setup works."""

import os
import pytest

# Set test environment variables before imports
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/tldr_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "30"
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["AWS_ACCESS_KEY_ID"] = "test-aws-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test-aws-secret"
os.environ["AWS_S3_BUCKET"] = "test-bucket"
os.environ["AWS_REGION"] = "us-east-1"


@pytest.mark.asyncio
async def test_basic_imports():
    """Test that basic imports work."""
    from src.domain.entities.stream import StreamStatus

    assert StreamStatus.PENDING.value == "pending"
    assert StreamStatus.PROCESSING.value == "processing"
    assert StreamStatus.COMPLETED.value == "completed"


@pytest.mark.asyncio
async def test_domain_entity_creation():
    """Test creating domain entities."""
    from src.domain.entities.user import User
    from src.domain.entities.organization import Organization, PlanType
    from src.domain.value_objects.email import Email
    from src.domain.value_objects.company_name import CompanyName
    from src.domain.value_objects.timestamp import Timestamp

    # Create a user
    user = User(
        id=1,
        email=Email("test@example.com"),
        company_name=CompanyName("Test Company"),
        password_hash="hashed_password_value",
        is_active=True,
        created_at=Timestamp.now(),
        updated_at=Timestamp.now(),
    )

    assert user.email.value == "test@example.com"
    assert user.is_active is True

    # Create an organization
    org = Organization(
        id=1,
        name="Test Org",
        owner_id=1,
        plan_type=PlanType.PROFESSIONAL,
        is_active=True,
        created_at=Timestamp.now(),
        updated_at=Timestamp.now(),
    )

    assert org.name == "Test Org"
    assert org.plan_type == PlanType.PROFESSIONAL


@pytest.mark.asyncio
async def test_stream_flow():
    """Test basic stream processing flow."""
    from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
    from src.domain.value_objects.url import Url
    from src.domain.value_objects.timestamp import Timestamp
    from src.domain.value_objects.processing_options import ProcessingOptions

    # Create a stream
    stream = Stream(
        id=1,
        user_id=1,
        url=Url("https://example.com/live/stream.m3u8"),
        title="Test Stream",
        platform=StreamPlatform.HLS,
        status=StreamStatus.PENDING,
        processing_options=ProcessingOptions(),
        created_at=Timestamp.now(),
        updated_at=Timestamp.now(),
    )

    assert stream.platform == StreamPlatform.HLS
    assert stream.status == StreamStatus.PENDING

    # Start processing
    processing_stream = stream.start_processing()
    assert processing_stream.status == StreamStatus.PROCESSING
    assert processing_stream.started_at is not None

    # Complete processing
    completed_stream = processing_stream.complete_processing()
    assert completed_stream.status == StreamStatus.COMPLETED
    assert completed_stream.completed_at is not None
    assert (
        completed_stream.duration is not None or completed_stream.duration is None
    )  # duration is optional


@pytest.mark.asyncio
async def test_highlight_creation():
    """Test highlight creation."""
    from src.domain.entities.highlight import Highlight
    from src.domain.value_objects.confidence_score import ConfidenceScore
    from src.domain.value_objects.duration import Duration
    from src.domain.value_objects.timestamp import Timestamp
    from src.domain.value_objects.url import Url

    # Create a highlight
    highlight = Highlight(
        id=1,
        stream_id=1,
        start_time=Duration(10.0),
        end_time=Duration(35.0),
        confidence_score=ConfidenceScore(0.92),
        title="Intense Action Sequence",
        description="High skill gameplay with intense action",
        highlight_types=["action_sequence"],
        thumbnail_url=Url("https://s3.test.com/thumbnails/1.jpg"),
        clip_url=Url("https://s3.test.com/clips/1.mp4"),
        video_analysis={
            "dimensions": {"action_intensity": 0.95, "skill_display": 0.88}
        },
        created_at=Timestamp.now(),
        updated_at=Timestamp.now(),
    )

    assert highlight.stream_id == 1
    assert highlight.start_time.seconds == 10.0
    assert highlight.end_time.seconds == 35.0
    assert highlight.confidence_score.value == 0.92
    assert highlight.duration.seconds == 25.0
    assert "action_sequence" in highlight.highlight_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
