"""Basic tests for database models.

This module contains basic tests to verify that all models
can be imported and instantiated correctly.
"""

from datetime import datetime, timedelta, timezone
from src.models import (
    APIKey,
    Base,
    Batch,
    BatchStatus,
    Highlight,
    Organization,
    PlanType,
    Stream,
    StreamPlatform,
    StreamStatus,
    UsageRecord,
    UsageRecordType,
    User,
    Webhook,
    WebhookEvent,
)


def test_model_imports():
    """Test that all models can be imported successfully."""
    assert Base is not None
    assert User is not None
    assert APIKey is not None
    assert Organization is not None
    assert Stream is not None
    assert Batch is not None
    assert Highlight is not None
    assert Webhook is not None
    assert UsageRecord is not None


def test_user_model():
    """Test User model instantiation and properties."""
    user = User(
        email="test@company.com",
        password_hash="hashed_password",
        company_name="Test Company",
    )

    assert user.email == "test@company.com"
    assert user.company_name == "Test Company"
    assert "Test Company" in repr(user)


def test_api_key_model():
    """Test APIKey model instantiation and methods."""
    api_key = APIKey(
        key="test_key_hash",
        name="Test Key",
        user_id=1,
        scopes=["stream:read", "batch:write"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )

    assert api_key.name == "Test Key"
    assert api_key.has_scope("stream:read")
    assert not api_key.has_scope("admin:all")
    assert not api_key.is_expired()


def test_organization_model():
    """Test Organization model instantiation and methods."""
    org = Organization(
        name="Test Organization", owner_id=1, plan_type=PlanType.PROFESSIONAL
    )

    assert org.plan_type == PlanType.PROFESSIONAL
    limits = org.plan_limits
    assert limits["monthly_streams"] == 1000


def test_stream_model():
    """Test Stream model instantiation and properties."""
    stream = Stream(
        source_url="https://twitch.tv/test",
        platform=StreamPlatform.TWITCH,
        status=StreamStatus.PENDING,
        user_id=1,
        options={"quality": "720p"},
    )

    assert stream.platform == StreamPlatform.TWITCH
    assert stream.is_active
    assert stream.processing_duration is None


def test_batch_model():
    """Test Batch model instantiation and properties."""
    batch = Batch(
        status=BatchStatus.PENDING,
        user_id=1,
        video_count=10,
        options={"quality": "1080p"},
    )

    assert batch.video_count == 10
    assert batch.is_active
    assert batch.progress_percentage == 0.0


def test_highlight_model():
    """Test Highlight model instantiation and methods."""
    highlight = Highlight(
        stream_id=1,
        title="Epic Moment",
        video_url="https://s3.example.com/highlight.mp4",
        duration=30.5,
        timestamp=1500,
        confidence_score=0.95,
        tags=["action", "gameplay"],
    )

    assert highlight.source_type == "stream"
    assert highlight.is_high_confidence()
    assert highlight.is_high_confidence(0.9)
    assert not highlight.is_high_confidence(0.99)


def test_webhook_model():
    """Test Webhook model instantiation and methods."""
    webhook = Webhook(
        user_id=1,
        url="https://api.company.com/webhooks",
        events=[WebhookEvent.STREAM_COMPLETED, WebhookEvent.HIGHLIGHT_CREATED],
        secret="webhook_secret",
    )

    assert webhook.is_subscribed_to(WebhookEvent.STREAM_COMPLETED)
    assert not webhook.is_subscribed_to(WebhookEvent.BATCH_STARTED)


def test_usage_record_model():
    """Test UsageRecord model instantiation and methods."""
    usage = UsageRecord(
        user_id=1,
        record_type=UsageRecordType.STREAM_PROCESSED,
        quantity=2,
        extra_metadata={"duration_hours": 4.5},
    )

    assert usage.get_unit() == "streams"
    assert usage.get_billing_amount() == 5.0  # 2 * 2.50
