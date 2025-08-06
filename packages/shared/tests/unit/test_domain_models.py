"""Unit tests for domain models."""

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from shared.domain.models.user import User, UserRole
from shared.domain.models.organization import Organization
from shared.domain.models.api_key import APIKey, APIScopes
from shared.domain.models.highlight import Highlight, DimensionScore
from shared.domain.models.wake_word import WakeWord
from shared.domain.models.stream import Stream, StreamStatus, StreamType


class TestUser:
    """Test User domain model."""

    def test_user_creation(self):
        """Test basic user creation."""
        user_id = uuid4()
        org_id = uuid4()
        
        user = User(
            id=user_id,
            organization_id=org_id,
            email="test@example.com",
            name="Test User",
            hashed_password="hashed123"
        )
        
        assert user.id == user_id
        assert user.organization_id == org_id
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.role == UserRole.MEMBER
        assert user.is_active is True

    def test_user_is_admin_property(self):
        """Test is_admin property."""
        user = User(email="admin@example.com", role=UserRole.ADMIN)
        assert user.is_admin is True
        
        user = User(email="member@example.com", role=UserRole.MEMBER)
        assert user.is_admin is False

    def test_user_can_manage_organization(self):
        """Test can_manage_organization property."""
        admin = User(email="admin@example.com", role=UserRole.ADMIN)
        assert admin.can_manage_organization is True
        
        member = User(email="member@example.com", role=UserRole.MEMBER)
        assert member.can_manage_organization is False

    def test_user_can_view_all_highlights(self):
        """Test can_view_all_highlights property."""
        active_user = User(email="active@example.com", is_active=True)
        assert active_user.can_view_all_highlights is True
        
        inactive_user = User(email="inactive@example.com", is_active=False)
        assert inactive_user.can_view_all_highlights is False

    def test_user_record_login(self):
        """Test record_login method."""
        user = User(email="test@example.com")
        assert user.last_login_at is None
        
        user.record_login()
        assert user.last_login_at is not None
        assert isinstance(user.last_login_at, datetime)

    def test_user_string_representation(self):
        """Test string representation."""
        user = User(email="test@example.com", name="John Doe")
        assert str(user) == "John Doe (test@example.com)"


class TestOrganization:
    """Test Organization domain model."""

    def test_organization_creation(self):
        """Test basic organization creation."""
        org_id = uuid4()
        
        org = Organization(
            id=org_id,
            name="Test Org",
            slug="test-org"
        )
        
        assert org.id == org_id
        assert org.name == "Test Org"
        assert org.slug == "test-org"
        assert org.is_active is True

    def test_organization_auto_slug_generation(self):
        """Test automatic slug generation."""
        org = Organization(name="Test Organization With Spaces")
        
        # Should generate slug automatically in __post_init__
        assert org.slug == "test-organization-with-spaces"

    def test_organization_has_webhook_configured(self):
        """Test has_webhook_configured property."""
        org = Organization(webhook_url="https://example.com/webhook")
        assert org.has_webhook_configured is True
        
        org = Organization(webhook_url="")
        assert org.has_webhook_configured is False
        
        org = Organization()
        assert org.has_webhook_configured is False

    def test_organization_has_custom_wake_words(self):
        """Test has_custom_wake_words property."""
        org = Organization()
        org.wake_words = {"hello", "hey"}
        assert org.has_custom_wake_words is True
        
        org.wake_words = set()
        assert org.has_custom_wake_words is False

    def test_organization_add_wake_word(self):
        """Test adding wake words."""
        org = Organization()
        
        org.add_wake_word("Hello World")
        assert "hello world" in org.wake_words
        
        org.add_wake_word("  ANOTHER WORD  ")
        assert "another word" in org.wake_words
        assert len(org.wake_words) == 2

    def test_organization_remove_wake_word(self):
        """Test removing wake words."""
        org = Organization()
        org.wake_words = {"hello", "world"}
        
        org.remove_wake_word("Hello")  # Case insensitive
        assert "hello" not in org.wake_words
        assert "world" in org.wake_words

    def test_organization_record_usage(self):
        """Test recording usage metrics."""
        org = Organization()
        
        org.record_usage(streams=2, highlights=5, seconds=120.5)
        assert org.total_streams_processed == 2
        assert org.total_highlights_generated == 5
        assert org.total_processing_seconds == 120.5
        
        org.record_usage(streams=1, highlights=3, seconds=60.0)
        assert org.total_streams_processed == 3
        assert org.total_highlights_generated == 8
        assert org.total_processing_seconds == 180.5


class TestAPIKey:
    """Test APIKey domain model."""

    def test_api_key_creation(self):
        """Test basic API key creation."""
        key_id = uuid4()
        org_id = uuid4()
        
        api_key = APIKey(
            id=key_id,
            organization_id=org_id,
            name="Test API Key",
            key_hash="hashed_key"
        )
        
        assert api_key.id == key_id
        assert api_key.organization_id == org_id
        assert api_key.name == "Test API Key"
        assert api_key.key_hash == "hashed_key"
        assert api_key.is_active is True

    def test_api_key_scopes(self):
        """Test API key scopes."""
        api_key = APIKey(
            name="Test Key",
            scopes={APIScopes.STREAMS_READ, APIScopes.HIGHLIGHTS_WRITE}
        )
        
        assert APIScopes.STREAMS_READ in api_key.scopes
        assert APIScopes.HIGHLIGHTS_WRITE in api_key.scopes
        assert APIScopes.STREAMS_WRITE not in api_key.scopes

    def test_api_scopes_class(self):
        """Test APIScopes class values."""
        assert APIScopes.STREAMS_READ == "streams:read"
        assert APIScopes.STREAMS_WRITE == "streams:write"
        assert APIScopes.HIGHLIGHTS_READ == "highlights:read"
        assert APIScopes.HIGHLIGHTS_WRITE == "highlights:write"
        
    def test_api_scopes_default(self):
        """Test default scopes."""
        defaults = APIScopes.default_scopes()
        assert APIScopes.STREAMS_READ in defaults
        assert APIScopes.STREAMS_WRITE in defaults
        assert APIScopes.HIGHLIGHTS_READ in defaults
        assert APIScopes.WEBHOOKS_READ in defaults

    def test_api_key_generate_key(self):
        """Test API key generation."""
        full_key, key_hash = APIKey.generate_key()
        
        # Check that both parts are returned
        assert isinstance(full_key, str)
        assert isinstance(key_hash, str)
        
        # Check key format
        assert full_key.startswith("tldr_")
        assert len(full_key) > 40  # prefix + underscore + 32 char key
        
        # Check hash format
        assert key_hash.startswith("hashed_")
        
        # Test uniqueness
        full_key2, key_hash2 = APIKey.generate_key()
        assert full_key != full_key2
        assert key_hash != key_hash2

    def test_api_key_has_scope_wildcard(self):
        """Test API key has_scope method with wildcard."""
        # Test with specific scopes
        api_key = APIKey(
            name="Test Key",
            scopes={APIScopes.STREAMS_READ}
        )
        
        assert api_key.has_scope(APIScopes.STREAMS_READ) is True
        assert api_key.has_scope(APIScopes.STREAMS_WRITE) is False
        
        # Test with wildcard scope
        api_key_wildcard = APIKey(
            name="Admin Key",
            scopes={"*"}
        )
        
        assert api_key_wildcard.has_scope(APIScopes.STREAMS_READ) is True
        assert api_key_wildcard.has_scope(APIScopes.STREAMS_WRITE) is True
        assert api_key_wildcard.has_scope("any:scope") is True

    def test_api_key_set_expiration(self):
        """Test setting API key expiration."""
        api_key = APIKey(name="Test Key")
        
        # Initially no expiration
        assert api_key.expires_at is None
        assert api_key.is_expired is False
        
        # Set expiration
        api_key.set_expiration(30)
        
        # Should have expiration set
        assert api_key.expires_at is not None
        assert api_key.is_expired is False

    def test_api_key_properties(self):
        """Test API key properties."""
        api_key = APIKey(name="Test Key")
        
        # Test is_valid (active, not expired, not revoked)
        assert api_key.is_valid is True
        
        # Test after revocation
        api_key.revoke()
        assert api_key.is_active is False
        assert api_key.revoked_at is not None
        assert api_key.is_valid is False

    def test_api_key_record_usage(self):
        """Test recording API key usage."""
        api_key = APIKey(name="Test Key")
        
        assert api_key.usage_count == 0
        assert api_key.last_used_at is None
        
        api_key.record_usage()
        
        assert api_key.usage_count == 1
        assert api_key.last_used_at is not None
        assert isinstance(api_key.last_used_at, datetime)


class TestHighlight:
    """Test Highlight domain model."""

    def test_highlight_creation(self):
        """Test basic highlight creation."""
        highlight_id = uuid4()
        stream_id = uuid4()
        org_id = uuid4()
        
        highlight = Highlight(
            id=highlight_id,
            stream_id=stream_id,
            organization_id=org_id,
            start_time=10.0,
            end_time=25.0,
            title="Test Highlight"
        )
        
        assert highlight.id == highlight_id
        assert highlight.stream_id == stream_id
        assert highlight.organization_id == org_id
        assert highlight.start_time == 10.0
        assert highlight.end_time == 25.0
        assert highlight.duration == 15.0  # Calculated in __post_init__

    def test_highlight_dimension_scores(self):
        """Test dimension scores functionality."""
        highlight = Highlight(title="Test")
        
        highlight.add_dimension_score("action", 0.8, 0.9)
        highlight.add_dimension_score("emotion", 0.6, 0.7)
        
        assert len(highlight.dimension_scores) == 2
        assert highlight.dimension_scores[0].name == "action"
        assert highlight.dimension_scores[0].score == 0.8
        assert highlight.dimension_scores[0].confidence == 0.9

    def test_highlight_is_high_confidence(self):
        """Test is_high_confidence property."""
        highlight = Highlight(title="Test")
        
        # No scores - should be False
        assert highlight.is_high_confidence is False
        
        # High confidence scores
        highlight.add_dimension_score("action", 0.8, 0.9)
        highlight.add_dimension_score("emotion", 0.7, 0.8)
        assert highlight.is_high_confidence is True
        
        # Add low confidence score
        highlight.add_dimension_score("other", 0.5, 0.3)
        # Average confidence should be lower now
        avg_conf = (0.9 + 0.8 + 0.3) / 3
        assert avg_conf < 0.7
        assert highlight.is_high_confidence is False

    def test_highlight_top_dimensions(self):
        """Test top_dimensions property."""
        highlight = Highlight(title="Test")
        
        highlight.add_dimension_score("low", 0.3, 0.5)
        highlight.add_dimension_score("high", 0.9, 0.8)
        highlight.add_dimension_score("medium", 0.6, 0.7)
        
        top_dims = highlight.top_dimensions
        assert len(top_dims) == 3
        assert top_dims[0].name == "high"  # Highest score
        assert top_dims[1].name == "medium"
        assert top_dims[2].name == "low"  # Lowest score

    def test_dimension_score_validation(self):
        """Test DimensionScore validation."""
        # Valid scores
        dim_score = DimensionScore(name="test", score=0.5, confidence=0.8)
        assert dim_score.score == 0.5
        assert dim_score.confidence == 0.8
        
        # Invalid score
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            DimensionScore(name="test", score=1.5, confidence=0.8)
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            DimensionScore(name="test", score=0.5, confidence=-0.1)

    def test_highlight_calculate_overall_score_edge_cases(self):
        """Test _recalculate_overall_score with edge cases."""
        # Test with no dimension scores (lines 130-131)
        highlight = Highlight(title="Test")
        assert highlight.dimension_scores == []
        
        # Trigger calculation (this should execute lines 130-131)
        highlight._recalculate_overall_score()
        assert highlight.overall_score == 0.0
        
        # Test with all zero confidence scores (line 141)
        highlight2 = Highlight(title="Test")
        highlight2.add_dimension_score("action", 0.8, 0.0)  # Zero confidence
        highlight2.add_dimension_score("emotion", 0.6, 0.0)  # Zero confidence
        
        # This should trigger the else clause on line 141 (simple average)
        highlight2._recalculate_overall_score()
        expected_average = (0.8 + 0.6) / 2
        assert highlight2.overall_score == expected_average


class TestWakeWord:
    """Test WakeWord domain model."""

    def test_wake_word_creation(self):
        """Test basic wake word creation."""
        wake_word = WakeWord(
            organization_id=uuid4(),
            phrase="hey assistant",
            similarity_threshold=0.8
        )
        
        assert wake_word.phrase == "hey assistant"
        assert wake_word.similarity_threshold == 0.8
        assert wake_word.is_active is True
        assert wake_word.case_sensitive is False

    def test_wake_word_phrase_normalization(self):
        """Test phrase normalization."""
        wake_word = WakeWord(
            phrase="  HEY ASSISTANT  ",
            case_sensitive=False
        )
        
        # Should normalize phrase in __post_init__
        assert wake_word.phrase == "hey assistant"

    def test_wake_word_case_sensitive_normalization(self):
        """Test case sensitive normalization."""
        wake_word = WakeWord(
            phrase="Hey Assistant",
            case_sensitive=True
        )
        
        # Should preserve case when case sensitive
        assert wake_word.phrase == "Hey Assistant"

    def test_wake_word_matching(self):
        """Test wake word matching."""
        wake_word = WakeWord(
            phrase="hello world",
            exact_match=True
        )
        
        # Exact match
        assert wake_word.matches("hello world") is True
        
        # Match with surrounding text
        assert wake_word.matches("I said hello world today") is True
        
        # Partial match should not work with exact_match=True
        assert wake_word.matches("hello") is False
        
        # Different phrase
        assert wake_word.matches("goodbye world") is False

    def test_wake_word_record_trigger(self):
        """Test recording wake word trigger."""
        wake_word = WakeWord(phrase="test")
        assert wake_word.last_triggered_at is None
        assert wake_word.trigger_count == 0
        
        wake_word.record_trigger()
        assert wake_word.last_triggered_at is not None
        assert wake_word.trigger_count == 1


class TestStream:
    """Test Stream domain model."""

    def test_stream_creation(self):
        """Test basic stream creation."""
        stream = Stream(
            organization_id=uuid4(),
            url="https://example.com/stream.m3u8",
            type=StreamType.VOD
        )
        
        assert stream.url == "https://example.com/stream.m3u8"
        assert stream.type == StreamType.VOD
        assert stream.status == StreamStatus.PENDING

    def test_stream_status_transitions(self):
        """Test stream status transitions."""
        stream = Stream(url="test")
        
        stream.start_processing()
        assert stream.status == StreamStatus.PROCESSING
        
        stream.mark_completed()
        assert stream.status == StreamStatus.COMPLETED
        
        stream.mark_failed("Test error")
        assert stream.status == StreamStatus.FAILED
        assert stream.error_message == "Test error"

    def test_stream_properties(self):
        """Test stream properties."""
        stream = Stream(url="test")
        
        assert stream.is_live is False  # Default
        assert stream.is_complete is False
        assert stream.has_failed is False
        
        stream.status = StreamStatus.PROCESSING
        assert stream.is_live is True
        
        stream.status = StreamStatus.COMPLETED
        assert stream.is_complete is True
        
        stream.status = StreamStatus.FAILED
        assert stream.has_failed is True

    def test_stream_counters(self):
        """Test stream counters."""
        stream = Stream(url="test")
        
        assert stream.segments_processed == 0
        assert stream.highlights_generated == 0
        
        stream.increment_segment_count()
        assert stream.segments_processed == 1
        
        stream.increment_highlight_count()
        assert stream.highlights_generated == 1

    def test_stream_processing_time(self):
        """Test stream processing time calculation."""
        from datetime import datetime, timezone, timedelta
        
        stream = Stream(url="test")
        
        # No started_at or completed_at - should return None
        assert stream.processing_time is None
        
        # Set both started_at and completed_at
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=120)  # 2 minutes
        
        stream.started_at = start_time
        stream.completed_at = end_time
        
        # Should calculate processing time
        processing_time = stream.processing_time
        assert processing_time is not None
        assert processing_time == 120.0  # 2 minutes in seconds

    def test_stream_enums(self):
        """Test stream enumeration values."""
        assert StreamType.LIVESTREAM.value == "livestream"
        assert StreamType.VOD.value == "vod"
        assert StreamType.FILE.value == "file"
        
        assert StreamStatus.PENDING.value == "pending"
        assert StreamStatus.QUEUED.value == "queued"
        assert StreamStatus.PROCESSING.value == "processing"
        assert StreamStatus.COMPLETED.value == "completed"
        assert StreamStatus.FAILED.value == "failed"