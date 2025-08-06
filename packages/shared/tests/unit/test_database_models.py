"""Unit tests for database models."""

import pytest
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import inspect

from shared.infrastructure.database.models import (
    OrganizationModel,
    UserModel,
    StreamModel,
    HighlightModel,
    APIKeyModel,
    WakeWordModel,
    Base,
)
from shared.domain.models.user import UserRole
from shared.domain.models.stream import StreamStatus, StreamSource


class TestDatabaseModels:
    """Test database model definitions."""

    def test_base_metadata_exists(self):
        """Test that Base metadata is properly configured."""
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')

    def test_organization_model_table_name(self):
        """Test OrganizationModel table configuration."""
        assert OrganizationModel.__tablename__ == "organizations"

    def test_organization_model_columns(self):
        """Test OrganizationModel has expected columns."""
        inspector = inspect(OrganizationModel)
        column_names = [col.name for col in inspector.columns]
        
        expected_columns = [
            'id', 'name', 'slug', 'is_active',
            'total_streams_processed', 'total_highlights_generated', 'total_processing_seconds',
            'webhook_url', 'webhook_secret', 'rubric_name',
            'created_at', 'updated_at'
        ]
        
        for col in expected_columns:
            assert col in column_names

    def test_user_model_table_name(self):
        """Test UserModel table configuration."""
        assert UserModel.__tablename__ == "users"

    def test_user_model_columns(self):
        """Test UserModel has expected columns."""
        inspector = inspect(UserModel)
        column_names = [col.name for col in inspector.columns]
        
        expected_columns = [
            'id', 'organization_id', 'email', 'name', 'hashed_password',
            'role', 'is_active', 'last_login_at', 'created_at', 'updated_at'
        ]
        
        for col in expected_columns:
            assert col in column_names

    def test_stream_model_table_name(self):
        """Test StreamModel table configuration."""
        assert StreamModel.__tablename__ == "streams"

    def test_stream_model_columns(self):
        """Test StreamModel has expected columns."""
        inspector = inspect(StreamModel)
        column_names = [col.name for col in inspector.columns]
        
        expected_columns = [
            'id', 'organization_id', 'url', 'name', 'type', 'status',
            'celery_task_id', 'metadata', 'stream_fingerprint', 'source_type',
            'started_at', 'completed_at', 'duration_seconds', 
            'segments_processed', 'highlights_generated', 'stats',
            'error_message', 'retry_count', 'created_at', 'updated_at'
        ]
        
        for col in expected_columns:
            assert col in column_names

    def test_highlight_model_table_name(self):
        """Test HighlightModel table configuration."""
        assert HighlightModel.__tablename__ == "highlights"

    def test_highlight_model_columns(self):
        """Test HighlightModel has expected columns."""
        inspector = inspect(HighlightModel)
        column_names = [col.name for col in inspector.columns]
        
        expected_columns = [
            'id', 'stream_id', 'organization_id', 'start_time', 'end_time', 'duration',
            'title', 'description', 'tags', 'dimension_scores', 'overall_score',
            'clip_path', 'thumbnail_path', 'transcript', 'wake_word_triggered', 
            'wake_word_detected', 'created_at', 'updated_at'
        ]
        
        for col in expected_columns:
            assert col in column_names

    def test_api_key_model_table_name(self):
        """Test APIKeyModel table configuration."""
        assert APIKeyModel.__tablename__ == "api_keys"

    def test_api_key_model_columns(self):
        """Test APIKeyModel has expected columns."""
        inspector = inspect(APIKeyModel)
        column_names = [col.name for col in inspector.columns]
        
        expected_columns = [
            'id', 'organization_id', 'name', 'key_hash', 'prefix', 'scopes',
            'description', 'is_active', 'expires_at', 'revoked_at', 'usage_count',
            'last_used_at', 'rate_limit', 'created_at', 'updated_at', 'created_by_user_id'
        ]
        
        for col in expected_columns:
            assert col in column_names

    def test_wake_word_model_table_name(self):
        """Test WakeWordModel table configuration."""
        assert WakeWordModel.__tablename__ == "wake_words"

    def test_wake_word_model_columns(self):
        """Test WakeWordModel has expected columns."""
        inspector = inspect(WakeWordModel)
        column_names = [col.name for col in inspector.columns]
        
        expected_columns = [
            'id', 'organization_id', 'phrase', 'is_active', 'case_sensitive',
            'exact_match', 'cooldown_seconds', 'max_edit_distance', 'similarity_threshold',
            'pre_roll_seconds', 'post_roll_seconds', 'trigger_count', 'last_triggered_at',
            'created_at', 'updated_at'
        ]
        
        for col in expected_columns:
            assert col in column_names

    def test_model_relationships(self):
        """Test that relationships are defined."""
        # Organization relationships
        org_relationships = OrganizationModel.__mapper__.relationships.keys()
        assert 'users' in org_relationships
        assert 'streams' in org_relationships
        assert 'api_keys' in org_relationships
        assert 'wake_word_configs' in org_relationships

        # User relationships
        user_relationships = UserModel.__mapper__.relationships.keys()
        assert 'organization' in user_relationships

        # Stream relationships
        stream_relationships = StreamModel.__mapper__.relationships.keys()
        assert 'organization' in stream_relationships
        assert 'highlights' in stream_relationships

    def test_model_defaults(self):
        """Test model default values."""
        # Test that models have proper default factories
        org = OrganizationModel()
        assert org.is_active is True
        assert org.total_streams_processed == 0
        assert org.total_highlights_generated == 0
        assert org.total_processing_seconds == 0.0

        user = UserModel()
        assert user.role == UserRole.MEMBER
        assert user.is_active is True

        stream = StreamModel()
        assert stream.status == StreamStatus.PENDING
        assert stream.segments_processed == 0
        assert stream.highlights_generated == 0
        assert stream.retry_count == 0

        highlight = HighlightModel()
        assert highlight.overall_score == 0.0
        assert highlight.wake_word_triggered is False

        api_key = APIKeyModel()
        assert api_key.is_active is True
        assert api_key.usage_count == 0
        assert api_key.rate_limit == 1000

        wake_word = WakeWordModel()
        assert wake_word.is_active is True
        assert wake_word.case_sensitive is False
        assert wake_word.exact_match is True
        assert wake_word.cooldown_seconds == 30

    def test_model_foreign_keys(self):
        """Test foreign key relationships."""
        # Check that foreign key columns exist
        user_inspector = inspect(UserModel)
        user_columns = {col.name: col for col in user_inspector.columns}
        assert 'organization_id' in user_columns
        # Verify it's a foreign key
        assert len(list(user_columns['organization_id'].foreign_keys)) > 0

        stream_inspector = inspect(StreamModel)
        stream_columns = {col.name: col for col in stream_inspector.columns}
        assert 'organization_id' in stream_columns
        assert len(list(stream_columns['organization_id'].foreign_keys)) > 0

        highlight_inspector = inspect(HighlightModel)
        highlight_columns = {col.name: col for col in highlight_inspector.columns}
        assert 'stream_id' in highlight_columns
        assert 'organization_id' in highlight_columns
        assert len(list(highlight_columns['stream_id'].foreign_keys)) > 0