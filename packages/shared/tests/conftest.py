"""Pytest configuration for shared package tests."""

import pytest
from shared.infrastructure.config.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Test settings for shared package tests."""
    return Settings(
        environment="test",
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        redis_url="redis://localhost:6379/15",
        jwt_secret_key="test_secret_key_for_testing_only",
        jwt_expiry_seconds=3600,
        cors_origins=["http://testserver"],
        gemini_api_key="test_gemini_key",
        aws_access_key_id="test_aws_key",
        aws_secret_access_key="test_aws_secret",
        aws_region="us-east-1",
        s3_bucket_name="test-bucket",
    )