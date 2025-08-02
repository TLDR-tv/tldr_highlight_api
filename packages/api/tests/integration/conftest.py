"""Additional fixtures for integration tests."""

import pytest
from unittest.mock import patch
import sys


@pytest.fixture(scope="session", autouse=True)
def setup_test_celery(redis_container):
    """Ensure Celery uses test Redis URL."""
    # Get Redis connection URL
    redis_host = redis_container.get_container_host_ip()
    redis_port = redis_container.get_exposed_port(6379)
    redis_url = f"redis://{redis_host}:{redis_port}/0"
    
    # Patch get_settings before importing anything that uses it
    from shared.infrastructure.config.config import Settings
    
    def get_test_settings():
        settings = Settings(
            environment="test",
            database_url="postgresql://test:test@localhost:5432/test",  # Will be overridden
            redis_url=redis_url,
            jwt_secret_key="test_secret_key_for_testing_only",
            jwt_expiry_seconds=3600,
            cors_origins=["http://testserver"],
            gemini_api_key="test_gemini_key",
            aws_access_key_id="test_aws_key",
            aws_secret_access_key="test_aws_secret",
            s3_region="us-east-1",
            s3_bucket_name="test-bucket",
        )
        return settings
    
    # Remove the celery_client module if already imported
    if "api.celery_client" in sys.modules:
        del sys.modules["api.celery_client"]
    
    # Patch get_settings
    with patch("shared.infrastructure.config.config.get_settings", get_test_settings):
        # Now import celery_client which will use our test settings
        import api.celery_client
        
        # Also patch in routes module
        with patch("api.routes.streams.celery_app", api.celery_client.celery_app):
            yield