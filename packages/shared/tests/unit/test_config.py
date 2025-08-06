"""Unit tests for configuration settings."""

import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError

from shared.infrastructure.config.config import Settings, get_settings


class TestSettings:
    """Test Settings configuration class."""

    def test_settings_initialization(self):
        """Test Settings can be initialized with defaults."""
        settings = Settings()
        
        # Environment defaults
        assert settings.environment == "development"
        assert settings.debug is True
        
        # Database defaults
        assert settings.database_url == "postgresql://postgres:postgres@localhost:5432/tldr_highlights"
        
        # Redis defaults
        assert settings.redis_url == "redis://localhost:6379/0"
        
        # Security defaults
        assert settings.jwt_secret_key == "your-secret-key-change-in-production"
        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_expiry_seconds == 3600
        
        # API defaults
        assert settings.api_prefix == "/api/v1"
        assert settings.cors_origins == ["http://localhost:3000", "http://localhost:8000"]

    def test_aws_s3_defaults(self):
        """Test AWS S3 configuration defaults."""
        settings = Settings()
        
        assert settings.s3_bucket_name == "tldr-highlights"
        assert settings.s3_region == "us-east-1"
        assert settings.aws_access_key_id is None
        assert settings.aws_secret_access_key is None
        assert settings.aws_endpoint_url is None

    def test_gemini_defaults(self):
        """Test Google Gemini configuration defaults."""
        settings = Settings()
        
        assert settings.gemini_api_key == ""
        assert settings.gemini_model == "gemini-2.0-flash"

    def test_stream_processing_defaults(self):
        """Test stream processing configuration defaults."""
        settings = Settings()
        
        assert settings.segment_duration_seconds == 120
        assert settings.segment_buffer_size == 10

    def test_celery_defaults(self):
        """Test Celery configuration defaults."""
        settings = Settings()
        
        assert settings.celery_broker_url == "redis://localhost:6379/1"
        assert settings.celery_result_backend == "redis://localhost:6379/2"

    def test_rate_limiting_defaults(self):
        """Test rate limiting configuration defaults."""
        settings = Settings()
        
        # General rate limiting
        assert settings.rate_limit_enabled is True
        assert settings.rate_limit_storage_url == "redis://localhost:6379/3"
        assert settings.rate_limit_global == "1000/minute"
        assert settings.rate_limit_burst == 20
        
        # Endpoint-specific rate limits
        assert settings.rate_limit_auth == "5/minute"
        assert settings.rate_limit_stream_create == "20/minute"
        assert settings.rate_limit_stream_process == "10/minute"
        assert settings.rate_limit_webhook == "5/hour"
        
        # Tier-specific rate limits
        assert settings.rate_limit_tier_free == "100/minute"
        assert settings.rate_limit_tier_pro == "1000/minute"
        assert settings.rate_limit_tier_enterprise == "10000/minute"

    def test_webhook_defaults(self):
        """Test webhook configuration defaults."""
        settings = Settings()
        
        assert settings.webhook_timeout_seconds == 30
        assert settings.webhook_max_retries == 3

    def test_logging_defaults(self):
        """Test logging configuration defaults."""
        settings = Settings()
        
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"

    def test_email_defaults(self):
        """Test email configuration defaults."""
        settings = Settings()
        
        assert settings.email_enabled is False
        assert settings.email_host == "smtp.gmail.com"
        assert settings.email_port == 587
        assert settings.email_username is None
        assert settings.email_password is None
        assert settings.email_from_address == "noreply@tldr-highlights.com"
        assert settings.email_from_name == "TLDR Highlights"
        assert settings.email_use_tls is True
        assert settings.email_use_ssl is False
        
        # Email URLs
        assert settings.frontend_url == "http://localhost:3000"
        assert settings.password_reset_url_path == "/auth/reset-password"
        assert settings.password_reset_token_expiry_hours == 24

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'DATABASE_URL': 'postgresql://prod:pass@prod-db:5432/prod_db',
            'JWT_SECRET_KEY': 'production-secret-key',
            'GEMINI_API_KEY': 'test-api-key',
        }):
            settings = Settings()
            
            assert settings.environment == "production"
            assert settings.debug is False
            assert settings.database_url == "postgresql://prod:pass@prod-db:5432/prod_db"
            assert settings.jwt_secret_key == "production-secret-key"
            assert settings.gemini_api_key == "test-api-key"

    def test_case_insensitive_env_vars(self):
        """Test that environment variables are case insensitive."""
        with patch.dict(os.environ, {
            'environment': 'staging',  # lowercase
            'DEBUG': 'False',  # uppercase
            'Database_Url': 'postgresql://test:test@test:5432/test',  # mixed case
        }):
            settings = Settings()
            
            assert settings.environment == "staging"
            assert settings.debug is False
            assert settings.database_url == "postgresql://test:test@test:5432/test"

    def test_integer_fields_from_env(self):
        """Test integer fields can be set from environment variables."""
        with patch.dict(os.environ, {
            'JWT_EXPIRY_SECONDS': '7200',
            'SEGMENT_DURATION_SECONDS': '180',
            'WEBHOOK_TIMEOUT_SECONDS': '60',
            'EMAIL_PORT': '25',
        }):
            settings = Settings()
            
            assert settings.jwt_expiry_seconds == 7200
            assert settings.segment_duration_seconds == 180
            assert settings.webhook_timeout_seconds == 60
            assert settings.email_port == 25

    def test_boolean_fields_from_env(self):
        """Test boolean fields can be set from environment variables."""
        with patch.dict(os.environ, {
            'RATE_LIMIT_ENABLED': 'false',
            'EMAIL_ENABLED': 'true',
            'EMAIL_USE_TLS': 'false',
            'EMAIL_USE_SSL': 'true',
        }):
            settings = Settings()
            
            assert settings.rate_limit_enabled is False
            assert settings.email_enabled is True
            assert settings.email_use_tls is False
            assert settings.email_use_ssl is True

    def test_list_fields_from_env(self):
        """Test list fields can be set from environment variables."""
        with patch.dict(os.environ, {
            'CORS_ORIGINS': '["http://example.com", "https://app.example.com"]'
        }):
            settings = Settings()
            
            assert settings.cors_origins == ["http://example.com", "https://app.example.com"]

    def test_model_config_attributes(self):
        """Test model configuration attributes."""
        settings = Settings()
        
        assert hasattr(settings.model_config, 'env_file')
        assert hasattr(settings.model_config, 'env_file_encoding')
        assert hasattr(settings.model_config, 'case_sensitive')
        assert hasattr(settings.model_config, 'extra')
        
        assert settings.model_config['env_file'] == ".env"
        assert settings.model_config['env_file_encoding'] == "utf-8"
        assert settings.model_config['case_sensitive'] is False
        assert settings.model_config['extra'] == "ignore"

    def test_optional_fields(self):
        """Test optional fields can be None."""
        settings = Settings()
        
        # These should be None by default
        assert settings.aws_access_key_id is None
        assert settings.aws_secret_access_key is None
        assert settings.aws_endpoint_url is None
        assert settings.email_username is None
        assert settings.email_password is None

    def test_field_descriptions_exist(self):
        """Test that all fields have descriptions."""
        settings = Settings()
        fields = settings.model_fields
        
        # Check a few key fields have descriptions
        assert 'environment' in fields
        assert 'description' in fields['environment'].json_schema()
        
        assert 'database_url' in fields
        assert 'description' in fields['database_url'].json_schema()
        
        assert 'jwt_secret_key' in fields
        assert 'description' in fields['jwt_secret_key'].json_schema()


class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """Test get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Test get_settings returns the same instance (cached)."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same object due to lru_cache
        assert settings1 is settings2

    def test_get_settings_cache_with_different_env(self):
        """Test caching behavior with environment changes."""
        # Clear the cache first
        get_settings.cache_clear()
        
        # Get settings with default environment
        settings1 = get_settings()
        original_env = settings1.environment
        
        # Settings should be cached
        settings2 = get_settings()
        assert settings1 is settings2
        assert settings2.environment == original_env

    def test_cache_clear_works(self):
        """Test that cache can be cleared."""
        # Get initial settings
        settings1 = get_settings()
        
        # Clear cache
        get_settings.cache_clear()
        
        # Get settings again - should be a new instance
        settings2 = get_settings()
        
        # Should be different instances but same values
        assert settings1 is not settings2
        assert settings1.environment == settings2.environment

    def test_get_settings_with_env_override(self):
        """Test get_settings with environment variable override."""
        # Clear cache to ensure fresh instance
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'test_environment'}):
            settings = get_settings()
            assert settings.environment == 'test_environment'
        
        # Clear cache again to clean up
        get_settings.cache_clear()