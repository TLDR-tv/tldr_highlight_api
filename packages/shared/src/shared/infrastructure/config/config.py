"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields to handle old env vars
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )
    debug: bool = Field(default=True, description="Debug mode")

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/tldr_highlights",
        description="PostgreSQL connection URL",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )

    # Security
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key for signing tokens",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_seconds: int = Field(
        default=3600, description="JWT token expiry in seconds"
    )

    # API
    api_prefix: str = Field(default="/api/v1", description="API route prefix")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )

    # AWS S3
    s3_bucket_name: str = Field(default="tldr-highlights", description="S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret key"
    )

    # Google Gemini
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.0-flash", description="Gemini model to use"
    )

    # Stream Processing
    segment_duration_seconds: int = Field(
        default=120, description="Stream segment duration (2 minutes)"
    )
    segment_buffer_size: int = Field(
        default=10, description="Number of segments to keep in buffer"
    )

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", description="Celery result backend URL"
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True, description="Enable rate limiting"
    )
    rate_limit_storage_url: str = Field(
        default="redis://localhost:6379/3", description="Rate limit storage backend URL"
    )
    
    # Global rate limits
    rate_limit_global: str = Field(
        default="1000/minute", description="Global rate limit per IP"
    )
    rate_limit_burst: int = Field(
        default=20, description="Burst size for rate limiting"
    )
    
    # Endpoint-specific rate limits
    rate_limit_auth: str = Field(
        default="5/minute", description="Auth endpoint rate limit"
    )
    rate_limit_stream_create: str = Field(
        default="20/minute", description="Stream creation rate limit"
    )
    rate_limit_stream_process: str = Field(
        default="10/minute", description="Stream processing rate limit"
    )
    rate_limit_webhook: str = Field(
        default="5/hour", description="Webhook configuration rate limit"
    )
    
    # Organization tier limits
    rate_limit_tier_free: str = Field(
        default="100/minute", description="Free tier rate limit"
    )
    rate_limit_tier_pro: str = Field(
        default="1000/minute", description="Pro tier rate limit"
    )
    rate_limit_tier_enterprise: str = Field(
        default="10000/minute", description="Enterprise tier rate limit"
    )

    # Webhooks
    webhook_timeout_seconds: int = Field(
        default=30, description="Webhook delivery timeout"
    )
    webhook_max_retries: int = Field(
        default=3, description="Maximum webhook retry attempts"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
