"""Configuration settings for the TL;DR Highlight API.

This module provides centralized configuration management using Pydantic Settings,
including environment variable support and Logfire observability configuration.
"""

from typing import Optional, Literal
from functools import lru_cache

from pydantic import Field, SecretStr, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    # Application settings
    app_name: str = Field(default="TL;DR Highlight API", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # API settings
    api_prefix: str = Field(default="/api/v1", description="API route prefix")
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 route prefix")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_allow_credentials: bool = Field(
        default=True, description="Allow CORS credentials"
    )
    cors_allow_methods: list[str] = Field(
        default=["*"], description="Allowed CORS methods"
    )
    cors_allow_headers: list[str] = Field(
        default=["*"], description="Allowed CORS headers"
    )
    allowed_hosts: list[str] = Field(default=["*"], description="Allowed host headers")

    # Database settings
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/tldr_highlights",
        description="PostgreSQL connection URL",
    )
    database_echo: bool = Field(default=False, description="Echo SQL statements")
    database_pool_size: int = Field(
        default=20, description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=10, description="Max overflow connections"
    )

    # Redis settings
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    redis_max_connections: int = Field(default=50, description="Redis max connections")
    redis_decode_responses: bool = Field(
        default=True, description="Decode Redis responses"
    )

    # Security settings
    jwt_secret_key: SecretStr = Field(
        default="your-secret-key-here", description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(
        default=30, description="JWT expiration in minutes"
    )

    # AWS S3 settings
    s3_access_key_id: str = Field(default="", description="AWS access key ID")
    s3_secret_access_key: SecretStr = Field(
        default="", description="AWS secret access key"
    )
    s3_region: str = Field(default="us-east-1", description="AWS region")
    s3_highlights_bucket: str = Field(
        default="tldr-highlights", description="S3 bucket for highlights"
    )
    s3_thumbnails_bucket: str = Field(
        default="tldr-thumbnails", description="S3 bucket for thumbnails"
    )
    s3_temp_bucket: str = Field(
        default="tldr-temp", description="S3 bucket for temporary files"
    )

    # Celery settings
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", description="Celery result backend URL"
    )

    # External API settings
    openai_api_key: SecretStr = Field(default="", description="OpenAI API key")
    google_gemini_api_key: SecretStr = Field(
        default="", description="Google Gemini API key"
    )
    twitch_client_id: str = Field(default="", description="Twitch client ID")
    twitch_client_secret: SecretStr = Field(
        default="", description="Twitch client secret"
    )
    youtube_api_key: SecretStr = Field(default="", description="YouTube API key")

    # Logfire observability settings
    logfire_enabled: bool = Field(
        default=True, description="Enable Logfire observability"
    )
    logfire_project_name: str = Field(
        default="tldr-highlight-api", description="Logfire project name"
    )
    logfire_api_key: Optional[SecretStr] = Field(
        default=None, description="Logfire API key (optional for local development)"
    )
    logfire_environment: Optional[str] = Field(
        default=None, description="Logfire environment (defaults to app environment)"
    )
    logfire_service_name: str = Field(
        default="tldr-api", description="Service name for Logfire"
    )
    logfire_service_version: Optional[str] = Field(
        default=None, description="Service version (defaults to app version)"
    )
    logfire_console_enabled: bool = Field(
        default=True, description="Enable Logfire console output"
    )
    logfire_log_level: str = Field(default="INFO", description="Logfire log level")
    logfire_capture_headers: bool = Field(
        default=True, description="Capture HTTP headers in Logfire"
    )
    logfire_capture_body: bool = Field(
        default=False,
        description="Capture HTTP request/response bodies (be careful with sensitive data)",
    )
    logfire_sql_enabled: bool = Field(
        default=True, description="Enable SQL query logging in Logfire"
    )
    logfire_redis_enabled: bool = Field(
        default=True, description="Enable Redis command logging in Logfire"
    )
    logfire_celery_enabled: bool = Field(
        default=True, description="Enable Celery task logging in Logfire"
    )
    logfire_system_metrics_enabled: bool = Field(
        default=True, description="Enable system metrics collection"
    )
    logfire_custom_metrics_enabled: bool = Field(
        default=True, description="Enable custom application metrics"
    )

    # Performance settings
    max_workers: int = Field(default=10, description="Max worker threads")
    worker_count: int = Field(default=1, description="Number of Uvicorn workers")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    log_level: str = Field(default="INFO", description="Application log level")

    # Feature flags
    enable_webhooks: bool = Field(
        default=True, description="Enable webhook functionality"
    )
    enable_batch_processing: bool = Field(
        default=True, description="Enable batch processing"
    )
    enable_real_time_processing: bool = Field(
        default=True, description="Enable real-time stream processing"
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def logfire_env(self) -> str:
        """Get Logfire environment, defaulting to app environment."""
        return self.logfire_environment or self.environment

    @property
    def logfire_version(self) -> str:
        """Get Logfire service version, defaulting to app version."""
        return self.logfire_service_version or self.app_version


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()


# Global settings instance
settings = get_settings()
