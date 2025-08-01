"""Configuration settings for the TL;DR Highlight API.

This module provides centralized configuration management using Pydantic Settings,
with logical grouping of related settings for better maintainability.
"""

from typing import Optional, Literal
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr, ConfigDict
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """Core application configuration."""

    name: str = Field(default="TL;DR Highlight API", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


class APIConfig(BaseModel):
    """API-specific configuration."""

    prefix: str = Field(default="/api/v1", description="API route prefix")
    v1_prefix: str = Field(default="/api/v1", description="API v1 route prefix")
    key_header: str = Field(default="X-API-Key", description="API key header name")
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


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/tldr_highlights",
        description="PostgreSQL connection URL",
    )
    echo: bool = Field(default=False, description="Echo SQL statements")
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")


class RedisConfig(BaseModel):
    """Redis configuration."""

    url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    max_connections: int = Field(default=50, description="Redis max connections")
    decode_responses: bool = Field(default=True, description="Decode Redis responses")


class SecurityConfig(BaseModel):
    """Security configuration."""

    jwt_secret_key: SecretStr = Field(
        default="your-secret-key-here", description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(
        default=30, description="JWT expiration in minutes"
    )


class AWSConfig(BaseModel):
    """AWS S3 configuration."""

    access_key_id: str = Field(default="", description="AWS access key ID")
    secret_access_key: SecretStr = Field(
        default="", description="AWS secret access key"
    )
    region: str = Field(default="us-east-1", description="AWS region")
    highlights_bucket: str = Field(
        default="tldr-highlights", description="S3 bucket for highlights"
    )
    thumbnails_bucket: str = Field(
        default="tldr-thumbnails", description="S3 bucket for thumbnails"
    )
    temp_bucket: str = Field(
        default="tldr-temp", description="S3 bucket for temporary files"
    )


class CeleryConfig(BaseModel):
    """Celery configuration."""

    broker_url: str = Field(
        default="redis://localhost:6379/1", description="Celery broker URL"
    )
    result_backend: str = Field(
        default="redis://localhost:6379/2", description="Celery result backend URL"
    )


class ExternalAPIsConfig(BaseModel):
    """External API keys configuration."""

    openai_api_key: SecretStr = Field(default="", description="OpenAI API key")
    google_gemini_api_key: SecretStr = Field(
        default="", description="Google Gemini API key"
    )
    twitch_client_id: str = Field(default="", description="Twitch client ID")
    twitch_client_secret: SecretStr = Field(
        default="", description="Twitch client secret"
    )
    youtube_api_key: SecretStr = Field(default="", description="YouTube API key")


class GeminiConfig(BaseModel):
    """Gemini AI configuration."""

    api_key: str = Field(
        env="GEMINI_API_KEY",
        description="Google Gemini API key (REQUIRED for highlight detection)",
    )
    model: str = Field(
        default="gemini-2.0-flash-exp",
        env="GEMINI_MODEL",
        description="Gemini model to use",
    )
    video_timeout: int = Field(
        default=300,
        env="GEMINI_VIDEO_TIMEOUT",
        description="Gemini video processing timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        env="GEMINI_MAX_RETRIES",
        description="Maximum retries for Gemini API calls",
    )
    enable_refinement: bool = Field(
        default=True,
        env="GEMINI_ENABLE_REFINEMENT",
        description="Enable highlight refinement step",
    )
    cache_ttl: int = Field(
        default=3600,
        env="GEMINI_CACHE_TTL",
        description="Cache TTL for Gemini analysis results in seconds",
    )

    def validate_config(self) -> None:
        """Validate that Gemini is properly configured.

        Raises:
            ValueError: If Gemini API key is not set
        """
        if not self.api_key or self.api_key == "your-gemini-api-key":
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Gemini is now the primary highlight detection method."
            )


class LogfireConfig(BaseModel):
    """Logfire observability configuration."""

    enabled: bool = Field(default=True, description="Enable Logfire observability")
    project_name: str = Field(
        default="tldr-highlight-api", description="Logfire project name"
    )
    api_key: Optional[SecretStr] = Field(
        default=None, description="Logfire API key (optional for local development)"
    )
    environment: Optional[str] = Field(
        default=None, description="Logfire environment (defaults to app environment)"
    )
    service_name: str = Field(
        default="tldr-api", description="Service name for Logfire"
    )
    service_version: Optional[str] = Field(
        default=None, description="Service version (defaults to app version)"
    )
    console_enabled: bool = Field(
        default=True, description="Enable Logfire console output"
    )
    log_level: str = Field(default="INFO", description="Logfire log level")
    capture_headers: bool = Field(
        default=True, description="Capture HTTP headers in Logfire"
    )
    capture_body: bool = Field(
        default=False,
        description="Capture HTTP request/response bodies (be careful with sensitive data)",
    )
    sql_enabled: bool = Field(
        default=True, description="Enable SQL query logging in Logfire"
    )
    redis_enabled: bool = Field(
        default=True, description="Enable Redis command logging in Logfire"
    )
    celery_enabled: bool = Field(
        default=True, description="Enable Celery task logging in Logfire"
    )
    system_metrics_enabled: bool = Field(
        default=True, description="Enable system metrics collection"
    )
    custom_metrics_enabled: bool = Field(
        default=True, description="Enable custom application metrics"
    )


class PerformanceConfig(BaseModel):
    """Performance and runtime configuration."""

    max_workers: int = Field(default=10, description="Max worker threads")
    worker_count: int = Field(default=1, description="Number of Uvicorn workers")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    log_level: str = Field(default="INFO", description="Application log level")


class FeatureFlagsConfig(BaseModel):
    """Feature flags configuration."""

    enable_webhooks: bool = Field(
        default=True, description="Enable webhook functionality"
    )
    enable_batch_processing: bool = Field(
        default=True, description="Enable batch processing"
    )
    enable_real_time_processing: bool = Field(
        default=True, description="Enable real-time stream processing"
    )
    use_gemini_for_video: bool = Field(
        default=True,
        env="USE_GEMINI_FOR_VIDEO",
        description="Use Gemini for video analysis (always True)",
    )
    gemini_fallback_enabled: bool = Field(
        default=False,
        env="GEMINI_FALLBACK_ENABLED",
        description="Enable fallback when Gemini fails (deprecated - Gemini required)",
    )


class Settings(BaseSettings):
    """Application settings with logical grouping."""

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    # Grouped configuration
    app: AppConfig = Field(default_factory=AppConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    external_apis: ExternalAPIsConfig = Field(default_factory=ExternalAPIsConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    logfire: LogfireConfig = Field(default_factory=LogfireConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    features: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)

    # Convenience properties for backward compatibility
    @property
    def database_url(self) -> str:
        """Get database URL for backward compatibility."""
        return self.database.url

    @property
    def redis_url(self) -> str:
        """Get Redis URL for backward compatibility."""
        return self.redis.url

    @property
    def gemini_api_key(self) -> str:
        """Get Gemini API key for backward compatibility."""
        return self.gemini.api_key

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.is_production

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app.is_development

    @property
    def logfire_env(self) -> str:
        """Get Logfire environment, defaulting to app environment."""
        return self.logfire.environment or self.app.environment

    @property
    def logfire_version(self) -> str:
        """Get Logfire service version, defaulting to app version."""
        return self.logfire.service_version or self.app.version

    def validate_gemini_config(self) -> None:
        """Validate that Gemini is properly configured.

        Raises:
            ValueError: If Gemini API key is not set
        """
        self.gemini.validate_config()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()


# Global settings instance
settings = get_settings()
