"""
Core configuration module for TL;DR Highlight API.

This module defines all configuration settings for the application using Pydantic Settings.
Configuration values are loaded from environment variables with sensible defaults.
"""

from typing import Optional, List, Literal
from pathlib import Path
from functools import lru_cache

from pydantic import Field, PostgresDsn, RedisDsn, HttpUrl, field_validator, model_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or a .env file.
    The settings are validated using Pydantic's type system.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application Settings
    app_name: str = Field(default="TL;DR Highlight API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=False, description="Debug mode flag")

    # API Settings
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 route prefix")
    api_key_header: str = Field(
        default="X-API-Key", description="Header name for API key authentication"
    )
    api_key_length: int = Field(default=32, description="Length of generated API keys")
    rate_limit_per_minute: int = Field(
        default=60, description="Default rate limit per minute per API key"
    )
    rate_limit_per_hour: int = Field(
        default=1000, description="Default rate limit per hour per API key"
    )
    pagination_default_limit: int = Field(
        default=20, description="Default pagination limit"
    )
    pagination_max_limit: int = Field(
        default=100, description="Maximum pagination limit"
    )

    # Security Settings
    jwt_secret_key: str = Field(
        description="Secret key for JWT token signing (required in production)"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_minutes: int = Field(
        default=60, description="JWT token expiration time in minutes"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH"],
        description="Allowed HTTP methods for CORS",
    )
    cors_allow_headers: List[str] = Field(
        default=["*"], description="Allowed headers for CORS"
    )
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed host headers")

    # Database Settings
    database_url: PostgresDsn = Field(description="PostgreSQL connection URL")
    database_pool_size: int = Field(
        default=20, description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=10, description="Maximum overflow connections for database pool"
    )
    database_pool_timeout: int = Field(
        default=30, description="Database pool timeout in seconds"
    )
    database_echo: bool = Field(
        default=False, description="Echo SQL queries (useful for debugging)"
    )
    database_echo_pool: bool = Field(
        default=False, description="Echo database pool events"
    )

    # Redis Settings
    redis_url: RedisDsn = Field(description="Redis connection URL")
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_max_connections: int = Field(
        default=100, description="Maximum Redis connections"
    )
    redis_decode_responses: bool = Field(
        default=True, description="Decode Redis responses to strings"
    )
    redis_socket_timeout: int = Field(
        default=5, description="Redis socket timeout in seconds"
    )
    redis_socket_connect_timeout: int = Field(
        default=5, description="Redis socket connection timeout in seconds"
    )
    redis_retry_on_timeout: bool = Field(
        default=True, description="Retry Redis operations on timeout"
    )

    # S3/Storage Settings
    s3_endpoint_url: Optional[HttpUrl] = Field(
        default=None, description="S3 endpoint URL (for S3-compatible services)"
    )
    s3_access_key_id: str = Field(description="S3 access key ID")
    s3_secret_access_key: str = Field(description="S3 secret access key")
    s3_region: str = Field(default="us-east-1", description="S3 region name")
    s3_region_name: str = Field(
        default="us-east-1", description="S3 region name (alias for s3_region)"
    )
    s3_highlights_bucket: str = Field(
        description="S3 bucket name for highlight storage"
    )
    s3_thumbnails_bucket: str = Field(
        description="S3 bucket name for thumbnail storage"
    )
    s3_temp_bucket: str = Field(description="S3 bucket name for temporary files")
    s3_presigned_url_expiration: int = Field(
        default=3600, description="S3 presigned URL expiration in seconds"
    )
    storage_max_file_size_mb: int = Field(
        default=500, description="Maximum file size for uploads in MB"
    )

    # Celery/RabbitMQ Settings
    celery_broker_url: str = Field(
        default="amqp://guest:guest@localhost:5672//",
        description="Celery broker URL (RabbitMQ)",
    )
    celery_result_backend: Optional[str] = Field(
        default=None, description="Celery result backend URL (Redis)"
    )
    celery_task_always_eager: bool = Field(
        default=False, description="Execute tasks synchronously (for testing)"
    )
    celery_task_acks_late: bool = Field(
        default=True, description="Acknowledge tasks after completion"
    )
    celery_worker_prefetch_multiplier: int = Field(
        default=1, description="Worker prefetch multiplier"
    )
    celery_task_time_limit: int = Field(
        default=3600, description="Task time limit in seconds"
    )
    celery_task_soft_time_limit: int = Field(
        default=3300, description="Task soft time limit in seconds"
    )
    celery_result_expires: int = Field(
        default=86400, description="Result expiration time in seconds"
    )

    # Stream Processing Settings
    stream_chunk_duration_seconds: int = Field(
        default=30, description="Duration of stream chunks for processing"
    )
    stream_buffer_size_mb: int = Field(
        default=100, description="Stream buffer size in MB"
    )
    stream_max_duration_hours: int = Field(
        default=12, description="Maximum stream duration in hours"
    )
    stream_reconnect_attempts: int = Field(
        default=3, description="Number of reconnection attempts for streams"
    )
    stream_reconnect_delay_seconds: int = Field(
        default=5, description="Delay between reconnection attempts"
    )

    # AI/ML Settings
    ai_provider: Literal["openai", "anthropic", "gemini", "custom"] = Field(
        default="openai", description="AI provider for analysis"
    )
    ai_api_key: Optional[str] = Field(
        default=None, description="API key for AI provider (OpenAI)"
    )
    gemini_api_key: Optional[str] = Field(
        default=None, description="API key for Google Gemini"
    )
    ai_model_name: str = Field(default="gpt-4", description="AI model name")
    ai_max_tokens: int = Field(
        default=1000, description="Maximum tokens for AI responses"
    )
    ai_temperature: float = Field(default=0.7, description="AI model temperature")
    ai_request_timeout: int = Field(
        default=30, description="AI request timeout in seconds"
    )

    # Webhook Settings
    webhook_timeout_seconds: int = Field(
        default=10, description="Webhook request timeout"
    )
    webhook_max_retries: int = Field(
        default=3, description="Maximum webhook retry attempts"
    )
    webhook_retry_delay_seconds: int = Field(
        default=60, description="Delay between webhook retries"
    )
    webhook_signature_header: str = Field(
        default="X-Webhook-Signature", description="Header name for webhook signatures"
    )

    # Logging Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    log_file_path: Optional[Path] = Field(
        default=None, description="Log file path (None for stdout only)"
    )
    log_file_max_bytes: int = Field(
        default=10485760,  # 10MB
        description="Maximum log file size in bytes",
    )
    log_file_backup_count: int = Field(
        default=5, description="Number of log file backups to keep"
    )
    log_json_format: bool = Field(default=False, description="Use JSON format for logs")

    # Performance Settings
    worker_count: Optional[int] = Field(
        default=None, description="Number of worker processes (None for auto)"
    )
    worker_class: str = Field(
        default="uvicorn.workers.UvicornWorker", description="Worker class for Gunicorn"
    )
    keepalive: int = Field(default=5, description="Keepalive timeout in seconds")

    # Platform API Settings
    twitch_client_id: Optional[str] = Field(
        default=None, description="Twitch API client ID"
    )
    twitch_client_secret: Optional[str] = Field(
        default=None, description="Twitch API client secret"
    )
    twitch_app_access_token: Optional[str] = Field(
        default=None, description="Twitch app access token (generated)"
    )
    twitch_api_base_url: str = Field(
        default="https://api.twitch.tv/helix", description="Twitch API base URL"
    )
    twitch_auth_url: str = Field(
        default="https://id.twitch.tv/oauth2/token", description="Twitch OAuth URL"
    )
    twitch_rate_limit_per_minute: int = Field(
        default=800, description="Twitch API rate limit per minute"
    )
    twitch_eventsub_websocket_url: str = Field(
        default="wss://eventsub.wss.twitch.tv/ws",
        description="Twitch EventSub WebSocket URL"
    )
    twitch_eventsub_reconnect_attempts: int = Field(
        default=5, description="EventSub reconnection attempts"
    )
    twitch_eventsub_reconnect_delay: float = Field(
        default=1.0, description="EventSub reconnection base delay in seconds"
    )
    twitch_eventsub_keepalive_timeout: int = Field(
        default=10, description="EventSub keepalive timeout in seconds"
    )

    youtube_api_key: Optional[str] = Field(
        default=None, description="YouTube Data API v3 key"
    )
    youtube_api_base_url: str = Field(
        default="https://www.googleapis.com/youtube/v3",
        description="YouTube API base URL",
    )
    youtube_rate_limit_per_day: int = Field(
        default=10000, description="YouTube API quota units per day"
    )
    youtube_rate_limit_per_100_seconds: int = Field(
        default=100, description="YouTube API requests per 100 seconds"
    )

    # RTMP Settings
    rtmp_buffer_size: int = Field(default=4096, description="RTMP buffer size in bytes")
    rtmp_connection_timeout: int = Field(
        default=30, description="RTMP connection timeout in seconds"
    )
    rtmp_read_timeout: int = Field(
        default=10, description="RTMP read timeout in seconds"
    )

    # Circuit Breaker Settings
    circuit_breaker_failure_threshold: int = Field(
        default=5, description="Circuit breaker failure threshold"
    )
    circuit_breaker_timeout_seconds: int = Field(
        default=60, description="Circuit breaker timeout in seconds"
    )
    circuit_breaker_recovery_timeout_seconds: int = Field(
        default=30, description="Circuit breaker recovery timeout in seconds"
    )

    # Feature Flags
    feature_batch_processing: bool = Field(
        default=True, description="Enable batch processing feature"
    )
    feature_webhooks: bool = Field(default=True, description="Enable webhooks feature")
    feature_analytics: bool = Field(
        default=True, description="Enable analytics feature"
    )
    feature_custom_models: bool = Field(
        default=False, description="Enable custom AI models feature"
    )

    # Validators for comma-separated string lists
    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    # Model-level validator for production settings
    @model_validator(mode="after")
    def validate_production_settings(self):
        """Ensure production-specific settings are properly configured."""
        if self.environment == "production":
            # JWT secret must be set in production
            if not self.jwt_secret_key or self.jwt_secret_key == "dev-secret-key-change-this":
                raise ValueError("JWT secret key must be properly set in production")
            
            # Force disable debug and database echo in production
            self.debug = False
            self.database_echo = False
            self.database_echo_pool = False
        else:
            # Set development defaults
            if not self.jwt_secret_key:
                self.jwt_secret_key = "dev-secret-key-change-this"
        
        # Set celery result backend default
        if not self.celery_result_backend:
            self.celery_result_backend = str(self.redis_url or "redis://localhost:6379/1")
        
        return self

    # Computed fields for environment checks
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @computed_field
    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment == "staging"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function returns a cached instance of the Settings class,
    ensuring that environment variables are only read once.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Create a module-level settings instance for easy import
settings = get_settings()
