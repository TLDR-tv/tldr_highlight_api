# Configuration Guide

This guide covers all configuration options for the TL;DR Highlight API.

## 🔧 Configuration Overview

The application uses environment-based configuration with validation through Pydantic Settings. Configuration is loaded from:

1. **Environment variables** (highest priority)
2. **`.env` files** (medium priority)
3. **Default values** (lowest priority)

## 📁 Configuration Files

### Environment Files

```bash
.env                    # Default environment file
.env.local             # Local overrides (not committed)
.env.development       # Development environment
.env.staging          # Staging environment
.env.production       # Production environment
```

### Loading Priority

```bash
# The application loads configuration in this order:
1. .env.{APP_ENVIRONMENT}  # Environment-specific
2. .env.local              # Local overrides
3. .env                    # Default
4. Environment variables   # System environment
```

## ⚙️ Core Application Settings

### Basic Application Configuration

```bash
# Application Identity
APP_NAME="TL;DR Highlight API"
APP_VERSION="1.0.0"
APP_ENVIRONMENT="development"  # development, staging, production
DEBUG=true                     # Enable debug mode (development only)
LOG_LEVEL="DEBUG"             # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Server Configuration
HOST="0.0.0.0"                # Server bind address
PORT=8000                     # Server port
WORKER_COUNT=1                # Number of worker processes (production)
RELOAD=true                   # Auto-reload on code changes (development)

# API Configuration
API_V1_PREFIX="/api/v1"       # API version prefix
API_KEY_HEADER="X-API-Key"    # API key header name
```

### Security Configuration

```bash
# Encryption Keys (MUST BE CHANGED IN PRODUCTION)
SECRET_KEY="your-super-secret-key-at-least-32-characters-long"
JWT_SECRET_KEY="your-jwt-secret-key-at-least-32-characters"
JWT_ALGORITHM="HS256"
JWT_EXPIRE_MINUTES=1440       # JWT token expiration (24 hours)

# Password Security
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SPECIAL=true

# API Key Configuration
API_KEY_PREFIX="tldr_sk_"
API_KEY_LENGTH=32
API_KEY_EXPIRE_DAYS=365       # Default API key expiration

# Webhook Security
WEBHOOK_SECRET_KEY="your-webhook-secret-key-32-chars-minimum"
WEBHOOK_SIGNATURE_HEADER="X-Webhook-Signature"
WEBHOOK_TIMESTAMP_HEADER="X-Webhook-Timestamp"
WEBHOOK_TOLERANCE_SECONDS=300  # Webhook timestamp tolerance
```

## 🗄️ Database Configuration

### PostgreSQL Settings

```bash
# Database Connection
DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/dbname"
DATABASE_POOL_SIZE=20         # Connection pool size
DATABASE_MAX_OVERFLOW=30      # Max overflow connections
DATABASE_POOL_TIMEOUT=30      # Connection timeout seconds
DATABASE_POOL_RECYCLE=3600    # Connection recycle time

# Alternative individual settings
DB_HOST="localhost"
DB_PORT=5432
DB_USER="postgres"
DB_PASSWORD="password"
DB_NAME="tldr_highlight_api"
DB_SCHEMA="public"

# SSL Configuration
DB_SSL_MODE="prefer"          # disable, allow, prefer, require
DB_SSL_CERT_PATH=""          # Path to SSL certificate
DB_SSL_KEY_PATH=""           # Path to SSL private key
DB_SSL_CA_PATH=""            # Path to SSL CA certificate

# Migration Settings
ALEMBIC_CONFIG="alembic.ini"
MIGRATION_TIMEOUT=300         # Migration timeout seconds
```

### Connection Pool Optimization

```bash
# Pool Settings by Environment
# Development
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Staging
DATABASE_POOL_SIZE=15
DATABASE_MAX_OVERFLOW=25

# Production
DATABASE_POOL_SIZE=25
DATABASE_MAX_OVERFLOW=50
```

## 🚀 Redis Configuration

### Cache and Queue Settings

```bash
# Redis Connection
REDIS_URL="redis://localhost:6379/0"
REDIS_PASSWORD=""             # Redis password (if required)
REDIS_SSL=false              # Enable SSL connection

# Alternative individual settings
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_DB=0
REDIS_USERNAME=""            # Redis 6+ username
REDIS_SOCKET_TIMEOUT=5       # Socket timeout seconds
REDIS_CONNECTION_TIMEOUT=10  # Connection timeout seconds

# Connection Pool
REDIS_POOL_SIZE=20           # Connection pool size
REDIS_POOL_MAX_CONNECTIONS=50 # Maximum connections

# Cache Settings
CACHE_TTL=3600               # Default cache TTL (1 hour)
CACHE_KEY_PREFIX="tldr:"     # Cache key prefix
CACHE_SERIALIZER="json"      # json, pickle

# Celery Redis Settings
CELERY_BROKER_URL="redis://localhost:6379/1"
CELERY_RESULT_BACKEND="redis://localhost:6379/2"
CELERY_BROKER_CONNECTION_RETRY=true
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP=true
```

## 🤖 AI Service Configuration

### Google Gemini Settings

```bash
# Gemini API Configuration
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-2.0-flash-exp"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com"
GEMINI_TIMEOUT=60            # Request timeout seconds
GEMINI_MAX_RETRIES=3         # Retry attempts
GEMINI_RETRY_DELAY=1         # Delay between retries

# Content Analysis Settings
GEMINI_MAX_TOKENS=8192       # Maximum tokens per request
GEMINI_TEMPERATURE=0.1       # Response creativity (0.0-1.0)
GEMINI_TOP_P=0.8            # Nucleus sampling parameter
GEMINI_TOP_K=40             # Top-K sampling parameter

# Rate Limiting
GEMINI_REQUESTS_PER_MINUTE=60
GEMINI_REQUESTS_PER_DAY=1000
```

### AI Analysis Configuration

```bash
# Multi-Modal Analysis
DEFAULT_VIDEO_ANALYSIS_FPS=1          # Frames per second for analysis
DEFAULT_AUDIO_ANALYSIS_INTERVAL=30    # Audio chunk interval (seconds)
DEFAULT_CONFIDENCE_THRESHOLD=0.7      # Minimum confidence score

# Processing Limits
MAX_VIDEO_DURATION_HOURS=6           # Maximum video duration
MAX_AUDIO_ANALYSIS_DURATION_HOURS=4  # Maximum audio analysis duration
MAX_HIGHLIGHTS_PER_STREAM=1000       # Maximum highlights per stream

# Dimension Analysis
DEFAULT_DIMENSION_WEIGHT=1.0         # Default dimension weight
MIN_DIMENSION_COUNT=1                # Minimum dimensions required
MAX_DIMENSION_COUNT=20               # Maximum dimensions allowed
```

## 💾 Storage Configuration

### AWS S3 Settings

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_SESSION_TOKEN=""         # For temporary credentials
AWS_REGION="us-east-1"       # AWS region

# S3 Configuration
AWS_S3_BUCKET="your-highlights-bucket"
AWS_S3_ENDPOINT_URL=""       # For S3-compatible services
AWS_S3_USE_SSL=true          # Use HTTPS for S3
AWS_S3_VERIFY_SSL=true       # Verify SSL certificates

# Bucket Structure
S3_HIGHLIGHTS_PREFIX="highlights/"
S3_THUMBNAILS_PREFIX="thumbnails/"
S3_TEMP_PREFIX="temp/"
S3_CLIPS_PREFIX="clips/"

# Upload Settings
S3_MULTIPART_THRESHOLD=64MB  # Multipart upload threshold
S3_MULTIPART_CHUNKSIZE=16MB  # Multipart chunk size
S3_MAX_CONCURRENCY=10        # Max concurrent uploads

# URL Settings
S3_PRESIGNED_URL_EXPIRY=86400 # Presigned URL expiry (24 hours)
S3_CUSTOM_DOMAIN=""          # Custom domain for S3 URLs
S3_USE_ACCELERATE=false      # Use S3 Transfer Acceleration
```

### Local Storage (Development)

```bash
# Local Storage Settings
LOCAL_STORAGE_PATH="/tmp/tldr-storage"
LOCAL_STORAGE_BASE_URL="http://localhost:8000/storage"
```

## 🔄 Async Processing Configuration

### Celery Settings

```bash
# Celery Configuration
CELERY_TASK_SERIALIZER="json"
CELERY_ACCEPT_CONTENT=["json"]
CELERY_RESULT_SERIALIZER="json"
CELERY_TIMEZONE="UTC"
CELERY_ENABLE_UTC=true

# Task Execution
CELERY_TASK_TRACK_STARTED=true
CELERY_TASK_TIME_LIMIT=3600        # Hard time limit (1 hour)
CELERY_TASK_SOFT_TIME_LIMIT=3300   # Soft time limit (55 minutes)
CELERY_TASK_ACKS_LATE=true
CELERY_TASK_REJECT_ON_WORKER_LOST=true

# Results
CELERY_RESULT_EXPIRES=86400        # Results expire after 24 hours
CELERY_RESULT_PERSISTENT=true
CELERY_RESULT_COMPRESSION="gzip"

# Worker Settings
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000
CELERY_WORKER_SEND_TASK_EVENTS=true
CELERY_WORKER_DISABLE_RATE_LIMITS=false

# Queue Configuration
CELERY_TASK_DEFAULT_QUEUE="default"
CELERY_TASK_CREATE_MISSING_QUEUES=true
CELERY_TASK_DEFAULT_RETRY_DELAY=60     # Default retry delay (1 minute)
CELERY_TASK_MAX_RETRIES=3              # Default max retries
```

### Queue Priorities

```bash
# Queue Priority Settings
HIGH_PRIORITY_QUEUE="high_priority"
DEFAULT_QUEUE="default"
LOW_PRIORITY_QUEUE="low_priority"
BATCH_QUEUE="batch"

# Processing Priorities
REALTIME_PRIORITY=10         # Real-time processing
HIGH_PRIORITY=8             # High priority streams
NORMAL_PRIORITY=5           # Normal processing
LOW_PRIORITY=2              # Low priority/cleanup
BATCH_PRIORITY=1            # Batch processing
```

## 🌐 Network Configuration

### CORS Settings

```bash
# CORS Configuration
CORS_ORIGINS="http://localhost:3000,http://localhost:8080"
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS="GET,POST,PUT,DELETE,OPTIONS,PATCH"
CORS_ALLOW_HEADERS="*"
CORS_EXPOSE_HEADERS=""
CORS_MAX_AGE=86400           # Preflight cache duration

# Production CORS (comma-separated)
CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com,https://admin.yourdomain.com"
```

### Security Headers

```bash
# Security Headers
SECURITY_HEADERS_ENABLED=true
HSTS_MAX_AGE=31536000              # HTTP Strict Transport Security
HSTS_INCLUDE_SUBDOMAINS=true
CONTENT_TYPE_OPTIONS="nosniff"     # X-Content-Type-Options
FRAME_OPTIONS="DENY"               # X-Frame-Options
XSS_PROTECTION="1; mode=block"     # X-XSS-Protection

# Trusted Hosts
ALLOWED_HOSTS="localhost,127.0.0.1,yourdomain.com"
```

## ⚡ Rate Limiting Configuration

### API Rate Limits

```bash
# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_STORAGE="redis"           # redis, memory
RATE_LIMIT_STORAGE_URL="redis://localhost:6379/3"

# Default Limits
RATE_LIMIT_PER_MINUTE=100           # Requests per minute
RATE_LIMIT_PER_HOUR=1000            # Requests per hour
RATE_LIMIT_BURST=20                 # Burst allowance

# Endpoint-Specific Limits
AUTH_RATE_LIMIT_PER_MINUTE=10       # Authentication endpoints
UPLOAD_RATE_LIMIT_PER_MINUTE=5      # Upload endpoints
WEBHOOK_RATE_LIMIT_PER_MINUTE=100   # Webhook endpoints

# Headers
RATE_LIMIT_HEADERS_ENABLED=true
RATE_LIMIT_HEADER_LIMIT="X-RateLimit-Limit"
RATE_LIMIT_HEADER_REMAINING="X-RateLimit-Remaining"
RATE_LIMIT_HEADER_RESET="X-RateLimit-Reset"
```

### Subscription-Based Limits

```bash
# Plan-Based Rate Limits
STARTER_RATE_LIMIT_PER_MINUTE=50
STARTER_RATE_LIMIT_PER_HOUR=500

PROFESSIONAL_RATE_LIMIT_PER_MINUTE=200
PROFESSIONAL_RATE_LIMIT_PER_HOUR=2000

ENTERPRISE_RATE_LIMIT_PER_MINUTE=1000
ENTERPRISE_RATE_LIMIT_PER_HOUR=10000
```

## 📊 Observability Configuration

### Logfire Settings

```bash
# Logfire Integration
LOGFIRE_TOKEN="your-logfire-token"
LOGFIRE_PROJECT_NAME="tldr-highlight-api"
LOGFIRE_ENVIRONMENT="production"    # Maps to APP_ENVIRONMENT
LOGFIRE_SERVICE_NAME="tldr-api"
LOGFIRE_SERVICE_VERSION="1.0.0"

# Data Collection
LOGFIRE_CAPTURE_HEADERS=false       # Capture HTTP headers
LOGFIRE_CAPTURE_BODY=false          # Capture request/response bodies
LOGFIRE_CAPTURE_SQL=true            # Capture SQL queries
LOGFIRE_CAPTURE_REDIS=true          # Capture Redis operations

# Sampling
LOGFIRE_SAMPLE_RATE=1.0             # Sample rate (0.0-1.0)
LOGFIRE_ERROR_SAMPLE_RATE=1.0       # Error sampling rate
LOGFIRE_SLOW_QUERY_THRESHOLD=1.0    # Slow query threshold (seconds)

# Export Settings
LOGFIRE_EXPORT_TIMEOUT=30           # Export timeout seconds
LOGFIRE_EXPORT_MAX_BATCH_SIZE=512   # Max batch size
LOGFIRE_EXPORT_SCHEDULE_DELAY=5000  # Schedule delay milliseconds
```

### Logging Configuration

```bash
# Logging Settings
LOG_LEVEL="INFO"                    # Global log level
LOG_FORMAT="json"                   # json, text
LOG_FILE_ENABLED=true               # Enable file logging
LOG_FILE_PATH="logs/app.log"        # Log file path
LOG_FILE_MAX_SIZE=10MB              # Max file size
LOG_FILE_BACKUP_COUNT=5             # Number of backup files

# Component Log Levels
DATABASE_LOG_LEVEL="WARNING"        # Database query logging
CELERY_LOG_LEVEL="INFO"            # Celery task logging
HTTP_LOG_LEVEL="INFO"              # HTTP request logging
AI_LOG_LEVEL="INFO"                # AI service logging

# Log Filtering
LOG_EXCLUDE_PATHS="/health,/metrics" # Exclude paths from logging
LOG_SENSITIVE_FIELDS="password,api_key,secret" # Sensitive field filtering
```

### Metrics Configuration

```bash
# Metrics Collection
METRICS_ENABLED=true
METRICS_PATH="/metrics"             # Prometheus metrics endpoint
METRICS_NAMESPACE="tldr"            # Metrics namespace
METRICS_SUBSYSTEM="api"             # Metrics subsystem

# Custom Metrics
BUSINESS_METRICS_ENABLED=true       # Business metrics collection
PERFORMANCE_METRICS_ENABLED=true    # Performance metrics
USAGE_METRICS_ENABLED=true          # Usage tracking metrics

# Export Settings
METRICS_EXPORT_INTERVAL=15          # Export interval seconds
METRICS_RETENTION_DAYS=30           # Metrics retention period
```

## 🔔 Webhook Configuration

### Webhook Delivery Settings

```bash
# Webhook Configuration
WEBHOOK_TIMEOUT_SECONDS=30          # HTTP request timeout
WEBHOOK_MAX_RETRIES=5               # Maximum retry attempts
WEBHOOK_RETRY_BACKOFF="exponential" # exponential, linear
WEBHOOK_RETRY_BASE_DELAY=60         # Base retry delay seconds
WEBHOOK_RETRY_MAX_DELAY=3600        # Maximum retry delay seconds

# Security
WEBHOOK_VERIFY_SSL=true             # Verify SSL certificates
WEBHOOK_USER_AGENT="TLDRHighlightAPI/1.0"
WEBHOOK_MAX_REDIRECTS=3             # Maximum HTTP redirects

# Event Configuration
WEBHOOK_EVENTS_ENABLED="stream.created,stream.completed,highlight.detected"
WEBHOOK_BATCH_SIZE=10               # Events per batch
WEBHOOK_BATCH_TIMEOUT=30            # Batch timeout seconds
```

## 🎥 Media Processing Configuration

### FFmpeg Settings

```bash
# FFmpeg Configuration
FFMPEG_BINARY_PATH="ffmpeg"         # FFmpeg binary path
FFMPEG_TIMEOUT=300                  # FFmpeg timeout seconds
FFMPEG_THREADS=4                    # Processing threads

# Video Processing
VIDEO_FRAME_RATE=1                  # Analysis frame rate
VIDEO_RESOLUTION="1280x720"         # Processing resolution
VIDEO_CODEC="libx264"               # Video codec
VIDEO_FORMAT="mp4"                  # Output format

# Audio Processing
AUDIO_SAMPLE_RATE=16000             # Audio sample rate
AUDIO_CHANNELS=1                    # Audio channels (mono)
AUDIO_CODEC="aac"                   # Audio codec
AUDIO_BITRATE="128k"                # Audio bitrate

# Thumbnail Generation
THUMBNAIL_WIDTH=640                 # Thumbnail width
THUMBNAIL_HEIGHT=360                # Thumbnail height
THUMBNAIL_FORMAT="jpg"              # Thumbnail format
THUMBNAIL_QUALITY=85                # JPEG quality (1-100)
```

### Stream Processing

```bash
# Stream Processing Limits
MAX_STREAM_DURATION_HOURS=6         # Maximum stream duration
MAX_CONCURRENT_STREAMS=10           # Max concurrent processing
STREAM_BUFFER_SIZE_MB=100           # Stream buffer size
STREAM_CHUNK_DURATION_SECONDS=30    # Processing chunk duration

# Platform Settings
TWITCH_CLIENT_ID="your-twitch-client-id"
TWITCH_CLIENT_SECRET="your-twitch-client-secret"
YOUTUBE_API_KEY="your-youtube-api-key"

# RTMP Settings
RTMP_SERVER_HOST="localhost"
RTMP_SERVER_PORT=1935
RTMP_CHUNK_SIZE=4096
RTMP_TIMEOUT=30
```

## 🔧 Environment-Specific Examples

### Development Configuration

```bash
# .env.development
APP_ENVIRONMENT="development"
DEBUG=true
LOG_LEVEL="DEBUG"

# Use local services
DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/tldr_dev"
REDIS_URL="redis://localhost:6379/0"

# Relaxed security for development
SECRET_KEY="dev-secret-key-not-for-production"
JWT_EXPIRE_MINUTES=10080  # 7 days

# Lower rate limits
RATE_LIMIT_PER_MINUTE=1000
RATE_LIMIT_PER_HOUR=10000

# Enable all logging
LOGFIRE_CAPTURE_HEADERS=true
LOGFIRE_CAPTURE_BODY=true
```

### Production Configuration

```bash
# .env.production
APP_ENVIRONMENT="production"
DEBUG=false
LOG_LEVEL="INFO"

# Production database with connection pooling
DATABASE_URL="postgresql+asyncpg://user:pass@prod-db:5432/tldr_prod"
DATABASE_POOL_SIZE=25
DATABASE_MAX_OVERFLOW=50

# Production Redis
REDIS_URL="redis://prod-redis:6379/0"
REDIS_PASSWORD="secure-redis-password"

# Strong security
SECRET_KEY="super-secure-production-key-64-characters-long-minimum"
JWT_EXPIRE_MINUTES=1440  # 24 hours

# Production rate limits
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000

# Minimal logging for performance
LOGFIRE_CAPTURE_HEADERS=false
LOGFIRE_CAPTURE_BODY=false
```

## ✅ Configuration Validation

### Required Settings

```bash
# These settings MUST be configured in production:
SECRET_KEY                  # Cryptographic key
DATABASE_URL               # Database connection
REDIS_URL                  # Redis connection
GEMINI_API_KEY            # AI service key
AWS_ACCESS_KEY_ID         # Storage access key
AWS_SECRET_ACCESS_KEY     # Storage secret key
LOGFIRE_TOKEN             # Observability token
```

### Security Checklist

- [ ] `SECRET_KEY` is at least 32 characters and cryptographically secure
- [ ] `JWT_SECRET_KEY` is unique and different from `SECRET_KEY`
- [ ] Database credentials use strong passwords
- [ ] API keys are properly secured and rotated
- [ ] SSL/TLS is enabled for all external connections
- [ ] CORS origins are restricted to trusted domains
- [ ] Rate limiting is appropriately configured
- [ ] Sensitive data is not logged in production

### Validation Script

```bash
#!/bin/bash
# validate-config.sh

echo "🔍 Validating configuration..."

# Check required environment variables
required_vars=(
    "SECRET_KEY"
    "DATABASE_URL" 
    "REDIS_URL"
    "GEMINI_API_KEY"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Missing required variable: $var"
        exit 1
    fi
done

# Check secret key length
if [ ${#SECRET_KEY} -lt 32 ]; then
    echo "❌ SECRET_KEY must be at least 32 characters"
    exit 1
fi

echo "✅ Configuration validation passed"
```

---

This configuration guide provides comprehensive coverage of all settings needed to run the TL;DR Highlight API in various environments.