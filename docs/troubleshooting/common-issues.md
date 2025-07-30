# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the TL;DR Highlight API.

## 🚨 Quick Diagnostic Commands

### Health Check
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Service Status
```bash
# Check all services
docker-compose ps

# Check application logs
docker-compose logs -f api

# Check worker logs
docker-compose logs -f worker
```

### Database Connection
```bash
# Test database connection
uv run python -c "
import asyncio
from src.infrastructure.database import init_db
try:
    asyncio.run(init_db())
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

### Redis Connection
```bash
# Test Redis connection
uv run python -c "
import asyncio
from src.infrastructure.cache import get_redis_cache
async def test():
    try:
        cache = await get_redis_cache()
        await cache.set('test', 'value')
        result = await cache.get('test')
        print(f'✅ Redis connection successful: {result}')
    except Exception as e:
        print(f'❌ Redis connection failed: {e}')
asyncio.run(test())
"
```

## 🔍 Common Issues and Solutions

### 1. Application Won't Start

#### **Issue**: Port Already in Use
```
Error: [Errno 48] Address already in use
```

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 $(lsof -t -i:8000)

# Or use a different port
uv run uvicorn src.api.main:app --port 8001
```

#### **Issue**: Import Errors
```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
```bash
# Ensure you're in the project root
pwd  # Should show project directory

# Reinstall dependencies
uv pip install -e ".[dev]"

# Check Python path
uv run python -c "import sys; print('\n'.join(sys.path))"
```

#### **Issue**: Environment Variables Not Loaded
```
ValidationError: SECRET_KEY field required
```

**Solution**:
```bash
# Check if .env file exists
ls -la .env*

# Copy example file
cp .env.example .env

# Verify environment loading
uv run python -c "
from src.infrastructure.config import settings
print(f'Environment: {settings.app_environment}')
print(f'Database URL: {settings.database_url[:30]}...')
"
```

### 2. Database Connection Issues

#### **Issue**: Database Connection Failed
```
asyncpg.exceptions.InvalidCatalogNameError: database "tldr_highlight_api" does not exist
```

**Solution**:
```bash
# Start PostgreSQL
docker-compose up -d postgres

# Create database
docker-compose exec postgres createdb -U postgres tldr_highlight_api

# Run migrations
uv run alembic upgrade head
```

#### **Issue**: Migration Failures
```
alembic.util.exc.CommandError: Can't locate revision identified by 'abc123'
```

**Solution**:
```bash
# Check migration history
uv run alembic history

# Reset to specific revision
uv run alembic downgrade base
uv run alembic upgrade head

# Or stamp current state (if database is manually updated)
uv run alembic stamp head
```

#### **Issue**: Connection Pool Exhausted
```
asyncpg.exceptions.TooManyConnectionsError: too many connections for role
```

**Solution**:
```bash
# Reduce pool size in .env
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Check active connections
docker-compose exec postgres psql -U postgres -c "
SELECT count(*) FROM pg_stat_activity WHERE datname = 'tldr_highlight_api';
"

# Kill idle connections
docker-compose exec postgres psql -U postgres -c "
SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
WHERE datname = 'tldr_highlight_api' AND state = 'idle';
"
```

### 3. Redis/Celery Issues

#### **Issue**: Redis Connection Refused
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379
```

**Solution**:
```bash
# Start Redis
docker-compose up -d redis

# Check Redis status
docker-compose exec redis redis-cli ping

# Test connection with different URL
REDIS_URL="redis://localhost:6379/0" uv run python -c "
import redis
r = redis.from_url('redis://localhost:6379/0')
print(r.ping())
"
```

#### **Issue**: Celery Workers Not Starting
```
kombu.exceptions.OperationalError: [Errno 111] Connection refused
```

**Solution**:
```bash
# Check Redis for Celery
docker-compose exec redis redis-cli -n 1 ping

# Start worker with debug
uv run celery -A src.infrastructure.queue.celery_app worker --loglevel=debug

# Check Celery configuration
uv run celery -A src.infrastructure.queue.celery_app inspect stats
```

#### **Issue**: Tasks Not Being Processed
```
Tasks stuck in PENDING state
```

**Solution**:
```bash
# Check active workers
uv run celery -A src.infrastructure.queue.celery_app inspect active

# Check queue status
uv run celery -A src.infrastructure.queue.celery_app inspect reserved

# Purge queue (caution: deletes all pending tasks)
uv run celery -A src.infrastructure.queue.celery_app purge

# Restart workers
docker-compose restart worker
```

### 4. AI Service Issues

#### **Issue**: Gemini API Authentication Failed
```
google.api_core.exceptions.Unauthenticated: 401 API key not valid
```

**Solution**:
```bash
# Verify API key
echo $GEMINI_API_KEY

# Test API key directly
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1beta/models"

# Update .env file
GEMINI_API_KEY="your-valid-api-key-here"
```

#### **Issue**: Gemini API Rate Limiting
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```

**Solution**:
```bash
# Reduce concurrent analysis
GEMINI_REQUESTS_PER_MINUTE=30
GEMINI_REQUESTS_PER_DAY=1000

# Add retry configuration
GEMINI_MAX_RETRIES=5
GEMINI_RETRY_DELAY=2

# Check quota usage in Google Cloud Console
```

#### **Issue**: Analysis Results Empty
```
No highlights detected despite content analysis
```

**Solution**:
```bash
# Lower confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD=0.5

# Check dimension configuration
uv run python -c "
from src.domain.services.industry_presets import get_gaming_preset
preset = get_gaming_preset()
print(f'Dimensions: {[d.name for d in preset.dimensions]}')
"

# Enable debug logging for AI service
AI_LOG_LEVEL="DEBUG"
```

### 5. Storage Issues

#### **Issue**: S3 Access Denied
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution**:
```bash
# Check AWS credentials
aws configure list

# Test S3 access
aws s3 ls s3://your-bucket-name

# Update .env file
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
AWS_REGION="us-east-1"

# Test with boto3
uv run python -c "
import boto3
s3 = boto3.client('s3')
print(s3.list_buckets())
"
```

#### **Issue**: File Upload Failures
```
ClientError: The specified bucket does not exist
```

**Solution**:
```bash
# Create S3 bucket
aws s3 mb s3://your-bucket-name

# Check bucket policy
aws s3api get-bucket-policy --bucket your-bucket-name

# Update bucket name in .env
AWS_S3_BUCKET="your-existing-bucket-name"
```

### 6. Performance Issues

#### **Issue**: Slow API Response Times
```
API responses taking > 5 seconds
```

**Diagnosis**:
```bash
# Check database performance
uv run python -c "
import asyncio
import time
from src.infrastructure.database import get_db
async def test():
    start = time.time()
    async with get_db() as db:
        await db.execute('SELECT 1')
    print(f'DB query time: {time.time() - start:.3f}s')
asyncio.run(test())
"

# Check Redis performance
uv run python -c "
import asyncio
import time
from src.infrastructure.cache import get_redis_cache
async def test():
    cache = await get_redis_cache()
    start = time.time()
    await cache.get('test')
    print(f'Redis query time: {time.time() - start:.3f}s')
asyncio.run(test())
"
```

**Solutions**:
```bash
# Increase connection pools
DATABASE_POOL_SIZE=25
REDIS_POOL_SIZE=20

# Enable query caching
CACHE_TTL=1800

# Add database indexes (check slow query log)
# Monitor with Logfire for detailed performance metrics
```

#### **Issue**: High Memory Usage
```
Worker processes consuming excessive memory
```

**Solutions**:
```bash
# Reduce worker concurrency
CELERY_WORKER_CONCURRENCY=2

# Limit tasks per child
CELERY_WORKER_MAX_TASKS_PER_CHILD=100

# Monitor memory usage
docker stats

# Increase available memory
docker-compose.yml:
  deploy:
    resources:
      limits:
        memory: 2G
```

### 7. Webhook Delivery Issues

#### **Issue**: Webhooks Not Being Delivered
```
Webhook delivery failures in logs
```

**Diagnosis**:
```bash
# Check webhook configuration
uv run python -c "
from src.infrastructure.persistence.repositories.webhook_repository import WebhookRepository
import asyncio
async def check():
    repo = WebhookRepository()
    webhooks = await repo.get_active_webhooks()
    for webhook in webhooks:
        print(f'Webhook: {webhook.url}, Events: {webhook.events}')
asyncio.run(check())
"

# Test webhook endpoint manually
curl -X POST https://your-webhook-url.com/webhook \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

**Solutions**:
```bash
# Increase timeout
WEBHOOK_TIMEOUT_SECONDS=60

# Reduce retry attempts for testing
WEBHOOK_MAX_RETRIES=2

# Check webhook URL accessibility
# Verify webhook signature validation
```

#### **Issue**: Webhook Signature Verification Failed
```
Invalid webhook signature
```

**Solution**:
```bash
# Verify webhook secret configuration
echo $WEBHOOK_SECRET_KEY

# Test signature generation
uv run python -c "
import hmac
import hashlib
import json

secret = 'your-webhook-secret'
payload = json.dumps({'test': 'data'})
signature = hmac.new(
    secret.encode(),
    payload.encode(),
    hashlib.sha256
).hexdigest()
print(f'Expected signature: sha256={signature}')
"
```

## 🔧 Performance Optimization

### Database Optimization

```sql
-- Add useful indexes
CREATE INDEX CONCURRENTLY idx_streams_organization_status 
ON streams(organization_id, status);

CREATE INDEX CONCURRENTLY idx_highlights_stream_confidence 
ON highlights(stream_id, confidence_score DESC);

CREATE INDEX CONCURRENTLY idx_usage_records_org_created 
ON usage_records(organization_id, created_at DESC);

-- Analyze table statistics
ANALYZE streams;
ANALYZE highlights;
ANALYZE organizations;
```

### Redis Optimization

```bash
# Redis configuration optimizations
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET maxmemory 1gb
redis-cli CONFIG SET timeout 300

# Monitor Redis performance
redis-cli --latency-history
redis-cli INFO memory
```

### Application Optimization

```bash
# Enable production optimizations
APP_ENVIRONMENT="production"
DEBUG=false

# Optimize logging
LOG_LEVEL="WARNING"
LOGFIRE_SAMPLE_RATE=0.1

# Enable caching
CACHE_TTL=3600
REDIS_POOL_SIZE=20
```

## 📊 Monitoring and Alerting

### Key Metrics to Monitor

```bash
# Application metrics
- API response times (p95 < 500ms)
- Error rates (< 1%)
- Database connection pool usage (< 80%)
- Redis memory usage (< 80%)
- Celery queue depth (< 100 pending tasks)

# Business metrics
- Stream processing success rate (> 95%)
- Highlight detection rate (varies by content)
- Webhook delivery success rate (> 99%)
- Storage usage growth
```

### Alerting Rules

```yaml
# Example alerting rules (Prometheus)
groups:
- name: tldr-critical
  rules:
  - alert: DatabaseDown
    expr: up{job="postgres"} == 0
    for: 1m
    
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 2m
    
  - alert: QueueBacklog
    expr: celery_queue_length > 100
    for: 5m
```

## 🆘 Emergency Procedures

### Service Recovery

```bash
# Complete service restart
docker-compose down
docker-compose up -d

# Database recovery
docker-compose exec postgres pg_dump tldr_highlight_api > backup.sql
docker-compose down -v postgres
docker-compose up -d postgres
# Restore from backup

# Clear Redis cache (if corrupted)
docker-compose exec redis redis-cli FLUSHALL
```

### Data Recovery

```bash
# Database backup restoration
uv run alembic downgrade base
psql -U postgres -d tldr_highlight_api < backup.sql
uv run alembic stamp head

# S3 data recovery
aws s3 sync s3://backup-bucket/ s3://production-bucket/ --delete
```

## 📞 Getting Help

### Log Analysis
```bash
# Collect logs for support
mkdir -p /tmp/tldr-logs
docker-compose logs api > /tmp/tldr-logs/api.log
docker-compose logs worker > /tmp/tldr-logs/worker.log
docker-compose logs postgres > /tmp/tldr-logs/postgres.log

# System information
uv run python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
"
```

### Support Checklist
- [ ] Error message and full stack trace
- [ ] Configuration file (with secrets redacted)
- [ ] Application logs from the time of issue
- [ ] System resource usage (CPU, memory, disk)
- [ ] Steps to reproduce the issue
- [ ] Expected vs actual behavior

### Contact Information
- **GitHub Issues**: https://github.com/tldr-tv/tldr-highlight-api/issues
- **Documentation**: https://docs.tldr.tv
- **Support Email**: support@tldr.tv
- **Status Page**: https://status.tldr.tv

---

This troubleshooting guide covers the most common issues. For additional help, please refer to the logs and monitoring dashboards, or contact support with detailed information about your specific issue.