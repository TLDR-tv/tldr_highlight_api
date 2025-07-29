# Infrastructure Setup Guide

This guide explains how to set up and run the local development infrastructure for the TL;DR Highlight API.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ with uv package manager
- Git

## Quick Start

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd tldr_highlight_api
   cp .env.example .env
   ```

2. **Start infrastructure services:**
   ```bash
   docker-compose up -d
   ```

3. **Install Python dependencies:**
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Verify services are running:**
   ```bash
   docker-compose ps
   ```

## Services Overview

The Docker Compose configuration includes the following services:

### PostgreSQL Database
- **Container:** `tldr_postgres`
- **Port:** 5433
- **Database:** `tldr_highlights`
- **User:** `tldr_user`
- **Password:** `tldr_password`
- **Health check:** PostgreSQL readiness probe

### Redis Cache
- **Container:** `tldr_redis`
- **Port:** 6379
- **Password:** `tldr_redis_password`
- **Persistence:** AOF enabled
- **Health check:** Connection test

### RabbitMQ Message Broker
- **Container:** `tldr_rabbitmq`
- **Ports:** 5673 (AMQP), 15673 (Management UI)
- **User:** `tldr_user`
- **Password:** `tldr_password`
- **VHost:** `tldr_vhost`
- **Management UI:** http://localhost:15673
- **Health check:** RabbitMQ diagnostics

### MinIO Object Storage
- **Container:** `tldr_minio`
- **Ports:** 9010 (API), 9011 (Console)
- **Access Key:** `tldr_minio_admin`
- **Secret Key:** `tldr_minio_password`
- **Console:** http://localhost:9011
- **Default Buckets:** `tldr-highlights`, `tldr-temp`
- **Health check:** MinIO health endpoint

## Infrastructure Modules

### Cache Module (`src/core/cache.py`)
- **Redis connection management** with async support
- **Connection pooling** for optimal performance
- **Automatic retry logic** for resilient connections
- **Caching decorators** for function result caching
- **Rate limiting** using Redis sliding window
- **JSON serialization/deserialization** for complex data types

Key features:
- Async context managers for connection handling
- Automatic reconnection on failures
- Multi-operation support (get_many, set_many)
- Built-in rate limiter for API protection

### Queue Module (`src/core/queue.py`)
- **Celery configuration** with RabbitMQ broker
- **Task routing** by priority and type
- **Enhanced logging** for task lifecycle
- **Retry policies** for failed tasks
- **Task management utilities** for monitoring

Key features:
- Multiple queue types (default, high_priority, low_priority, batch)
- Custom task classes with enhanced error handling
- Task manager for administrative operations
- Decorators for common task patterns

### Storage Module (`src/services/storage.py`)
- **S3/MinIO operations** with async support
- **Multipart upload** for large files
- **Presigned URLs** for direct access
- **File metadata management**
- **Bucket lifecycle policies** for temp files

Key features:
- Connection pooling and retry logic
- Helper utilities for key generation and URL parsing
- Support for both AWS S3 and MinIO
- Automatic content type detection

## Environment Configuration

The application uses environment variables for configuration. Key settings include:

```bash
# Database
DATABASE_URL="postgresql://tldr_user:tldr_password@localhost:5433/tldr_highlights"

# Redis
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_PASSWORD="tldr_redis_password"

# S3/MinIO
S3_ENDPOINT_URL="http://localhost:9010"
S3_ACCESS_KEY_ID="tldr_minio_admin"
S3_SECRET_ACCESS_KEY="tldr_minio_password"

# Celery
CELERY_BROKER_URL="amqp://tldr_user:tldr_password@localhost:5673/tldr_vhost"
CELERY_RESULT_BACKEND="redis://:tldr_redis_password@localhost:6379/1"
```

## Development Workflow

### Starting Services
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d postgres redis

# View logs
docker-compose logs -f
```

### Stopping Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (data loss!)
docker-compose down -v
```

### Service Health Checks
All services include health checks that can be monitored:

```bash
# Check service health
docker-compose ps

# View detailed health status
docker inspect tldr_postgres --format='{{.State.Health.Status}}'
```

### Accessing Services

- **PostgreSQL:** Connect using any PostgreSQL client to `localhost:5433`
- **Redis:** Use Redis CLI: `redis-cli -h localhost -p 6379 -a tldr_redis_password`
- **RabbitMQ Management:** http://localhost:15673 (tldr_user/tldr_password)
- **MinIO Console:** http://localhost:9011 (tldr_minio_admin/tldr_minio_password)

## Data Persistence

All services use Docker volumes for data persistence:
- `postgres_data`: PostgreSQL database files
- `redis_data`: Redis persistence files
- `rabbitmq_data`: RabbitMQ data and configuration
- `minio_data`: MinIO object storage

## Networking

Services communicate through a custom Docker network (`tldr_network`) which provides:
- Service discovery by container name
- Isolated network environment
- Consistent networking across development environments

## Troubleshooting

### Service Won't Start
1. Check if ports are already in use: `netstat -tulpn | grep <port>`
2. Check Docker logs: `docker-compose logs <service-name>`
3. Verify environment variables in `.env` file

### Connection Issues
1. Ensure services are healthy: `docker-compose ps`
2. Check network connectivity: `docker-compose exec <service> ping <other-service>`
3. Verify credentials match between services and application config

### Performance Issues
1. Monitor resource usage: `docker stats`
2. Check service logs for errors or warnings
3. Adjust connection pool sizes in configuration

### Data Issues
1. Check volume mounts: `docker volume ls`
2. Backup data before troubleshooting: `docker-compose exec postgres pg_dump...`
3. Reset data if needed: `docker-compose down -v && docker-compose up -d`

## Production Considerations

For production deployment:

1. **Use external managed services** when possible (RDS, ElastiCache, etc.)
2. **Configure proper security groups** and network isolation
3. **Set up monitoring and alerting** for all services
4. **Use secrets management** for sensitive configuration
5. **Implement backup strategies** for data persistence
6. **Scale services** based on load requirements
7. **Use container orchestration** (Kubernetes, ECS) for high availability

## Monitoring

The infrastructure includes basic health checks. For production, consider:
- Prometheus metrics collection
- Grafana dashboards
- Log aggregation (ELK stack)
- Application performance monitoring (APM)
- Resource usage monitoring