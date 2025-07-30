# Deployment Guide

This guide covers production deployment strategies for the TL;DR Highlight API.

## 🎯 Deployment Options

### Option 1: Docker Compose (Recommended for Small-Medium Scale)
- **Use Case**: Single server deployments, development, staging
- **Pros**: Simple setup, cost-effective, easy maintenance
- **Cons**: Limited scalability, single point of failure
- **Capacity**: Up to 50 concurrent streams

### Option 2: Kubernetes (Recommended for Enterprise Scale)
- **Use Case**: Multi-server, high-availability, auto-scaling
- **Pros**: High availability, horizontal scaling, advanced orchestration
- **Cons**: Complex setup, higher infrastructure costs
- **Capacity**: 100+ concurrent streams

### Option 3: Cloud Platform-as-a-Service
- **Use Case**: Minimal infrastructure management
- **Platforms**: AWS ECS, Google Cloud Run, Azure Container Instances
- **Pros**: Managed infrastructure, auto-scaling
- **Cons**: Platform lock-in, higher costs

## 🐋 Docker Compose Deployment

### Production Docker Compose Setup

```yaml
version: '3.8'

services:
  # Application Services
  api:
    image: tldr/highlight-api:latest
    ports:
      - "8000:8000"
    environment:
      - APP_ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://tldr_user:${POSTGRES_PASSWORD}@postgres:5432/tldr_highlights
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/1
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - LOGFIRE_TOKEN=${LOGFIRE_TOKEN}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    image: tldr/highlight-api:latest
    command: celery -A src.infrastructure.queue.celery_app worker --loglevel=info --concurrency=4
    environment:
      - APP_ENVIRONMENT=production
      - DATABASE_URL=postgresql+asyncpg://tldr_user:${POSTGRES_PASSWORD}@postgres:5432/tldr_highlights
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/1
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - /tmp:/tmp
    restart: unless-stopped
    deploy:
      replicas: 2

  scheduler:
    image: tldr/highlight-api:latest
    command: celery -A src.infrastructure.queue.celery_app beat --loglevel=info
    environment:
      - APP_ENVIRONMENT=production
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/1
    depends_on:
      - redis
    restart: unless-stopped

  # Infrastructure Services
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=tldr_highlights
      - POSTGRES_USER=tldr_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5433:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tldr_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Monitoring and Management
  flower:
    image: mher/flower:0.9.7
    command: flower --broker=redis://:${REDIS_PASSWORD}@redis:6379/1 --port=5555
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/1
    depends_on:
      - redis
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Production Environment File

```bash
# .env.production
APP_NAME="TL;DR Highlight API"
APP_VERSION="0.1.0"
APP_ENVIRONMENT="production"
DEBUG=false
LOG_LEVEL="INFO"

# Security
SECRET_KEY="your-super-secret-production-key-64-chars-minimum"
JWT_SECRET_KEY="your-jwt-secret-production-key"
WEBHOOK_SECRET_KEY="your-webhook-secret-production-key"

# Database
POSTGRES_PASSWORD="secure-postgres-password"
DATABASE_URL="postgresql+asyncpg://tldr_user:${POSTGRES_PASSWORD}@postgres:5432/tldr_highlights"

# Redis
REDIS_PASSWORD="secure-redis-password"
REDIS_URL="redis://:${REDIS_PASSWORD}@redis:6379/0"
CELERY_BROKER_URL="redis://:${REDIS_PASSWORD}@redis:6379/1"
CELERY_RESULT_BACKEND="redis://:${REDIS_PASSWORD}@redis:6379/2"

# AI Services
GEMINI_API_KEY="your-production-gemini-api-key"
GEMINI_MODEL="gemini-2.0-flash-exp"

# AWS Storage
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_S3_BUCKET="your-production-highlights-bucket"
AWS_REGION="us-east-1"

# Observability
LOGFIRE_TOKEN="your-production-logfire-token"
LOGFIRE_PROJECT_NAME="tldr-highlight-api-prod"
LOGFIRE_CAPTURE_HEADERS=false
LOGFIRE_CAPTURE_BODY=false

# CORS (Production domains)
CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
CORS_ALLOW_CREDENTIALS=true

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=1000
RATE_LIMIT_PER_HOUR=10000
```

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # Authentication routes (stricter rate limiting)
        location /api/v1/auth/ {
            limit_req zone=auth burst=10 nodelay;
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health checks
        location /health {
            proxy_pass http://api;
            access_log off;
        }

        # Static files (if needed)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## ☸️ Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tldr-highlight-api

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tldr-config
  namespace: tldr-highlight-api
data:
  APP_NAME: "TL;DR Highlight API"
  APP_ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379/0"
  CELERY_BROKER_URL: "redis://redis-service:6379/1"
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: tldr-secrets
  namespace: tldr-highlight-api
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret>
  DATABASE_URL: <base64-encoded-db-url>
  GEMINI_API_KEY: <base64-encoded-gemini-key>
  AWS_ACCESS_KEY_ID: <base64-encoded-aws-key>
  AWS_SECRET_ACCESS_KEY: <base64-encoded-aws-secret>
  LOGFIRE_TOKEN: <base64-encoded-logfire-token>
```

### API Deployment

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tldr-api
  namespace: tldr-highlight-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tldr-api
  template:
    metadata:
      labels:
        app: tldr-api
    spec:
      containers:
      - name: api
        image: tldr/highlight-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: tldr-config
        - secretRef:
            name: tldr-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: tldr-api-service
  namespace: tldr-highlight-api
spec:
  selector:
    app: tldr-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Worker Deployment

```yaml
# worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tldr-worker
  namespace: tldr-highlight-api
spec:
  replicas: 4
  selector:
    matchLabels:
      app: tldr-worker
  template:
    metadata:
      labels:
        app: tldr-worker
    spec:
      containers:
      - name: worker
        image: tldr/highlight-api:latest
        command: ["celery"]
        args: ["-A", "src.infrastructure.queue.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
        envFrom:
        - configMapRef:
            name: tldr-config
        - secretRef:
            name: tldr-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: temp-storage
          mountPath: /tmp
      volumes:
      - name: temp-storage
        emptyDir: {}
```

### PostgreSQL Deployment

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: tldr-highlight-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:16-alpine
        env:
        - name: POSTGRES_DB
          value: "tldr_highlights"
        - name: POSTGRES_USER
          value: "tldr_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: tldr-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: tldr-highlight-api
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: tldr-highlight-api
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tldr-ingress
  namespace: tldr-highlight-api
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: tldr-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tldr-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tldr-api-hpa
  namespace: tldr-highlight-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tldr-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tldr-worker-hpa
  namespace: tldr-highlight-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tldr-worker
  minReplicas: 4
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
```

## 🚀 Deployment Scripts

### Build and Deploy Script

```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
REGISTRY="your-registry.com"
IMAGE_NAME="tldr/highlight-api"
VERSION=${1:-latest}
ENVIRONMENT=${2:-production}

echo "🚀 Deploying TL;DR Highlight API v${VERSION} to ${ENVIRONMENT}"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME}:${VERSION} .
docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}

# Push to registry
echo "📤 Pushing to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}

# Deploy based on environment
if [ "$ENVIRONMENT" = "kubernetes" ]; then
    echo "☸️ Deploying to Kubernetes..."
    
    # Update image in deployments
    kubectl set image deployment/tldr-api api=${REGISTRY}/${IMAGE_NAME}:${VERSION} -n tldr-highlight-api
    kubectl set image deployment/tldr-worker worker=${REGISTRY}/${IMAGE_NAME}:${VERSION} -n tldr-highlight-api
    
    # Wait for rollout
    kubectl rollout status deployment/tldr-api -n tldr-highlight-api
    kubectl rollout status deployment/tldr-worker -n tldr-highlight-api
    
    echo "✅ Kubernetes deployment complete"
    
elif [ "$ENVIRONMENT" = "docker-compose" ]; then
    echo "🐋 Deploying with Docker Compose..."
    
    # Update image version in docker-compose
    sed -i.bak "s|image: tldr/highlight-api:.*|image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}|g" docker-compose.prod.yml
    
    # Deploy
    docker-compose -f docker-compose.prod.yml pull
    docker-compose -f docker-compose.prod.yml up -d
    
    echo "✅ Docker Compose deployment complete"
fi

# Run health check
echo "🏥 Running health check..."
sleep 30
curl -f http://localhost:8000/health || echo "⚠️ Health check failed"

echo "🎉 Deployment complete!"
```

### Database Migration Script

```bash
#!/bin/bash
# migrate.sh

set -e

ENVIRONMENT=${1:-production}

echo "🗄️ Running database migrations for ${ENVIRONMENT}"

if [ "$ENVIRONMENT" = "kubernetes" ]; then
    # Run migration job in Kubernetes
    kubectl create job --from=deployment/tldr-api migration-$(date +%s) -n tldr-highlight-api
    kubectl wait --for=condition=complete job/migration-$(date +%s) -n tldr-highlight-api --timeout=300s
elif [ "$ENVIRONMENT" = "docker-compose" ]; then
    # Run migration with Docker Compose
    docker-compose -f docker-compose.prod.yml exec api uv run alembic upgrade head
fi

echo "✅ Database migrations complete"
```

## 📊 Production Monitoring

### Health Check Endpoints

```python
# Health check configuration
HEALTH_CHECKS = {
    "database": check_database_connection,
    "redis": check_redis_connection, 
    "storage": check_s3_connectivity,
    "ai_service": check_gemini_api,
    "queue": check_celery_broker
}
```

### Monitoring Stack

```yaml
# monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'tldr-api'
      static_configs:
      - targets: ['tldr-api-service:80']
      metrics_path: /metrics
      
    - job_name: 'postgres'
      static_configs:
      - targets: ['postgres-service:5432']
```

### Alerting Rules

```yaml
# alerts.yaml
groups:
- name: tldr-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      
  - alert: DatabaseDown
    expr: up{job="postgres"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Database is down
      
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High memory usage detected
```

## 🔒 Security Hardening

### Container Security

```dockerfile
# Security-hardened Dockerfile
FROM python:3.13-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```


### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tldr-network-policy
  namespace: tldr-highlight-api
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Database backups created
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Monitoring configured

### Deployment
- [ ] Docker images built and pushed
- [ ] Database migrations applied
- [ ] Configuration deployed
- [ ] Services started
- [ ] Health checks passing
- [ ] Load balancer configured

### Post-Deployment
- [ ] Application health verified
- [ ] Monitoring alerts active
- [ ] Performance metrics baseline
- [ ] Backup verification
- [ ] Documentation updated
- [ ] Team notified

## 🔄 Rollback Procedures

### Docker Compose Rollback

```bash
#!/bin/bash
# rollback.sh
PREVIOUS_VERSION=${1:-previous}

echo "🔄 Rolling back to version: ${PREVIOUS_VERSION}"
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
echo "✅ Rollback complete"
```

### Kubernetes Rollback

```bash
# Rollback deployment
kubectl rollout undo deployment/tldr-api -n tldr-highlight-api
kubectl rollout undo deployment/tldr-worker -n tldr-highlight-api

# Check status
kubectl rollout status deployment/tldr-api -n tldr-highlight-api
```

---

This deployment guide provides comprehensive production deployment strategies for various scales and requirements.