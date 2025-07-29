#!/bin/bash

# TL;DR Highlight API Infrastructure Startup Script

set -e

echo "🚀 Starting TL;DR Highlight API Infrastructure..."
echo

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. You may want to customize the configuration."
    echo
fi

# Start infrastructure services
echo "📦 Starting infrastructure services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

echo
echo "✅ Infrastructure startup complete!"
echo
echo "📊 Service URLs:"
echo "  - PostgreSQL:   localhost:5433"
echo "  - Redis:        localhost:6379"
echo "  - RabbitMQ:     localhost:5673"
echo "  - RabbitMQ UI:  http://localhost:15673 (tldr_user/tldr_password)"
echo "  - MinIO API:    localhost:9010"
echo "  - MinIO Console: http://localhost:9011 (tldr_minio_admin/tldr_minio_password)"
echo
echo "💡 To stop services: docker-compose down"
echo "💡 To view logs: docker-compose logs -f [service-name]"
echo "💡 To check status: docker-compose ps"