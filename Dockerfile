# Multi-stage Dockerfile for API and Worker

FROM python:3.13-slim as base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# API stage
FROM base AS api

# Install API dependencies
RUN uv sync --package api --no-dev

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/packages/api/src:/app/packages/shared/src
ENV PYTHONUNBUFFERED=1

# Run API
CMD ["uv", "run", "--package", "api", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Worker stage
FROM base AS worker

# Install worker dependencies
RUN uv sync --package worker --no-dev

# Set environment variables
ENV PYTHONPATH=/app/packages/worker/src:/app/packages/shared/src
ENV PYTHONUNBUFFERED=1
ENV C_FORCE_ROOT=1

# Run worker (command specified in docker-compose)
CMD ["uv", "run", "--package", "worker", "celery", "-A", "worker.app", "worker", "--loglevel=info"]