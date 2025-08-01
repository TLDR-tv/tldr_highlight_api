# TLDR Highlight API

Enterprise B2B API for AI-powered highlight extraction from livestreams and video content.

## Quick Start

1. Install dependencies:
```bash
uv pip install -e .
```

2. Set up environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the application:
```bash
uv run uvicorn src.api.main:app --reload
```

4. Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

This application follows clean architecture principles with:
- Domain layer: Business logic and entities
- Application layer: Use cases and workflows
- Infrastructure layer: External services and persistence
- API layer: FastAPI routes and middleware

## Key Features

- Multi-tenant organization support
- Secure API key authentication
- Stream fingerprinting for unique streamer identification
- JWT-based content delivery with flexible access patterns
- Multi-dimensional AI scoring framework
- Real-time stream processing with FFmpeg
- Webhook support for event notifications