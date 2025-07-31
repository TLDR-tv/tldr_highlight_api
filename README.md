# TL;DR Highlight API Documentation

Welcome to the comprehensive documentation for the TL;DR Highlight API - an enterprise B2B service for AI-powered highlight extraction from livestreams and video content.

## 📚 Documentation Structure

### Getting Started
- [**Development Setup**](./development-setup.md) - Local development environment setup and requirements
- [**Quick Start Guide**](./quick-start.md) - Get up and running in minutes
- [**Configuration Guide**](./configuration.md) - Environment variables and settings

### Architecture & Design
- [**Architecture Overview**](./architecture/overview.md) - High-level system architecture
- [**Domain Model**](./architecture/domain-model.md) - Core business entities and concepts
- [**API Design**](./api/overview.md) - RESTful API structure and patterns
- [**Database Schema**](./architecture/database-schema.md) - Data model and relationships

### API Reference
- [**API Overview**](./api/overview.md) - General API concepts and authentication
- [**Authentication**](./api/authentication.md) - API keys and authentication methods
- [**Streams API**](./api/streams.md) - Stream processing endpoints
- [**Highlights API**](./api/highlights.md) - Highlight retrieval and management
- [**Organizations API**](./api/organizations.md) - Multi-tenant organization management
- [**Webhooks API**](./api/webhooks.md) - Event notifications and webhooks
- [**Error Handling**](./api/errors.md) - Error codes and troubleshooting

### AI & Processing
- [**Highlight Detection System**](./ai/highlight-detection.md) - AI-powered analysis pipeline
- [**Flexible Analysis Framework**](./ai/analysis-framework.md) - Custom dimensions and strategies
- [**Gemini Integration**](./ai/gemini-integration.md) - Google Gemini AI configuration
- [**Multi-Modal Analysis**](./ai/multi-modal.md) - Video, audio, and chat processing

### Infrastructure
- [**Streaming Infrastructure**](./infrastructure/streaming.md) - Universal stream support via FFmpeg
- [**Asynchronous Processing**](./infrastructure/async-processing.md) - Celery tasks and workflows
- [**Caching & Storage**](./infrastructure/storage.md) - Redis caching and S3 storage
- [**Observability**](./infrastructure/observability.md) - Monitoring, metrics, and logging

### Deployment & Operations
- [**Deployment Guide**](./deployment/overview.md) - Production deployment strategies
- [**Docker Setup**](./deployment/docker.md) - Containerization and Docker Compose
- [**Monitoring & Alerting**](./deployment/monitoring.md) - Production monitoring setup
- [**Backup & Recovery**](./deployment/backup.md) - Data backup and disaster recovery

### Development & Testing
- [**Development Workflow**](./development/workflow.md) - Git workflow and code standards
- [**Testing Strategy**](./development/testing.md) - Unit, integration, and E2E testing
- [**Code Style Guide**](./development/code-style.md) - Python conventions and best practices
- [**Contributing Guide**](./development/contributing.md) - How to contribute to the project

### Troubleshooting & Support
- [**Troubleshooting Guide**](./troubleshooting/common-issues.md) - Common problems and solutions
- [**Performance Tuning**](./troubleshooting/performance.md) - Optimization techniques
- [**FAQ**](./troubleshooting/faq.md) - Frequently asked questions
- [**Support**](./troubleshooting/support.md) - Getting help and support resources

## 🚀 Quick Links

- **[Development Setup](./development-setup.md)** - Start here for local development
- **[API Overview](./api/overview.md)** - Understanding the API structure
- **[Architecture Overview](./architecture/overview.md)** - System design and patterns
- **[Deployment Guide](./deployment/overview.md)** - Production deployment

## 🎯 Key Features

- **Enterprise B2B Focus**: Multi-tenant architecture with organization-level isolation
- **AI-Powered Analysis**: Google Gemini integration for sophisticated content understanding
- **Flexible Detection**: Customizable highlight dimensions and detection strategies
- **Real-Time Processing**: Live stream ingestion with low-latency processing
- **Comprehensive Monitoring**: Full observability with Logfire integration
- **Scalable Infrastructure**: Async processing with Celery and Redis
- **Universal Stream Support**: Any format FFmpeg can handle (RTMP, HLS, HTTP, local files, etc.)

## 🏗️ Architecture Highlights

- **Domain-Driven Design**: Clean architecture with clear separation of concerns
- **Hexagonal Architecture**: Pluggable external service integrations
- **Event-Driven**: Webhook-based notifications and async processing
- **Multi-Modal**: Video, audio, and chat analysis capabilities
- **Enterprise-Ready**: Rate limiting, authentication, usage tracking

## 📋 Requirements

- Python 3.13+
- PostgreSQL 16+
- Redis 7+
- FFmpeg 4.4+
- Docker & Docker Compose (for local development)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./development/contributing.md) for details on how to get started.

## 📜 License

This project is proprietary software. See the LICENSE file for details.

---

*Last updated: 2025-07-30*