# TL;DR Highlight API - Complete Feature Documentation

## Overview

The TL;DR Highlight API is an enterprise-grade B2B service that provides AI-powered highlight extraction from livestreams and video content. Built with a domain-driven design architecture, it offers comprehensive features for businesses to integrate automated highlight detection into their platforms and workflows.

## Table of Contents

1. [Core Capabilities](#core-capabilities)
2. [Authentication & Authorization](#authentication--authorization)
3. [Stream Processing](#stream-processing)
4. [Batch Processing](#batch-processing)
5. [Highlight Management](#highlight-management)
6. [AI-Powered Detection](#ai-powered-detection)
7. [Organization Management](#organization-management)
8. [Webhook System](#webhook-system)
9. [Infrastructure & Integrations](#infrastructure--integrations)
10. [Monitoring & Observability](#monitoring--observability)
11. [Security Features](#security-features)
12. [Enterprise Features](#enterprise-features)

## Core Capabilities

### Architecture
- **Domain-Driven Design (DDD)** - Clean architecture with separated concerns
- **Multi-tenant Architecture** - Complete isolation between organizations
- **Asynchronous Processing** - Scalable background job processing
- **RESTful API Design** - Standard REST patterns with OpenAPI documentation
- **Event-Driven Architecture** - Webhook-based event notifications

### Supported Platforms
- **Twitch** - Full integration with Twitch livestreams
- **YouTube** - Support for YouTube live and VOD content
- **RTMP** - Direct RTMP stream ingestion
- **Custom URLs** - Support for any HLS/HTTP stream source

## Authentication & Authorization

### User Management

#### Registration & Authentication
```
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/logout
POST /api/v1/auth/refresh
```

- Email/password based authentication
- JWT token generation with refresh tokens
- Secure session management
- Organization association on registration

#### Profile Management
```
GET /api/v1/users/me
PUT /api/v1/users/me
POST /api/v1/users/me/change-password
```

- View and update user profile
- Change password with current password verification
- Profile includes organization membership

### API Key Management

#### Key Operations
```
POST /api/v1/users/me/api-keys
GET /api/v1/users/me/api-keys
DELETE /api/v1/users/me/api-keys/{key_id}
POST /api/v1/users/me/api-keys/{key_id}/rotate
```

#### Key Features
- **Scoped Permissions** - Fine-grained access control:
  - `streams:read` - Read stream data
  - `streams:write` - Create/update streams
  - `highlights:read` - Read highlights
  - `highlights:write` - Delete highlights
  - `webhooks:write` - Manage webhooks
  - `organization:admin` - Full organization access
- **Expiration Support** - Optional key expiration dates
- **Key Rotation** - Rotate keys with grace period
- **Rate Limiting** - Per-key rate limits

## Stream Processing

### Live Stream Management

#### Start Processing
```
POST /api/v1/streams
{
  "url": "https://twitch.tv/username",
  "title": "My Gaming Stream",
  "processing_options": {
    "detect_gameplay": true,
    "detect_reactions": true,
    "detect_funny_moments": true,
    "min_highlight_duration": 5.0,
    "max_highlight_duration": 120.0,
    "confidence_threshold": 0.7
  }
}
```

#### Stream Operations
```
GET /api/v1/streams
GET /api/v1/streams/{stream_id}
PUT /api/v1/streams/{stream_id}
DELETE /api/v1/streams/{stream_id}
POST /api/v1/streams/{stream_id}/stop
```

### Processing Options

#### Detection Types
- **Gameplay Highlights** - Key gameplay moments (kills, wins, clutches)
- **Reaction Moments** - Streamer reactions and surprises
- **Funny Moments** - Humor and entertainment value
- **Skillful Plays** - High-skill gameplay demonstrations
- **Emotional Moments** - Emotional peaks and reactions
- **Climactic Moments** - Story or match climaxes

#### Configuration Parameters
- **Confidence Threshold** (0.0-1.0) - Minimum confidence for highlight creation
- **Duration Limits** - Min/max highlight duration in seconds
- **Detection Toggles** - Enable/disable specific detection types
- **Priority Settings** - Prioritize certain highlight types

### Stream Status Tracking
```json
{
  "id": 123,
  "status": "processing",
  "progress_percentage": 45.5,
  "highlights_detected": 7,
  "started_at": "2024-01-15T10:00:00Z",
  "estimated_completion": "2024-01-15T11:30:00Z"
}
```

## Batch Processing

### Batch Video Operations

#### Create Batch Job
```
POST /api/v1/batches
{
  "videos": [
    {
      "url": "https://youtube.com/watch?v=...",
      "title": "Video 1",
      "processing_options": {...}
    }
  ],
  "parallel_limit": 3
}
```

#### Batch Management
```
GET /api/v1/batches
GET /api/v1/batches/{batch_id}
DELETE /api/v1/batches/{batch_id}
```

### Batch Features
- **Parallel Processing** - Process multiple videos simultaneously
- **Progress Tracking** - Individual progress for each video
- **Error Handling** - Continue processing on individual failures
- **Resource Management** - Configurable resource allocation

## Highlight Management

### Highlight Access

#### List Highlights
```
GET /api/v1/highlights?stream_id=123&min_confidence=0.8&highlight_type=gameplay
```

Query Parameters:
- `stream_id` - Filter by stream
- `min_confidence` - Minimum confidence score
- `max_confidence` - Maximum confidence score
- `highlight_type` - Filter by type
- `page` - Pagination page
- `page_size` - Results per page

#### Highlight Operations
```
GET /api/v1/highlights/{highlight_id}
GET /api/v1/highlights/{highlight_id}/download
DELETE /api/v1/highlights/{highlight_id}
```

### Highlight Data Structure
```json
{
  "id": 456,
  "stream_id": 123,
  "title": "Epic Clutch Win",
  "description": "Player clutches 1v3 situation",
  "start_time": 1250.5,
  "end_time": 1280.5,
  "duration": 30.0,
  "confidence_score": 0.92,
  "highlight_type": "gameplay",
  "video_url": "https://...",
  "thumbnail_url": "https://...",
  "metadata": {
    "tags": ["clutch", "victory", "intense"],
    "sentiment_score": 0.85,
    "engagement_level": "high",
    "detection_details": {...}
  }
}
```

## AI-Powered Detection

### Multi-Modal Analysis Engine

#### Video Analysis
- **Scene Detection** - Identify significant scene changes
- **Action Recognition** - Detect specific gameplay actions
- **Visual Excitement** - Measure visual intensity
- **Player Tracking** - Track player movements and actions

#### Audio Analysis
- **Audio Peak Detection** - Identify audio excitement spikes
- **Voice Analysis** - Detect streamer reactions
- **Music Detection** - Identify background music changes
- **Sound Effect Recognition** - Game-specific audio cues

#### Chat Analysis
- **Sentiment Analysis** - Real-time chat sentiment scoring
- **Emote Detection** - Track emote usage patterns
- **Message Velocity** - Detect chat activity spikes
- **Keyword Detection** - Identify highlight-related keywords

### AI Integration

#### Gemini Pro Vision
- Advanced multimodal understanding
- Frame-by-frame analysis
- Context-aware detection
- Natural language descriptions

#### Scoring System
```json
{
  "final_score": 0.88,
  "components": {
    "video_score": 0.85,
    "audio_score": 0.90,
    "chat_score": 0.92,
    "context_score": 0.87
  },
  "confidence": 0.88,
  "detection_method": "multi_modal_fusion"
}
```

## Organization Management

### Organization Features

#### Organization Operations
```
GET /api/v1/organizations/current
PUT /api/v1/organizations/current
GET /api/v1/organizations/current/usage
```

#### Member Management
```
GET /api/v1/organizations/current/members
POST /api/v1/organizations/current/members
DELETE /api/v1/organizations/current/members/{user_id}
```

### Subscription Plans

#### Free Tier
- 10 concurrent streams
- 100 hours/month processing
- 1,000 API calls/day
- 7-day data retention

#### Pro Tier ($299/month)
- 50 concurrent streams
- 1,000 hours/month processing
- 10,000 API calls/day
- 30-day data retention
- Priority processing

#### Enterprise Tier (Custom)
- Unlimited concurrent streams
- Custom processing limits
- Unlimited API calls
- Custom retention
- Dedicated support
- SLA guarantees

### Usage Tracking
```json
{
  "organization_id": "org_123",
  "current_period": {
    "streams_processed": 245,
    "processing_hours": 89.5,
    "api_calls": 15420,
    "storage_gb": 125.3
  },
  "limits": {
    "concurrent_streams": 50,
    "monthly_hours": 1000,
    "daily_api_calls": 10000,
    "storage_gb": 500
  }
}
```

## Webhook System

### Webhook Configuration

#### Webhook Management
```
POST /api/v1/webhooks
GET /api/v1/webhooks
GET /api/v1/webhooks/{webhook_id}
PUT /api/v1/webhooks/{webhook_id}
DELETE /api/v1/webhooks/{webhook_id}
POST /api/v1/webhooks/{webhook_id}/test
```

#### Configuration Options
```json
{
  "url": "https://your-app.com/webhooks/tldr",
  "events": ["stream.completed", "highlight.detected"],
  "secret": "your-webhook-secret",
  "active": true,
  "headers": {
    "X-Custom-Header": "value"
  }
}
```

### Webhook Events

#### Stream Events
- `stream.started` - Processing began
- `stream.completed` - Processing finished successfully
- `stream.failed` - Processing encountered error

#### Highlight Events
- `highlight.detected` - New highlight found
- `highlight.processed` - Highlight clip ready

#### Batch Events
- `batch.completed` - All videos processed
- `batch.failed` - Batch processing failed

### Webhook Security
- **HMAC Signatures** - Verify webhook authenticity
- **Retry Logic** - Automatic retries with backoff
- **Event Deduplication** - Prevent duplicate deliveries
- **Delivery Logs** - Track webhook delivery status

## Infrastructure & Integrations

### Stream Ingestion

#### Protocol Support
- **HLS (HTTP Live Streaming)** - M3U8 playlist parsing
- **RTMP** - Real-time messaging protocol
- **HTTP/HTTPS** - Direct video file URLs
- **Custom Protocols** - Extensible adapter system

#### Media Processing Pipeline
1. **Stream Discovery** - Detect stream format and properties
2. **Segment Extraction** - Extract video segments
3. **Frame Extraction** - Extract frames for analysis
4. **Audio Extraction** - Separate audio track
5. **Transcoding** - Normalize formats for processing

### Storage Architecture

#### S3 Integration
- **Highlights Storage** - Final highlight videos
- **Thumbnails** - Preview images
- **Temporary Files** - Processing intermediates
- **Multi-region Support** - CDN distribution

#### Database Design
- **PostgreSQL** - Primary data store
- **Redis** - Caching and queues
- **Optimized Indexes** - Fast query performance
- **Connection Pooling** - Efficient resource usage

### External Integrations

#### Streaming Platforms
- **Twitch API** - Stream metadata and chat
- **YouTube API** - Video information
- **100ms** - Webhook integration
- **Custom Platforms** - Adapter interface

#### AI Services
- **Google Gemini** - Advanced AI analysis
- **Custom Models** - Pluggable AI interface
- **Model Versioning** - A/B testing support

## Monitoring & Observability

### Logfire Integration

#### Distributed Tracing
- End-to-end request tracing
- Cross-service correlation
- Performance bottleneck identification
- Error root cause analysis

#### Metrics Collection
```
- Request rate and latency
- Stream processing duration
- Highlight detection accuracy
- API usage by organization
- Resource utilization
- Error rates by service
```

#### Custom Dashboards
1. **API Overview** - Request rates, latencies, errors
2. **Stream Processing** - Pipeline performance metrics
3. **Business Metrics** - Usage, costs, adoption
4. **Infrastructure Health** - System resources
5. **Error Tracking** - Exception monitoring

### Health Monitoring

#### Service Health Checks
```
GET /health
GET /health/auth
GET /health/stream-processing
GET /health/batch-processing
GET /health/highlights
GET /health/webhooks
```

#### Monitoring Features
- **Real-time Alerts** - Proactive issue detection
- **SLA Tracking** - Uptime and performance
- **Resource Monitoring** - CPU, memory, disk usage
- **Queue Monitoring** - Job processing status

## Security Features

### API Security

#### Authentication Methods
- **API Key** - Header-based authentication
- **JWT Tokens** - Bearer token authentication
- **Webhook Signatures** - Request verification

#### Rate Limiting
- **Per API Key** - Configurable limits
- **Per Endpoint** - Operation-specific limits
- **Burst Protection** - Short-term spike handling
- **Quota Enforcement** - Monthly/daily limits

### Data Protection

#### Encryption
- **TLS 1.3** - All API communications
- **At-rest Encryption** - Database and S3
- **Secure Keys** - AWS KMS integration

#### Privacy & Compliance
- **Data Isolation** - Multi-tenant separation
- **Access Logging** - Audit trail
- **GDPR Support** - Data export/deletion
- **SOC2 Compliance** - Security controls

## Enterprise Features

### Performance & Scalability

#### Infrastructure
- **Auto-scaling** - Dynamic resource allocation
- **Load Balancing** - Request distribution
- **CDN Integration** - Global content delivery
- **Queue Priority** - Plan-based processing

#### Reliability
- **99.9% SLA** - Uptime guarantee
- **Disaster Recovery** - Multi-region backup
- **Zero-downtime Deploys** - Rolling updates
- **Circuit Breakers** - Fault tolerance

### Custom Integration Support

#### Professional Services
- **Integration Assistance** - Implementation help
- **Custom Development** - Feature additions
- **Training** - Team onboarding
- **Dedicated Support** - Priority assistance

#### Advanced Features
- **Custom AI Models** - Bring your own models
- **White-label Options** - Branded experience
- **On-premise Deployment** - Self-hosted option
- **Custom Workflows** - Tailored processing

### Analytics & Reporting

#### Usage Analytics
- **Processing Statistics** - Detailed metrics
- **Cost Analysis** - Usage-based breakdown
- **Performance Reports** - SLA compliance
- **Trend Analysis** - Usage patterns

#### Business Intelligence
- **ROI Metrics** - Highlight engagement
- **User Behavior** - Feature adoption
- **Quality Metrics** - Detection accuracy
- **Custom Reports** - Tailored insights

## Getting Started

### Quick Start Guide
1. **Sign Up** - Create organization account
2. **Get API Key** - Generate authentication key
3. **Test API** - Try example requests
4. **Process Stream** - Submit first stream
5. **View Highlights** - Access detected highlights

### Integration Examples
- Python SDK usage
- Node.js integration
- Webhook handling
- Error handling patterns
- Best practices guide

### Support Resources
- **API Documentation** - OpenAPI/Swagger specs
- **Developer Portal** - Guides and tutorials
- **Support Tickets** - Technical assistance
- **Community Forum** - User discussions
- **Status Page** - Service availability

---

The TL;DR Highlight API provides a comprehensive, enterprise-ready solution for automated highlight extraction, designed to scale with your business needs while maintaining high performance and reliability.