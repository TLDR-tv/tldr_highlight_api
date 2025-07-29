# B2B Highlight API - CrowdCover Integration Sprint

## Project Overview

**Project Name**: B2B Highlight API - CrowdCover Integration  
**Timeline**: Sprint 1 - 2 Weeks (10 Working Days)  
**Status**: Ready for Development  
**Team**: Engineering  
**Priority**: P0 - First Enterprise Client  

### Executive Summary

Build a standalone B2B Highlight API service for CrowdCover, a sports betting livestreaming platform using 100ms. The API will provide AI-powered highlight extraction using the same multi-dimensional scoring system and YAML configuration as the tldrtv platform, with enterprise-grade API key authentication and webhook integration.

### Business Context

**Client**: CrowdCover - "Twitch for sports betting"
- Live streaming platform where users must place a bet to join streams
- Uses 100ms for video infrastructure
- Needs automated highlight detection for sports betting moments
- First B2B client for the highlight API service

---

## 🎯 Sprint Epics & Issues

### Epic 1: Core API Infrastructure & Authentication (3 days)

#### TLDR-B2B-001: Initialize B2B API Project Structure
**Priority**: P0 | **Points**: 3 | **Assignee**: TBD
- Set up FastAPI project structure mirroring tldrtv patterns
- Configure pyproject.toml with dependencies
- Create base configuration system (src/api/config.py)
- Set up logging infrastructure
- Initialize AWS Copilot application

#### TLDR-B2B-002: Implement API Key Authentication System
**Priority**: P0 | **Points**: 5 | **Assignee**: TBD
- Create API key model and database schema
- Implement API key generation and management endpoints
- Build authentication middleware for request validation
- Add rate limiting per API key
- Create tenant isolation decorators

#### TLDR-B2B-003: Design Multi-Tenant Database Schema
**Priority**: P0 | **Points**: 3 | **Assignee**: TBD
- Create enterprise schema for CrowdCover in Aurora
- Design tables: api_keys, streams, highlights, usage_metrics
- Implement yoyo migrations for schema creation
- Add row-level security policies
- Create database connection pooling

---

### Epic 2: Highlight Detection Core (4 days)

#### TLDR-B2B-004: Port Multi-Dimensional Scoring System
**Priority**: P0 | **Points**: 5 | **Assignee**: TBD
- Copy HighlightScore model with 8 dimensions:
  - skill_execution, game_impact, rarity, visual_spectacle
  - emotional_intensity, narrative_value, timing_importance, momentum_shift
- Implement calculate_combined_score with game weights
- Port HighlightCandidate model with timestamp handling
- Create score validation and normalization
- Add backward compatibility for legacy scores

#### TLDR-B2B-005: Implement YAML Game Configuration System
**Priority**: P0 | **Points**: 3 | **Assignee**: TBD
- Create game_configs directory structure
- Port existing game configs (valorant.yaml, etc.)
- Implement config loader with fuzzy matching
- Add default.yaml fallback configuration
- Create sports betting specific configs

#### TLDR-B2B-006: Build Highlight Selection Logic
**Priority**: P0 | **Points**: 5 | **Assignee**: TBD
- Port HighlightSelector class with time-based filtering
- Implement quality thresholds and similarity detection
- Add spacing requirements and window limits
- Create pending/confirmed highlight tracking
- Implement exceptional highlight overrides

#### TLDR-B2B-007: Create Video Analysis Service
**Priority**: P0 | **Points**: 8 | **Assignee**: TBD
- Implement VideoAnalyzer class for Gemini integration
- Port multi-step analysis logic
- Add segment validation and timestamp sanitization
- Implement highlight refinement for short clips
- Create game-specific tag extraction

---

### Epic 3: 100ms Webhook Integration (2 days)

#### TLDR-B2B-008: Build 100ms Webhook Handler
**Priority**: P0 | **Points**: 5 | **Assignee**: TBD
- Create webhook endpoint /webhooks/100ms
- Implement HMAC signature verification
- Add IP whitelisting for 100ms NAT gateways:
  ```
  34.100.213.146/32, 35.200.143.211/32, 34.100.191.162/32
  34.100.132.35/32, 34.93.93.114/32, 34.131.109.150/32
  ```
- Parse webhook events (hls.started, hls.stopped, etc.)
- Queue stream processing jobs

#### TLDR-B2B-009: Implement Stream Processing Pipeline
**Priority**: P0 | **Points**: 5 | **Assignee**: TBD
- Create stream ingestion from HLS URLs
- Implement video segment buffering
- Add clip assembly service integration
- Connect to existing RabbitMQ for job queuing
- Handle stream lifecycle events

---

### Epic 4: API Endpoints & Business Logic (2 days)

#### TLDR-B2B-010: Core B2B API Endpoints
**Priority**: P0 | **Points**: 5 | **Assignee**: TBD
- POST /streams/analyze - Submit stream for processing
- GET /highlights/list - Retrieve highlights with filters
- GET /highlights/{id} - Get specific highlight details
- POST /webhooks/subscribe - Manage webhook subscriptions
- GET /usage/stats - Usage analytics per tenant

#### TLDR-B2B-011: Implement Highlight Export Features
**Priority**: P1 | **Points**: 3 | **Assignee**: TBD
- Add clip generation with watermarking
- Implement S3 upload to shared clips bucket
- Create CDN URLs for clip delivery
- Add batch export capabilities
- Support multiple export formats

---

### Epic 5: Integration & Testing (1 day)

#### TLDR-B2B-012: AWS Copilot Cross-App Integration
**Priority**: P0 | **Points**: 5 | **Assignee**: TBD
- Configure shared resource access (Aurora, S3, MQ)
- Set up CloudFormation exports/imports
- Implement service discovery
- Create environment addons
- Test cross-application connectivity

#### TLDR-B2B-013: End-to-End Testing Suite
**Priority**: P0 | **Points**: 3 | **Assignee**: TBD
- Create integration tests for webhook flow
- Test multi-dimensional scoring accuracy
- Validate game config loading
- Load test with concurrent streams
- API contract testing

---

## 🏗️ Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    B2B Highlight API                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                 │
│  │   API Gateway    │    │  Auth Service    │                 │
│  ├──────────────────┤    ├──────────────────┤                 │
│  │ • Rate Limiting  │    │ • API Keys       │                 │
│  │ • Load Balancing │    │ • Tenant Isolation│                 │
│  │ • Request Routing│    │ • Usage Tracking  │                 │
│  └─────────┬────────┘    └─────────┬────────┘                 │
│            │                       │                          │
│  ┌─────────▼───────────────────────▼────────┐                 │
│  │         100ms Webhook Handler            │                 │
│  │    • HMAC Verification                   │                 │
│  │    • Event Processing                    │                 │
│  │    • Job Queuing                         │                 │
│  └─────────────────┬────────────────────────┘                 │
│                    │                                          │
│  ┌─────────────────▼────────────────────────┐                 │
│  │    Multi-Dimensional Highlight Engine    │                 │
│  │    • 8-Dimension Scoring                 │                 │
│  │    • YAML Game Configs                   │                 │
│  │    • Highlight Selection                 │                 │
│  └─────────────────┬────────────────────────┘                 │
│                    │                                          │
│  ┌─────────────────▼────────────────────────┐                 │
│  │    Shared Infrastructure (tldrtv)        │                 │
│  │    • Aurora PostgreSQL                   │                 │
│  │    • RabbitMQ                            │                 │
│  │    • S3 Clips Bucket                     │                 │
│  └──────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components from tldrtv to Port

#### 1. Multi-Dimensional Scoring System
```python
# 8 dimensions for comprehensive highlight evaluation
- skill_execution     # Technical skill displayed
- game_impact        # Impact on match outcome
- rarity            # Uniqueness of the moment
- visual_spectacle  # Visual excitement
- emotional_intensity # Emotional impact
- narrative_value   # Story/comeback value
- timing_importance # Criticality of timing
- momentum_shift    # Game momentum change
```

#### 2. YAML Game Configuration
```yaml
# Per-game configuration with custom weights
dimension_weights:
  skill_execution: 0.25
  game_impact: 0.20
  rarity: 0.15
  # ... other dimensions

score_thresholds:
  exceptional: 0.95
  high_quality: 0.85
  standard: 0.70
```

#### 3. Highlight Selection Algorithm
- Time-window based filtering (5-minute windows)
- Minimum spacing between highlights (30 seconds)
- Similarity detection to avoid duplicates
- Exceptional highlight overrides
- Pending/confirmed state tracking

---

## 📊 Sprint Metrics

### Velocity Targets
- **Total Points**: 59
- **Daily Velocity**: 6 points
- **Buffer**: 1 day for integration issues

### Success Criteria
- [ ] Process 100ms webhooks successfully
- [ ] Generate highlights with multi-dimensional scoring
- [ ] API key authentication working
- [ ] Integration with tldrtv infrastructure
- [ ] 80%+ unit test coverage

### Risk Mitigation
- **100ms Integration**: Early webhook testing with mock data
- **AWS Copilot**: Spike on cross-app resource sharing
- **Gemini API**: Reuse existing integration patterns
- **Database Schema**: Collaborate with DBA on multi-tenancy

---

## 🚀 Definition of Done

### Code Quality
- [ ] Follows tldrtv code patterns
- [ ] Type hints on all functions
- [ ] Comprehensive error handling
- [ ] Logging with correlation IDs

### Testing
- [ ] Unit tests for all components
- [ ] Integration tests for webhooks
- [ ] Load testing with 10+ streams
- [ ] API contract tests

### Documentation
- [ ] OpenAPI specification
- [ ] README with setup instructions
- [ ] API integration guide
- [ ] Webhook configuration guide

### Deployment
- [ ] AWS Copilot manifests
- [ ] Environment configurations
- [ ] Database migrations tested
- [ ] Monitoring dashboards

---

## 📋 Sprint Retrospective Topics

### Week 1 Review
- API authentication implementation
- Multi-dimensional scoring accuracy
- YAML configuration flexibility
- Database multi-tenancy approach

### Week 2 Review
- 100ms webhook reliability
- Highlight selection performance
- Cross-app integration challenges
- CrowdCover specific requirements

---

*This Linear project defines the sprint plan for building the B2B Highlight API with CrowdCover as the first enterprise client.*