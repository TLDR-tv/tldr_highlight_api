# Architecture Overview

The TL;DR Highlight API is built using modern software architecture principles with a focus on scalability, maintainability, and enterprise-grade reliability.

## 🏗️ High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Clients]
        SDK[SDKs]
        WEBHOOK[Webhook Consumers]
    end

    subgraph "API Gateway Layer"
        LB[Load Balancer]
        API[FastAPI Application]
        MW[Middleware Stack]
    end

    subgraph "Application Layer"
        UC[Use Cases]
        SVC[Domain Services]
        REPO[Repositories]
    end

    subgraph "Infrastructure Layer"
        DB[(PostgreSQL)]
        REDIS[(Redis Cache)]
        S3[(S3 Storage)]
        CELERY[Celery Workers]
        AI[Gemini AI]
    end

    subgraph "External Services"
        TWITCH[Twitch API]
        YOUTUBE[YouTube API]
        RTMP[RTMP Streams]
        LOGFIRE[Logfire Observability]
    end

    WEB --> LB
    SDK --> LB
    LB --> API
    API --> MW
    MW --> UC
    UC --> SVC
    SVC --> REPO
    REPO --> DB
    REPO --> REDIS
    REPO --> S3
    
    SVC --> CELERY
    CELERY --> AI
    CELERY --> TWITCH
    CELERY --> YOUTUBE
    CELERY --> RTMP
    
    API --> WEBHOOK
    API --> LOGFIRE
```

## 🎯 Architectural Principles

### Domain-Driven Design (DDD)
- **Entities**: Core business objects with identity (Stream, Highlight, Organization)
- **Value Objects**: Immutable objects (Timestamp, Duration, ConfidenceScore)
- **Aggregates**: Consistency boundaries around related entities
- **Domain Services**: Complex business logic that doesn't belong to a single entity
- **Repositories**: Data access abstractions

### Hexagonal Architecture (Ports & Adapters)
- **Core Domain**: Pure business logic without infrastructure dependencies
- **Application Services**: Orchestrate domain operations
- **Adapters**: Interface with external systems (databases, APIs, file systems)
- **Ports**: Define contracts between layers

### Clean Architecture Layers

#### 1. Domain Layer (`src/domain/`)
**The Core Business Logic**
- **Entities**: Stream, Highlight, Organization, User, Webhook
- **Value Objects**: Timestamp, Duration, URL, ConfidenceScore
- **Domain Services**: Complex business operations
- **Repository Interfaces**: Data access contracts
- **Domain Exceptions**: Business rule violations

**Key Characteristics:**
- No dependencies on infrastructure
- Contains all business rules and invariants
- Framework-independent
- Highly testable

#### 2. Application Layer (`src/application/`)
**Use Case Orchestration**
- **Use Cases**: Application-specific business rules
- **DTOs**: Data transfer objects for use case boundaries
- **Application Services**: Coordinate domain operations

**Key Characteristics:**
- Depends only on domain layer
- Orchestrates domain operations
- Handles application-specific workflows
- Transaction boundaries

#### 3. Infrastructure Layer (`src/infrastructure/`)
**External Integrations and Technical Concerns**
- **Database**: PostgreSQL with SQLAlchemy
- **Caching**: Redis for performance and rate limiting
- **Storage**: S3 for file storage
- **Queue**: Celery for async processing
- **AI Integration**: Google Gemini for content analysis
- **Monitoring**: Logfire for observability

#### 4. API Layer (`src/api/`)
**External Interface**
- **REST Controllers**: FastAPI routers
- **DTOs**: Request/response models
- **Middleware**: Cross-cutting concerns
- **Exception Handlers**: Error management

## 🔄 Data Flow Architecture

### Stream Processing Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant UseCase
    participant DomainService
    participant Queue
    participant Worker
    participant AI
    participant Storage

    Client->>API: POST /streams
    API->>UseCase: StreamProcessingUseCase
    UseCase->>DomainService: StreamProcessingService
    DomainService->>Queue: Queue processing task
    Queue->>Worker: Execute stream processing
    
    Worker->>AI: Analyze content (Gemini)
    AI-->>Worker: Analysis results
    Worker->>Storage: Store highlights
    Storage-->>Worker: Storage URLs
    
    Worker->>DomainService: Update stream status
    DomainService->>API: Webhook notification
    API-->>Client: Processing complete
```

### Multi-Modal Analysis Flow

```mermaid
graph LR
    subgraph "Content Ingestion"
        STREAM[Live Stream]
        VIDEO[Video Frames]
        AUDIO[Audio Track]
        CHAT[Chat Messages]
    end

    subgraph "Processing Pipeline"
        FFmpeg[FFmpeg Processing]
        GEMINI[Gemini AI Analysis]
        SENTIMENT[Sentiment Analysis]
    end

    subgraph "Analysis & Fusion"
        SCORER[Multi-Modal Scorer]
        FUSION[Signal Fusion]
        DETECTOR[Highlight Detector]
    end

    subgraph "Output"
        HIGHLIGHTS[(Highlights DB)]
        S3[(S3 Storage)]
        WEBHOOK[Webhook Delivery]
    end

    STREAM --> VIDEO
    STREAM --> AUDIO
    STREAM --> CHAT
    
    VIDEO --> FFmpeg
    AUDIO --> FFmpeg
    FFmpeg --> GEMINI
    CHAT --> SENTIMENT
    
    GEMINI --> SCORER
    SENTIMENT --> SCORER
    SCORER --> FUSION
    FUSION --> DETECTOR
    
    DETECTOR --> HIGHLIGHTS
    DETECTOR --> S3
    DETECTOR --> WEBHOOK
```

## 🏢 Multi-Tenant Architecture

### Organization Isolation

```mermaid
graph TB
    subgraph "Shared Infrastructure"
        API[API Gateway]
        AUTH[Authentication]
        DB[(Database)]
        REDIS[(Redis)]
    end

    subgraph "Organization A"
        DIMSET_A[Dimension Sets A]
        TYPES_A[Highlight Types A]
        STORAGE_A[S3 Bucket A]
        CONFIG_A[Processing Config A]
    end

    subgraph "Organization B"
        DIMSET_B[Dimension Sets B]
        TYPES_B[Highlight Types B]
        STORAGE_B[S3 Bucket B]
        CONFIG_B[Processing Config B]
    end

    API --> AUTH
    AUTH --> DB
    AUTH --> REDIS
    
    AUTH --> DIMSET_A
    AUTH --> DIMSET_B
    
    DIMSET_A --> STORAGE_A
    DIMSET_B --> STORAGE_B
```

**Key Isolation Mechanisms:**
- Database-level tenant isolation with organization_id filtering
- Separate S3 buckets per organization
- Tenant-specific dimension sets and highlight type registries
- Isolated processing configurations and quotas
- API key scoping to organizations

## 🚀 Scalability Patterns

### Horizontal Scaling
- **Stateless API**: Multiple API instances behind load balancer
- **Worker Scaling**: Multiple Celery workers for parallel processing
- **Database Scaling**: Read replicas and connection pooling
- **Cache Scaling**: Redis cluster for distributed caching

### Performance Optimizations
- **Async Processing**: Non-blocking I/O throughout the stack
- **Connection Pooling**: Database and Redis connection reuse
- **Caching Strategy**: Multi-layer caching (Redis, CDN)
- **Batch Processing**: Efficient bulk operations

### Resource Management
- **Queue Prioritization**: Different priority levels for customer tiers
- **Rate Limiting**: Per-organization API quotas
- **Resource Quotas**: Processing limits per subscription plan
- **Circuit Breakers**: Fault tolerance for external services

## 🔒 Security Architecture

### Authentication & Authorization
- **API Key Authentication**: Secure key-based access
- **JWT Tokens**: Stateless session management
- **Scoped Permissions**: Fine-grained access control
- **Organization Isolation**: Multi-tenant security

### Data Security
- **Encryption at Rest**: Database and file storage encryption
- **Encryption in Transit**: TLS for all communications
- **Webhook Security**: HMAC signature validation
- **Input Validation**: Comprehensive request validation

### Infrastructure Security
- **Network Isolation**: VPC and security groups
- **Secrets Management**: Encrypted configuration storage
- **Audit Logging**: Comprehensive activity tracking
- **Security Headers**: HTTP security headers

## 📊 Observability Architecture

### Monitoring Stack
- **Logfire Integration**: Comprehensive application monitoring
- **Distributed Tracing**: Request flow tracking
- **Custom Metrics**: Business and technical metrics
- **Health Checks**: Service availability monitoring

### Logging Strategy
- **Structured Logging**: JSON-formatted logs
- **Correlation IDs**: Request tracking across services
- **Log Aggregation**: Centralized log collection
- **Alerting**: Automated incident detection

## 🔧 Technology Stack

### Core Technologies
- **Python 3.11+**: Modern Python with async support
- **FastAPI**: High-performance web framework
- **SQLAlchemy 2.0**: Modern ORM with async support
- **Celery**: Distributed task queue
- **Redis**: Caching and message broker
- **PostgreSQL**: Primary data store

### AI & Media Processing
- **Google Gemini**: Multi-modal AI analysis
- **FFmpeg**: Video and audio processing
- **OpenCV**: Computer vision operations
- **scikit-learn**: Machine learning utilities

### Infrastructure & Deployment
- **Docker**: Containerization
- **AWS S3**: File storage
- **Logfire**: Observability platform
- **Alembic**: Database migrations

## 📈 Performance Characteristics

### Response Times (SLA)
- **API Endpoints**: < 200ms (95th percentile)
- **Stream Processing**: < 5 minutes for 30-minute stream
- **Webhook Delivery**: < 1 second
- **Highlight Detection**: < 2 minutes per highlight

### Throughput
- **Concurrent Streams**: 100+ simultaneous processing jobs
- **API Requests**: 10,000+ requests per minute
- **Highlight Generation**: 1,000+ highlights per hour
- **Storage Operations**: 50+ GB per hour

### Reliability
- **Uptime SLA**: 99.9% availability
- **Data Durability**: 99.999999999% (S3)
- **Processing Success Rate**: 99.5%
- **Webhook Delivery**: 99.9% success rate

## 🔮 Future Architecture Considerations

### Scalability Enhancements
- **Microservices Migration**: Breaking down monolith into focused services
- **Event Sourcing**: Audit trail and state reconstruction
- **CQRS**: Command-Query Responsibility Segregation
- **GraphQL**: Flexible API query interface

### AI/ML Improvements
- **Model Serving**: Dedicated ML model serving infrastructure
- **Real-Time Inference**: Sub-second AI analysis
- **Custom Models**: Organization-specific trained models
- **Edge Processing**: Distributed AI processing

### Infrastructure Evolution
- **Kubernetes**: Container orchestration
- **Service Mesh**: Advanced networking and security
- **Multi-Region**: Geographic distribution
- **Auto-Scaling**: Dynamic resource allocation

---

This architecture provides a solid foundation for enterprise-scale operations while maintaining flexibility for future growth and evolution.