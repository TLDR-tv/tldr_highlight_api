# Application Flows Documentation

This document provides a comprehensive overview of all the flows in the TL;DR Highlight API application. Each flow represents a complete user journey or system process from initiation to completion.

## Table of Contents

1. [User and Organization Management Flows](#user-and-organization-management-flows)
2. [Authentication and Authorization Flows](#authentication-and-authorization-flows)
3. [Stream Processing Flows](#stream-processing-flows)
4. [Highlight Detection and Analysis Flows](#highlight-detection-and-analysis-flows)
5. [Storage and Media Handling Flows](#storage-and-media-handling-flows)
6. [Webhook and Notification Flows](#webhook-and-notification-flows)
7. [Usage Tracking and Billing Flows](#usage-tracking-and-billing-flows)
8. [Error Handling and Recovery Flows](#error-handling-and-recovery-flows)
9. [Monitoring and Observability Flows](#monitoring-and-observability-flows)

---

## 1. User and Organization Management Flows

### 1.1 User Registration Flow

**Purpose**: Allow new enterprise customers to create accounts and organizations.

**Flow Steps**:
1. **API Request**: POST `/api/v1/auth/register`
   - Receives email, password, company name
   - Validates email format and uniqueness
   - Validates password strength requirements
   
2. **User Creation**:
   - Create User entity with hashed password
   - Generate email verification token
   - Store user in database via UserRepository
   
3. **Organization Setup**:
   - Create Organization entity linked to user
   - Set default plan type (STARTER)
   - Assign user as organization owner
   
4. **API Key Generation**:
   - Generate initial API key for authentication
   - Set default scopes based on plan
   - Store hashed key in database
   
5. **Welcome Workflow**:
   - Send welcome email with verification link
   - Create initial usage quotas
   - Initialize webhook endpoints (if provided)

**Success Response**: User object with organization details and API key

### 1.2 Organization Plan Upgrade Flow

**Purpose**: Enable organizations to upgrade their subscription plan.

**Flow Steps**:
1. **Authentication**: Verify user has organization admin permissions
2. **Plan Selection**: Validate target plan (STARTER → PROFESSIONAL → ENTERPRISE)
3. **Usage Check**: Verify current usage doesn't exceed new plan limits
4. **Payment Processing**: (External) Process payment via payment provider
5. **Plan Update**:
   - Update Organization entity with new plan
   - Adjust rate limits and quotas
   - Update concurrent stream limits
6. **Notification**: Send confirmation email and webhook event

---

## 2. Authentication and Authorization Flows

### 2.1 API Key Authentication Flow

**Purpose**: Authenticate API requests using API keys.

**Flow Steps**:
1. **Header Extraction**: Extract API key from `X-API-Key` header
2. **Key Validation**:
   - Hash provided key
   - Look up in APIKeyRepository
   - Verify key is active and not expired
3. **Organization Context**:
   - Load associated user and organization
   - Check organization is active
4. **Scope Verification**:
   - Extract requested operation (e.g., STREAMS_WRITE)
   - Verify API key has required scope
5. **Rate Limiting**:
   - Check Redis for current request count
   - Apply plan-based or custom rate limits
   - Return 429 if limit exceeded
6. **Context Injection**: Add user/org context to request

**Error Cases**: 401 (invalid key), 403 (insufficient scope), 429 (rate limited)

### 2.2 JWT Token Authentication Flow

**Purpose**: Authenticate web dashboard and admin panel access.

**Flow Steps**:
1. **Login Request**: POST `/api/v1/auth/login` with email/password
2. **Credential Verification**:
   - Look up user by email
   - Verify password hash
   - Check user is active
3. **Token Generation**:
   - Create JWT with user ID and role claims
   - Set appropriate expiration (30 min access, 7 day refresh)
   - Sign with secret key
4. **Token Refresh**:
   - Validate refresh token
   - Check user still active
   - Issue new access token
5. **Logout**: Invalidate refresh token in Redis blacklist

---

## 3. Stream Processing Flows

### 3.1 Stream Submission Flow

**Purpose**: Submit a livestream or video for highlight extraction.

**Flow Steps**:
1. **API Request**: POST `/api/v1/streams`
   ```json
   {
     "source_url": "https://example.com/stream.m3u8",
     "title": "My Stream",
     "processing_options": {...}
   }
   ```

2. **URL Validation**:
   - Parse URL using Url value object
   - Detect platform/protocol automatically
   - Verify URL is accessible (HEAD request)

3. **Stream Entity Creation**:
   - Create Stream with PENDING status
   - Assign to user's organization
   - Set processing options with defaults

4. **Async Job Queuing**:
   - Create Celery task with stream ID
   - Queue to appropriate priority queue
   - Return job ID for tracking

5. **Response**: Stream object with ID and status

### 3.2 Stream Processing Pipeline Flow

**Purpose**: Process submitted streams to extract highlights.

**Flow Steps**:
1. **Task Initialization** (Celery Worker):
   - Load stream from database
   - Update status to PROCESSING
   - Send STREAM_STARTED webhook

2. **Stream Adapter Selection**:
   - StreamAdapterFactory determines adapter type
   - FFmpegStreamAdapter handles any format
   - Initialize connection to stream source

3. **Media Ingestion**:
   - UnifiedIngestionPipeline manages data flow
   - Split stream into video segments for analysis
   - No frame extraction - segments passed directly to Gemini

4. **Gemini Video Analysis**:
   - Upload video segments to Gemini API
   - Gemini analyzes video with native understanding
   - Processes visual, audio, and speech content internally
   - Returns structured analysis with dimension scores

5. **B2B Agent Processing**:
   - B2BStreamAgent coordinates Gemini analysis
   - Apply organization's dimension set
   - Identify highlights based on dimension scores
   - Calculate confidence levels

6. **Highlight Creation**:
   - Create Highlight entities for each detection
   - Generate clips using FFmpeg
   - Create thumbnails
   - Upload to S3 storage

7. **Completion**:
   - Update stream status to COMPLETED
   - Calculate total duration
   - Send STREAM_COMPLETED webhook
   - Update usage records

### 3.3 Generic Stream Ingestion Flow

**Purpose**: Support any FFmpeg-compatible stream format.

**Flow Steps**:
1. **URL Analysis**:
   - Parse URL scheme (rtmp://, rtsp://, srt://, etc.)
   - Map to StreamPlatform enum
   - Default to CUSTOM for unknown formats

2. **FFmpeg Configuration**:
   - Build appropriate input parameters
   - Set codec preferences
   - Configure error handling

3. **Stream Connection**:
   - Attempt connection with retries
   - Handle authentication if needed
   - Verify stream is live/accessible

4. **Data Extraction**:
   - Read stream segments
   - Demux audio/video tracks
   - Handle format-specific metadata

---

## 4. Highlight Detection and Analysis Flows

### 4.1 AI-Powered Highlight Detection Flow

**Purpose**: Use Gemini's native video understanding to identify highlight moments.

**Flow Steps**:
1. **Segment Preparation**:
   - Split video into analysis segments (10-30 seconds)
   - Upload video segments to Gemini API
   - No frame extraction needed - Gemini analyzes video directly

2. **Dimension Scoring**:
   - Load organization's dimension set
   - Build comprehensive prompt with dimension definitions
   - Send video segment to Gemini's video understanding model
   - Gemini analyzes visual, audio, and contextual elements natively
   - Parse dimension scores from structured response

3. **Score Aggregation**:
   - Apply temporal smoothing
   - Weight scores by dimension importance
   - Calculate overall confidence

4. **Highlight Identification**:
   - Find peaks above threshold
   - Extend boundaries for context
   - Merge nearby highlights

5. **Type Classification**:
   - Match score patterns to highlight types
   - Apply rule-based type assignment
   - Prioritize by confidence

---

## 5. Storage and Media Handling Flows

### 5.1 Highlight Clip Generation Flow

**Purpose**: Create video clips from detected highlights.

**Flow Steps**:
1. **Clip Boundaries**:
   - Calculate start/end with context padding
   - Ensure segment boundaries are clean
   - Validate against stream duration

2. **FFmpeg Processing**:
   - Extract segment from source
   - Transcode to standard format (H.264/AAC)
   - Apply quality settings
   - Add watermark if configured

3. **Thumbnail Generation**:
   - Extract frame at peak moment
   - Resize to standard dimensions
   - Optimize for web delivery

4. **S3 Upload**:
   - Generate unique S3 keys
   - Upload with appropriate ACLs
   - Set CDN cache headers
   - Generate signed URLs

5. **Database Update**:
   - Store S3 URLs in Highlight entity
   - Update clip metadata
   - Mark as ready for delivery

### 5.2 Storage Lifecycle Management Flow

**Purpose**: Manage storage costs and retention.

**Flow Steps**:
1. **Usage Monitoring**:
   - Track storage per organization
   - Monitor bandwidth usage
   - Calculate costs

2. **Retention Policy**:
   - Apply plan-based retention periods
   - Move old clips to glacier storage
   - Delete expired content

3. **Quota Enforcement**:
   - Check storage limits before upload
   - Send warnings at 80% usage
   - Block uploads when exceeded

---

## 6. Webhook and Notification Flows

### 6.1 Webhook Registration Flow

**Purpose**: Configure webhook endpoints for event notifications.

**Flow Steps**:
1. **Endpoint Configuration**:
   - POST `/api/v1/webhooks`
   - Validate URL is HTTPS
   - Select event types to subscribe

2. **Verification**:
   - Generate webhook secret
   - Send test event with signature
   - Verify endpoint responds correctly

3. **Storage**:
   - Create Webhook entity
   - Store encrypted secret
   - Set to ACTIVE status

### 6.2 Webhook Delivery Flow

**Purpose**: Deliver events to configured webhooks.

**Flow Steps**:
1. **Event Trigger**:
   - Stream started/completed
   - Highlight detected
   - Error occurred

2. **Webhook Loading**:
   - Query active webhooks for event type
   - Load delivery configuration

3. **Payload Construction**:
   - Build event-specific payload
   - Add metadata and timestamps
   - Sign with webhook secret

4. **Delivery Attempt**:
   - POST to webhook URL
   - Include signature header
   - Set appropriate timeout

5. **Retry Logic**:
   - Exponential backoff on failure
   - Maximum 3 retry attempts
   - Disable webhook after repeated failures

6. **Delivery Tracking**:
   - Log response status and time
   - Update delivery statistics
   - Store for debugging

---

## 7. Usage Tracking and Billing Flows

### 7.1 Usage Recording Flow

**Purpose**: Track resource consumption for billing.

**Flow Steps**:
1. **Consumption Events**:
   - Stream processing minutes
   - Storage usage (GB)
   - API calls
   - Bandwidth usage

2. **Real-time Tracking**:
   - Create UsageRecord entities
   - Update counters in Redis
   - Aggregate by time period

3. **Quota Checking**:
   - Compare against plan limits
   - Send warnings near limits
   - Enforce hard limits

4. **Billing Integration**:
   - Export usage to billing system
   - Calculate costs by tier
   - Generate invoices

### 7.2 Usage Analytics Flow

**Purpose**: Provide usage insights to customers.

**Flow Steps**:
1. **Data Aggregation**:
   - Query usage records by period
   - Group by resource type
   - Calculate trends

2. **Report Generation**:
   - Daily/weekly/monthly summaries
   - Cost breakdowns
   - Highlight statistics

3. **API Access**:
   - GET `/api/v1/usage/summary`
   - Filterable by date range
   - Exportable formats (CSV, JSON)

---

## 8. Error Handling and Recovery Flows

### 8.1 Stream Processing Error Recovery Flow

**Purpose**: Handle and recover from processing failures.

**Flow Steps**:
1. **Error Detection**:
   - Catch exceptions in processing pipeline
   - Categorize error type
   - Log with full context

2. **Error Classification**:
   - **Transient**: Network issues, temporary unavailability
   - **Permanent**: Invalid format, authentication failure
   - **Resource**: Out of memory, quota exceeded

3. **Recovery Strategy**:
   - **Transient Errors**:
     - Retry with exponential backoff
     - Maximum 3 attempts
     - Preserve partial results
   - **Permanent Errors**:
     - Mark stream as FAILED
     - Send failure webhook
     - Provide detailed error message
   - **Resource Errors**:
     - Queue for later retry
     - Adjust resource allocation
     - Alert operations team

4. **Partial Result Handling**:
   - Save any highlights detected before failure
   - Mark stream as PARTIALLY_COMPLETED
   - Allow manual retry from checkpoint

### 8.2 Circuit Breaker Flow

**Purpose**: Prevent cascading failures in external dependencies.

**Flow Steps**:
1. **Health Monitoring**:
   - Track success/failure rates
   - Monitor response times
   - Set thresholds (e.g., 50% failure rate)

2. **Circuit States**:
   - **CLOSED**: Normal operation
   - **OPEN**: Blocking all requests
   - **HALF_OPEN**: Testing recovery

3. **State Transitions**:
   - CLOSED → OPEN: After threshold failures
   - OPEN → HALF_OPEN: After cooldown period
   - HALF_OPEN → CLOSED: After successful test
   - HALF_OPEN → OPEN: If test fails

4. **Fallback Behavior**:
   - Return cached results if available
   - Use degraded mode
   - Queue for later processing

---

## 9. Monitoring and Observability Flows

### 9.1 Request Tracing Flow

**Purpose**: Track requests through the entire system.

**Flow Steps**:
1. **Trace Initiation**:
   - Generate trace ID at API gateway
   - Add to request headers
   - Initialize span

2. **Span Creation**:
   - Create child spans for each service
   - Add relevant metadata
   - Track timing information

3. **Context Propagation**:
   - Pass trace context to async tasks
   - Include in webhook deliveries
   - Add to log entries

4. **Trace Assembly**:
   - Collect spans in Logfire
   - Build request timeline
   - Identify bottlenecks

### 9.2 Performance Monitoring Flow

**Purpose**: Track system performance and health.

**Flow Steps**:
1. **Metric Collection**:
   - API response times
   - Database query performance
   - Queue depths
   - Resource utilization

2. **Aggregation**:
   - Calculate percentiles (p50, p95, p99)
   - Track error rates
   - Monitor throughput

3. **Alerting**:
   - Define SLO thresholds
   - Configure alert rules
   - Route to appropriate teams

4. **Dashboard Updates**:
   - Real-time metric display
   - Historical trending
   - Anomaly detection

---

## Flow Interactions and Dependencies

### Critical Path Dependencies

1. **Authentication → All Flows**: Every API request requires authentication
2. **User → Organization → Streams**: Hierarchical ownership model
3. **Stream → Highlights → Webhooks**: Event-driven notification chain
4. **Processing → Storage → Delivery**: Media handling pipeline

### Asynchronous Boundaries

- **API → Celery**: Stream processing is always async
- **Processing → Webhooks**: Delivered via separate tasks
- **Storage → CDN**: Eventually consistent delivery

### Resource Constraints

- **Concurrent Streams**: Limited by organization plan
- **Storage Quotas**: Enforced before upload
- **Rate Limits**: Applied per API key
- **Processing Priority**: Based on plan tier

---

## Best Practices for Flow Implementation

1. **Idempotency**: All flows should be idempotent where possible
2. **Partial Failure Handling**: Save progress and allow resumption
3. **Timeout Management**: Set appropriate timeouts at each step
4. **Error Context**: Include sufficient information for debugging
5. **Monitoring Coverage**: Instrument all critical paths
6. **Testing**: Each flow should have integration tests
7. **Documentation**: Keep flow documentation synchronized with code

---

## Conclusion

This document covers all major flows in the TL;DR Highlight API application. Each flow is designed to be:
- **Resilient**: Handle failures gracefully
- **Scalable**: Support concurrent operations
- **Observable**: Provide visibility into operations
- **Secure**: Enforce authentication and authorization
- **Efficient**: Optimize for performance and cost

For implementation details of specific flows, refer to the corresponding source code and integration tests.