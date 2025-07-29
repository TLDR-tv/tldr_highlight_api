# TL;DR Highlight API - Technical Implementation Guide

## Project Structure

```
tldr_highlight_api/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── dependencies.py         # Dependency injection
│   │   ├── exceptions.py           # Custom exceptions
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py            # API key authentication
│   │   │   ├── rate_limit.py      # Rate limiting middleware
│   │   │   ├── logging.py         # Request/response logging
│   │   │   └── cors.py            # CORS configuration
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── streams.py          # Stream processing endpoints
│   │       ├── batch.py            # Batch processing endpoints
│   │       ├── highlights.py       # Highlight retrieval endpoints
│   │       ├── webhooks.py         # Webhook management
│   │       ├── api_keys.py         # API key management
│   │       └── usage.py            # Usage tracking endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Base SQLAlchemy models
│   │   ├── api_key.py             # API key models
│   │   ├── stream.py              # Stream processing models
│   │   ├── batch.py               # Batch processing models
│   │   ├── highlight.py           # Highlight models
│   │   ├── webhook.py             # Webhook models
│   │   └── usage.py               # Usage tracking models
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── api_key.py             # Pydantic schemas for API keys
│   │   ├── stream.py              # Stream request/response schemas
│   │   ├── batch.py               # Batch request/response schemas
│   │   ├── highlight.py           # Highlight schemas
│   │   ├── webhook.py             # Webhook schemas
│   │   └── common.py              # Common schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth.py                # Authentication service
│   │   ├── stream_processor.py    # Stream processing logic
│   │   ├── batch_processor.py     # Batch processing logic
│   │   ├── highlight_extractor.py # Highlight extraction service
│   │   ├── storage.py             # S3/Storage service
│   │   ├── webhook.py             # Webhook delivery service
│   │   └── usage_tracker.py       # Usage tracking service
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── tldrtv/
│   │   │   ├── __init__.py
│   │   │   ├── client.py          # TL;DR TV service client
│   │   │   ├── stream_adapter.py  # Stream processing adapter
│   │   │   └── models.py          # TL;DR TV model mappings
│   │   ├── platforms/
│   │   │   ├── __init__.py
│   │   │   ├── twitch.py          # Twitch integration
│   │   │   ├── youtube.py         # YouTube integration
│   │   │   └── rtmp.py            # Generic RTMP support
│   │   └── ai/
│   │       ├── __init__.py
│   │       ├── base.py             # Base AI provider
│   │       ├── premium.py          # Premium quality analysis
│   │       ├── standard.py         # Standard quality analysis
│   │       └── custom.py           # Custom model support
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── database.py            # Database connection
│   │   ├── cache.py               # Redis cache
│   │   ├── queue.py               # Task queue setup
│   │   └── security.py            # Security utilities
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── stream_tasks.py        # Celery tasks for streams
│   │   ├── batch_tasks.py         # Celery tasks for batch
│   │   └── webhook_tasks.py       # Webhook delivery tasks
│   └── utils/
│       ├── __init__.py
│       ├── validators.py          # Input validators
│       ├── pagination.py          # Pagination utilities
│       └── helpers.py             # General helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── e2e/                       # End-to-end tests
├── migrations/
│   └── alembic/                   # Database migrations
├── scripts/
│   ├── setup_db.py               # Database setup
│   └── seed_data.py              # Seed test data
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Core Implementation Details

### 1. FastAPI Application Setup

```python
# src/api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logfire

from .middleware.auth import APIKeyMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .routes import streams, batch, highlights, webhooks, api_keys, usage
from ..core.config import settings
from ..core.database import engine, Base
from ..core.cache import redis_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await redis_client.initialize()
    logfire.configure(service_name="tldr-highlight-api")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Shutdown
    await redis_client.close()
    await engine.dispose()

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="TL;DR Highlight API",
        description="AI-powered highlight extraction for livestreams and videos",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(APIKeyMiddleware)
    
    # Add logfire instrumentation
    logfire.instrument_fastapi(app)
    
    # Include routers
    app.include_router(streams.router, prefix="/v1/streams", tags=["streams"])
    app.include_router(batch.router, prefix="/v1/batch", tags=["batch"])
    app.include_router(highlights.router, prefix="/v1/highlights", tags=["highlights"])
    app.include_router(webhooks.router, prefix="/v1/webhooks", tags=["webhooks"])
    app.include_router(api_keys.router, prefix="/v1/api-keys", tags=["api-keys"])
    app.include_router(usage.router, prefix="/v1/usage", tags=["usage"])
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "tldr-highlight-api",
            "version": "1.0.0"
        }
    
    return app

app = create_app()
```

### 2. Authentication Middleware

```python
# src/api/middleware/auth.py
from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time

from ...services.auth import AuthService
from ...core.cache import redis_client

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication"""
    
    # Skip auth for these paths
    SKIP_PATHS = ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip authentication for certain paths
        if any(request.url.path.startswith(path) for path in self.SKIP_PATHS):
            return await call_next(request)
        
        # Extract API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")
        
        # Check cache first
        cache_key = f"api_key:{api_key}"
        cached_data = await redis_client.get(cache_key)
        
        if cached_data:
            api_key_data = cached_data
        else:
            # Validate API key
            auth_service = AuthService()
            api_key_data = await auth_service.validate_api_key(api_key)
            
            if not api_key_data:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Cache for 5 minutes
            await redis_client.setex(cache_key, 300, api_key_data)
        
        # Add API key data to request state
        request.state.api_key = api_key_data
        request.state.client_id = api_key_data["client_id"]
        
        # Track usage
        await self._track_request(request.state.client_id)
        
        # Process request
        response = await call_next(request)
        
        return response
    
    async def _track_request(self, client_id: str):
        """Track API request for usage"""
        usage_key = f"usage:{client_id}:{time.strftime('%Y-%m-%d')}"
        await redis_client.hincrby(usage_key, "requests", 1)
        await redis_client.expire(usage_key, 86400 * 31)  # Keep for 31 days
```

### 3. Stream Processing Implementation

```python
# src/schemas/stream.py
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

class AnalysisQuality(str, Enum):
    PREMIUM = "premium"
    HIGH = "high"
    STANDARD = "standard"
    FAST = "fast"
    CUSTOM = "custom"

class Sensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Platform(str, Enum):
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    RTMP = "rtmp"
    CUSTOM = "custom"

class StreamOptions(BaseModel):
    analysis_quality: AnalysisQuality = Field(default=AnalysisQuality.HIGH)
    sensitivity: Sensitivity = Field(default=Sensitivity.MEDIUM)
    clip_duration: int = Field(default=30, ge=10, le=120)
    include_chat: bool = Field(default=True)
    include_audio: bool = Field(default=True)
    custom_tags: Optional[List[str]] = Field(default=None)
    webhook_url: Optional[HttpUrl] = Field(default=None)
    frame_interval: float = Field(default=1.0, ge=0.1, le=10.0)
    max_highlights: Optional[int] = Field(default=None, ge=1, le=100)

class StreamCreateRequest(BaseModel):
    source_url: HttpUrl
    platform: Optional[Platform] = Field(default=None)
    options: StreamOptions = Field(default_factory=StreamOptions)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class StreamStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class StreamResponse(BaseModel):
    id: str
    status: StreamStatus
    source_url: str
    platform: Platform
    options: StreamOptions
    created_at: datetime
    updated_at: datetime
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    class Config:
        from_attributes = True
```

```python
# src/api/routes/streams.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from typing import List
import uuid

from ...schemas.stream import StreamCreateRequest, StreamResponse, StreamStatus
from ...services.stream_processor import StreamProcessorService
from ...services.webhook import WebhookService
from ...tasks.stream_tasks import process_stream_task
from ..dependencies import get_current_client, get_db

router = APIRouter()

@router.post("/", response_model=StreamResponse)
async def create_stream(
    request: StreamCreateRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(get_current_client),
    db = Depends(get_db)
):
    """Start processing a livestream"""
    # Generate stream ID
    stream_id = f"str_{uuid.uuid4().hex[:12]}"
    
    # Create stream record
    stream_service = StreamProcessorService(db)
    stream = await stream_service.create_stream(
        stream_id=stream_id,
        client_id=client_id,
        request=request
    )
    
    # Queue stream processing task
    process_stream_task.delay(stream_id, dict(request))
    
    # Schedule webhook if provided
    if request.options.webhook_url:
        webhook_service = WebhookService()
        await webhook_service.schedule_webhook(
            client_id=client_id,
            event_type="stream.started",
            url=str(request.options.webhook_url),
            payload=stream.dict()
        )
    
    return stream

@router.get("/{stream_id}", response_model=StreamResponse)
async def get_stream(
    stream_id: str,
    client_id: str = Depends(get_current_client),
    db = Depends(get_db)
):
    """Get stream processing status"""
    stream_service = StreamProcessorService(db)
    stream = await stream_service.get_stream(stream_id, client_id)
    
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return stream

@router.post("/{stream_id}/stop")
async def stop_stream(
    stream_id: str,
    client_id: str = Depends(get_current_client),
    db = Depends(get_db)
):
    """Stop stream processing"""
    stream_service = StreamProcessorService(db)
    success = await stream_service.stop_stream(stream_id, client_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return {"message": "Stream processing stopped"}

@router.get("/{stream_id}/highlights")
async def get_stream_highlights(
    stream_id: str,
    page: int = 1,
    per_page: int = 10,
    client_id: str = Depends(get_current_client),
    db = Depends(get_db)
):
    """Get highlights from a processed stream"""
    stream_service = StreamProcessorService(db)
    
    # Verify stream ownership
    stream = await stream_service.get_stream(stream_id, client_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Get highlights
    highlights = await stream_service.get_highlights(
        stream_id=stream_id,
        page=page,
        per_page=per_page
    )
    
    return highlights
```

### 4. TL;DR TV Integration

```python
# src/integrations/tldrtv/client.py
import asyncio
from typing import Dict, Any, Optional
import aiohttp
from common.message.model import StreamOnlineMessage, StreamOfflineMessage
from common.platform.twitch.api import TwitchApi
from stream_processing_service.llm import create_llm

class TLDRTVClient:
    """Client for integrating with TL;DR TV services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.rabbitmq_url = config["rabbitmq_url"]
        self.s3_bucket = config["s3_bucket"]
        self.twitch_api = TwitchApi(config=config["twitch"])
        self.analysis_engines = {}
        
    async def initialize(self):
        """Initialize analysis engines"""
        # Initialize quality-based analysis engines
        self.analysis_engines["premium"] = create_llm(
            model_id="premium-analysis",
            model_kwargs={"api_key": self.config["api_key"]}
        )
        
        self.analysis_engines["high"] = create_llm(
            model_id="high-quality-analysis",
            model_kwargs={"api_key": self.config["api_key"]}
        )
    
    async def start_stream_processing(
        self,
        stream_url: str,
        platform: str,
        options: Dict[str, Any]
    ) -> str:
        """Start processing a stream using TL;DR TV infrastructure"""
        # Parse stream URL to get channel info
        channel_info = await self._parse_stream_url(stream_url, platform)
        
        # Create stream online message
        message = StreamOnlineMessage(
            stream_id=channel_info["stream_id"],
            user_id=channel_info["user_id"],
            game_name=channel_info.get("game_name", "Unknown"),
            channel_name=channel_info["channel_name"],
            started_at=channel_info["started_at"],
            platform=platform
        )
        
        # Send to RabbitMQ for processing
        await self._send_to_queue("stream.online", message)
        
        return channel_info["stream_id"]
    
    async def stop_stream_processing(self, stream_id: str):
        """Stop processing a stream"""
        # Create stream offline message
        message = StreamOfflineMessage(
            stream_id=stream_id,
            user_id="",  # Will be looked up by stream_id
            platform="twitch"
        )
        
        # Send to RabbitMQ
        await self._send_to_queue("stream.offline", message)
    
    async def _parse_stream_url(self, url: str, platform: str) -> Dict[str, Any]:
        """Parse stream URL to extract channel information"""
        if platform == "twitch":
            # Extract username from URL
            username = url.split("/")[-1]
            
            # Get stream info from Twitch API
            stream_info = await self.twitch_api.get_stream_by_username(username)
            
            return {
                "stream_id": stream_info["id"],
                "user_id": stream_info["user_id"],
                "channel_name": username,
                "game_name": stream_info.get("game_name", "Unknown"),
                "started_at": stream_info["started_at"]
            }
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    async def _send_to_queue(self, routing_key: str, message: Any):
        """Send message to RabbitMQ"""
        # Implementation would use aio-pika or similar
        pass
```

```python
# src/integrations/tldrtv/stream_adapter.py
from typing import Dict, Any, List
import asyncio
from stream_processing_service.highlight_processor import HighlightProcessor
from stream_processing_service.clip_assembly import ClipAssembler

class StreamAdapter:
    """Adapter for TL;DR TV stream processing"""
    
    def __init__(self, tldrtv_client, storage_service):
        self.tldrtv_client = tldrtv_client
        self.storage_service = storage_service
        self.active_processors = {}
    
    async def process_stream(
        self,
        stream_id: str,
        options: Dict[str, Any]
    ):
        """Process stream using TL;DR TV infrastructure"""
        # Create processing queues
        highlight_queue = asyncio.Queue()
        clip_queue = asyncio.Queue()
        
        # Initialize processors based on options
        analysis_engine = self.tldrtv_client.analysis_engines[options.get("analysis_quality", "high")]
        
        # Create highlight processor
        highlight_processor = HighlightProcessor(
            llm=analysis_engine,
            highlight_queue=highlight_queue,
            clip_processing_queue=clip_queue,
            chat_message_dao=None,  # Will be injected
            batch_size=1,
            batch_timeout=60.0
        )
        
        # Store processor reference
        self.active_processors[stream_id] = {
            "highlight_processor": highlight_processor,
            "highlight_queue": highlight_queue,
            "clip_queue": clip_queue,
            "task": asyncio.create_task(highlight_processor.run())
        }
        
        # Monitor for highlights
        asyncio.create_task(self._monitor_highlights(stream_id))
    
    async def _monitor_highlights(self, stream_id: str):
        """Monitor and store highlights as they're generated"""
        processor_info = self.active_processors.get(stream_id)
        if not processor_info:
            return
        
        clip_queue = processor_info["clip_queue"]
        
        while stream_id in self.active_processors:
            try:
                # Wait for highlights with timeout
                clip_request = await asyncio.wait_for(
                    clip_queue.get(),
                    timeout=30.0
                )
                
                # Store highlight
                await self._store_highlight(stream_id, clip_request)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error monitoring highlights: {e}")
    
    async def _store_highlight(self, stream_id: str, clip_request):
        """Store highlight in our system"""
        # Upload to S3
        video_url = await self.storage_service.upload_highlight(
            stream_id=stream_id,
            clip_data=clip_request
        )
        
        # Store metadata in database
        highlight_data = {
            "stream_id": stream_id,
            "video_url": video_url,
            "timestamp": clip_request.start_time,
            "duration": clip_request.end_time - clip_request.start_time,
            "confidence_score": getattr(clip_request, "score", 0.5),
            "metadata": {
                "kind": clip_request.kind.value,
                "source_dir": clip_request.source_dir
            }
        }
        
        # Save to database (implementation depends on your models)
        # await self.db.save_highlight(highlight_data)
```

### 5. Celery Task Implementation

```python
# src/tasks/stream_tasks.py
from celery import Celery, Task
from typing import Dict, Any
import asyncio

from ..integrations.tldrtv.client import TLDRTVClient
from ..services.stream_processor import StreamProcessorService
from ..services.webhook import WebhookService
from ..core.config import settings

# Initialize Celery
celery_app = Celery(
    "tldr_highlight_api",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

class StreamTask(Task):
    """Base task with database connection"""
    _db = None
    _tldrtv_client = None
    
    @property
    def db(self):
        if self._db is None:
            from ..core.database import SessionLocal
            self._db = SessionLocal()
        return self._db
    
    @property
    def tldrtv_client(self):
        if self._tldrtv_client is None:
            self._tldrtv_client = TLDRTVClient({
                "rabbitmq_url": settings.RABBITMQ_URL,
                "s3_bucket": settings.S3_BUCKET,
                "twitch": {
                    "client_id": settings.TWITCH_CLIENT_ID,
                    "client_secret": settings.TWITCH_CLIENT_SECRET
                }
            })
        return self._tldrtv_client

@celery_app.task(base=StreamTask, bind=True)
def process_stream_task(self, stream_id: str, request_data: Dict[str, Any]):
    """Process a stream asynchronously"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Start stream processing
        loop.run_until_complete(
            self._process_stream(stream_id, request_data)
        )
    except Exception as e:
        # Update stream status to error
        loop.run_until_complete(
            self._handle_error(stream_id, str(e))
        )
        raise
    finally:
        loop.close()

async def _process_stream(self, stream_id: str, request_data: Dict[str, Any]):
    """Async stream processing logic"""
    stream_service = StreamProcessorService(self.db)
    webhook_service = WebhookService()
    
    try:
        # Update status to processing
        await stream_service.update_status(stream_id, "processing")
        
        # Start TL;DR TV processing
        tldrtv_stream_id = await self.tldrtv_client.start_stream_processing(
            stream_url=request_data["source_url"],
            platform=request_data["platform"],
            options=request_data["options"]
        )
        
        # Store TL;DR TV stream ID mapping
        await stream_service.update_external_id(stream_id, tldrtv_stream_id)
        
        # Monitor stream until completion
        await self._monitor_stream(stream_id, tldrtv_stream_id)
        
        # Update status to completed
        await stream_service.update_status(stream_id, "completed")
        
        # Send completion webhook
        if request_data["options"].get("webhook_url"):
            stream = await stream_service.get_stream(stream_id)
            await webhook_service.send_webhook(
                url=request_data["options"]["webhook_url"],
                event_type="stream.completed",
                payload=stream.dict()
            )
            
    except Exception as e:
        await self._handle_error(stream_id, str(e))
        raise

async def _monitor_stream(self, stream_id: str, tldrtv_stream_id: str):
    """Monitor stream processing progress"""
    # This would poll TL;DR TV for status updates
    # and update our database accordingly
    pass

async def _handle_error(self, stream_id: str, error: str):
    """Handle processing errors"""
    stream_service = StreamProcessorService(self.db)
    await stream_service.update_status(stream_id, "error", error=error)
```

### 6. Rate Limiting Implementation

```python
# src/api/middleware/rate_limit.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time

from ...core.cache import redis_client
from ...core.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Get client ID from request state (set by auth middleware)
        client_id = getattr(request.state, "client_id", None)
        if not client_id:
            return await call_next(request)
        
        # Get rate limit for client
        api_key_data = getattr(request.state, "api_key", {})
        rate_limit = api_key_data.get("rate_limit", settings.DEFAULT_RATE_LIMIT)
        
        # Check rate limit
        is_allowed = await self._check_rate_limit(client_id, rate_limit)
        
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + 3600)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_id, rate_limit)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 3600)
        
        return response
    
    async def _check_rate_limit(self, client_id: str, limit: int) -> bool:
        """Check if request is within rate limit"""
        key = f"rate_limit:{client_id}:{int(time.time() // 3600)}"
        
        # Increment counter
        current = await redis_client.incr(key)
        
        # Set expiry on first request
        if current == 1:
            await redis_client.expire(key, 3600)
        
        return current <= limit
    
    async def _get_remaining_requests(self, client_id: str, limit: int) -> int:
        """Get remaining requests in current window"""
        key = f"rate_limit:{client_id}:{int(time.time() // 3600)}"
        current = await redis_client.get(key) or 0
        return max(0, limit - int(current))
```

### 7. Database Models

```python
# src/models/stream.py
from sqlalchemy import Column, String, DateTime, JSON, Enum, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from .base import Base

class StreamStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class Stream(Base):
    __tablename__ = "streams"
    
    id = Column(String(50), primary_key=True)
    client_id = Column(String(50), ForeignKey("clients.id"), nullable=False)
    external_id = Column(String(100), nullable=True)  # TL;DR TV stream ID
    source_url = Column(String(500), nullable=False)
    platform = Column(String(50), nullable=False)
    status = Column(Enum(StreamStatus), default=StreamStatus.PENDING)
    options = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=True)
    error = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="streams")
    highlights = relationship("Highlight", back_populates="stream", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_stream_client_created", "client_id", "created_at"),
        Index("idx_stream_status", "status"),
        Index("idx_stream_external", "external_id"),
    )
```

```python
# src/models/highlight.py
from sqlalchemy import Column, String, DateTime, Float, Integer, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import Base

class Highlight(Base):
    __tablename__ = "highlights"
    
    id = Column(String(50), primary_key=True)
    stream_id = Column(String(50), ForeignKey("streams.id"), nullable=False)
    title = Column(String(500), nullable=True)
    description = Column(String(2000), nullable=True)
    video_url = Column(String(500), nullable=False)
    thumbnail_url = Column(String(500), nullable=True)
    duration = Column(Integer, nullable=False)  # seconds
    timestamp = Column(Integer, nullable=False)  # seconds from stream start
    confidence_score = Column(Float, nullable=False)
    tags = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stream = relationship("Stream", back_populates="highlights")
    
    # Indexes
    __table_args__ = (
        Index("idx_highlight_stream", "stream_id"),
        Index("idx_highlight_score", "confidence_score"),
        Index("idx_highlight_created", "created_at"),
    )
```

### 8. Configuration Management

```python
# src/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "TL;DR Highlight API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_PREFIX: str = "/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 0
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Authentication
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT: int = 1000  # requests per hour
    
    # Storage
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY_ID: str
    S3_SECRET_ACCESS_KEY: str
    S3_BUCKET_NAME: str
    S3_REGION: str = "us-east-1"
    
    # TL;DR TV Integration
    RABBITMQ_URL: str
    TWITCH_CLIENT_ID: str
    TWITCH_CLIENT_SECRET: str
    
    # Analysis Engine
    ANALYSIS_API_KEY: Optional[str] = None
    ANALYSIS_ENDPOINT: Optional[str] = None
    
    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # Monitoring
    LOGFIRE_TOKEN: Optional[str] = None
    SENTRY_DSN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

### 9. Docker Configuration

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
RUN uv pip install -r requirements.txt

# Copy application code
COPY src/ src/
COPY migrations/ migrations/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/tldr_highlight
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
    depends_on:
      - postgres
      - redis
      - rabbitmq
    volumes:
      - ../src:/app/src
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

  celery:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/tldr_highlight
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    command: celery -A src.tasks.stream_tasks worker --loglevel=info

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=tldr_highlight
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"

volumes:
  postgres_data:
```

### 10. Testing Setup

```python
# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.core.database import Base, get_db
from src.core.config import settings

# Test database
TEST_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(engine):
    """Create a new database session for each test"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture(scope="function")
def client(db_session):
    """Create test client with database override"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

@pytest.fixture
def api_key():
    """Create test API key"""
    return "test_api_key_123"

@pytest.fixture
def auth_headers(api_key):
    """Create authentication headers"""
    return {"X-API-Key": api_key}
```

```python
# tests/unit/test_stream_api.py
import pytest
from unittest.mock import patch, MagicMock

def test_create_stream(client, auth_headers):
    """Test stream creation endpoint"""
    request_data = {
        "source_url": "https://twitch.tv/teststream",
        "platform": "twitch",
        "options": {
            "analysis_quality": "high",
            "sensitivity": "medium",
            "clip_duration": 30
        }
    }
    
    with patch('src.tasks.stream_tasks.process_stream_task.delay') as mock_task:
        response = client.post(
            "/v1/streams",
            json=request_data,
            headers=auth_headers
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["source_url"] == request_data["source_url"]
    assert mock_task.called

def test_get_stream_status(client, auth_headers):
    """Test getting stream status"""
    stream_id = "str_test123"
    
    # Mock the stream service
    with patch('src.services.stream_processor.StreamProcessorService.get_stream') as mock_get:
        mock_get.return_value = {
            "id": stream_id,
            "status": "processing",
            "progress": {"processed_duration": 300, "highlights_found": 3}
        }
        
        response = client.get(
            f"/v1/streams/{stream_id}",
            headers=auth_headers
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["progress"]["highlights_found"] == 3

def test_rate_limiting(client, auth_headers):
    """Test rate limiting"""
    # Make many requests quickly
    responses = []
    for _ in range(10):
        response = client.get("/v1/streams", headers=auth_headers)
        responses.append(response)
    
    # Check that we get rate limited eventually
    # (This is a simplified test; real implementation would use Redis)
    assert any(r.status_code == 429 for r in responses)
```

This technical implementation guide provides a complete foundation for building the TL;DR Highlight API. The architecture is designed to be:

1. **Scalable** - Using Celery for async processing and Redis for caching
2. **Reliable** - With proper error handling and retry mechanisms  
3. **Extensible** - Easy to add new AI models and platforms
4. **Enterprise-ready** - With authentication, rate limiting, and monitoring
5. **Well-integrated** - Leveraging the existing TL;DR TV infrastructure

The implementation focuses on making it as easy as possible for enterprise clients to integrate while maintaining the robustness and features they expect from a production API.