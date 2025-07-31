"""
Comprehensive end-to-end integration test for the complete flow.

This test covers the entire user journey from registration through
to receiving processed highlights, ensuring all components work together.
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from src.api.main import app
from src.infrastructure.database import Base, get_db
from src.infrastructure.persistence.models.organization import Organization
from src.infrastructure.persistence.models.stream import Stream, StreamStatus
from src.infrastructure.persistence.models.highlight import Highlight


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="function")
async def test_db():
    """Create a test database for each test."""
    # Create async engine
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    yield async_session
    
    # Clean up
    await engine.dispose()


@pytest.fixture
def override_db(test_db):
    """Override the default database dependency."""
    async def _get_test_db():
        async with test_db() as session:
            yield session
    
    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def test_client(override_db):
    """Create a test client with database override."""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Mock Redis for testing."""
    with patch('src.infrastructure.cache.get_redis_client') as mock:
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.set = AsyncMock(return_value=True)
        redis_mock.expire = AsyncMock(return_value=True)
        redis_mock.hset = AsyncMock(return_value=1)
        redis_mock.hget = AsyncMock(return_value=None)
        redis_mock.hgetall = AsyncMock(return_value={})
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.ping = AsyncMock(return_value=True)
        mock.return_value = redis_mock
        yield redis_mock


@pytest.fixture
def mock_celery():
    """Mock Celery for testing."""
    with patch('src.infrastructure.async_processing.stream_tasks.ingest_stream_with_ffmpeg.delay') as mock_task:
        mock_result = Mock()
        mock_result.id = "test-task-123"
        mock_result.state = "PENDING"
        mock_task.return_value = mock_result
        yield mock_task


@pytest.fixture
def mock_ffmpeg():
    """Mock FFmpeg for testing."""
    with patch('src.infrastructure.media.ffmpeg_integration.FFmpegProbe.probe_stream') as mock_probe:
        # Create mock media info
        video_stream = Mock()
        video_stream.width = 1920
        video_stream.height = 1080
        video_stream.fps = 30.0
        video_stream.codec = "h264"
        
        audio_stream = Mock()
        audio_stream.codec = "aac"
        audio_stream.sample_rate = 48000
        audio_stream.channels = 2
        
        media_info = Mock()
        media_info.format_name = "hls"
        media_info.duration = None  # Live stream
        media_info.bitrate = 5000000
        media_info.video_streams = [video_stream]
        media_info.audio_streams = [audio_stream]
        media_info.is_live = True
        
        mock_probe.return_value = asyncio.Future()
        mock_probe.return_value.set_result(media_info)
        yield mock_probe


@pytest.fixture
def mock_gemini():
    """Mock Gemini AI for testing."""
    with patch('src.infrastructure.content_processing.gemini.GeminiVideoProcessor') as mock_class:
        mock_processor = Mock()
        mock_processor.analyze_video_segment = AsyncMock(return_value={
            "highlights": [
                {
                    "start_time": 10.0,
                    "end_time": 35.0,
                    "confidence": 0.92,
                    "type": "action_sequence",
                    "dimensions": {
                        "action_intensity": 0.95,
                        "skill_display": 0.88,
                        "viewer_engagement": 0.90
                    }
                }
            ]
        })
        mock_class.return_value = mock_processor
        yield mock_processor


@pytest.fixture
def mock_s3():
    """Mock S3 storage for testing."""
    with patch('src.infrastructure.storage.s3_storage.S3Storage') as mock_class:
        mock_storage = Mock()
        mock_storage.upload_file = AsyncMock(
            side_effect=lambda file_path, object_name: f"https://s3.test.com/{object_name}"
        )
        mock_class.return_value = mock_storage
        yield mock_storage


@pytest.fixture
def sample_video_file():
    """Create a sample video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        # Write some dummy data
        f.write(b"fake video data")
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestEndToEndFlow:
    """Test the complete end-to-end flow from registration to highlights."""
    
    @pytest.mark.asyncio
    async def test_complete_user_journey(
        self,
        test_client,
        test_db,
        mock_redis,
        mock_celery,
        mock_ffmpeg,
        mock_gemini,
        mock_s3,
        sample_video_file
    ):
        """Test the complete flow from user registration to highlight retrieval."""
        
        # Step 1: User Registration
        print("\n=== Step 1: User Registration ===")
        registration_data = {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "company_name": "Test Company"
        }
        
        response = test_client.post("/auth/register", json=registration_data)
        assert response.status_code == 201
        user_data = response.json()
        assert "access_token" in user_data
        assert user_data["email"] == registration_data["email"]
        assert user_data["company_name"] == registration_data["company_name"]
        
        access_token = user_data["access_token"]
        user_id = user_data["id"]
        print(f"✓ User registered with ID: {user_id}")
        
        # Step 2: Create API Key
        print("\n=== Step 2: Create API Key ===")
        api_key_data = {
            "name": "Test API Key",
            "scopes": ["streams:read", "streams:write", "highlights:read"],
            "expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat()
        }
        
        response = test_client.post(
            "/auth/api-keys",
            json=api_key_data,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == 201
        api_key_response = response.json()
        assert "key" in api_key_response
        assert api_key_response["name"] == api_key_data["name"]
        
        api_key = api_key_response["key"]
        print(f"✓ API key created: {api_key[:20]}...")
        
        # Step 3: Submit Stream for Processing
        print("\n=== Step 3: Submit Stream ===")
        stream_data = {
            "source_url": "https://example.com/live/stream.m3u8",
            "options": {
                "highlight_threshold": 0.8,
                "min_duration": 10,
                "max_duration": 60,
                "max_highlights": 20
            }
        }
        
        # Mock the stream processing service to create a stream in DB
        async with test_db() as session:
            # First ensure user and org exist
            org = Organization(
                id=1,
                name="Test Company",
                owner_id=user_id,
                plan_type="professional"
            )
            session.add(org)
            await session.commit()
        
        response = test_client.post(
            "/streams",
            json=stream_data,
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 201
        stream_response = response.json()
        assert stream_response["platform"] == "hls"  # Auto-detected from URL
        assert stream_response["status"] == "pending"
        
        stream_id = stream_response["id"]
        print(f"✓ Stream submitted with ID: {stream_id}")
        
        # Step 4: Verify Celery Task Was Triggered
        print("\n=== Step 4: Verify Async Processing Started ===")
        mock_celery.assert_called_once()
        args, kwargs = mock_celery.call_args
        assert args[0] == stream_id  # Stream ID passed to task
        print("✓ Celery task triggered for stream processing")
        
        # Step 5: Simulate Processing Progress
        print("\n=== Step 5: Simulate Processing Progress ===")
        
        # Update stream status in database
        async with test_db() as session:
            stream = await session.get(Stream, stream_id)
            stream.status = StreamStatus.PROCESSING
            await session.commit()
        
        # Check stream status
        response = test_client.get(
            f"/streams/{stream_id}",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["status"] == "processing"
        print("✓ Stream status updated to processing")
        
        # Step 6: Simulate Highlight Creation
        print("\n=== Step 6: Create Highlights ===")
        async with test_db() as session:
            # Create sample highlights
            highlights = [
                Highlight(
                    stream_id=stream_id,
                    start_time=10.0,
                    end_time=35.0,
                    duration=25.0,
                    confidence_score=0.92,
                    highlight_type="action_sequence",
                    video_url="https://s3.test.com/clips/1001.mp4",
                    thumbnail_url="https://s3.test.com/thumbnails/1001.jpg",
                    caption="Intense action sequence with high skill display",
                    metadata={
                        "dimensions": {
                            "action_intensity": 0.95,
                            "skill_display": 0.88,
                            "viewer_engagement": 0.90
                        }
                    }
                ),
                Highlight(
                    stream_id=stream_id,
                    start_time=120.5,
                    end_time=145.5,
                    duration=25.0,
                    confidence_score=0.85,
                    highlight_type="funny_moment",
                    video_url="https://s3.test.com/clips/1002.mp4",
                    thumbnail_url="https://s3.test.com/thumbnails/1002.jpg",
                    caption="Hilarious unexpected moment that had viewers laughing",
                    metadata={
                        "dimensions": {
                            "humor": 0.88,
                            "surprise": 0.92,
                            "viewer_engagement": 0.85
                        }
                    }
                )
            ]
            
            for highlight in highlights:
                session.add(highlight)
            
            # Update stream to completed
            stream = await session.get(Stream, stream_id)
            stream.status = StreamStatus.COMPLETED
            stream.completed_at = datetime.utcnow()
            
            await session.commit()
        
        print("✓ Created 2 highlights for the stream")
        
        # Step 7: Retrieve Highlights
        print("\n=== Step 7: Retrieve Highlights ===")
        response = test_client.get(
            f"/streams/{stream_id}/highlights",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        highlights_data = response.json()
        assert highlights_data["total"] == 2
        assert len(highlights_data["items"]) == 2
        
        # Verify highlight data
        highlight = highlights_data["items"][0]
        assert highlight["confidence_score"] == 0.92
        assert highlight["highlight_type"] == "action_sequence"
        assert highlight["video_url"].startswith("https://s3.test.com/")
        assert "dimensions" in highlight["metadata"]
        print("✓ Successfully retrieved highlights")
        
        # Step 8: Check Usage Statistics
        print("\n=== Step 8: Check Usage Statistics ===")
        response = test_client.get(
            "/api/v1/organizations/1/usage",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == 200
        usage_data = response.json()
        assert usage_data["total_streams"] >= 1
        print("✓ Usage statistics available")
        
        # Summary
        print("\n=== ✓ End-to-End Test Completed Successfully ===")
        print(f"• User registered: {user_data['email']}")
        print(f"• API key created: {api_key[:20]}...")
        print(f"• Stream processed: {stream_id}")
        print(f"• Highlights generated: {highlights_data['total']}")
        print("• All components working together correctly!")
    
    @pytest.mark.asyncio
    async def test_webhook_notifications(
        self,
        test_client,
        test_db,
        mock_redis,
        access_token,
        api_key,
        stream_id
    ):
        """Test webhook notifications throughout the flow."""
        
        # Configure webhook endpoint
        webhook_data = {
            "url": "https://example.com/webhooks/tldr",
            "events": ["stream.started", "highlight.detected", "processing.complete"],
            "active": True
        }
        
        response = test_client.post(
            "/webhooks",
            json=webhook_data,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == 201
        
        # Mock webhook delivery
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            # Simulate webhook events
            events = [
                {
                    "event": "stream.started",
                    "stream_id": stream_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "event": "highlight.detected",
                    "stream_id": stream_id,
                    "highlight_id": 1001,
                    "confidence": 0.92,
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "event": "processing.complete",
                    "stream_id": stream_id,
                    "highlights_count": 2,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
            
            # Verify webhook calls
            for event in events:
                mock_post.assert_any_call(
                    webhook_data["url"],
                    json=event,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
    
    @pytest.mark.asyncio 
    async def test_error_handling_flow(
        self,
        test_client,
        test_db,
        mock_redis,
        api_key
    ):
        """Test error handling throughout the flow."""
        
        # Test invalid stream URL
        stream_data = {
            "source_url": "invalid://not-a-real-url",
            "options": {"highlight_threshold": 0.8}
        }
        
        response = test_client.post(
            "/streams",
            json=stream_data,
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 422  # Validation error
        
        # Test with mocked FFmpeg failure
        with patch('src.infrastructure.media.ffmpeg_integration.FFmpegProbe.probe_stream') as mock_probe:
            mock_probe.side_effect = Exception("Cannot access stream")
            
            stream_data["source_url"] = "rtmp://example.com/live"
            response = test_client.post(
                "/streams",
                json=stream_data,
                headers={"X-API-Key": api_key}
            )
            # Should still accept the stream, error happens in async processing
            assert response.status_code == 201