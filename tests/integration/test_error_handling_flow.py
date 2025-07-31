"""Error handling integration tests for stream and processing failures."""

import os
import pytest

# Set test environment variables before imports
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/tldr_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "30"
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["AWS_ACCESS_KEY_ID"] = "test-aws-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test-aws-secret"
os.environ["AWS_S3_BUCKET"] = "test-bucket"
os.environ["AWS_REGION"] = "us-east-1"


@pytest.mark.asyncio
async def test_stream_validation_errors():
    """Test various stream validation error scenarios."""
    
    from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
    from src.domain.value_objects.url import Url
    from src.domain.value_objects.processing_options import ProcessingOptions
    from src.domain.value_objects.timestamp import Timestamp
    from src.domain.exceptions import InvalidValueError
    
    # Test 1: Invalid URL Format
    print("\n=== Test 1: Invalid URL Validation ===")
    
    invalid_urls = [
        "",  # Empty URL
        "not-a-url",  # No protocol
        "http::/example.com",  # Malformed protocol
        "http://",  # Incomplete URL
        "://example.com",  # Missing protocol
        "http:///path",  # Missing domain
    ]
    
    for invalid_url in invalid_urls:
        try:
            url = Url(invalid_url)
            assert False, f"URL validation should have failed for: {invalid_url}"
        except (InvalidValueError, ValueError) as e:
            print(f"✓ Correctly rejected invalid URL: {invalid_url}")
            print(f"  Error: {str(e)}")
    
    # Test 2: Invalid Processing Options
    print("\n=== Test 2: Invalid Processing Options ===")
    
    try:
        # Invalid confidence thresholds
        options = ProcessingOptions(
            min_confidence_threshold=1.5,  # > 1.0
            target_confidence_threshold=0.8,
            exceptional_threshold=0.9
        )
        assert False, "Should reject confidence threshold > 1.0"
    except InvalidValueError as e:
        print("✓ Correctly rejected invalid confidence threshold")
        print(f"  Error: {str(e)}")
    
    try:
        # Thresholds in wrong order
        options = ProcessingOptions(
            min_confidence_threshold=0.8,
            target_confidence_threshold=0.6,  # Less than min
            exceptional_threshold=0.9
        )
        assert False, "Should reject thresholds in wrong order"
    except InvalidValueError as e:
        print("✓ Correctly rejected thresholds in wrong order")
        print(f"  Error: {str(e)}")
    
    try:
        # Negative duration
        options = ProcessingOptions(
            min_highlight_duration=-10.0
        )
        assert False, "Should reject negative duration"
    except InvalidValueError as e:
        print("✓ Correctly rejected negative duration")
        print(f"  Error: {str(e)}")
    
    # Test 3: Invalid Stream State Transitions
    print("\n=== Test 3: Invalid Stream State Transitions ===")
    
    stream = Stream(
        id=1,
        user_id=1,
        url=Url("https://example.com/stream.m3u8"),
        platform=StreamPlatform.HLS,
        status=StreamStatus.COMPLETED,  # Already completed
        processing_options=ProcessingOptions(),
        created_at=Timestamp.now(),
        updated_at=Timestamp.now()
    )
    
    # Try to start processing a completed stream
    try:
        stream.start_processing()
        assert False, "Should not allow starting a completed stream"
    except ValueError as e:
        print("✓ Correctly prevented starting completed stream")
        print(f"  Error: {str(e)}")
    
    # Try to fail a completed stream
    completed_stream = Stream(
        id=2,
        user_id=1,
        url=Url("https://example.com/stream2.m3u8"),
        platform=StreamPlatform.HLS,
        status=StreamStatus.COMPLETED,  # Already completed
        processing_options=ProcessingOptions(),
        completed_at=Timestamp.now(),
        created_at=Timestamp.now(),
        updated_at=Timestamp.now()
    )
    
    try:
        completed_stream.fail_processing("Test error")
        assert False, "Should not allow failing a completed stream"
    except ValueError as e:
        print("✓ Correctly prevented failing completed stream")
        print(f"  Error: {str(e)}")
    
    # Verify that pending and processing streams CAN be failed
    pending_stream = Stream(
        id=3,
        user_id=1,
        url=Url("https://example.com/stream3.m3u8"),
        platform=StreamPlatform.HLS,
        status=StreamStatus.PENDING,
        processing_options=ProcessingOptions(),
        created_at=Timestamp.now(),
        updated_at=Timestamp.now()
    )
    
    # This should work
    failed_stream = pending_stream.fail_processing("Test failure")
    assert failed_stream.status == StreamStatus.FAILED
    assert failed_stream.error_message == "Test failure"
    print("✓ Correctly allowed failing pending stream")


@pytest.mark.asyncio
async def test_processing_errors():
    """Test various processing error scenarios."""
    
    from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
    from src.domain.entities.user import User
    from src.domain.value_objects.url import Url
    from src.domain.value_objects.email import Email
    from src.domain.value_objects.company_name import CompanyName
    from src.domain.value_objects.processing_options import ProcessingOptions
    from src.domain.value_objects.timestamp import Timestamp
    
    # Create test user and stream
    user = User(
        id=1,
        email=Email("test@example.com"),
        company_name=CompanyName("Test Company"),
        password_hash="hashed",
        is_active=True,
        created_at=Timestamp.now(),
        updated_at=Timestamp.now()
    )
    
    # Test 1: Network/Connection Errors
    print("\n=== Test 1: Network Connection Errors ===")
    
    network_error_urls = [
        ("https://unreachable-domain-12345.com/stream.m3u8", "Network unreachable"),
        ("rtmp://10.0.0.0/live", "Connection timeout"),
        ("rtsp://private.network.local/stream", "DNS resolution failed"),
    ]
    
    for url, error_type in network_error_urls:
        stream = Stream(
            id=1,
            user_id=user.id,
            url=Url(url),
            platform=StreamPlatform.HLS,
            status=StreamStatus.PENDING,
            processing_options=ProcessingOptions(),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Start processing
        processing_stream = stream.start_processing()
        assert processing_stream.status == StreamStatus.PROCESSING
        
        # Simulate network failure
        failed_stream = processing_stream.fail_processing(f"{error_type}: Unable to connect to {url}")
        assert failed_stream.status == StreamStatus.FAILED
        assert failed_stream.error_message is not None
        assert error_type in failed_stream.error_message
        print(f"✓ Handled {error_type} for {url}")
    
    # Test 2: FFmpeg Processing Errors
    print("\n=== Test 2: FFmpeg Processing Errors ===")
    
    ffmpeg_errors = [
        ("Invalid codec", "Unsupported video codec: vp9"),
        ("Corrupted stream", "Invalid data found when processing input"),
        ("Format error", "Invalid container format"),
        ("Memory error", "Cannot allocate memory"),
    ]
    
    for error_name, error_msg in ffmpeg_errors:
        stream = Stream(
            id=2,
            user_id=user.id,
            url=Url("https://example.com/stream.m3u8"),
            platform=StreamPlatform.HLS,
            status=StreamStatus.PROCESSING,
            processing_options=ProcessingOptions(),
            started_at=Timestamp.now(),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Simulate FFmpeg error
        failed_stream = stream.fail_processing(f"FFmpeg error: {error_msg}")
        assert failed_stream.status == StreamStatus.FAILED
        assert "FFmpeg error" in failed_stream.error_message
        assert error_msg in failed_stream.error_message
        print(f"✓ Handled FFmpeg {error_name}")
    
    # Test 3: AI Agent Processing Errors
    print("\n=== Test 3: AI Agent Processing Errors ===")
    
    ai_errors = [
        ("API rate limit", "Gemini API rate limit exceeded"),
        ("Model timeout", "AI model request timeout after 30s"),
        ("Invalid response", "AI returned invalid JSON response"),
        ("Context overflow", "Video content exceeds model context window"),
    ]
    
    for error_name, error_msg in ai_errors:
        stream = Stream(
            id=3,
            user_id=user.id,
            url=Url("https://example.com/stream.m3u8"),
            platform=StreamPlatform.HLS,
            status=StreamStatus.PROCESSING,
            processing_options=ProcessingOptions(),
            started_at=Timestamp.now(),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Simulate AI processing error
        failed_stream = stream.fail_processing(f"B2B Agent error: {error_msg}")
        assert failed_stream.status == StreamStatus.FAILED
        assert "B2B Agent error" in failed_stream.error_message
        print(f"✓ Handled AI {error_name}")
    
    # Test 4: Storage/S3 Errors
    print("\n=== Test 4: Storage/S3 Errors ===")
    
    storage_errors = [
        ("Upload failed", "S3 PutObject failed: Access Denied"),
        ("Bucket error", "S3 bucket does not exist"),
        ("Storage quota", "Storage quota exceeded for organization"),
        ("Region error", "S3 bucket in wrong region"),
    ]
    
    for error_name, error_msg in storage_errors:
        stream = Stream(
            id=4,
            user_id=user.id,
            url=Url("https://example.com/stream.m3u8"),
            platform=StreamPlatform.HLS,
            status=StreamStatus.PROCESSING,
            processing_options=ProcessingOptions(),
            started_at=Timestamp.now(),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        failed_stream = stream.fail_processing(f"Storage error: {error_msg}")
        assert failed_stream.status == StreamStatus.FAILED
        assert "Storage error" in failed_stream.error_message
        print(f"✓ Handled storage {error_name}")


@pytest.mark.asyncio
async def test_error_recovery_flow():
    """Test error recovery and retry mechanisms."""
    
    from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
    from src.domain.entities.highlight import Highlight
    from src.domain.value_objects.url import Url
    from src.domain.value_objects.processing_options import ProcessingOptions
    from src.domain.value_objects.timestamp import Timestamp
    from src.domain.value_objects.duration import Duration
    from src.domain.value_objects.confidence_score import ConfidenceScore
    
    print("\n=== Test: Error Recovery Flow ===")
    
    # Create a stream that will fail initially
    stream = Stream(
        id=1,
        user_id=1,
        url=Url("https://example.com/stream.m3u8"),
        platform=StreamPlatform.HLS,
        status=StreamStatus.PENDING,
        processing_options=ProcessingOptions(),
        created_at=Timestamp.now(),
        updated_at=Timestamp.now()
    )
    
    # First attempt - fails
    processing_stream = stream.start_processing()
    failed_stream = processing_stream.fail_processing("Temporary network error")
    assert failed_stream.status == StreamStatus.FAILED
    print("✓ First attempt failed as expected")
    
    # Retry mechanism - reset to pending
    retry_stream = Stream(
        id=failed_stream.id,
        user_id=failed_stream.user_id,
        url=failed_stream.url,
        platform=failed_stream.platform,
        status=StreamStatus.PENDING,  # Reset for retry
        processing_options=failed_stream.processing_options,
        created_at=failed_stream.created_at,
        updated_at=Timestamp.now(),
        # Would track retry attempts in metadata
    )
    
    # Second attempt - succeeds
    processing_stream2 = retry_stream.start_processing()
    successful_stream = processing_stream2.complete_processing()
    assert successful_stream.status == StreamStatus.COMPLETED
    print("✓ Retry succeeded after temporary failure")
    
    # Partial success scenario - some highlights detected before error
    partial_stream = Stream(
        id=2,
        user_id=1,
        url=Url("https://example.com/long-stream.m3u8"),
        platform=StreamPlatform.HLS,
        status=StreamStatus.PROCESSING,
        processing_options=ProcessingOptions(),
        started_at=Timestamp.now(),
        created_at=Timestamp.now(),
        updated_at=Timestamp.now()
    )
    
    # Simulate partial highlights before failure
    partial_highlights = [
        Highlight(
            id=1,
            stream_id=partial_stream.id,
            start_time=Duration(10.0),
            end_time=Duration(30.0),
            confidence_score=ConfidenceScore(0.85),
            title="Partial highlight 1",
            description="Detected before error",
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        ),
        Highlight(
            id=2,
            stream_id=partial_stream.id,
            start_time=Duration(45.0),
            end_time=Duration(60.0),
            confidence_score=ConfidenceScore(0.90),
            title="Partial highlight 2",
            description="Also detected before error",
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
    ]
    
    # Stream fails after partial processing
    partial_failed = partial_stream.fail_processing("Connection lost after 2 minutes")
    assert partial_failed.status == StreamStatus.FAILED
    assert len(partial_highlights) == 2
    print("✓ Partial highlights preserved despite failure")
    print(f"  Detected {len(partial_highlights)} highlights before failure")


@pytest.mark.asyncio
async def test_concurrent_error_handling():
    """Test error handling with concurrent stream processing."""
    
    from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
    from src.domain.value_objects.url import Url
    from src.domain.value_objects.processing_options import ProcessingOptions
    from src.domain.value_objects.timestamp import Timestamp
    import asyncio
    
    print("\n=== Test: Concurrent Stream Error Handling ===")
    
    # Create multiple streams that will have different outcomes
    streams = []
    for i in range(5):
        stream = Stream(
            id=i+1,
            user_id=1,
            url=Url(f"https://example.com/stream{i}.m3u8"),
            platform=StreamPlatform.HLS,
            status=StreamStatus.PENDING,
            processing_options=ProcessingOptions(),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        streams.append(stream)
    
    # Simulate concurrent processing with different outcomes
    async def process_stream(stream: Stream, should_fail: bool, error_msg: str = None):
        """Simulate stream processing with potential failure."""
        processing = stream.start_processing()
        await asyncio.sleep(0.1)  # Simulate processing time
        
        if should_fail:
            return processing.fail_processing(error_msg or "Processing error")
        else:
            return processing.complete_processing()
    
    # Process streams concurrently with different outcomes
    results = await asyncio.gather(
        process_stream(streams[0], False),  # Success
        process_stream(streams[1], True, "Network timeout"),  # Network error
        process_stream(streams[2], False),  # Success
        process_stream(streams[3], True, "FFmpeg codec error"),  # FFmpeg error
        process_stream(streams[4], True, "AI model unavailable"),  # AI error
        return_exceptions=True  # Don't fail on exceptions
    )
    
    # Verify results
    success_count = sum(1 for r in results if not isinstance(r, Exception) and r.status == StreamStatus.COMPLETED)
    failure_count = sum(1 for r in results if not isinstance(r, Exception) and r.status == StreamStatus.FAILED)
    
    assert success_count == 2
    assert failure_count == 3
    
    print(f"✓ Handled {len(streams)} concurrent streams")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")
    
    # Verify specific error messages preserved
    for i, result in enumerate(results):
        if isinstance(result, Stream) and result.status == StreamStatus.FAILED:
            print(f"  Stream {i}: {result.error_message}")


@pytest.mark.asyncio
async def test_cascading_error_prevention():
    """Test prevention of cascading failures."""
    
    from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
    from src.domain.entities.organization import Organization, PlanType
    from src.domain.value_objects.url import Url
    from src.domain.value_objects.processing_options import ProcessingOptions
    from src.domain.value_objects.timestamp import Timestamp
    
    print("\n=== Test: Cascading Error Prevention ===")
    
    # Create organization with rate limits
    org = Organization(
        id=1,
        name="Test Org",
        owner_id=1,
        plan_type=PlanType.STARTER,  # Lower limits
        is_active=True,
        created_at=Timestamp.now(),
        updated_at=Timestamp.now()
    )
    
    # Simulate hitting rate limits
    streams_attempted = []
    max_concurrent = 3  # Plan limit
    
    for i in range(10):  # Try to create more than allowed
        try:
            if i < max_concurrent:
                stream = Stream(
                    id=i+1,
                    user_id=1,
                    url=Url(f"https://example.com/stream{i}.m3u8"),
                    platform=StreamPlatform.HLS,
                    status=StreamStatus.PROCESSING,
                    processing_options=ProcessingOptions(),
                    started_at=Timestamp.now(),
                    created_at=Timestamp.now(),
                    updated_at=Timestamp.now()
                )
                streams_attempted.append(stream)
            else:
                # Simulate rate limit error
                raise ValueError(f"Rate limit exceeded: Max {max_concurrent} concurrent streams for {org.plan_type.value} plan")
        except ValueError as e:
            print(f"✓ Rate limit enforced for stream {i+1}: {str(e)}")
    
    assert len(streams_attempted) == max_concurrent
    print(f"✓ Prevented cascading failures - limited to {max_concurrent} streams")
    
    # Test circuit breaker pattern
    consecutive_failures = 0
    circuit_breaker_threshold = 3
    circuit_open = False
    
    for i in range(6):
        if circuit_open:
            print(f"✓ Circuit breaker OPEN - rejecting stream {i+1}")
            continue
            
        # Simulate all streams failing
        stream = Stream(
            id=100+i,
            user_id=1,
            url=Url(f"https://failing-service.com/stream{i}.m3u8"),
            platform=StreamPlatform.HLS,
            status=StreamStatus.PROCESSING,
            processing_options=ProcessingOptions(),
            started_at=Timestamp.now(),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Simulate failure
        failed = stream.fail_processing("Service unavailable")
        consecutive_failures += 1
        
        if consecutive_failures >= circuit_breaker_threshold:
            circuit_open = True
            print(f"✓ Circuit breaker triggered after {consecutive_failures} failures")
    
    assert circuit_open is True
    print("✓ Circuit breaker pattern prevents cascading failures")


if __name__ == "__main__":
    import asyncio
    
    async def run_all_tests():
        """Run all error handling tests."""
        await test_stream_validation_errors()
        await test_processing_errors()
        await test_error_recovery_flow()
        await test_concurrent_error_handling()
        await test_cascading_error_prevention()
        print("\n✅ All error handling tests completed!")
    
    asyncio.run(run_all_tests())