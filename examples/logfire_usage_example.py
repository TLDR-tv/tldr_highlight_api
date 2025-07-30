"""Example demonstrating Logfire observability integration in TL;DR Highlight API.

This example shows how to use the Logfire integration for:
- Distributed tracing
- Custom metrics
- Structured logging
- Performance monitoring
"""

import asyncio
import time
from typing import Dict, Any

import logfire
from src.infrastructure.observability import (
    traced,
    timed,
    with_span,
    metrics,
    traced_service_method,
    traced_repository_method,
)


# Example 1: Using decorators for automatic tracing
@traced(name="process_video_highlight", capture_args=True, capture_result=True)
async def process_video_highlight(video_id: str, platform: str) -> Dict[str, Any]:
    """Example function showing automatic tracing with decorators."""
    
    # Add custom attributes to the current span
    logfire.set_attribute("video.id", video_id)
    logfire.set_attribute("video.platform", platform)
    
    # Simulate some processing
    await asyncio.sleep(0.1)
    
    # Track business metrics
    metrics.increment_highlights_detected(
        count=3,
        platform=platform,
        organization_id="org_123",
        detection_method="ai_multimodal"
    )
    
    return {
        "video_id": video_id,
        "highlights_found": 3,
        "processing_time": 0.1
    }


# Example 2: Using timing decorator for performance monitoring
@timed(
    name="extract_video_frames",
    metric_name="video.frame_extraction.duration",
    platform="youtube"
)
async def extract_video_frames(video_url: str) -> int:
    """Example showing timing decorator for performance metrics."""
    
    # Log the start of frame extraction
    logfire.info(
        "Starting frame extraction",
        video_url=video_url,
        extraction_method="ffmpeg"
    )
    
    # Simulate frame extraction
    await asyncio.sleep(0.2)
    frame_count = 150
    
    # Record custom metric
    metrics.record_highlight_processing_time(
        duration_seconds=0.2,
        stage="frame_extraction",
        platform="youtube"
    )
    
    return frame_count


# Example 3: Using context manager for custom spans
async def analyze_stream_sentiment(stream_id: str, chat_messages: list) -> float:
    """Example using context manager for creating custom spans."""
    
    with with_span(
        "sentiment_analysis",
        stream_id=stream_id,
        message_count=len(chat_messages)
    ):
        # Log structured data
        logfire.info(
            "Analyzing chat sentiment",
            stream_id=stream_id,
            message_count=len(chat_messages)
        )
        
        # Simulate sentiment analysis
        total_sentiment = 0.0
        
        for i, message in enumerate(chat_messages):
            with with_span(
                "analyze_message",
                message_index=i,
                message_length=len(message)
            ):
                # Simulate processing
                await asyncio.sleep(0.01)
                sentiment = 0.7  # Mock sentiment score
                total_sentiment += sentiment
        
        avg_sentiment = total_sentiment / len(chat_messages) if chat_messages else 0.0
        
        # Track metric
        metrics.record_highlight_confidence(
            confidence=avg_sentiment,
            detection_method="chat_sentiment",
            platform="twitch"
        )
        
        return avg_sentiment


# Example 4: Service layer with comprehensive observability
class HighlightDetectionService:
    """Example service class with full observability integration."""
    
    @traced_service_method()
    async def detect_highlights(
        self,
        stream_id: str,
        organization_id: str,
        platform: str
    ) -> Dict[str, Any]:
        """Detect highlights with comprehensive tracking."""
        
        # Start processing
        start_time = time.time()
        
        # Add context for all operations
        logfire.set_attribute("stream.id", stream_id)
        logfire.set_attribute("organization.id", organization_id)
        logfire.set_attribute("platform", platform)
        
        # Track stream started
        metrics.increment_stream_started(
            platform=platform,
            organization_id=organization_id,
            stream_type="live"
        )
        
        try:
            # Step 1: Extract frames
            with with_span("extract_frames_step"):
                frame_count = await extract_video_frames(f"stream://{stream_id}")
                logfire.info(f"Extracted {frame_count} frames")
            
            # Step 2: Analyze video
            with with_span("video_analysis_step"):
                video_highlights = await self._analyze_video(stream_id)
                logfire.info(f"Found {len(video_highlights)} video highlights")
            
            # Step 3: Analyze chat
            with with_span("chat_analysis_step"):
                chat_sentiment = await analyze_stream_sentiment(
                    stream_id,
                    ["Great play!", "Amazing!", "Wow!"]
                )
                logfire.info(f"Chat sentiment score: {chat_sentiment}")
            
            # Step 4: Combine results
            with with_span("combine_results_step"):
                highlights = self._combine_highlights(video_highlights, chat_sentiment)
                
                # Track highlights detected
                metrics.increment_highlights_detected(
                    count=len(highlights),
                    platform=platform,
                    organization_id=organization_id,
                    detection_method="multimodal"
                )
            
            # Track success
            duration = time.time() - start_time
            metrics.increment_stream_completed(
                platform=platform,
                organization_id=organization_id,
                success=True
            )
            metrics.record_stream_duration(
                duration_seconds=duration,
                platform=platform,
                organization_id=organization_id
            )
            
            # Log completion event
            logfire.info(
                "Stream processing completed",
                highlights_count=len(highlights),
                duration_seconds=duration,
                status="success"
            )
            
            return {
                "stream_id": stream_id,
                "highlights": highlights,
                "processing_time": duration,
                "frame_count": frame_count,
                "sentiment_score": chat_sentiment
            }
            
        except Exception as e:
            # Track failure
            metrics.increment_stream_completed(
                platform=platform,
                organization_id=organization_id,
                success=False
            )
            
            # Log error with context
            logfire.error(
                "Stream processing failed",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True
            )
            
            raise
    
    async def _analyze_video(self, stream_id: str) -> list:
        """Mock video analysis."""
        await asyncio.sleep(0.3)
        return [
            {"timestamp": 30, "confidence": 0.9},
            {"timestamp": 120, "confidence": 0.85},
        ]
    
    def _combine_highlights(self, video_highlights: list, sentiment: float) -> list:
        """Combine highlights from different sources."""
        combined = []
        for vh in video_highlights:
            combined.append({
                **vh,
                "combined_score": (vh["confidence"] + sentiment) / 2
            })
        return combined


# Example 5: Repository layer with tracing
class StreamRepository:
    """Example repository with observability."""
    
    @traced_repository_method()
    async def get_stream(self, stream_id: str) -> Dict[str, Any]:
        """Get stream with tracing."""
        # This would normally query the database
        await asyncio.sleep(0.05)
        
        return {
            "id": stream_id,
            "platform": "twitch",
            "status": "processing"
        }
    
    @traced_repository_method()
    async def update_stream_status(self, stream_id: str, status: str) -> None:
        """Update stream status with tracing."""
        logfire.info(
            "Updating stream status",
            stream_id=stream_id,
            new_status=status
        )
        
        # This would normally update the database
        await asyncio.sleep(0.05)


# Example 6: API endpoint simulation
async def api_endpoint_example(organization_id: str, api_key_id: str):
    """Example simulating an API endpoint with full observability."""
    
    # Track API usage
    metrics.increment_api_key_usage(
        api_key_id=api_key_id,
        organization_id=organization_id,
        endpoint="/api/v1/streams"
    )
    
    # Create service instances
    service = HighlightDetectionService()
    repository = StreamRepository()
    
    # Process request
    with with_span(
        "api.process_stream_request",
        organization_id=organization_id,
        api_key_id=api_key_id
    ):
        # Get stream info
        stream = await repository.get_stream("stream_123")
        
        # Process highlights
        result = await service.detect_highlights(
            stream_id=stream["id"],
            organization_id=organization_id,
            platform=stream["platform"]
        )
        
        # Update status
        await repository.update_stream_status(stream["id"], "completed")
        
        # Track API latency
        metrics.record_api_latency(
            latency_seconds=0.8,
            endpoint="/api/v1/streams",
            method="POST",
            status_code=200
        )
        
        return result


# Example 7: Background task with observability
async def background_cleanup_task():
    """Example background task with observability."""
    
    with with_span("background.cleanup_task", task_type="maintenance"):
        logfire.info("Starting cleanup task")
        
        # Simulate cleanup
        deleted_count = 0
        
        for i in range(5):
            with with_span(f"cleanup_batch_{i}"):
                await asyncio.sleep(0.1)
                deleted_count += 10
                
                # Track progress
                logfire.info(
                    f"Cleaned up batch {i}",
                    batch_number=i,
                    items_deleted=10
                )
        
        # Track task completion
        metrics.increment_task_executed(
            task_name="cleanup_old_highlights",
            organization_id=None,
            success=True
        )
        
        logfire.info(
            "Cleanup task completed",
            total_deleted=deleted_count
        )


# Main execution example
async def main():
    """Run all examples to demonstrate Logfire integration."""
    
    print("Starting Logfire integration examples...")
    
    # Initialize Logfire (normally done in app startup)
    logfire.configure(
        service_name="tldr-api-example",
        service_version="1.0.0",
        console=True
    )
    
    try:
        # Example 1: Process video highlight
        print("\n1. Processing video highlight...")
        result = await process_video_highlight("video_123", "youtube")
        print(f"   Result: {result}")
        
        # Example 2: Extract frames
        print("\n2. Extracting video frames...")
        frames = await extract_video_frames("https://example.com/video.mp4")
        print(f"   Extracted {frames} frames")
        
        # Example 3: Analyze sentiment
        print("\n3. Analyzing chat sentiment...")
        sentiment = await analyze_stream_sentiment(
            "stream_123",
            ["Amazing play!", "That was incredible!", "Best stream ever!"]
        )
        print(f"   Sentiment score: {sentiment}")
        
        # Example 4: Full API request
        print("\n4. Simulating API request...")
        api_result = await api_endpoint_example("org_123", "key_abc")
        print(f"   API result: {api_result}")
        
        # Example 5: Background task
        print("\n5. Running background task...")
        await background_cleanup_task()
        
        print("\n✅ All examples completed successfully!")
        print("\nCheck your Logfire dashboard to see:")
        print("- Distributed traces for each operation")
        print("- Custom metrics for business KPIs")
        print("- Structured logs with context")
        print("- Performance measurements")
        
    except Exception as e:
        logfire.error(
            "Example failed",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True
        )
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())