"""Example showing how to instrument an existing domain service with Logfire.

This demonstrates the minimal changes needed to add comprehensive observability
to existing services in the TL;DR Highlight API.
"""

from typing import List
import asyncio
from celery import Task

# Original imports
from src.domain.entities import Stream, Highlight
from src.domain.repositories import StreamRepository

# New observability imports
from src.infrastructure.observability import (
    traced_service_method,
    with_span,
    metrics,
)
import logfire


# Example: Instrumenting the StreamProcessingService
class StreamProcessingService:
    """Example of instrumenting an existing service with Logfire."""

    def __init__(self, stream_repository: StreamRepository):
        self.stream_repository = stream_repository

    # Before: Original method without observability
    # async def process_stream(self, stream_id: str, organization_id: str) -> List[Highlight]:
    #     stream = await self.stream_repository.get_by_id(stream_id)
    #     # ... processing logic ...
    #     return highlights

    # After: Method with comprehensive observability
    @traced_service_method(name="process_stream")
    async def process_stream(
        self, stream_id: str, organization_id: str
    ) -> List[Highlight]:
        """Process stream with full observability."""

        # Add context that will be included in all child spans
        logfire.set_attribute("stream.id", stream_id)
        logfire.set_attribute("organization.id", organization_id)

        # Track business metric
        metrics.increment_stream_started(
            platform="unknown",  # Will be updated after fetching stream
            organization_id=organization_id,
            stream_type="live",
        )

        try:
            # Step 1: Fetch stream with span
            with with_span("fetch_stream"):
                stream = await self.stream_repository.get_by_id(stream_id)
                logfire.set_attribute("stream.platform", stream.platform)
                logfire.info(f"Processing {stream.platform} stream")

            # Step 2: Process with monitoring
            highlights = await self._process_stream_content(stream)

            # Track success metrics
            metrics.increment_stream_completed(
                platform=stream.platform, organization_id=organization_id, success=True
            )

            metrics.increment_highlights_detected(
                count=len(highlights),
                platform=stream.platform,
                organization_id=organization_id,
                detection_method="multimodal",
            )

            logfire.info(
                "Stream processing completed",
                highlights_count=len(highlights),
                status="success",
            )

            return highlights

        except Exception as e:
            # Track failure metrics
            metrics.increment_stream_completed(
                platform=stream.platform if "stream" in locals() else "unknown",
                organization_id=organization_id,
                success=False,
            )

            logfire.error("Stream processing failed", error=str(e), exc_info=True)
            raise

    async def _process_stream_content(self, stream: Stream) -> List[Highlight]:
        """Process stream content with detailed tracking."""

        highlights = []

        # Process each segment with timing
        with with_span("process_segments", segment_count=len(stream.segments)):
            for i, segment in enumerate(stream.segments):
                with with_span(f"process_segment_{i}", segment_index=i):
                    # Simulate processing
                    await asyncio.sleep(0.1)

                    # Track processing time per segment
                    metrics.record_highlight_processing_time(
                        duration_seconds=0.1,
                        stage="segment_analysis",
                        platform=stream.platform,
                    )

                    # Create highlight
                    highlight = Highlight(timestamp=segment.timestamp, confidence=0.85)
                    highlights.append(highlight)

        return highlights


# Example: Quick instrumentation with minimal changes
class ExistingService:
    """Example showing minimal instrumentation."""

    # Just add the decorator - that's it!
    @traced_service_method()
    async def existing_method(self, param1: str, param2: int) -> dict:
        """This method now has automatic tracing."""
        # Original logic remains unchanged
        result = await self._do_work(param1, param2)
        return result

    async def _do_work(self, param1: str, param2: int) -> dict:
        await asyncio.sleep(0.1)
        return {"status": "completed", "value": param2}


# Example: Adding custom metrics to existing code
class WebhookService:
    """Example of adding metrics to webhook delivery."""

    async def deliver_webhook(
        self, url: str, payload: dict, organization_id: str, event_type: str
    ) -> bool:
        """Deliver webhook with metrics tracking."""

        start_time = asyncio.get_event_loop().time()

        try:
            # Simulate webhook delivery
            await asyncio.sleep(0.2)
            success = True

            # Track success metric
            metrics.increment_webhook_sent(
                organization_id=organization_id, event_type=event_type, success=True
            )

            # Track latency
            latency = asyncio.get_event_loop().time() - start_time
            metrics.record_webhook_latency(
                latency_seconds=latency,
                organization_id=organization_id,
                event_type=event_type,
            )

            return success

        except Exception as e:
            # Track failure
            metrics.increment_webhook_sent(
                organization_id=organization_id, event_type=event_type, success=False
            )

            logfire.error(f"Webhook delivery failed: {e}")
            return False


# Example: Instrumenting a Celery task


class InstrumentedTask(Task):
    """Base class for instrumented Celery tasks."""

    def __call__(self, *args, **kwargs):
        """Wrap task execution with observability."""

        task_name = self.name
        organization_id = kwargs.get("organization_id")

        with with_span(f"celery.task.{task_name}", task_name=task_name):
            try:
                result = super().__call__(*args, **kwargs)

                # Track success
                metrics.increment_task_executed(
                    task_name=task_name, organization_id=organization_id, success=True
                )

                return result

            except Exception as e:
                # Track failure
                metrics.increment_task_executed(
                    task_name=task_name, organization_id=organization_id, success=False
                )

                logfire.error(
                    f"Celery task failed: {task_name}", error=str(e), exc_info=True
                )
                raise


# Usage example
async def main():
    """Demonstrate the instrumented services."""

    # Initialize Logfire
    logfire.configure(service_name="service-instrumentation-demo", console=True)

    print("Running instrumented service examples...")

    # Example 1: Existing service with minimal changes
    service = ExistingService()
    result = await service.existing_method("test", 42)
    print(f"✅ Existing service result: {result}")

    # Example 2: Webhook delivery with metrics
    webhook_service = WebhookService()
    success = await webhook_service.deliver_webhook(
        url="https://example.com/webhook",
        payload={"event": "stream.completed"},
        organization_id="org_123",
        event_type="stream.completed",
    )
    print(f"✅ Webhook delivered: {success}")

    print("\nObservability features added:")
    print("- Automatic distributed tracing")
    print("- Business metrics tracking")
    print("- Error logging with context")
    print("- Performance monitoring")


if __name__ == "__main__":
    asyncio.run(main())
