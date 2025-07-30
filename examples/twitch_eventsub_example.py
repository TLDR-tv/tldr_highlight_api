"""Example of using Twitch EventSub WebSocket with stream adapter.

This example demonstrates how to:
1. Connect to a Twitch stream using the stream adapter
2. Connect to Twitch EventSub for real-time chat events
3. Synchronize chat events with the stream timeline
4. Process events for highlight detection
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.stream_adapters.twitch import TwitchAdapter
from src.services.chat_adapters.twitch_eventsub import TwitchEventSubAdapter
from src.services.chat_adapters.stream_sync import StreamChatSynchronizer
from src.services.chat_adapters.base import ChatEventType
from src.core.config import get_settings


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HighlightDetector:
    """Simple highlight detector based on chat activity."""

    def __init__(self):
        self.potential_highlights = []
        self.chat_velocity_window = []
        self.excitement_threshold = 10  # Messages per 10 seconds

    async def process_sync_event(self, sync_event):
        """Process a synchronized event for highlight detection."""
        if (
            sync_event.chat_event
            and sync_event.chat_event.type == ChatEventType.MESSAGE
        ):
            # Track chat velocity
            self.chat_velocity_window.append(sync_event)

            # Remove old events (keep 10 second window)
            cutoff_time = sync_event.stream_time - 10.0
            self.chat_velocity_window = [
                e for e in self.chat_velocity_window if e.stream_time > cutoff_time
            ]

            # Check if we have high chat activity
            if len(self.chat_velocity_window) >= self.excitement_threshold:
                highlight = {
                    "timestamp": sync_event.timestamp,
                    "stream_time": sync_event.stream_time,
                    "reason": "high_chat_activity",
                    "chat_count": len(self.chat_velocity_window),
                    "sample_messages": [
                        e.chat_event.data["message"].text
                        for e in self.chat_velocity_window[-5:]
                        if e.chat_event and e.chat_event.type == ChatEventType.MESSAGE
                    ],
                }
                self.potential_highlights.append(highlight)
                logger.info(
                    f"Potential highlight detected at {sync_event.stream_time:.1f}s: High chat activity"
                )

        # Check for special events
        elif sync_event.chat_event and sync_event.chat_event.type in [
            ChatEventType.SUBSCRIBE,
            ChatEventType.RAID,
            ChatEventType.HYPE_TRAIN_BEGIN,
        ]:
            highlight = {
                "timestamp": sync_event.timestamp,
                "stream_time": sync_event.stream_time,
                "reason": sync_event.chat_event.type.value,
                "event_data": sync_event.chat_event.data,
            }
            self.potential_highlights.append(highlight)
            logger.info(
                f"Potential highlight detected at {sync_event.stream_time:.1f}s: {sync_event.chat_event.type.value}"
            )


async def main():
    """Main example function."""
    # Get configuration
    settings = get_settings()

    # Get Twitch credentials from environment or config
    channel_url = input(
        "Enter Twitch channel URL (e.g., https://twitch.tv/username): "
    ).strip()
    access_token = (
        os.getenv("TWITCH_ACCESS_TOKEN") or input("Enter Twitch access token: ").strip()
    )

    # Extract username from URL
    username = channel_url.split("/")[-1]

    # Create stream adapter
    stream_adapter = TwitchAdapter(
        url=channel_url,
        client_id=settings.twitch_client_id,
        client_secret=settings.twitch_client_secret,
    )

    # Create chat adapter
    chat_adapter = TwitchEventSubAdapter(
        channel_id=None,  # Will be set after getting user info
        access_token=access_token,
        client_id=settings.twitch_client_id,
    )

    # Create synchronizer
    synchronizer = StreamChatSynchronizer(
        stream_adapter=stream_adapter,
        chat_adapter=chat_adapter,
        buffer_seconds=30.0,
        sync_interval=1.0,
    )

    # Create highlight detector
    detector = HighlightDetector()
    synchronizer.on_sync_event(detector.process_sync_event)

    try:
        logger.info(f"Starting Twitch EventSub example for channel: {username}")

        # Start stream adapter
        await stream_adapter.start()

        # Get channel ID from stream metadata
        metadata = await stream_adapter.get_metadata()
        if not metadata.platform_id:
            logger.error("Could not get channel ID from stream")
            return

        chat_adapter.channel_id = metadata.platform_id
        logger.info(f"Channel ID: {metadata.platform_id}")

        # Subscribe to chat events
        await chat_adapter.subscribe_to_events(
            [
                ChatEventType.MESSAGE,
                ChatEventType.FOLLOW,
                ChatEventType.SUBSCRIBE,
                ChatEventType.CHEER,
                ChatEventType.RAID,
                ChatEventType.HYPE_TRAIN_BEGIN,
                ChatEventType.HYPE_TRAIN_PROGRESS,
                ChatEventType.HYPE_TRAIN_END,
            ]
        )

        # Start chat adapter
        await chat_adapter.start()

        # Start synchronizer
        await synchronizer.start()

        logger.info("Connected successfully! Monitoring for highlights...")
        logger.info("Press Ctrl+C to stop")

        # Monitor for events
        start_time = datetime.now()
        last_status = datetime.now()

        while True:
            await asyncio.sleep(10)

            # Print status every 30 seconds
            if (datetime.now() - last_status).total_seconds() > 30:
                runtime = (datetime.now() - start_time).total_seconds()
                metrics = synchronizer.get_metrics()

                logger.info(f"Status after {runtime:.0f}s:")
                logger.info(f"  Stream time: {metrics['stream_current_time']:.1f}s")
                logger.info(f"  Events synced: {metrics['events_synced']}")
                logger.info(f"  Buffer size: {metrics['buffer_size']}")
                logger.info(
                    f"  Highlights detected: {len(detector.potential_highlights)}"
                )

                last_status = datetime.now()

    except KeyboardInterrupt:
        logger.info("\nStopping...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Print detected highlights
        if detector.potential_highlights:
            logger.info(
                f"\nDetected {len(detector.potential_highlights)} potential highlights:"
            )
            for i, highlight in enumerate(detector.potential_highlights, 1):
                logger.info(f"\n{i}. Time: {highlight['stream_time']:.1f}s")
                logger.info(f"   Reason: {highlight['reason']}")
                if "chat_count" in highlight:
                    logger.info(f"   Chat count: {highlight['chat_count']}")
                if "sample_messages" in highlight:
                    logger.info(
                        f"   Sample messages: {highlight['sample_messages'][:3]}"
                    )
                if "event_data" in highlight:
                    logger.info(f"   Event data: {highlight['event_data']}")

        # Clean up
        await synchronizer.stop()
        await chat_adapter.stop()
        await stream_adapter.stop()

        logger.info("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
