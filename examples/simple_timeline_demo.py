"""
Simple demonstration of the timeline synchronization system.

This script shows basic usage without complex dependencies.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from src.services.chat_adapters.timeline_sync import (
    TimelineSynchronizer,
    SyncStrategy,
    ChatSourceType,
    StreamType,
    TimestampOffset,
)
from src.services.chat_adapters.base import (
    ChatEvent,
    ChatEventType,
    ChatMessage,
    ChatUser,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demonstrate_basic_sync():
    """Demonstrate basic timeline synchronization."""
    logger.info("🚀 Starting Basic Timeline Synchronization Demo")

    # Create a timeline synchronizer for Twitch
    sync = TimelineSynchronizer(
        stream_id="demo_stream",
        chat_source=ChatSourceType.TWITCH_EVENTSUB,
        video_type=StreamType.TWITCH_HLS,
        sync_strategy=SyncStrategy.OFFSET_BASED,
    )

    # Track detected highlights
    highlights = []

    def on_highlight(highlight):
        logger.info(
            f"🎯 Highlight detected: {highlight.duration:.1f}s, confidence: {highlight.confidence:.2f}"
        )
        highlights.append(highlight)

    sync.add_highlight_callback(on_highlight)

    try:
        # Start the synchronizer
        await sync.start()
        logger.info("✅ Timeline synchronizer started")

        # Check initial sync status
        status = sync.get_sync_status()
        logger.info(
            f"📊 Initial sync status: synchronized={status['is_synchronized']}, "
            f"offset={status['timestamp_offset']['offset_seconds']:.1f}s"
        )

        # Create some test users
        users = [
            ChatUser(id="user1", username="viewer1", display_name="Viewer One"),
            ChatUser(id="user2", username="gamer2", display_name="Pro Gamer"),
            ChatUser(id="user3", username="fan3", display_name="Big Fan"),
        ]

        # Simulate regular chat
        logger.info("💬 Simulating regular chat activity...")
        for i in range(5):
            user = users[i % len(users)]
            message = ChatMessage(
                id=f"msg_{i}",
                user=user,
                text=f"Regular message {i}",
                timestamp=datetime.now(timezone.utc),
            )

            event = ChatEvent(
                id=f"event_{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message},
            )

            await sync.add_chat_event(event)
            await asyncio.sleep(0.5)

        # Simulate a raid event
        logger.info("🎉 Simulating raid event...")
        raid_event = ChatEvent(
            id="raid_1",
            type=ChatEventType.RAID,
            timestamp=datetime.now(timezone.utc),
            data={"viewers": 500, "from_channel": "big_streamer"},
        )
        await sync.add_chat_event(raid_event)

        # Simulate hype messages
        logger.info("🔥 Simulating hype burst...")
        hype_messages = [
            "RAID HYPE!!!",
            "POGGERS!!!",
            "WELCOME RAIDERS!!!",
            "500 VIEWERS!!!",
            "LETS GOOO!!!",
            "BEST STREAMER!!!",
        ]

        for i, text in enumerate(hype_messages):
            user = users[i % len(users)]
            message = ChatMessage(
                id=f"hype_{i}",
                user=user,
                text=text,
                timestamp=datetime.now(timezone.utc),
                emotes=[{"name": "PogChamp"}],
            )

            event = ChatEvent(
                id=f"hype_event_{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message},
            )

            await sync.add_chat_event(event)
            await asyncio.sleep(0.3)

        # Wait for processing
        await asyncio.sleep(2.0)

        # Check final status
        final_status = sync.get_sync_status()
        logger.info("📊 Final status:")
        logger.info(f"   Synchronized: {final_status['is_synchronized']}")
        logger.info(f"   Buffered events: {final_status['buffered_events']}")
        logger.info(f"   Detected highlights: {final_status['detected_highlights']}")

        # Get synchronized events for a time range
        current_time = datetime.now(timezone.utc).timestamp()
        sync_events = await sync.get_synchronized_events(
            current_time - 30.0,  # Last 30 seconds
            current_time,
        )

        logger.info(f"🔄 Synchronized events in last 30s: {len(sync_events)}")

        # Get all highlights
        all_highlights = await sync.get_highlights(min_confidence=0.5)
        logger.info(f"🏆 Total highlights found: {len(all_highlights)}")

        for i, highlight in enumerate(all_highlights, 1):
            logger.info(
                f"   {i}. Duration: {highlight.duration:.1f}s, "
                f"Confidence: {highlight.confidence:.2f}, "
                f"Category: {highlight.category}"
            )

        logger.info("✅ Demo completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

    finally:
        await sync.stop()
        logger.info("🛑 Timeline synchronizer stopped")


async def demonstrate_timestamp_offset():
    """Demonstrate timestamp offset functionality."""
    logger.info("\n🕒 Demonstrating Timestamp Offset Functionality")

    # Create different offsets for different scenarios
    scenarios = [
        ("Twitch Live", -2.0, 0.9),
        ("YouTube Live", -1.5, 0.85),
        ("YouTube VOD", 0.0, 1.0),
        ("Generic Stream", -1.0, 0.7),
    ]

    for platform, offset_sec, confidence in scenarios:
        offset = TimestampOffset(offset_seconds=offset_sec, confidence=confidence)

        # Test timestamp
        original_time = 1000.0
        adjusted_time = offset.apply(original_time)

        logger.info(
            f"📺 {platform:15} | "
            f"Offset: {offset_sec:+5.1f}s | "
            f"Confidence: {confidence:4.2f} | "
            f"Time: {original_time:.1f} → {adjusted_time:.1f}"
        )

    # Demonstrate drift correction
    logger.info("\n⏱️  Demonstrating Drift Correction:")

    # Create offset with drift
    offset = TimestampOffset(
        offset_seconds=2.0,
        confidence=0.9,
        drift_rate=0.01,  # 10ms per second drift
        last_update=datetime.now(timezone.utc) - timedelta(seconds=10),
    )

    original_time = 1000.0
    adjusted_time = offset.apply(original_time, datetime.now(timezone.utc))

    logger.info(f"   Original time: {original_time:.3f}s")
    logger.info(f"   Base offset: {offset.offset_seconds:.3f}s")
    logger.info(f"   Drift rate: {offset.drift_rate:.6f}s/s")
    logger.info(f"   Drift correction: ~{0.01 * 10:.3f}s (for 10s elapsed)")
    logger.info(f"   Final adjusted time: {adjusted_time:.3f}s")


async def main():
    """Run all demonstrations."""
    try:
        await demonstrate_timestamp_offset()
        await demonstrate_basic_sync()
    except Exception as e:
        logger.error(f"❌ Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
