"""
Minimal demonstration of timeline synchronization without complex dependencies.

This shows the core timeline sync functionality working independently.
"""

import asyncio
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockStreamBufferManager:
    """Mock stream buffer manager for demonstration."""

    def __init__(self):
        self.streams = {}

    def add_stream(self, stream_id, adapter, stream_type):
        logger.info(f"Mock: Added stream {stream_id} of type {stream_type}")
        return True

    def remove_stream(self, stream_id):
        logger.info(f"Mock: Removed stream {stream_id}")
        return True

    async def close(self):
        logger.info("Mock: Closed stream buffer manager")


class MockFrameSynchronizer:
    """Mock frame synchronizer for demonstration."""

    def __init__(self):
        self.streams = {}

    def register_stream(
        self, stream_id, buffer_format, timestamp_format, is_reference=False
    ):
        logger.info(f"Mock: Registered stream {stream_id} (reference: {is_reference})")
        self.streams[stream_id] = {
            "format": buffer_format,
            "timestamp_format": timestamp_format,
            "is_reference": is_reference,
        }

    def normalize_timestamp(self, stream_id, timestamp, target_format):
        # Just return the timestamp as-is for demo
        return timestamp

    async def close(self):
        logger.info("Mock: Closed frame synchronizer")


class MockSentimentAnalyzer:
    """Mock sentiment analyzer for demonstration."""

    def __init__(self):
        self.events_processed = 0
        self.message_count = 0

    async def process_event(self, event):
        self.events_processed += 1
        logger.debug(
            f"Mock: Processed event {event.id} (total: {self.events_processed})"
        )

    async def analyze_window(self, messages, start_time, end_time, platform="generic"):
        self.message_count += len(messages)

        # Mock sentiment analysis based on message content
        hype_keywords = ["POGGERS", "HYPE", "INSANE", "LETS GO", "RAID"]
        sentiment_score = 0.0
        intensity = 0.0

        for msg in messages:
            text_upper = msg.text.upper()
            for keyword in hype_keywords:
                if keyword in text_upper:
                    sentiment_score += 0.8
                    intensity += 0.9

        # Normalize
        if messages:
            sentiment_score = min(sentiment_score / len(messages), 1.0)
            intensity = min(intensity / len(messages), 1.0)

        # Create mock window sentiment
        from dataclasses import dataclass

        @dataclass
        class MockWindowSentiment:
            start_time: datetime
            end_time: datetime
            message_count: int
            unique_users: int
            avg_sentiment: float
            sentiment_variance: float
            dominant_category: str
            intensity: float
            momentum: float
            confidence: float
            emote_density: float
            keyword_density: float
            spam_ratio: float

        return MockWindowSentiment(
            start_time=start_time,
            end_time=end_time,
            message_count=len(messages),
            unique_users=len(set(msg.user.id for msg in messages)),
            avg_sentiment=sentiment_score,
            sentiment_variance=0.1,
            dominant_category="hype" if sentiment_score > 0.7 else "neutral",
            intensity=intensity,
            momentum=0.5 if sentiment_score > 0.5 else 0.0,
            confidence=0.8 if sentiment_score > 0.6 else 0.4,
            emote_density=1.0,
            keyword_density=2.0,
            spam_ratio=0.0,
        )

    def get_highlight_confidence(
        self, window_sentiment, velocity_metrics, event_impact=0.0
    ):
        # Calculate based on mock sentiment
        confidence = 0.0

        if hasattr(window_sentiment, "avg_sentiment"):
            confidence += window_sentiment.avg_sentiment * 0.4

        if hasattr(window_sentiment, "intensity"):
            confidence += window_sentiment.intensity * 0.3

        if (
            hasattr(velocity_metrics, "spike_detected")
            and velocity_metrics.spike_detected
        ):
            confidence += 0.3

        confidence += event_impact * 0.2

        return min(confidence, 1.0)


async def demonstrate_timeline_sync():
    """Demonstrate timeline synchronization with mocked dependencies."""
    logger.info("🚀 Starting Timeline Synchronization Demo with Mock Dependencies")

    # Import after setting up mocks
    import sys

    sys.path.insert(0, "/Users/parkergabel/PycharmProjects/tldr_highlight_api")

    from src.services.chat_adapters.timeline_sync import (
        TimelineSynchronizer,
        SyncStrategy,
        ChatSourceType,
        StreamType,
    )
    from src.services.chat_adapters.base import (
        ChatEvent,
        ChatEventType,
        ChatMessage,
        ChatUser,
    )

    # Create timeline synchronizer with mocked dependencies
    sync = TimelineSynchronizer(
        stream_id="demo_stream",
        chat_source=ChatSourceType.TWITCH_EVENTSUB,
        video_type=StreamType.TWITCH_HLS,
        buffer_manager=MockStreamBufferManager(),
        frame_synchronizer=MockFrameSynchronizer(),
        sentiment_analyzer=MockSentimentAnalyzer(),
        sync_strategy=SyncStrategy.OFFSET_BASED,
    )

    # Track highlights
    highlights = []

    def on_highlight(highlight):
        logger.info("🎯 HIGHLIGHT DETECTED!")
        logger.info(f"   Duration: {highlight.duration:.1f}s")
        logger.info(f"   Confidence: {highlight.confidence:.2f}")
        logger.info(f"   Category: {highlight.category}")
        logger.info(f"   Sentiment: {highlight.sentiment_score:.2f}")
        highlights.append(highlight)

    sync.add_highlight_callback(on_highlight)

    try:
        # Start the synchronizer
        await sync.start()
        logger.info("✅ Timeline synchronizer started")

        # Check sync status
        status = sync.get_sync_status()
        logger.info("📊 Sync Status:")
        logger.info(f"   Synchronized: {status['is_synchronized']}")
        logger.info(f"   Strategy: {status['sync_strategy']}")
        logger.info(f"   Offset: {status['timestamp_offset']['offset_seconds']:.1f}s")
        logger.info(f"   Confidence: {status['timestamp_offset']['confidence']:.2f}")

        # Create test users
        users = [
            ChatUser(id="user1", username="viewer1", display_name="Excited Viewer"),
            ChatUser(id="user2", username="raider1", display_name="Raid Leader"),
            ChatUser(id="user3", username="fan1", display_name="Long Time Fan"),
        ]

        # Phase 1: Regular chat
        logger.info("\n💬 Phase 1: Regular chat activity")
        for i in range(3):
            user = users[i % len(users)]
            message = ChatMessage(
                id=f"regular_{i}",
                user=user,
                text=f"Hello everyone! Message {i}",
                timestamp=datetime.now(timezone.utc),
            )

            event = ChatEvent(
                id=f"regular_event_{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message},
            )

            await sync.add_chat_event(event)
            logger.info(f"   Added regular message: '{message.text}'")
            await asyncio.sleep(0.5)

        # Phase 2: Raid event
        logger.info("\n🎉 Phase 2: Raid event occurs")
        raid_event = ChatEvent(
            id="raid_event",
            type=ChatEventType.RAID,
            timestamp=datetime.now(timezone.utc),
            data={"viewers": 1000, "from_channel": "big_streamer"},
        )
        await sync.add_chat_event(raid_event)
        logger.info("   💥 RAID EVENT: 1000 viewers from big_streamer!")

        # Phase 3: Hype explosion
        logger.info("\n🔥 Phase 3: Chat goes crazy!")
        hype_messages = [
            "POGGERS RAID!!!",
            "HOLY RAID BATMAN!!!",
            "1000 VIEWERS HYPE!!!",
            "LETS GO CHAT!!!",
            "INSANE RAID!!!",
            "BEST STREAMER EVER!!!",
            "RAID TRAIN COMING!!!",
            "POGGERS IN CHAT!!!",
        ]

        for i, text in enumerate(hype_messages):
            user = users[i % len(users)]
            message = ChatMessage(
                id=f"hype_{i}",
                user=user,
                text=text,
                timestamp=datetime.now(timezone.utc),
                emotes=[{"name": "PogChamp"}, {"name": "Kreygasm"}],
            )

            event = ChatEvent(
                id=f"hype_event_{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message},
            )

            await sync.add_chat_event(event)
            logger.info(f"   🔥 HYPE: {text}")
            await asyncio.sleep(0.2)

        # Phase 4: Donation for more hype
        logger.info("\n💰 Phase 4: Big donation!")
        donation_event = ChatEvent(
            id="donation_event",
            type=ChatEventType.CHEER,
            timestamp=datetime.now(timezone.utc),
            data={"bits": 5000, "message": "TAKE MY BITS!! BEST STREAMER!!!"},
        )
        await sync.add_chat_event(donation_event)
        logger.info("   💸 BIG DONATION: 5000 bits!")

        # More excitement
        more_hype = [
            "5000 BITS POGGERS!!!",
            "RICHEST VIEWER EVER!!!",
            "MONEY RAIN!!!",
            "INSANE DONATION!!!",
        ]

        for i, text in enumerate(more_hype):
            user = users[i % len(users)]
            message = ChatMessage(
                id=f"donation_hype_{i}",
                user=user,
                text=text,
                timestamp=datetime.now(timezone.utc),
            )

            event = ChatEvent(
                id=f"donation_hype_event_{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message},
            )

            await sync.add_chat_event(event)
            logger.info(f"   💰 DONATION HYPE: {text}")
            await asyncio.sleep(0.3)

        # Phase 5: Calm down
        logger.info("\n😌 Phase 5: Chat calms down")
        calm_messages = [
            "That was amazing!",
            "Best stream ever!",
            "Thanks for the entertainment!",
        ]

        for i, text in enumerate(calm_messages):
            user = users[i % len(users)]
            message = ChatMessage(
                id=f"calm_{i}",
                user=user,
                text=text,
                timestamp=datetime.now(timezone.utc),
            )

            event = ChatEvent(
                id=f"calm_event_{i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message},
            )

            await sync.add_chat_event(event)
            logger.info(f"   💬 Calm: {text}")
            await asyncio.sleep(1.0)

        # Wait for processing
        logger.info("\n⏳ Processing timeline data...")
        await asyncio.sleep(3.0)

        # Check final status
        final_status = sync.get_sync_status()
        logger.info("\n📊 Final Results:")
        logger.info(f"   Events buffered: {final_status['buffered_events']}")
        logger.info(f"   Messages buffered: {final_status['buffered_messages']}")
        logger.info(f"   Highlights detected: {final_status['detected_highlights']}")

        # Get all highlights
        all_highlights = await sync.get_highlights(min_confidence=0.3)
        logger.info(f"\n🏆 Highlight Summary: {len(all_highlights)} highlights found")

        for i, highlight in enumerate(all_highlights, 1):
            logger.info(
                f"   {i}. {highlight.start_timestamp:.1f}s - {highlight.end_timestamp:.1f}s"
            )
            logger.info(f"      Duration: {highlight.duration:.1f}s")
            logger.info(f"      Confidence: {highlight.confidence:.2f}")
            logger.info(f"      Category: {highlight.category}")
            logger.info(f"      Chat events: {len(highlight.chat_events)}")

        # Test timestamp synchronization
        logger.info("\n🕐 Testing Timestamp Synchronization:")
        current_time = datetime.now(timezone.utc).timestamp()

        # Get events from last 30 seconds
        sync_events = await sync.get_synchronized_events(
            current_time - 30.0, current_time
        )

        logger.info(f"   Synchronized events in last 30s: {len(sync_events)}")
        for event in sync_events[:3]:  # Show first 3
            if event.is_synchronized:
                logger.info(
                    f"   📍 Event '{event.event.id}' synced to video time {event.video_timestamp:.2f}s"
                )

        logger.info("\n✅ Demo completed successfully!")

        if highlights:
            logger.info(
                f"\n🎯 SUCCESS: Detected {len(highlights)} highlights during the demo!"
            )
        else:
            logger.info(
                "\n⚠️  No highlights were automatically detected during this run."
            )
            logger.info("   This could be due to timing or threshold settings.")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        await sync.stop()
        logger.info("🛑 Timeline synchronizer stopped")


if __name__ == "__main__":
    asyncio.run(demonstrate_timeline_sync())
