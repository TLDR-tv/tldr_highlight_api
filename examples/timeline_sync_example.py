"""
Example usage of the chat-video timeline synchronization system.

This script demonstrates how to set up and use the timeline synchronization
system for various streaming scenarios.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from src.services.chat_adapters.timeline_sync import (
    TimelineSynchronizer,
    MultiStreamTimelineSynchronizer,
    SyncStrategy,
    ChatSourceType,
    HighlightCandidate
)
from src.services.chat_adapters.base import ChatEvent, ChatEventType, ChatMessage, ChatUser
from src.services.content_processing.stream_buffer_manager import StreamBufferManager, StreamType
from src.utils.frame_synchronizer import FrameSynchronizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleChatGenerator:
    """Generates example chat events for demonstration."""
    
    def __init__(self):
        self.message_id = 0
        self.users = [
            ChatUser(id=f"user_{i}", username=f"user{i}", display_name=f"User {i}")
            for i in range(20)
        ]
    
    def generate_regular_message(self, text: str = None) -> ChatEvent:
        """Generate a regular chat message."""
        user = self.users[self.message_id % len(self.users)]
        text = text or f"Regular message {self.message_id}"
        
        message = ChatMessage(
            id=f"msg_{self.message_id}",
            user=user,
            text=text,
            timestamp=datetime.now(timezone.utc)
        )
        
        event = ChatEvent(
            id=f"event_{self.message_id}",
            type=ChatEventType.MESSAGE,
            timestamp=message.timestamp,
            user=user,
            data={"message": message}
        )
        
        self.message_id += 1
        return event
    
    def generate_hype_burst(self, duration: float = 5.0) -> List[ChatEvent]:
        """Generate a burst of hype messages."""
        events = []
        hype_messages = [
            "POGGERS!!!", "NO WAY!!!", "INSANE!!!", "LETS GO!!!",
            "HOLY SHIT!!!", "UNREAL!!!", "CLIP IT!!!", "HYPE!!!",
            "BEST PLAY EVER!!!", "GODLIKE!!!", "5HEAD!!!", "EZ CLAP!!!"
        ]
        
        start_time = datetime.now(timezone.utc)
        
        for i in range(int(duration * 5)):  # 5 messages per second
            text = hype_messages[i % len(hype_messages)]
            user = self.users[(self.message_id + i) % len(self.users)]
            
            message = ChatMessage(
                id=f"hype_msg_{self.message_id + i}",
                user=user,
                text=text,
                timestamp=start_time + timedelta(seconds=i * 0.2),
                emotes=[{"name": "PogChamp"}, {"name": "Kreygasm"}]
            )
            
            event = ChatEvent(
                id=f"hype_event_{self.message_id + i}",
                type=ChatEventType.MESSAGE,
                timestamp=message.timestamp,
                user=user,
                data={"message": message}
            )
            
            events.append(event)
        
        self.message_id += len(events)
        return events
    
    def generate_special_event(self, event_type: ChatEventType, **kwargs) -> ChatEvent:
        """Generate a special event (raid, cheer, etc.)."""
        event = ChatEvent(
            id=f"special_{self.message_id}",
            type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=kwargs
        )
        
        self.message_id += 1
        return event


async def example_single_stream_sync():
    """Example: Single stream synchronization."""
    logger.info("=== Single Stream Synchronization Example ===")
    
    # Create components
    buffer_manager = StreamBufferManager()
    frame_synchronizer = FrameSynchronizer()
    
    # Create timeline synchronizer for Twitch stream
    timeline_sync = TimelineSynchronizer(
        stream_id="example_twitch_stream",
        chat_source=ChatSourceType.TWITCH_EVENTSUB,
        video_type=StreamType.TWITCH_HLS,
        buffer_manager=buffer_manager,
        frame_synchronizer=frame_synchronizer,
        sync_strategy=SyncStrategy.OFFSET_BASED
    )
    
    # Track highlights
    highlights: List[HighlightCandidate] = []
    
    def on_highlight_detected(highlight: HighlightCandidate):
        logger.info(f"🎯 Highlight detected: {highlight.duration:.1f}s @ {highlight.start_timestamp:.1f}s")
        logger.info(f"   Confidence: {highlight.confidence:.2f}, Category: {highlight.category}")
        highlights.append(highlight)
    
    timeline_sync.add_highlight_callback(on_highlight_detected)
    
    # Create chat generator
    chat_gen = ExampleChatGenerator()
    
    try:
        # Start synchronizer
        await timeline_sync.start()
        logger.info("Timeline synchronizer started")
        
        # Phase 1: Regular chat activity
        logger.info("Phase 1: Regular chat activity (10 seconds)")
        for i in range(10):
            event = chat_gen.generate_regular_message()
            await timeline_sync.add_chat_event(event)
            await asyncio.sleep(1.0)
        
        # Phase 2: Raid event + hype
        logger.info("Phase 2: Raid event with hype burst")
        raid_event = chat_gen.generate_special_event(
            ChatEventType.RAID,
            viewers=500,
            from_channel="big_streamer"
        )
        await timeline_sync.add_chat_event(raid_event)
        
        # Generate hype burst
        hype_events = chat_gen.generate_hype_burst(duration=8.0)
        for event in hype_events:
            await timeline_sync.add_chat_event(event)
            await asyncio.sleep(0.2)
        
        # Phase 3: Return to normal
        logger.info("Phase 3: Activity declines")
        for i in range(15):
            if i % 3 == 0:  # Sparse messages
                event = chat_gen.generate_regular_message("Cool raid!")
                await timeline_sync.add_chat_event(event)
            await asyncio.sleep(1.0)
        
        # Phase 4: Big donation + excitement
        logger.info("Phase 4: Big donation creates another highlight")
        donation_event = chat_gen.generate_special_event(
            ChatEventType.CHEER,
            bits=5000,
            message="TAKE MY MONEY! BEST STREAMER!"
        )
        await timeline_sync.add_chat_event(donation_event)
        
        # More hype
        hype_events = chat_gen.generate_hype_burst(duration=6.0)
        for event in hype_events:
            await timeline_sync.add_chat_event(event)
            await asyncio.sleep(0.3)
        
        # Final calm period
        logger.info("Phase 5: Calm down period")
        for i in range(10):
            event = chat_gen.generate_regular_message("That was amazing!")
            await timeline_sync.add_chat_event(event)
            await asyncio.sleep(1.5)
        
        # Get final status
        status = timeline_sync.get_sync_status()
        logger.info(f"Final sync status: {status['is_synchronized']}")
        logger.info(f"Buffered events: {status['buffered_events']}")
        logger.info(f"Detected highlights: {status['detected_highlights']}")
        
        # Get all highlights
        all_highlights = await timeline_sync.get_highlights(min_confidence=0.6)
        logger.info(f"\n📊 Summary: {len(all_highlights)} highlights detected")
        
        for i, highlight in enumerate(all_highlights, 1):
            logger.info(f"  {i}. {highlight.start_timestamp:.1f}s - {highlight.end_timestamp:.1f}s "
                       f"({highlight.duration:.1f}s) - {highlight.category} "
                       f"(confidence: {highlight.confidence:.2f})")
        
    finally:
        await timeline_sync.stop()
        await buffer_manager.close()
        await frame_synchronizer.close()


async def example_multi_stream_sync():
    """Example: Multi-stream synchronization."""
    logger.info("\n=== Multi-Stream Synchronization Example ===")
    
    # Create multi-stream synchronizer
    multi_sync = MultiStreamTimelineSynchronizer()
    
    # Track multi-stream highlights
    multi_highlights: List[Dict[str, Any]] = []
    
    try:
        # Add multiple streams (tournament scenario)
        streams = {
            "player1": (ChatSourceType.TWITCH_EVENTSUB, StreamType.TWITCH_HLS),
            "player2": (ChatSourceType.TWITCH_EVENTSUB, StreamType.TWITCH_HLS),
            "main_stream": (ChatSourceType.TWITCH_EVENTSUB, StreamType.TWITCH_HLS),
            "observer": (ChatSourceType.YOUTUBE_LIVE, StreamType.YOUTUBE_HLS),
        }
        
        synchronizers = {}
        chat_generators = {}
        
        for stream_id, (chat_source, video_type) in streams.items():
            sync = await multi_sync.add_stream(stream_id, chat_source, video_type)
            synchronizers[stream_id] = sync
            chat_generators[stream_id] = ExampleChatGenerator()
            logger.info(f"Added stream: {stream_id}")
        
        # Simulate tournament moment - all streams react to same event
        logger.info("\nSimulating tournament highlight moment...")
        
        # Main event happens
        event_time = datetime.now(timezone.utc)
        
        # Each stream's chat reacts with different timing and intensity
        reactions = {
            "player1": {
                "delay": 0.0,
                "messages": ["WHAT?!", "NO WAY I HIT THAT!", "INSANE SHOT!", "LETS GOOOO!"],
                "intensity": "high"
            },
            "player2": {
                "delay": 0.5,
                "messages": ["Dude how?", "That was sick", "GG", "Respect"],
                "intensity": "medium"
            },
            "main_stream": {
                "delay": 0.2,
                "messages": ["CLIP THAT!", "BEST PLAY OF THE TOURNAMENT!", "LEGENDARY!", "5HEAD PLAY!"],
                "intensity": "high"
            },
            "observer": {
                "delay": 1.0,
                "messages": ["Incredible play!", "Tournament defining moment", "This will be remembered"],
                "intensity": "medium"
            }
        }
        
        # Generate reactions for each stream
        tasks = []
        for stream_id, reaction in reactions.items():
            task = asyncio.create_task(
                simulate_stream_reaction(
                    synchronizers[stream_id],
                    chat_generators[stream_id],
                    reaction["delay"],
                    reaction["messages"],
                    reaction["intensity"]
                )
            )
            tasks.append(task)
        
        # Wait for all reactions to complete
        await asyncio.gather(*tasks)
        
        # Wait for processing
        await asyncio.sleep(5.0)
        
        # Check for multi-stream highlights
        multi_highlights = await multi_sync.get_multi_stream_highlights()
        
        logger.info(f"\n🎯 Multi-stream highlights detected: {len(multi_highlights)}")
        
        for i, highlight in enumerate(multi_highlights, 1):
            logger.info(f"  {i}. Streams: {list(highlight['streams'].keys())}")
            logger.info(f"     Duration: {highlight['end_timestamp'] - highlight['start_timestamp']:.1f}s")
            logger.info(f"     Confidence: {highlight['confidence']:.2f}")
        
        # Get overall status
        status = multi_sync.get_status()
        logger.info(f"\n📊 Multi-stream status:")
        logger.info(f"  Active streams: {status['active_streams']}")
        logger.info(f"  Multi-stream highlights: {status['multi_stream_highlights']}")
        
        for stream_id, stream_status in status["stream_status"].items():
            logger.info(f"  {stream_id}: {stream_status['buffered_events']} events, "
                       f"sync: {stream_status['is_synchronized']}")
        
    finally:
        # Cleanup all streams
        for stream_id in list(synchronizers.keys()):
            await multi_sync.remove_stream(stream_id)


async def simulate_stream_reaction(sync: TimelineSynchronizer, 
                                 chat_gen: ExampleChatGenerator,
                                 delay: float,
                                 messages: List[str],
                                 intensity: str):
    """Simulate a stream's chat reaction to an event."""
    await asyncio.sleep(delay)
    
    if intensity == "high":
        # High intensity: rapid messages with emotes
        for message_text in messages:
            event = chat_gen.generate_regular_message(message_text)
            if hasattr(event.data["message"], "emotes"):
                event.data["message"].emotes = [{"name": "PogChamp"}, {"name": "5Head"}]
            await sync.add_chat_event(event)
            await asyncio.sleep(0.3)
        
        # Add extra hype messages
        hype_events = chat_gen.generate_hype_burst(duration=3.0)
        for event in hype_events:
            await sync.add_chat_event(event)
            await asyncio.sleep(0.2)
    
    else:
        # Medium intensity: slower messages
        for message_text in messages:
            event = chat_gen.generate_regular_message(message_text)
            await sync.add_chat_event(event)
            await asyncio.sleep(1.0)


async def example_vod_analysis():
    """Example: VOD (Video on Demand) analysis."""
    logger.info("\n=== VOD Analysis Example ===")
    
    # Create components for VOD
    buffer_manager = StreamBufferManager()
    frame_synchronizer = FrameSynchronizer()
    
    # VOD synchronizer (perfect sync)
    vod_sync = TimelineSynchronizer(
        stream_id="example_vod",
        chat_source=ChatSourceType.YOUTUBE_VOD,
        video_type=StreamType.YOUTUBE_HLS,
        buffer_manager=buffer_manager,
        frame_synchronizer=frame_synchronizer,
        sync_strategy=SyncStrategy.OFFSET_BASED  # VOD has zero offset
    )
    
    highlights: List[HighlightCandidate] = []
    vod_sync.add_highlight_callback(lambda h: highlights.append(h))
    
    chat_gen = ExampleChatGenerator()
    
    try:
        await vod_sync.start()
        logger.info("VOD synchronizer started")
        
        # Simulate VOD replay with known highlights
        vod_start = datetime.now(timezone.utc)
        
        # Define VOD moments with timestamps
        vod_moments = [
            (0, 30, "intro", 1.0),      # 0-30s: intro, low activity
            (30, 45, "buildup", 3.0),   # 30-45s: building tension
            (45, 55, "climax", 10.0),   # 45-55s: main highlight moment
            (55, 70, "aftermath", 4.0), # 55-70s: reaction to climax
            (70, 90, "analysis", 2.0),  # 70-90s: analyzing what happened
            (90, 100, "outro", 1.0),    # 90-100s: outro
        ]
        
        logger.info("Processing VOD with timed moments...")
        
        for start_time, end_time, moment_type, activity_rate in vod_moments:
            logger.info(f"Processing {moment_type} ({start_time}s-{end_time}s)")
            
            duration = end_time - start_time
            num_messages = int(duration * activity_rate)
            
            for i in range(num_messages):
                # Calculate timestamp within the moment
                progress = i / max(num_messages - 1, 1)
                timestamp = vod_start + timedelta(seconds=start_time + duration * progress)
                
                # Generate appropriate message for the moment
                if moment_type == "buildup":
                    text = f"Something's happening... {i}"
                elif moment_type == "climax":
                    hype_messages = [
                        "HOLY SHIT!", "NO WAY!", "INSANE!", "BEST MOMENT EVER!",
                        "CLIP IT!", "LEGENDARY!", "POGGERS!", "UNREAL!"
                    ]
                    text = hype_messages[i % len(hype_messages)]
                elif moment_type == "aftermath":
                    reaction_messages = [
                        "I can't believe that happened", "Replay please!",
                        "How is that even possible?", "Mind blown",
                        "Best play I've ever seen"
                    ]
                    text = reaction_messages[i % len(reaction_messages)]
                elif moment_type == "analysis":
                    text = f"Technical analysis point {i}"
                else:
                    text = f"{moment_type} message {i}"
                
                # Create event with VOD timestamp
                user = chat_gen.users[i % len(chat_gen.users)]
                message = ChatMessage(
                    id=f"vod_msg_{start_time}_{i}",
                    user=user,
                    text=text,
                    timestamp=timestamp
                )
                
                event = ChatEvent(
                    id=f"vod_event_{start_time}_{i}",
                    type=ChatEventType.MESSAGE,
                    timestamp=timestamp,
                    user=user,
                    data={"message": message}
                )
                
                await vod_sync.add_chat_event(event)
                
                # Small delay for processing
                await asyncio.sleep(0.01)
            
            # Add special events for certain moments
            if moment_type == "climax":
                special_event = chat_gen.generate_special_event(
                    ChatEventType.CHEER,
                    bits=10000,
                    message="TAKE ALL MY BITS!"
                )
                special_event.timestamp = vod_start + timedelta(seconds=start_time + 5)
                await vod_sync.add_chat_event(special_event)
        
        # Process everything
        await asyncio.sleep(2.0)
        
        # Analyze results
        final_highlights = await vod_sync.get_highlights(min_confidence=0.5)
        
        logger.info(f"\n📊 VOD Analysis Results:")
        logger.info(f"Detected highlights: {len(final_highlights)}")
        
        for i, highlight in enumerate(final_highlights, 1):
            start_sec = highlight.start_timestamp - vod_start.timestamp()
            end_sec = highlight.end_timestamp - vod_start.timestamp()
            logger.info(f"  {i}. {start_sec:.1f}s - {end_sec:.1f}s "
                       f"({highlight.duration:.1f}s) - {highlight.category}")
            logger.info(f"     Confidence: {highlight.confidence:.2f}, "
                       f"Sentiment: {highlight.sentiment_score:.2f}")
        
        # The climax moment should be detected as the top highlight
        if final_highlights:
            top_highlight = max(final_highlights, key=lambda h: h.confidence)
            top_start = top_highlight.start_timestamp - vod_start.timestamp()
            logger.info(f"\n🏆 Top highlight: {top_start:.1f}s "
                       f"(confidence: {top_highlight.confidence:.2f})")
    
    finally:
        await vod_sync.stop()
        await buffer_manager.close()
        await frame_synchronizer.close()


async def main():
    """Run all examples."""
    logger.info("🚀 Starting Timeline Synchronization Examples")
    
    try:
        # Run examples in sequence
        await example_single_stream_sync()
        await example_multi_stream_sync()
        await example_vod_analysis()
        
        logger.info("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())