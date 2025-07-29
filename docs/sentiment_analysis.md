# Chat Sentiment Analysis System

## Overview

The TL;DR Highlight API includes a comprehensive chat sentiment analysis system that provides advanced highlight detection through multi-modal sentiment analysis. This system analyzes chat messages using multiple techniques to identify exciting moments in livestreams.

## Key Features

### 1. Multi-Modal Sentiment Analysis
- **Text Sentiment**: Uses VADER, TextBlob, and transformer models to analyze text sentiment
- **Emote-Based Sentiment**: Platform-specific emote meanings (Twitch: PogChamp, Kappa, etc.)
- **Combined Analysis**: Weights text and emote sentiment appropriately

### 2. Chat Velocity and Acceleration
- **Message Frequency**: Tracks messages per second
- **Spike Detection**: Identifies sudden increases in chat activity
- **Acceleration Tracking**: Monitors rate of change in velocity
- **Momentum Calculation**: Detects sustained increases in activity

### 3. Event Impact Scoring
- **Raids**: Highest impact, scaled by viewer count
- **Cheers/Donations**: High impact, scaled by amount
- **Subscriptions**: Moderate impact, scaled by tier
- **Follows**: Lower impact
- **Hype Trains**: Long-lasting impact with gradual decay

### 4. Temporal Analysis
- **Sliding Windows**: Configurable window sizes for analysis
- **Exponential Decay**: Older messages have less weight
- **Cooldown Periods**: Prevents rapid re-triggering after spikes

### 5. Advanced Features
- **Emote Learning**: Learns sentiment of new emotes from usage context
- **Multi-Language Support**: Handles multiple languages gracefully
- **Spam Filtering**: Filters out spam and repetitive messages
- **Keyword Detection**: Identifies hype keywords and phrases

## Usage

### Basic Chat Detector (Enhanced Version)

```python
from src.services.highlight_detection.chat_detector_v2 import EnhancedChatDetector, EnhancedChatDetectionConfig

# Configure the detector
config = EnhancedChatDetectionConfig(
    min_score=0.3,
    min_confidence=0.4,
    sentiment_window_size=10.0,  # 10-second windows
    velocity_spike_weight=0.3,   # 30% weight for velocity spikes
    event_impact_weight=0.2      # 20% weight for special events
)

# Create detector
detector = EnhancedChatDetector(config)

# Detect highlights from segments
results = await detector.detect_highlights(segments)
```

### Direct Sentiment Analysis

```python
from src.services.chat_adapters.sentiment_analyzer import SentimentAnalyzer, ChatMessage, ChatUser

# Create analyzer
analyzer = SentimentAnalyzer(
    window_size_seconds=10.0,
    decay_factor=0.95,
    min_confidence=0.3
)

# Analyze a single message
message = ChatMessage(
    id="msg1",
    user=ChatUser(id="user1", username="viewer", display_name="Viewer"),
    text="That was amazing! PogChamp",
    timestamp=datetime.now(timezone.utc),
    emotes=[{"name": "PogChamp"}]
)

sentiment = await analyzer.analyze_message(message, platform="twitch")
print(f"Sentiment: {sentiment.combined_sentiment:.2f}")
print(f"Category: {sentiment.category}")
print(f"Intensity: {sentiment.intensity:.2f}")
```

### Window Analysis

```python
# Analyze a window of messages
window_sentiment = await analyzer.analyze_window(
    messages,
    start_time,
    end_time,
    platform="twitch"
)

# Get velocity metrics
velocity = analyzer.temporal_analyzer.calculate_velocity_metrics(timestamp)

# Get event impact
event_impact = analyzer.event_tracker.get_current_impact(timestamp)

# Calculate highlight confidence
confidence = analyzer.get_highlight_confidence(
    window_sentiment,
    velocity,
    event_impact
)
```

### Processing Events

```python
from src.services.chat_adapters.base import ChatEvent, ChatEventType

# Process a raid event
raid_event = ChatEvent(
    id="raid1",
    type=ChatEventType.RAID,
    timestamp=datetime.now(timezone.utc),
    data={"viewers": 500}
)

await analyzer.process_event(raid_event)
```

## Configuration Options

### Sentiment Categories and Weights

The system recognizes these sentiment categories with default weights:

- **HYPE**: 1.0 (maximum weight)
- **EXCITEMENT**: 0.9
- **VERY_POSITIVE**: 0.7
- **POSITIVE**: 0.5
- **NEUTRAL**: 0.2
- **NEGATIVE**: 0.1
- **VERY_NEGATIVE**: 0.0
- **DISAPPOINTMENT**: 0.1
- **SURPRISE**: 0.6

### Customizing Emotes

```python
# Add custom emote
analyzer.emote_db.add_custom_emote(
    "CustomHype",
    sentiment_value=0.95,  # Very positive
    intensity=1.0,         # High intensity
    category=SentimentCategory.HYPE,
    platform="custom"
)

# Learn emote from context
contexts = [0.8, 0.9, 0.7, 0.85, 0.75]  # Sentiment values when emote was used
analyzer.emote_db.learn_emote_from_context("NewEmote", contexts)
```

## Performance Considerations

1. **Async Processing**: All analysis is done asynchronously for real-time performance
2. **Sliding Windows**: Efficient sliding window calculations minimize redundant processing
3. **Memory Management**: Old messages are automatically cleaned up
4. **Caching**: Recent analysis results are cached for faster repeated queries

## Metrics and Monitoring

Get current metrics:

```python
metrics = analyzer.get_metrics_summary()
print(f"Messages analyzed: {metrics['total_messages_analyzed']}")
print(f"Active event impacts: {metrics['active_event_impacts']}")
print(f"Current velocity: {metrics['velocity_metrics']['current_mps']} msg/s")
```

## Integration with Highlight Detection

The sentiment analyzer is fully integrated with the highlight detection system:

1. Chat messages are analyzed for sentiment
2. Velocity and acceleration are tracked
3. Special events add impact scores
4. All signals are combined to generate highlight confidence
5. Highlights are ranked and filtered based on thresholds

## Best Practices

1. **Window Size**: Use 5-15 second windows for most streams
2. **Confidence Thresholds**: Set minimum confidence to 0.3-0.5
3. **Event Weights**: Adjust based on your stream type
4. **Spam Filtering**: Enable for public streams
5. **Baseline Reset**: Reset baselines between different streams

## Examples

### Detecting a Raid Highlight

```python
# Messages before raid
normal_messages = [...]  # Regular chat

# Raid happens
raid_event = ChatEvent(
    id="raid1",
    type=ChatEventType.RAID,
    timestamp=raid_time,
    data={"viewers": 1000}
)

# Messages after raid
excited_messages = [...]  # "Welcome raiders!", "PogChamp", etc.

# The system will:
# 1. Detect the velocity spike from increased messages
# 2. Apply high impact score from the raid event
# 3. Analyze positive sentiment from welcome messages
# 4. Generate high confidence highlight score
```

### Custom Highlight Criteria

```python
# Configure for competitive gaming streams
gaming_config = EnhancedChatDetectionConfig(
    sentiment_category_weights={
        SentimentCategory.HYPE: 1.0,
        SentimentCategory.EXCITEMENT: 0.95,
        SentimentCategory.SURPRISE: 0.8,  # Higher weight for surprising plays
        # ... other categories
    },
    velocity_spike_weight=0.4,  # Higher weight for sudden reactions
    min_velocity_spike_intensity=0.4
)

# Configure for casual/variety streams
variety_config = EnhancedChatDetectionConfig(
    sentiment_category_weights={
        SentimentCategory.POSITIVE: 0.7,  # Higher weight for general positivity
        SentimentCategory.HYPE: 0.8,
        # ... other categories
    },
    event_impact_weight=0.3,  # Higher weight for community events
    min_event_impact=0.3
)
```

## Troubleshooting

### Low Confidence Scores
- Check if spam filtering is too aggressive
- Verify emotes are being detected correctly
- Ensure sufficient message volume for analysis

### Missing Highlights
- Lower confidence thresholds
- Adjust sentiment category weights
- Check event processing is working

### Too Many False Positives
- Increase minimum thresholds
- Reduce weights for less important signals
- Enable stricter spam filtering