# Gemini-Based Highlight Processing Flow

This document provides a comprehensive overview of how the TL;DR Highlight API uses Google's Gemini model for AI-powered highlight detection, integrated with the flexible dimension framework.

## Overview

The Gemini-based processing flow represents a significant advancement in highlight detection, combining:
- Google's state-of-the-art multimodal video understanding
- The flexible dimension framework for customizable scoring
- Enterprise-grade processing with comprehensive observability
- Streamlined architecture optimized for production use

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Stream Source                             │
│              (YouTube, Twitch, RTMP, HLS, etc.)                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   FFmpeg Ingestion Layer                         │
│  • Hardware-accelerated processing                              │
│  • Adaptive chunking (30-60s segments)                          │
│  • Multi-format support                                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    B2B Stream Agent                              │
│  • Loads organization's dimension set                           │
│  • Manages processing configuration                             │
│  • Orchestrates highlight detection                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                Gemini Video Processor                            │
│  • Uploads video segments to Gemini                            │
│  • Builds dimension-aware prompts                               │
│  • Processes structured responses                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│              Dimension Score Validation                          │
│  • Ensures all dimensions are scored                            │
│  • Validates score ranges (0.0-1.0)                            │
│  • Calculates weighted final scores                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│             Highlight Type Classification                        │
│  • Matches scores against type criteria                         │
│  • Applies organization's type registry                         │
│  • Generates appropriate tags                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  Storage & Delivery                              │
│  • S3 clip generation                                           │
│  • Webhook notifications                                        │
│  • API response preparation                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Processing Flow

### 1. Stream Ingestion and Segmentation

The process begins when a stream is submitted for processing:

```python
# Stream submission via API
POST /api/v1/streams
{
    "source_url": "https://youtube.com/watch?v=example",
    "title": "Product Launch Event",
    "processing_options": {
        "dimension_set_id": "corporate_events",
        "confidence_threshold": 0.75,
        "max_highlights": 50
    }
}
```

The FFmpeg ingestion layer:
1. **Probes the stream** to determine format and codec information
2. **Creates adaptive segments** based on content characteristics
3. **Extracts keyframes** for visual analysis
4. **Prepares audio tracks** for transcription

### 2. B2B Stream Agent Initialization

The B2BStreamAgent is initialized with:

```python
# Agent initialization
agent = B2BStreamAgent(
    stream=stream_entity,
    agent_config=organization_config,
    gemini_processor=gemini_processor,
    dimension_set=organization_dimension_set,
    processing_options=processing_options
)
```

Key responsibilities:
- **Configuration Management**: Loads organization-specific settings
- **Dimension Set Loading**: Retrieves the appropriate dimension definitions
- **Context Tracking**: Maintains processing context for better analysis
- **Metrics Collection**: Tracks performance and quality metrics

### 3. Dimension-Aware Prompt Construction

The system builds sophisticated prompts that incorporate dimension definitions:

```python
def _build_dimension_aware_prompt(self, dimension_set, agent_config, segment_info):
    """
    Constructs a comprehensive prompt that:
    1. Includes base instructions from agent config
    2. Adds dimension-specific scoring guidance
    3. Provides context about the content type
    4. Specifies output format requirements
    """
    
    # Example generated prompt structure:
    prompt = f"""
    You are analyzing a {content_type} video for highlight detection.
    
    ## Dimension-Based Scoring Framework
    
    ### Decision Point (ID: decision_point)
    - Weight: 0.30 (High importance)
    - Description: Moment where key decision is made
    - Type: binary
    - Threshold: 1.0
    - Scoring Guidance: Mark 1.0 if a concrete decision is made, 0.0 otherwise
    
    ### Stakeholder Alignment (ID: stakeholder_alignment)
    - Weight: 0.25 (High importance)
    - Description: Level of agreement among participants
    - Type: numeric
    - Threshold: 0.7
    - Scoring Guidance: Rate 0-1 based on verbal agreement and body language
    - Examples:
      - High (0.9-1.0): Clear consensus, positive body language
      - Medium (0.5-0.8): General agreement with minor concerns
      - Low (0.0-0.4): Disagreement or lack of engagement
    
    [Additional dimensions...]
    
    ## Analysis Requirements
    1. Score EVERY dimension listed above
    2. Ensure scores are between 0.0 and 1.0
    3. Calculate weighted score using provided weights
    4. Focus on moments scoring highly across multiple dimensions
    
    ## Output Format
    Return structured JSON with:
    - highlights: Array of detected highlights
    - dimension_scores: Dictionary mapping dimension IDs to scores
    - confidence: Overall confidence in the detection
    - ranking_score: Weighted combination of dimension scores
    """
```

### 4. Gemini Video Analysis

The GeminiVideoProcessor performs the core analysis:

```python
async def analyze_video_with_dimensions(
    self,
    video_path: str,
    segment_info: Dict[str, Any],
    dimension_set: DimensionSet,
    agent_config: HighlightAgentConfig,
) -> GeminiVideoAnalysis:
    """
    Core Gemini analysis process:
    """
    # 1. Upload video segment to Gemini
    uploaded_file = await self._upload_video_with_retry(
        video_path, 
        segment_info.get("id")
    )
    
    # 2. Generate structured response
    response = await self._generate_with_structured_output(
        file_uri=uploaded_file.uri,
        prompt=dimension_aware_prompt,
        response_schema=GeminiVideoAnalysis,
        config={
            "temperature": 0.2,  # Low for consistency
            "top_p": 0.8,
            "top_k": 40,
            "response_mime_type": "application/json"
        }
    )
    
    # 3. Validate dimension scores
    validated_response = self._validate_dimension_scores(
        response, 
        dimension_set
    )
    
    return validated_response
```

#### Gemini's Multimodal Understanding

Gemini processes multiple modalities simultaneously:

1. **Visual Analysis**
   - Object detection and tracking
   - Action recognition
   - Scene understanding
   - Body language analysis
   - Visual quality assessment

2. **Audio Analysis**
   - Speech transcription
   - Speaker diarization
   - Emotion detection
   - Audio event recognition
   - Music and sound effect identification

3. **Contextual Understanding**
   - Temporal relationships
   - Cause-and-effect reasoning
   - Narrative flow comprehension
   - Multi-speaker conversation tracking

### 5. Structured Response Processing

Gemini returns structured responses matching our schema:

```json
{
    "highlights": [
        {
            "start_time": "01:23",
            "end_time": "02:15",
            "confidence": 0.92,
            "type": "strategic_decision",
            "description": "CEO announces pivot to AI-first strategy with board approval",
            "ranking_score": 0.88,
            "dimension_scores": {
                "decision_point": 1.0,
                "stakeholder_alignment": 0.85,
                "financial_impact": 0.9,
                "strategic_importance": 0.95
            },
            "key_moments": ["01:35", "01:58", "02:10"],
            "transcript_excerpt": "After careful consideration, we're committing to an AI-first approach...",
            "viewer_impact": "Major strategic shift likely to generate significant interest"
        }
    ],
    "segment_quality": {
        "overall_score": 0.85,
        "has_highlights": true,
        "audio_quality": "excellent",
        "video_quality": "good",
        "content_richness": 0.9,
        "engagement_potential": 0.88
    }
}
```

### 6. Dimension Score Validation

The system ensures data integrity:

```python
def _validate_dimension_scores(self, analysis, dimension_set):
    """
    Validation process:
    1. Ensures all required dimensions have scores
    2. Adds missing dimensions with 0.0 scores
    3. Removes any extra dimensions not in the set
    4. Recalculates final scores using dimension weights
    """
    
    for highlight in analysis.highlights:
        # Ensure completeness
        for dim_id in dimension_set.dimensions.keys():
            if dim_id not in highlight.dimension_scores:
                highlight.dimension_scores[dim_id] = 0.0
        
        # Remove extras
        valid_dims = set(dimension_set.dimensions.keys())
        highlight.dimension_scores = {
            k: v for k, v in highlight.dimension_scores.items()
            if k in valid_dims
        }
        
        # Recalculate weighted score
        highlight.ranking_score = dimension_set.calculate_score(
            highlight.dimension_scores
        )
```

### 7. Highlight Candidate Creation

Validated highlights are converted to candidates:

```python
def convert_to_highlight_candidates(
    self,
    analysis: GeminiVideoAnalysis,
    dimension_set: DimensionSet,
    segment_info: Dict[str, Any],
    min_confidence: float = 0.7,
) -> List[HighlightCandidate]:
    """
    Conversion process includes:
    1. Confidence threshold filtering
    2. Dimension threshold validation
    3. Timestamp adjustment to stream time
    4. Metadata enrichment
    5. Keyword extraction
    """
    
    candidates = []
    
    for highlight in analysis.highlights:
        # Check thresholds
        if highlight.confidence < min_confidence:
            continue
            
        # Validate dimension requirements
        meets_thresholds = sum(
            1 for dim_id, score in highlight.dimension_scores.items()
            if dimension_set.dimensions[dim_id].meets_threshold(score)
        )
        
        if meets_thresholds < dimension_set.minimum_dimensions_required:
            continue
        
        # Create candidate with full metadata
        candidate = HighlightCandidate(
            id=str(uuid.uuid4()),
            start_time=segment_start + highlight.start_time,
            end_time=segment_start + highlight.end_time,
            description=highlight.description,
            confidence=highlight.confidence,
            dimensions=highlight.dimension_scores,
            final_score=highlight.ranking_score,
            metadata={
                "gemini_analysis": True,
                "dimension_set": dimension_set.name,
                "segment_id": segment_info["id"],
                "meets_thresholds": meets_thresholds,
                "transcript": highlight.transcript_excerpt,
                "key_moments": highlight.key_moments
            }
        )
        
        candidates.append(candidate)
    
    return sorted(candidates, key=lambda c: c.final_score, reverse=True)
```

### 8. Highlight Type Classification

The system matches candidates against the organization's type registry:

```python
def classify_highlight_type(candidate, type_registry):
    """
    Classification process:
    1. Evaluate candidate against each type's criteria
    2. Check min_scores requirements
    3. Verify any_of_scores conditions
    4. Apply score formulas if defined
    5. Select highest priority matching type
    """
    
    matching_types = []
    
    for highlight_type in type_registry.get_active_types():
        if meets_criteria(candidate, highlight_type.criteria):
            matching_types.append({
                "type": highlight_type,
                "priority": highlight_type.priority,
                "score": calculate_type_score(candidate, highlight_type)
            })
    
    # Sort by priority (lower is higher) then score
    matching_types.sort(key=lambda x: (x["priority"], -x["score"]))
    
    return matching_types[0]["type"] if matching_types else None
```

### 9. Final Processing and Storage

Once highlights are created:

1. **Clip Generation**
   ```python
   # Generate video clips for each highlight
   clip_url = await clip_generator.generate_clip(
       stream_url=stream.source_url,
       start_time=highlight.start_time,
       end_time=highlight.end_time,
       output_format="mp4",
       quality_preset="high"
   )
   ```

2. **Thumbnail Creation**
   ```python
   # Extract representative thumbnail
   thumbnail_url = await thumbnail_generator.generate_thumbnail(
       video_url=clip_url,
       timestamp=highlight.peak_time,
       size=(1280, 720)
   )
   ```

3. **Caption Generation**
   ```python
   # Generate AI captions
   captions = await caption_generator.generate_captions(
       video_url=clip_url,
       style="professional",
       include_speaker_labels=True
   )
   ```

4. **Webhook Notifications**
   ```python
   # Send webhook to client
   await webhook_dispatcher.dispatch_webhook(
       stream_id=stream.id,
       event=WebhookEvent.HIGHLIGHT_DETECTED,
       data={
           "highlight_id": highlight.id,
           "type": highlight.type,
           "confidence": highlight.confidence,
           "dimension_scores": highlight.dimension_scores,
           "clip_url": clip_url,
           "thumbnail_url": thumbnail_url
       }
   )
   ```

## Performance Optimizations

### 1. Parallel Processing

The system leverages parallelism at multiple levels:

```python
# Parallel segment analysis
async def process_segments_parallel(segments):
    tasks = []
    for segment in segments:
        task = asyncio.create_task(
            analyze_segment_with_gemini(segment)
        )
        tasks.append(task)
    
    # Process up to 5 segments concurrently
    results = []
    for batch in chunked(tasks, 5):
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    
    return results
```

### 2. Intelligent Caching

While the streamlined system removes complex caching, it implements smart strategies:

- **Dimension Set Caching**: Frequently used dimension sets are cached in memory
- **Configuration Caching**: Agent configurations are cached per organization
- **Type Registry Caching**: Highlight type definitions are cached with TTL

### 3. Progressive Analysis

The system can perform quick pre-filtering:

```python
def should_analyze_segment(segment_metadata):
    """
    Quick checks before expensive Gemini analysis:
    1. Audio level threshold (skip silent segments)
    2. Motion detection (skip static segments)
    3. Scene change detection (focus on dynamic content)
    """
    
    if segment_metadata["avg_audio_level"] < 0.1:
        return False
    
    if segment_metadata["motion_score"] < 0.05:
        return False
    
    return True
```

### 4. Resource Management

Efficient resource utilization:

```python
class GeminiVideoProcessor:
    def __init__(self):
        # Connection pooling
        self.client = genai.Client(
            api_key=api_key,
            connection_pool_size=10,
            timeout=30
        )
        
        # File cleanup tracking
        self.uploaded_files = {}
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Automatic cleanup of uploaded files"""
        await self.cleanup_all_files()
```

## Observability and Monitoring

### 1. Comprehensive Tracing

Every step is traced with Logfire:

```python
with logfire.span("gemini_dimension_analysis") as span:
    span.set_attribute("segment.id", segment_id)
    span.set_attribute("dimension_set.name", dimension_set.name)
    span.set_attribute("dimension_count", len(dimensions))
    
    # Process segment
    result = await process_segment()
    
    span.set_attribute("highlights.found", len(result.highlights))
    span.set_attribute("processing.duration_ms", processing_time)
    span.set_attribute("confidence.avg", avg_confidence)
```

### 2. Metrics Collection

Key metrics tracked:

- **Processing Metrics**
  - Segments processed per minute
  - Average processing time per segment
  - Gemini API response times
  - Error rates and retry counts

- **Quality Metrics**
  - Average confidence scores
  - Dimension score distributions
  - Highlight detection rates
  - Type classification accuracy

- **Business Metrics**
  - Highlights per stream
  - Processing costs per organization
  - API usage by dimension set
  - Popular dimension configurations

### 3. Error Tracking

Comprehensive error handling:

```python
@traced_service_method(name="analyze_with_gemini")
async def analyze_segment(self, segment):
    try:
        # Processing logic
        pass
    except GeminiAPIError as e:
        logfire.error(
            "gemini.api.error",
            error_type=type(e).__name__,
            error_message=str(e),
            segment_id=segment.id,
            retry_count=retry_count
        )
        raise
    except Exception as e:
        logfire.error(
            "gemini.processing.error",
            error=str(e),
            segment_id=segment.id,
            traceback=traceback.format_exc()
        )
        raise ProcessingError(f"Gemini analysis failed: {str(e)}")
```

## Configuration Examples

### Gaming Content Configuration

```python
# Gaming-optimized configuration
gaming_config = {
    "dimension_set": {
        "id": "competitive_gaming",
        "dimensions": {
            "skill_display": {"weight": 0.3},
            "clutch_moment": {"weight": 0.25},
            "audience_reaction": {"weight": 0.2},
            "game_state_importance": {"weight": 0.15},
            "entertainment_value": {"weight": 0.1}
        }
    },
    "processing_options": {
        "segment_duration": 30,
        "confidence_threshold": 0.7,
        "max_highlights_per_hour": 20,
        "prefer_action_sequences": true
    },
    "gemini_config": {
        "model": "gemini-2.0-flash-exp",
        "temperature": 0.3,
        "focus_modalities": ["video", "audio"]
    }
}
```

### Corporate Meeting Configuration

```python
# Corporate meeting configuration
corporate_config = {
    "dimension_set": {
        "id": "corporate_meetings",
        "dimensions": {
            "decision_point": {"weight": 0.35},
            "action_item": {"weight": 0.25},
            "stakeholder_consensus": {"weight": 0.2},
            "financial_discussion": {"weight": 0.2}
        }
    },
    "processing_options": {
        "segment_duration": 60,
        "confidence_threshold": 0.8,
        "max_highlights_per_hour": 10,
        "prioritize_speech": true
    },
    "gemini_config": {
        "model": "gemini-2.0-flash-exp",
        "temperature": 0.2,
        "focus_modalities": ["audio", "text"],
        "enable_speaker_diarization": true
    }
}
```

## Integration Points

### 1. API Integration

```python
# Create stream with Gemini processing
POST /api/v1/streams
{
    "source_url": "https://example.com/stream",
    "processing_options": {
        "dimension_set_id": "custom_dimensions_v2",
        "detection_strategy": "gemini",
        "confidence_threshold": 0.75
    }
}

# Response includes Gemini processing status
{
    "id": "stream_123",
    "status": "processing",
    "processing_details": {
        "processor": "gemini-2.0-flash-exp",
        "dimension_set": "custom_dimensions_v2",
        "estimated_completion": "2024-01-20T15:30:00Z"
    }
}
```

### 2. Webhook Events

```json
{
    "event": "highlight.detected",
    "timestamp": "2024-01-20T14:23:45Z",
    "data": {
        "highlight_id": "hl_abc123",
        "stream_id": "stream_123",
        "detection_method": "gemini",
        "confidence": 0.92,
        "dimension_scores": {
            "action_intensity": 0.95,
            "skill_display": 0.88,
            "audience_reaction": 0.91
        },
        "type": "epic_gameplay",
        "clip_url": "https://cdn.example.com/clips/hl_abc123.mp4"
    }
}
```

### 3. SDK Usage

```python
from tldr_highlight_api import TLDRClient

client = TLDRClient(api_key="your_api_key")

# Process stream with Gemini
stream = client.streams.create(
    source_url="https://youtube.com/watch?v=example",
    options={
        "dimension_set_id": "gaming_default",
        "processor": "gemini",
        "gemini_options": {
            "model": "gemini-2.0-flash-exp",
            "enable_refinement": False,
            "parallel_segments": 3
        }
    }
)

# Monitor processing
async for update in client.streams.monitor(stream.id):
    if update.event == "segment.processed":
        print(f"Processed segment: {update.data.segment_id}")
        print(f"Highlights found: {update.data.highlight_count}")
```

## Best Practices

### 1. Dimension Set Design for Gemini

- **Clear Descriptions**: Provide detailed descriptions that Gemini can interpret
- **Concrete Examples**: Include specific examples in dimension definitions
- **Balanced Weights**: Avoid over-weighting single dimensions
- **Appropriate Thresholds**: Set thresholds based on testing with Gemini

### 2. Prompt Engineering

- **Structured Format**: Use consistent formatting in prompts
- **Context Inclusion**: Provide relevant context about content type
- **Output Specification**: Clearly define expected output format
- **Iterative Refinement**: Test and refine prompts based on results

### 3. Performance Optimization

- **Segment Duration**: Balance between context and processing time (30-60s optimal)
- **Parallel Processing**: Process multiple segments concurrently when possible
- **Pre-filtering**: Use quick checks to skip low-value segments
- **Resource Cleanup**: Always clean up uploaded files

### 4. Error Handling

- **Retry Logic**: Implement exponential backoff for API failures
- **Fallback Strategies**: Have backup processing options
- **Partial Failures**: Handle individual segment failures gracefully
- **Monitoring**: Track error rates and patterns

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Review dimension scoring prompts
   - Check segment quality (audio/video)
   - Verify appropriate model selection
   - Adjust temperature settings

2. **Missing Dimensions**
   - Ensure dimension set is properly loaded
   - Check prompt includes all dimensions
   - Verify Gemini response parsing
   - Review validation logic

3. **Performance Issues**
   - Monitor Gemini API latency
   - Check segment sizes
   - Review parallel processing settings
   - Optimize file upload/cleanup

4. **Inconsistent Results**
   - Verify prompt consistency
   - Check temperature settings
   - Review dimension weights
   - Monitor model versions

## Future Enhancements

1. **Multi-Model Ensemble**: Combine multiple Gemini models for improved accuracy
2. **Real-time Processing**: Stream processing for live events
3. **Custom Model Fine-tuning**: Organization-specific model adaptations
4. **Advanced Caching**: Intelligent result caching based on content similarity
5. **Predictive Analysis**: Anticipate highlight moments before they occur

## Conclusion

The Gemini-based highlight processing flow represents a powerful, flexible system for AI-powered content analysis. By combining Google's advanced video understanding capabilities with our customizable dimension framework, organizations can define precisely what constitutes a "highlight" for their specific use cases while leveraging state-of-the-art AI technology.

The streamlined architecture ensures reliable, scalable processing while maintaining the flexibility needed for diverse industry requirements. With comprehensive observability and monitoring, organizations can continuously optimize their highlight detection strategies based on real-world performance data.

---

*Related Documentation:*
- [Customizable Dimensions Framework](./customizable-dimensions-framework.md)
- [Flexible Highlight Detection](./flexible-highlight-detection.md)
- [API Reference](../api/overview.md)
- [Architecture Overview](../architecture/overview.md)