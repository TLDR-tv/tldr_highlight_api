# Customizable Dimension Framework

The TL;DR Highlight API's customizable dimension framework is a groundbreaking system that allows organizations to define exactly what constitutes a "highlight" for their specific content and use cases. This document provides a comprehensive guide to understanding and implementing custom dimensions.

## Framework Overview

The dimension framework is built on four core pillars:

1. **Dimensions**: Measurable aspects of content (e.g., action intensity, educational value)
2. **Dimension Sets**: Grouped collections of dimensions for specific use cases
3. **Highlight Types**: Dynamic definitions of what constitutes a highlight based on dimension scores
4. **Analysis Strategies**: Methods for scoring content against dimensions

## Understanding Dimensions

### What is a Dimension?

A dimension is a single measurable characteristic of content. Unlike traditional systems that use fixed categories like "funny" or "exciting," dimensions allow you to define precise, measurable criteria that matter to your specific use case.

### Dimension Anatomy

Every dimension consists of:

```json
{
    "id": "engagement_level",
    "name": "Audience Engagement Level",
    "type": "numeric",
    "description": "Measures audience interaction and engagement with content",
    "range": [0.0, 1.0],
    "weight": 0.25,
    "modalities": ["video", "audio", "text"],
    "scoring_prompt": "Analyze audience engagement based on:\n- Viewer reactions and expressions\n- Audio cues (gasps, laughter, exclamations)\n- Chat velocity and sentiment\nScore from 0 (no engagement) to 1 (maximum engagement)",
    "examples": {
        "high": "Audience visibly excited, lots of positive chat messages, audible reactions",
        "medium": "Some viewer reactions, moderate chat activity",
        "low": "Minimal visible or audible audience response"
    },
    "aggregation": "weighted_mean",
    "threshold": 0.3,
    "metadata": {
        "category": "audience",
        "requires_live_data": true
    }
}
```

### Dimension Types

#### Numeric Dimensions
- Score range: Typically 0.0 to 1.0
- Use case: Gradual measurements (intensity, quality, clarity)
- Example: `action_intensity`, `educational_value`, `production_quality`

#### Binary Dimensions
- Score values: 0 (false) or 1 (true)
- Use case: Yes/no criteria
- Example: `contains_profanity`, `brand_visible`, `goal_scored`

#### Categorical Dimensions
- Score values: Mapped to numeric values
- Use case: Classification-based scoring
- Example: `content_type`, `emotion_displayed`, `game_phase`

### Dimension Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `name` | string | Yes | Human-readable name |
| `type` | enum | Yes | `numeric`, `binary`, or `categorical` |
| `description` | string | Yes | Detailed description of what this measures |
| `weight` | float | Yes | Importance weight (0.0-1.0) |
| `modalities` | array | Yes | Which input types to analyze |
| `scoring_prompt` | string | Yes | Instructions for AI/rule-based scoring |
| `examples` | object | No | Examples of high/medium/low scores |
| `aggregation` | enum | No | How to combine scores over time |
| `threshold` | float | No | Minimum score to consider |
| `metadata` | object | No | Additional configuration |

## Dimension Sets

### Purpose

Dimension sets group related dimensions for specific use cases. They allow you to:
- Organize dimensions by industry or content type
- Apply consistent weighting across related dimensions
- Share configurations across teams
- Switch between different analysis profiles

### Structure

```json
{
    "id": "esports_tournament",
    "name": "Esports Tournament Analysis",
    "description": "Optimized for competitive gaming tournaments",
    "dimensions": {
        "skill_display": {
            "weight": 0.3,
            "threshold": 0.7
        },
        "clutch_moment": {
            "weight": 0.25,
            "threshold": 0.8
        },
        "team_coordination": {
            "weight": 0.2,
            "threshold": 0.6
        },
        "crowd_reaction": {
            "weight": 0.15,
            "threshold": 0.7
        },
        "game_changing_play": {
            "weight": 0.1,
            "threshold": 0.9
        }
    },
    "normalize_weights": true,
    "min_dimensions_required": 3,
    "fallback_set_id": "gaming_default",
    "metadata": {
        "optimized_for": ["fps", "moba", "fighting"],
        "typical_stream_length": "3-6 hours"
    }
}
```

### Weight Normalization

When `normalize_weights` is true, the system automatically adjusts weights to sum to 1.0:

```python
# Original weights
weights = {"A": 0.5, "B": 0.3, "C": 0.4}  # Sum = 1.2

# After normalization
normalized = {"A": 0.417, "B": 0.25, "C": 0.333}  # Sum = 1.0
```

## Multi-Modal Analysis

### Modality Configuration

Dimensions can analyze different input types:

```json
{
    "visual_complexity": {
        "modalities": ["video"],
        "modality_config": {
            "video": {
                "frame_sampling_rate": 5,
                "color_analysis": true,
                "motion_detection": true
            }
        }
    },
    "dialogue_clarity": {
        "modalities": ["audio", "text"],
        "modality_config": {
            "audio": {
                "noise_reduction": true,
                "speaker_diarization": true
            },
            "text": {
                "source": "transcription",
                "language_detection": true
            }
        }
    }
}
```

### Fusion Strategies

#### Weighted Fusion
Combines scores from different modalities with specified weights:

```python
# Configuration
fusion_config = {
    "strategy": "weighted",
    "weights": {
        "video": 0.5,
        "audio": 0.3,
        "text": 0.2
    }
}

# Result: dimension_score = (video_score * 0.5) + (audio_score * 0.3) + (text_score * 0.2)
```

#### Consensus Fusion
Requires agreement between modalities:

```python
# Configuration
fusion_config = {
    "strategy": "consensus",
    "min_agreement": 2,
    "agreement_threshold": 0.6
}

# Only includes dimension if at least 2 modalities score >= 0.6
```

#### Cascade Fusion
Processes modalities in order, stopping when confident:

```python
# Configuration
fusion_config = {
    "strategy": "cascade",
    "order": ["video", "audio", "text"],
    "confidence_threshold": 0.8
}

# Checks video first, only processes audio if video confidence < 0.8
```

#### Max Confidence Fusion
Takes the highest score across modalities:

```python
# Configuration
fusion_config = {
    "strategy": "max_confidence"
}

# Result: dimension_score = max(video_score, audio_score, text_score)
```

## Dynamic Highlight Types

### Type Definition

Highlight types define what combinations of dimension scores constitute a highlight:

```json
{
    "id": "educational_breakthrough",
    "name": "Educational Breakthrough Moment",
    "description": "Key learning moment with clear understanding",
    "criteria": {
        "min_scores": {
            "concept_clarity": 0.8,
            "student_understanding": 0.7
        },
        "any_of_scores": {
            "aha_moment": 0.9,
            "problem_solved": 0.8,
            "misconception_corrected": 0.85
        },
        "required_dimensions": ["concept_clarity"],
        "min_total_score": 0.75,
        "score_formula": "(concept_clarity * 0.4) + (student_understanding * 0.3) + (max(aha_moment, problem_solved, misconception_corrected) * 0.3)",
        "min_duration": 15,
        "max_duration": 180,
        "cooldown_period": 30
    },
    "priority": 1,
    "auto_tag": ["education", "breakthrough", "learning"],
    "metadata_template": {
        "highlight_color": "#4CAF50",
        "thumbnail_overlay": "lightbulb_icon",
        "export_format": "with_context"
    }
}
```

### Criteria Types

#### Absolute Requirements (`min_scores`)
All specified dimensions must meet minimum thresholds:

```json
"min_scores": {
    "accuracy": 0.9,
    "complexity": 0.7
}
```

#### Alternative Requirements (`any_of_scores`)
At least one dimension must meet threshold:

```json
"any_of_scores": {
    "surprise_factor": 0.8,
    "plot_twist": 0.9,
    "revelation": 0.85
}
```

#### Formula-Based (`score_formula`)
Custom mathematical expressions:

```json
"score_formula": "(action * 0.4) + (skill * 0.3) + (sqrt(intensity) * 0.3)"
```

Supported operations:
- Basic math: `+`, `-`, `*`, `/`, `^`
- Functions: `sqrt()`, `min()`, `max()`, `avg()`, `abs()`
- Conditionals: `if(condition, true_value, false_value)`

### Priority and Conflict Resolution

When multiple highlight types match:

1. **Priority Order**: Lower priority numbers win (1 = highest)
2. **Score Tiebreaker**: Higher total scores win
3. **Specificity**: More specific criteria win
4. **Timestamp**: Earlier detection wins

## Analysis Strategies

### AI-Only Strategy

Uses large language models for nuanced understanding:

```python
class AIAnalysisStrategy:
    def __init__(self, model="gemini-1.5-pro", temperature=0.3):
        self.model = model
        self.temperature = temperature
    
    async def analyze_segment(self, segment, dimensions):
        # Build structured prompt
        prompt = self._build_analysis_prompt(dimensions, segment.metadata)
        
        # Prepare multimodal input
        inputs = {
            "video": segment.video_frames,
            "audio": segment.audio_transcript,
            "metadata": segment.metadata
        }
        
        # Get AI analysis
        response = await self.llm.analyze(
            prompt=prompt,
            inputs=inputs,
            response_format="structured_json",
            temperature=self.temperature
        )
        
        return self._validate_scores(response.scores, dimensions)
```

Best for:
- Subjective criteria (humor, emotion, quality)
- Complex pattern recognition
- Nuanced understanding required

### Rule-Based Strategy

Deterministic analysis using defined rules:

```python
class RuleBasedStrategy:
    def analyze_segment(self, segment, dimensions):
        scores = {}
        
        for dim_id, dimension in dimensions.items():
            if dim_id == "motion_intensity":
                scores[dim_id] = self._calculate_motion(segment.frames)
            elif dim_id == "audio_peaks":
                scores[dim_id] = self._analyze_audio_levels(segment.audio)
            elif dim_id == "scene_change":
                scores[dim_id] = self._detect_scene_changes(segment.frames)
            elif dim_id == "keyword_mentioned":
                scores[dim_id] = self._check_keywords(segment.transcript)
        
        return scores
    
    def _calculate_motion(self, frames):
        # Optical flow analysis
        motion_vectors = cv2.calcOpticalFlowFarneback(frames[:-1], frames[1:])
        return np.mean(np.abs(motion_vectors))
```

Best for:
- Objective measurements
- Performance-critical applications
- Consistent, repeatable results

### Hybrid Strategy

Combines AI and rules for optimal performance:

```python
class HybridStrategy:
    def __init__(self, rule_weight=0.3, ai_weight=0.7):
        self.rule_strategy = RuleBasedStrategy()
        self.ai_strategy = AIAnalysisStrategy()
        self.weights = {"rule": rule_weight, "ai": ai_weight}
    
    async def analyze_segment(self, segment, dimensions):
        # Categorize dimensions
        rule_dims = {d: v for d, v in dimensions.items() 
                     if v.get("prefer_rules", False)}
        ai_dims = {d: v for d, v in dimensions.items() 
                   if d not in rule_dims}
        
        # Parallel analysis
        rule_scores, ai_scores = await asyncio.gather(
            self.rule_strategy.analyze_segment(segment, rule_dims),
            self.ai_strategy.analyze_segment(segment, ai_dims)
        )
        
        # Merge results
        return {**rule_scores, **ai_scores}
```

## Implementation Examples

### Gaming Industry

```python
# Create gaming-specific dimensions
gaming_dimensions = {
    "mechanical_skill": {
        "type": "numeric",
        "weight": 0.3,
        "modalities": ["video"],
        "scoring_prompt": "Rate player mechanical skill (0-1) based on:\n- Precision of inputs\n- Reaction time\n- Complex combo execution\n- Movement efficiency",
        "examples": {
            "high": "Frame-perfect inputs, instant reactions, flawless execution",
            "medium": "Good timing, occasional mistakes, solid fundamentals",
            "low": "Slow reactions, frequent errors, basic execution"
        }
    },
    "strategic_depth": {
        "type": "numeric",
        "weight": 0.25,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Evaluate strategic decision-making (0-1) considering:\n- Resource management\n- Positioning choices\n- Risk/reward assessment\n- Adaptation to opponent"
    },
    "clutch_factor": {
        "type": "binary",
        "weight": 0.2,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Is this a clutch moment where player overcomes significant disadvantage to succeed?"
    }
}

# Define highlight types
gaming_highlights = [
    {
        "id": "skill_ceiling_play",
        "criteria": {
            "min_scores": {
                "mechanical_skill": 0.9,
                "strategic_depth": 0.7
            },
            "min_total_score": 0.85
        }
    },
    {
        "id": "comeback_victory",
        "criteria": {
            "min_scores": {
                "clutch_factor": 1.0,
                "health_deficit": 0.7
            },
            "score_formula": "clutch_factor * (1 + health_deficit)"
        }
    }
]
```

### Education Sector

```python
# Educational content dimensions
education_dimensions = {
    "conceptual_clarity": {
        "type": "numeric",
        "weight": 0.35,
        "modalities": ["audio", "video"],
        "scoring_prompt": "Rate how clearly the concept is explained (0-1):\n- Logical progression\n- Use of examples\n- Visual aids effectiveness\n- Addressing common misconceptions"
    },
    "student_engagement": {
        "type": "numeric",
        "weight": 0.25,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Measure student engagement level (0-1):\n- Visible attention/focus\n- Questions asked\n- Active participation\n- Body language"
    },
    "knowledge_checkpoint": {
        "type": "binary",
        "weight": 0.2,
        "modalities": ["audio", "text"],
        "scoring_prompt": "Does this segment contain a knowledge check, quiz, or comprehension verification?"
    },
    "practical_application": {
        "type": "numeric",
        "weight": 0.2,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Rate the practical applicability shown (0-1):\n- Real-world examples\n- Hands-on demonstration\n- Problem-solving shown"
    }
}

# Educational highlight types
education_highlights = [
    {
        "id": "key_concept_explained",
        "criteria": {
            "min_scores": {
                "conceptual_clarity": 0.8,
                "student_engagement": 0.6
            }
        }
    },
    {
        "id": "breakthrough_moment",
        "criteria": {
            "min_scores": {
                "student_engagement": 0.9
            },
            "any_of_scores": {
                "aha_vocalization": 0.8,
                "understanding_demonstrated": 0.9
            }
        }
    }
]
```

### Corporate/Business Use

```python
# Corporate meeting dimensions
corporate_dimensions = {
    "decision_point": {
        "type": "binary",
        "weight": 0.3,
        "modalities": ["audio", "text"],
        "scoring_prompt": "Was a concrete decision made or agreed upon in this segment?"
    },
    "action_item_created": {
        "type": "binary",
        "weight": 0.25,
        "modalities": ["audio", "text"],
        "scoring_prompt": "Was a specific action item assigned with owner and deadline?"
    },
    "stakeholder_alignment": {
        "type": "numeric",
        "weight": 0.2,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Rate stakeholder alignment (0-1):\n- Verbal agreement\n- Body language\n- Consensus building\n- Objections resolved"
    },
    "financial_impact": {
        "type": "numeric",
        "weight": 0.25,
        "modalities": ["audio", "text"],
        "scoring_prompt": "Rate financial significance discussed (0-1):\n- Dollar amounts mentioned\n- Budget implications\n- ROI discussed\n- Cost/benefit analysis"
    }
}

# Corporate highlight types
corporate_highlights = [
    {
        "id": "strategic_decision",
        "criteria": {
            "min_scores": {
                "decision_point": 1.0,
                "stakeholder_alignment": 0.7
            },
            "any_of_scores": {
                "financial_impact": 0.8,
                "strategic_importance": 0.8
            }
        }
    },
    {
        "id": "action_plan_created",
        "criteria": {
            "min_scores": {
                "action_item_created": 1.0
            },
            "min_total_score": 0.6
        }
    }
]
```

## Advanced Configuration

### Temporal Analysis

Control how dimensions are evaluated over time:

```json
{
    "temporal_config": {
        "window_size": 30,
        "step_size": 5,
        "aggregation_method": "weighted_mean",
        "weights_decay": "exponential",
        "min_segment_duration": 10,
        "max_segment_duration": 300,
        "boundary_detection": {
            "method": "score_change",
            "threshold": 0.3,
            "smoothing": "gaussian"
        }
    }
}
```

### Confidence Scoring

Configure confidence thresholds:

```json
{
    "confidence_config": {
        "min_confidence": 0.5,
        "confidence_bands": {
            "high": {"min": 0.8, "label": "Definite Highlight"},
            "medium": {"min": 0.6, "label": "Probable Highlight"},
            "low": {"min": 0.4, "label": "Possible Highlight"}
        },
        "confidence_factors": {
            "modality_agreement": 0.3,
            "score_consistency": 0.3,
            "dimension_coverage": 0.2,
            "analysis_quality": 0.2
        }
    }
}
```

### Custom Modality Processors

Register custom processors for specialized content:

```python
class ChessAnalysisProcessor:
    def process(self, segment):
        # Extract chess position
        position = self.extract_board_position(segment.frames)
        
        # Analyze with chess engine
        analysis = self.chess_engine.analyze(position)
        
        return {
            "position_complexity": analysis.complexity_score,
            "move_brilliancy": analysis.brilliancy_score,
            "tactical_opportunity": analysis.has_tactics,
            "endgame_transition": analysis.is_endgame
        }

# Register processor
dimension_config = {
    "chess_brilliancy": {
        "type": "numeric",
        "modalities": ["custom:chess"],
        "custom_processor": ChessAnalysisProcessor(),
        "weight": 0.4
    }
}
```

## Performance Optimization

### Dimension Set Caching

```python
# Cache frequently used dimension sets
cache_config = {
    "dimension_set_cache": {
        "max_size": 100,
        "ttl": 3600,
        "preload": ["gaming_default", "education_default"]
    }
}
```

### Progressive Analysis

```python
# Analyze in stages for efficiency
progressive_config = {
    "stages": [
        {
            "name": "quick_filter",
            "dimensions": ["motion_level", "audio_peak"],
            "threshold": 0.3,
            "strategy": "rules"
        },
        {
            "name": "detailed_analysis",
            "dimensions": "all",
            "strategy": "hybrid",
            "condition": "quick_filter.max_score > 0.3"
        }
    ]
}
```

### Batch Processing

```python
# Process multiple segments efficiently
batch_config = {
    "batch_size": 10,
    "parallel_workers": 4,
    "gpu_acceleration": true,
    "dimension_grouping": {
        "video_heavy": ["action", "motion", "visual_quality"],
        "audio_heavy": ["dialogue", "music", "sound_effects"],
        "text_heavy": ["keywords", "sentiment", "topics"]
    }
}
```

## Monitoring and Analytics

### Dimension Performance Tracking

```json
{
    "analytics": {
        "dimension_performance": {
            "skill_display": {
                "avg_score": 0.62,
                "std_deviation": 0.18,
                "score_distribution": {
                    "0.0-0.2": 0.05,
                    "0.2-0.4": 0.15,
                    "0.4-0.6": 0.35,
                    "0.6-0.8": 0.30,
                    "0.8-1.0": 0.15
                },
                "highlight_correlation": 0.78,
                "false_positive_rate": 0.12,
                "processing_time_ms": 45
            }
        },
        "highlight_type_performance": {
            "epic_gameplay": {
                "total_detected": 1250,
                "user_validated": 1100,
                "precision": 0.88,
                "recall": 0.92,
                "f1_score": 0.90
            }
        }
    }
}
```

### A/B Testing Framework

```python
# Test different dimension configurations
ab_test_config = {
    "test_id": "engagement_optimization_v2",
    "variants": {
        "control": {
            "dimension_set": "current_production",
            "traffic_percentage": 50
        },
        "variant_a": {
            "dimension_set": "engagement_focused_v2",
            "traffic_percentage": 25,
            "changes": {
                "audience_reaction.weight": 0.35,
                "chat_velocity.weight": 0.25
            }
        },
        "variant_b": {
            "dimension_set": "ai_optimized",
            "traffic_percentage": 25,
            "strategy": "ai_only"
        }
    },
    "metrics": ["highlight_accuracy", "processing_time", "user_satisfaction"],
    "minimum_sample_size": 1000
}
```

## Best Practices

### Dimension Design Guidelines

1. **Specificity**: Make dimensions measurable and unambiguous
2. **Independence**: Minimize overlap between dimensions
3. **Balance**: Avoid over-weighting any single dimension
4. **Examples**: Provide clear examples for each score level
5. **Validation**: Test dimensions on sample content before deployment

### Highlight Type Configuration

1. **Clear Criteria**: Use precise, testable conditions
2. **Appropriate Thresholds**: Balance precision vs recall
3. **Priority Management**: Handle overlapping types gracefully
4. **Meaningful Tags**: Enable easy filtering and search
5. **Duration Limits**: Set reasonable min/max durations

### Strategy Selection

1. **Content Appropriate**: Match strategy to content type
2. **Performance Balance**: Consider processing time vs accuracy
3. **Fallback Options**: Have backup strategies for edge cases
4. **Resource Management**: Monitor CPU/GPU usage
5. **Cost Optimization**: Balance API calls with quality needs

## Integration Patterns

### SDK Usage

```python
from tldr_highlight_api import TLDRClient, DimensionSet

client = TLDRClient(api_key="your_api_key")

# Create custom dimension set
dimension_set = DimensionSet(
    name="product_review_analysis",
    dimensions={
        "product_focus": {...},
        "opinion_clarity": {...},
        "comparison_made": {...}
    }
)

# Register with organization
client.dimensions.create_set(
    organization_id=123,
    dimension_set=dimension_set
)

# Process content with custom dimensions
stream = client.streams.create(
    source_url="https://youtube.com/watch?v=...",
    options={
        "dimension_set_id": dimension_set.id,
        "detection_strategy": "hybrid"
    }
)
```

### Webhook Integration

```javascript
// Listen for dimension score updates
app.post('/webhooks/tldr', (req, res) => {
    const { event, data } = req.body;
    
    if (event === 'dimension.scores.calculated') {
        const { segment_id, dimensions, scores } = data;
        
        // Process dimension scores
        for (const [dim_id, score] of Object.entries(scores)) {
            if (score > dimensions[dim_id].alert_threshold) {
                notifyTeam(`High ${dim_id} score: ${score}`);
            }
        }
    }
    
    res.status(200).send('OK');
});
```

## Migration Strategies

### From Fixed to Flexible

1. **Audit Current System**: Identify implicit dimensions in existing code
2. **Map to Dimensions**: Convert hardcoded logic to dimension definitions
3. **Create Highlight Types**: Define types based on dimension combinations
4. **Parallel Testing**: Run both systems side-by-side
5. **Gradual Migration**: Move content types incrementally
6. **Performance Validation**: Ensure no regression in quality/speed

### Example Migration

```python
# Legacy fixed system
def detect_funny_moment(segment):
    laugh_track = detect_laughter(segment.audio)
    audience_reaction = analyze_chat(segment.chat)
    
    if laugh_track > 0.7 and audience_reaction > 0.6:
        return "funny_moment"
    return None

# Migrated flexible system
funny_dimensions = {
    "laughter_detected": {
        "type": "numeric",
        "modalities": ["audio"],
        "scoring_prompt": "Detect and rate laughter intensity"
    },
    "audience_positive_reaction": {
        "type": "numeric",
        "modalities": ["text"],
        "scoring_prompt": "Analyze chat for positive reactions"
    }
}

funny_highlight_type = {
    "id": "funny_moment",
    "criteria": {
        "min_scores": {
            "laughter_detected": 0.7,
            "audience_positive_reaction": 0.6
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Low Dimension Scores**
   - Check scoring prompts clarity
   - Verify modality configuration
   - Review example quality
   - Test with different strategies

2. **Inconsistent Results**
   - Ensure weight normalization
   - Check temporal aggregation settings
   - Verify confidence thresholds
   - Monitor modality fusion

3. **Performance Problems**
   - Use progressive analysis
   - Implement caching
   - Optimize batch sizes
   - Consider rule-based pre-filtering

4. **False Positives/Negatives**
   - Adjust dimension thresholds
   - Refine scoring prompts
   - Add more specific dimensions
   - Implement confidence bands

## Future Roadmap

### Planned Enhancements

1. **Auto-ML Optimization**: Automatic dimension weight tuning
2. **Cross-Organization Learning**: Shared dimension insights
3. **Real-time Adaptation**: Dynamic threshold adjustment
4. **Custom ML Models**: Bring your own models for dimensions
5. **Dimension Marketplace**: Share and monetize dimension sets

## Conclusion

The customizable dimension framework empowers organizations to define precisely what matters for their content. By moving from fixed categories to flexible dimensions, the TL;DR Highlight API can serve any industry, content type, or use case while maintaining consistency and quality.

---

*Related Documentation:*
- [API Reference](../api/overview.md)
- [Integration Guide](../quick-start.md)
- [Architecture Overview](../architecture/overview.md)
- [Flexible Highlight Detection](./flexible-highlight-detection.md)