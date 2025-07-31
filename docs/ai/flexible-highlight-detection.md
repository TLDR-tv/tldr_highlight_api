# Flexible Highlight Detection System

The TL;DR Highlight API features a revolutionary flexible highlight detection system that allows organizations to define custom scoring dimensions and highlight types tailored to their specific content and business needs.

## Overview

Traditional highlight detection systems use fixed categories like "funny moments" or "action scenes." Our flexible system allows you to:

- **Define Custom Dimensions**: Create scoring criteria specific to your content
- **Configure Highlight Types**: Define what constitutes a highlight for your use case  
- **Choose Analysis Strategies**: Select AI-only, rule-based, or hybrid approaches
- **Customize Fusion Methods**: Control how multi-modal signals are combined
- **Use Industry Presets**: Start with pre-configured templates

## Core Concepts

### 1. Dimensions

Dimensions are the fundamental scoring criteria used to evaluate content. Each dimension represents a measurable aspect of the content.

```python
{
    "action_intensity": {
        "type": "numeric",
        "description": "Level of action, movement, and excitement",
        "range": [0.0, 1.0],
        "weight": 0.3,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Rate the intensity of action from 0-1 based on movement, pace, and energy",
        "examples": {
            "high": "Fast combat, explosions, chase scenes",
            "medium": "Sports plays, quick gameplay",
            "low": "Slow exploration, dialogue"
        }
    }
}
```

### 2. Dimension Sets

Dimension sets group related dimensions for specific use cases:

```python
{
    "id": "gaming_default",
    "name": "Gaming Content Default",
    "dimensions": {
        "action_intensity": {"weight": 0.3},
        "skill_display": {"weight": 0.25},
        "audience_reaction": {"weight": 0.2},
        "emotional_peak": {"weight": 0.15},
        "narrative_moment": {"weight": 0.1}
    },
    "normalize_weights": true,
    "min_dimensions_required": 3
}
```

### 3. Highlight Types

Dynamic highlight types replace hardcoded categories:

```python
{
    "id": "epic_gameplay",
    "name": "Epic Gameplay Moment",
    "criteria": {
        "min_scores": {
            "action_intensity": 0.8,
            "skill_display": 0.7
        },
        "required_dimensions": ["action_intensity", "skill_display"],
        "min_total_score": 0.75
    },
    "priority": 1,
    "auto_tag": ["epic", "gameplay", "skilled"]
}
```

### 4. Analysis Strategies

Different strategies for scoring content:

- **AI-Only**: Uses LLM to score all dimensions
- **Rule-Based**: Applies deterministic rules
- **Hybrid**: Combines AI and rules for optimal results

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│                  Stream Input                        │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Dimension Manager                       │
│  • Load organization's dimension set                 │
│  • Validate dimension configurations                 │
│  • Apply weights and normalization                  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│             Analysis Strategy                        │
│  • AI Strategy → Gemini Analysis                   │
│  • Rule Strategy → Pattern Matching                │
│  • Hybrid Strategy → Combined Approach             │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Score Aggregation                       │
│  • Temporal smoothing                               │
│  • Multi-modal fusion                              │
│  • Confidence calculation                          │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│            Type Classification                       │
│  • Match scores to highlight types                  │
│  • Apply priority ordering                          │
│  • Auto-tag generation                             │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Highlight Output                        │
└─────────────────────────────────────────────────────┘
```

## Dimension Configuration

### Dimension Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | enum | `numeric`, `binary`, `categorical` |
| `description` | string | Human-readable description |
| `weight` | float | Importance weight (0.0-1.0) |
| `modalities` | array | Which modalities to analyze: `video`, `audio`, `text` |
| `scoring_prompt` | string | Instructions for AI scoring |
| `examples` | object | Example high/medium/low scores |
| `aggregation` | enum | How to combine over time: `max`, `mean`, `weighted_mean` |
| `threshold` | float | Minimum score to consider |

### Example Dimensions

#### Gaming Dimensions

```json
{
    "skill_display": {
        "type": "numeric",
        "description": "Technical skill and expertise demonstrated",
        "weight": 0.25,
        "modalities": ["video"],
        "scoring_prompt": "Rate the level of gaming skill shown (0-1) based on precision, strategy, and difficulty",
        "aggregation": "max"
    },
    "clutch_moment": {
        "type": "binary",
        "description": "High-pressure situation with successful outcome",
        "weight": 0.3,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Is this a clutch moment where the player overcomes significant odds?"
    }
}
```

#### Educational Dimensions

```json
{
    "concept_clarity": {
        "type": "numeric",
        "description": "How clearly a concept is explained",
        "weight": 0.35,
        "modalities": ["audio", "video"],
        "scoring_prompt": "Rate clarity of explanation (0-1) based on structure, examples, and comprehension"
    },
    "key_insight": {
        "type": "binary",
        "description": "Contains important learning moment",
        "weight": 0.4,
        "modalities": ["audio", "text"],
        "scoring_prompt": "Does this segment contain a key insight or 'aha' moment?"
    }
}
```

#### Entertainment Dimensions

```json
{
    "humor_level": {
        "type": "numeric",
        "description": "Comedy and entertainment value",
        "weight": 0.3,
        "modalities": ["video", "audio", "text"],
        "scoring_prompt": "Rate humor level (0-1) based on comedic timing, audience reaction, and content"
    },
    "surprise_factor": {
        "type": "numeric",
        "description": "Unexpected or surprising elements",
        "weight": 0.2,
        "modalities": ["video", "audio"],
        "scoring_prompt": "Rate how surprising or unexpected this moment is (0-1)"
    }
}
```

## Highlight Type Registry

### Type Definition Structure

```json
{
    "id": "educational_keypoint",
    "name": "Educational Key Point",
    "description": "Important educational or instructional moment",
    "criteria": {
        "min_scores": {
            "concept_clarity": 0.7,
            "key_insight": 1.0
        },
        "any_of_scores": {
            "visual_demonstration": 0.8,
            "verbal_emphasis": 0.8
        },
        "min_total_score": 0.75,
        "min_duration": 10,
        "max_duration": 120
    },
    "priority": 2,
    "auto_tag": ["education", "key-point", "learning"],
    "metadata_template": {
        "highlight_color": "#4CAF50",
        "icon": "lightbulb"
    }
}
```

### Criteria Options

| Field | Description |
|-------|-------------|
| `min_scores` | All specified dimensions must meet minimum |
| `any_of_scores` | At least one dimension must meet minimum |
| `required_dimensions` | Dimensions that must be present |
| `min_total_score` | Weighted average must exceed threshold |
| `score_formula` | Custom formula for complex logic |

## Analysis Strategies

### AI-Only Strategy

Uses Gemini's multimodal understanding to score all dimensions:

```python
class AIAnalysisStrategy:
    async def analyze_segment(self, segment: VideoSegment, dimensions: Dict) -> Scores:
        # Build comprehensive prompt
        prompt = self._build_dimension_prompt(dimensions)
        
        # Send to Gemini with video
        response = await gemini.analyze_video(
            video=segment.data,
            prompt=prompt,
            response_format="structured"
        )
        
        # Parse structured scores
        return self._parse_scores(response)
```

### Rule-Based Strategy

Applies deterministic rules for specific patterns:

```python
class RuleBasedStrategy:
    def analyze_segment(self, segment: VideoSegment, dimensions: Dict) -> Scores:
        scores = {}
        
        # Audio peak detection
        if "audio_intensity" in dimensions:
            scores["audio_intensity"] = self._analyze_audio_peaks(segment.audio)
        
        # Motion detection
        if "motion_level" in dimensions:
            scores["motion_level"] = self._calculate_motion(segment.frames)
        
        # Chat velocity (if available)
        if "audience_engagement" in dimensions:
            scores["audience_engagement"] = self._analyze_chat_velocity(segment.chat)
        
        return scores
```

### Hybrid Strategy

Combines AI and rules for optimal performance:

```python
class HybridStrategy:
    async def analyze_segment(self, segment: VideoSegment, dimensions: Dict) -> Scores:
        # Quick rule-based pre-filtering
        rule_scores = self.rule_strategy.analyze_segment(segment, dimensions)
        
        # Skip AI if below threshold
        if max(rule_scores.values()) < 0.3:
            return rule_scores
        
        # AI analysis for promising segments
        ai_scores = await self.ai_strategy.analyze_segment(segment, dimensions)
        
        # Weighted combination
        return self._fuse_scores(rule_scores, ai_scores, weights={
            "rules": 0.3,
            "ai": 0.7
        })
```

## Fusion Strategies

### Multi-Modal Signal Fusion

Different methods to combine signals from video, audio, and text:

#### Weighted Fusion
```python
def weighted_fusion(scores: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
    """Weight modalities differently per dimension."""
    fused = {}
    for dimension in scores.get("video", {}):
        dim_score = 0
        total_weight = 0
        
        for modality, weight in weights.items():
            if dimension in scores.get(modality, {}):
                dim_score += scores[modality][dimension] * weight
                total_weight += weight
        
        if total_weight > 0:
            fused[dimension] = dim_score / total_weight
    
    return fused
```

#### Consensus Fusion
```python
def consensus_fusion(scores: Dict[str, Dict[str, float]], threshold: float = 0.6) -> Dict[str, float]:
    """Require agreement between modalities."""
    fused = {}
    for dimension in get_all_dimensions(scores):
        modality_scores = [
            scores[mod][dimension] 
            for mod in scores 
            if dimension in scores[mod]
        ]
        
        if len(modality_scores) >= 2:
            # Require consensus above threshold
            if all(s >= threshold for s in modality_scores):
                fused[dimension] = np.mean(modality_scores)
            else:
                fused[dimension] = 0.0
    
    return fused
```

## Processing Options

### Configuration Example

```json
{
    "dimension_set_id": "gaming_default",
    "custom_dimensions": {
        "brand_visibility": {
            "type": "binary",
            "weight": 0.1,
            "scoring_prompt": "Is the sponsor brand clearly visible?"
        }
    },
    "detection_strategy": "hybrid",
    "fusion_strategy": "weighted",
    "fusion_weights": {
        "video": 0.5,
        "audio": 0.3,
        "text": 0.2
    },
    "confidence_thresholds": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    },
    "temporal_smoothing": {
        "window_size": 5,
        "method": "gaussian"
    }
}
```

## Industry Presets

### Gaming Preset

```json
{
    "id": "gaming_default",
    "name": "Gaming Content",
    "dimensions": {
        "action_intensity": {"weight": 0.25},
        "skill_display": {"weight": 0.2},
        "clutch_moment": {"weight": 0.2},
        "audience_reaction": {"weight": 0.15},
        "funny_fail": {"weight": 0.1},
        "achievement": {"weight": 0.1}
    },
    "highlight_types": [
        "epic_gameplay",
        "clutch_play",
        "funny_moment",
        "skill_showcase",
        "achievement_unlock"
    ],
    "detection_strategy": "hybrid",
    "typical_duration": {
        "min": 15,
        "max": 60
    }
}
```

### Education Preset

```json
{
    "id": "education_default",
    "name": "Educational Content",
    "dimensions": {
        "concept_clarity": {"weight": 0.3},
        "key_insight": {"weight": 0.25},
        "example_quality": {"weight": 0.2},
        "student_engagement": {"weight": 0.15},
        "summary_moment": {"weight": 0.1}
    },
    "highlight_types": [
        "key_concept",
        "important_example",
        "student_question",
        "summary_point",
        "aha_moment"
    ],
    "detection_strategy": "ai",
    "typical_duration": {
        "min": 30,
        "max": 180
    }
}
```

### Corporate Preset

```json
{
    "id": "corporate_default",
    "name": "Corporate/Meeting Content",
    "dimensions": {
        "decision_point": {"weight": 0.3},
        "action_item": {"weight": 0.25},
        "key_metric": {"weight": 0.2},
        "stakeholder_input": {"weight": 0.15},
        "consensus_moment": {"weight": 0.1}
    },
    "highlight_types": [
        "decision_made",
        "action_item_assigned",
        "metric_revealed",
        "important_question",
        "agreement_reached"
    ],
    "detection_strategy": "hybrid",
    "typical_duration": {
        "min": 20,
        "max": 120
    }
}
```

## Implementation Examples

### Creating Custom Dimensions

```python
from tldr_highlight_api import TLDRClient, DimensionBuilder

client = TLDRClient(api_key="tldr_sk_your_api_key")

# Create a custom dimension for product placement
product_dimension = DimensionBuilder()\
    .set_type("numeric")\
    .set_name("product_placement_quality")\
    .set_description("Quality and prominence of product placement")\
    .set_weight(0.4)\
    .set_modalities(["video"])\
    .set_scoring_prompt(
        "Rate the product placement quality (0-1) based on:\n"
        "- Product visibility and screen time\n"
        "- Natural integration into content\n"
        "- Brand message clarity"
    )\
    .add_examples(
        high="Product clearly visible, naturally integrated, brand message clear",
        medium="Product visible but not prominent, somewhat integrated",
        low="Product barely visible or forced placement"
    )\
    .build()

# Add to organization's dimension set
client.dimensions.add_custom_dimension(
    organization_id=123,
    dimension=product_dimension
)
```

### Configuring Highlight Types

```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({ apiKey: 'tldr_sk_your_api_key' });

// Define a new highlight type for product reviews
const reviewHighlight = {
  id: 'product_review_moment',
  name: 'Product Review Highlight',
  criteria: {
    minScores: {
      'product_focus': 0.8,
      'opinion_expressed': 0.7,
      'detail_level': 0.6
    },
    anyOfScores: {
      'positive_sentiment': 0.8,
      'negative_sentiment': 0.8,
      'comparison_made': 0.7
    },
    minTotalScore: 0.75,
    minDuration: 15,
    maxDuration: 90
  },
  priority: 1,
  autoTag: ['review', 'product', 'opinion']
};

// Register the highlight type
await client.highlightTypes.create({
  organizationId: 123,
  highlightType: reviewHighlight
});
```

### Processing with Custom Configuration

```python
# Process a stream with custom dimensions and settings
stream = client.streams.create(
    source_url="https://youtube.com/watch?v=example",
    options={
        "dimension_set_id": "custom_review_set",
        "custom_dimensions": {
            "sponsor_mention": {
                "type": "binary",
                "weight": 0.2,
                "scoring_prompt": "Is a sponsor mentioned or shown?"
            }
        },
        "detection_strategy": "hybrid",
        "fusion_strategy": "consensus",
        "confidence_thresholds": {
            "high": 0.85,
            "medium": 0.65,
            "low": 0.45
        },
        "highlight_types": [
            "product_review_moment",
            "comparison_moment",
            "recommendation"
        ]
    }
)
```

## Best Practices

### 1. Dimension Design

- **Be Specific**: Clear, measurable criteria work best
- **Provide Examples**: Help the AI understand expectations
- **Balance Weights**: Avoid over-weighting single dimensions
- **Test Iteratively**: Refine based on results

### 2. Highlight Type Configuration

- **Clear Criteria**: Unambiguous scoring requirements
- **Appropriate Thresholds**: Balance precision vs recall
- **Meaningful Tags**: Enable easy filtering later
- **Priority Ordering**: Handle overlapping types

### 3. Strategy Selection

- **AI for Subjective**: Complex judgments need AI
- **Rules for Objective**: Simple patterns use rules
- **Hybrid for Production**: Best balance of speed/accuracy

### 4. Performance Optimization

- **Cache Dimension Sets**: Avoid repeated lookups
- **Batch Processing**: Analyze multiple segments together
- **Progressive Analysis**: Quick rules first, AI if promising

## Monitoring and Optimization

### Dimension Performance Metrics

Track how well dimensions perform:

```json
{
    "dimension_analytics": {
        "action_intensity": {
            "average_score": 0.62,
            "score_distribution": [0.1, 0.2, 0.3, 0.25, 0.15],
            "correlation_with_selection": 0.78,
            "false_positive_rate": 0.12
        }
    }
}
```

### Continuous Improvement

1. **A/B Testing**: Compare dimension configurations
2. **Feedback Loop**: Incorporate user feedback
3. **Score Calibration**: Adjust thresholds based on results
4. **Model Updates**: Retrain with organization-specific data

## Migration Guide

### From Fixed to Flexible System

1. **Map Existing Types**: Convert hardcoded types to dimension-based
2. **Define Dimensions**: Create dimensions for your criteria
3. **Configure Types**: Set up highlight type definitions
4. **Test Thoroughly**: Validate on sample content
5. **Gradual Rollout**: Migrate incrementally

### Example Migration

```python
# Old fixed system
if highlight_type == "funny_moment":
    process_funny_moment()

# New flexible system
dimensions = {
    "humor_level": {"weight": 0.4, "threshold": 0.7},
    "unexpectedness": {"weight": 0.3, "threshold": 0.6},
    "audience_laughter": {"weight": 0.3, "threshold": 0.8}
}

highlight_types = [{
    "id": "funny_moment",
    "criteria": {
        "min_scores": {
            "humor_level": 0.7,
            "unexpectedness": 0.6
        }
    }
}]
```

## Conclusion

The flexible highlight detection system empowers organizations to:

- Define what matters for their content
- Adapt to changing requirements without code changes
- Optimize for their specific use cases
- Maintain consistency across content types
- Scale to new content categories easily

This industry-agnostic approach ensures the TL;DR Highlight API can serve any content type, from gaming streams to corporate meetings, educational content to live events.

---

*See also: [AI Integration](./gemini-integration.md) | [API Overview](../api/overview.md) | [Architecture](../architecture/overview.md)*