"""Multi-dimensional scoring framework for LLM-based evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, Dict, List
from collections import defaultdict
import re


class DimensionType(Enum):
    """Types of scoring dimensions for evaluation.
    
    Defines the different formats and ranges for dimensional scoring
    to optimize LLM evaluation performance.
    """

    NUMERIC = "numeric"  # 0-1 continuous scale
    BINARY = "binary"  # Yes/No
    CATEGORICAL = "categorical"  # Multiple choice
    ORDINAL = "ordinal"  # Ordered categories (low/medium/high)
    SCALE_1_4 = "scale_1_4"  # 1-4 scale (proven more reliable for LLMs)


class AggregationMethod(Enum):
    """Methods for aggregating scores across time segments.
    
    Defines how individual segment scores should be combined
    to create an overall score for the entire content.
    """

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    WEIGHTED_MEAN = "weighted_mean"
    PERCENTILE = "percentile"


@dataclass
class DimensionExample:
    """Example for few-shot prompting.
    
    Provides concrete examples to improve LLM understanding
    and consistency in dimensional scoring.
    """

    input_description: str
    expected_score: float
    reasoning: str


@dataclass
class DimensionDefinition:
    """Definition of a scoring dimension.
    
    Represents a single dimension for evaluation including its type,
    scoring criteria, examples, and aggregation method.
    """

    name: str
    description: str
    type: DimensionType = DimensionType.NUMERIC
    weight: float = 1.0  # For weighted aggregation

    # LLM prompting
    scoring_prompt: str = ""
    evaluation_criteria: list[str] = field(default_factory=list)
    examples: list[DimensionExample] = field(default_factory=list)

    # Aggregation
    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    temporal_weight_decay: float = 0.0  # How much to decay older scores

    # Constraints
    min_score: float = 0.0
    max_score: float = 1.0
    requires_context: bool = False  # If true, needs previous segments

    def __post_init__(self) -> None:
        """Validate dimension definition.
        
        Ensures weight is within valid range and sets appropriate
        score bounds for binary dimensions.
        
        Raises:
            ValueError: If weight is outside the valid range [0, 10].

        """
        if not 0 <= self.weight <= 10:
            raise ValueError(f"Weight must be between 0 and 10, got {self.weight}")
        if self.type == DimensionType.BINARY:
            self.min_score = 0.0
            self.max_score = 1.0

    def build_evaluation_prompt(self, video_timestamp: str = "") -> str:
        """Build optimized evaluation prompt for Gemini video analysis.
        
        Creates a structured prompt following Gemini best practices including
        clear task definition, evaluation criteria, examples, and output format.

        Args:
            video_timestamp: Optional timestamp in MM:SS format for specific moments.

        Returns:
            Structured prompt optimized for Gemini with JSON output format.

        """
        # Build structured prompt following Gemini best practices
        prompt_parts = []

        # Clear task definition
        prompt_parts.extend(
            [
                f"Task: Evaluate this video segment for the dimension '{self.name}'",
                f"Definition: {self.description}",
                "",
            ]
        )

        # Evaluation rubric with clear criteria
        if self.evaluation_criteria:
            prompt_parts.append("Evaluation Rubric:")
            for i, criterion in enumerate(self.evaluation_criteria, 1):
                prompt_parts.append(f"{i}. {criterion}")
            prompt_parts.append("")

        # Examples for few-shot learning
        if self.examples:
            prompt_parts.append("Reference Examples:")
            for i, example in enumerate(self.examples, 1):
                prompt_parts.extend(
                    [
                        f"\nExample {i}:",
                        f"Scenario: {example.input_description}",
                        f"Score: {example.expected_score}",
                        f"Reasoning: {example.reasoning}",
                    ]
                )
            prompt_parts.append("")

        # Scoring instructions based on type
        if self.type == DimensionType.SCALE_1_4:
            prompt_parts.extend(
                [
                    "Scoring Scale:",
                    "1 = Poor/None - Dimension is absent or very weak",
                    "2 = Fair - Some presence but below average",
                    "3 = Good - Clear presence, above average",
                    "4 = Excellent - Strong presence, exceptional quality",
                    "",
                ]
            )
        elif self.type == DimensionType.NUMERIC:
            prompt_parts.append(
                f"Score Range: {self.min_score} to {self.max_score} (decimal values allowed)"
            )
            prompt_parts.append("")
        elif self.type == DimensionType.BINARY:
            prompt_parts.extend(
                ["Binary Scoring:", "0 = No/Absent", "1 = Yes/Present", ""]
            )

        # Video-specific instructions
        if video_timestamp:
            prompt_parts.append(f"Focus on the moment at timestamp {video_timestamp}")
        else:
            prompt_parts.append("Analyze the entire video segment provided")

        # Chain of thought prompting
        prompt_parts.extend(
            [
                "",
                "Instructions:",
                "1. Watch the video segment carefully",
                "2. Consider each evaluation criterion",
                "3. Think step-by-step about how the content meets each criterion",
                "4. Provide your evaluation in the following JSON format:",
                "",
                "```json",
                "{",
                f'  "dimension": "{self.name}",',
                '  "score": <numeric_score>,',
                '  "confidence": <0.0-1.0>,',
                '  "reasoning": "<brief explanation of your scoring decision>",',
                '  "key_moments": ["<timestamp MM:SS if specific moments influenced score>"]',
                "}",
                "```",
            ]
        )

        # Custom scoring prompt if provided
        if self.scoring_prompt:
            prompt_parts.extend(["", "Additional Instructions:", self.scoring_prompt])

        return "\n".join(prompt_parts)


@dataclass
class ScoringRubric:
    """Complete rubric for multi-dimensional scoring.
    
    Contains a collection of dimensions with global settings for
    normalization, thresholds, and highlight detection criteria.
    """

    name: str
    description: str
    dimensions: list[DimensionDefinition] = field(default_factory=list)

    # Global settings
    requires_all_dimensions: bool = False
    min_confidence_threshold: float = 0.5
    normalization_enabled: bool = True

    # Highlight detection thresholds
    highlight_threshold: float = 0.7  # Minimum normalized score for highlight
    highlight_confidence_threshold: float = 0.6  # Minimum confidence

    def __post_init__(self) -> None:
        """Validate rubric configuration.
        
        Ensures all dimension names are unique within the rubric.
        
        Raises:
            ValueError: If dimension names are not unique.

        """
        dimension_names = [d.name for d in self.dimensions]
        if len(dimension_names) != len(set(dimension_names)):
            raise ValueError("Dimension names must be unique")

    @property
    def total_weight(self) -> float:
        """Calculate total weight of all dimensions.
        
        Returns:
            Sum of all dimension weights in the rubric.

        """
        return sum(d.weight for d in self.dimensions)

    def get_normalized_weights(self) -> dict[str, float]:
        """Get normalized weights for each dimension.
        
        Returns:
            Dictionary mapping dimension names to normalized weights.
            If normalization is disabled, returns raw weights.

        """
        if not self.normalization_enabled:
            return {d.name: d.weight for d in self.dimensions}

        total = self.total_weight
        if total == 0:
            return {d.name: 1.0 / len(self.dimensions) for d in self.dimensions}

        return {d.name: d.weight / total for d in self.dimensions}

    def add_dimension(self, dimension: DimensionDefinition) -> None:
        """Add a dimension to the rubric."""
        if any(d.name == dimension.name for d in self.dimensions):
            raise ValueError(f"Dimension '{dimension.name}' already exists")
        self.dimensions.append(dimension)

    def remove_dimension(self, name: str) -> None:
        """Remove a dimension from the rubric."""
        self.dimensions = [d for d in self.dimensions if d.name != name]

    def build_multi_dimensional_prompt(self) -> str:
        """Build a single prompt for evaluating all dimensions at once.

        This is more efficient than multiple API calls and allows the model
        to consider relationships between dimensions.
        """
        prompt_parts = [
            f"Task: Evaluate this video segment using the '{self.name}' scoring rubric",
            f"Description: {self.description}",
            "",
            "You will evaluate the video across multiple dimensions. For each dimension:",
            "1. Watch the video with that specific dimension in mind",
            "2. Consider the evaluation criteria provided",
            "3. Think step-by-step about how well the content meets the criteria",
            "4. Assign a score with confidence level",
            "",
            "DIMENSIONS TO EVALUATE:",
            "",
        ]

        # Add each dimension's details
        for i, dim in enumerate(self.dimensions, 1):
            prompt_parts.extend(
                [
                    f"### Dimension {i}: {dim.name}",
                    f"Description: {dim.description}",
                    "",
                ]
            )

            if dim.evaluation_criteria:
                prompt_parts.append("Criteria:")
                for criterion in dim.evaluation_criteria:
                    prompt_parts.append(f"- {criterion}")
                prompt_parts.append("")

            # Scoring scale based on type
            if dim.type == DimensionType.SCALE_1_4:
                prompt_parts.append("Scale: 1=Poor, 2=Fair, 3=Good, 4=Excellent")
            elif dim.type == DimensionType.NUMERIC:
                prompt_parts.append(f"Scale: {dim.min_score} to {dim.max_score}")
            elif dim.type == DimensionType.BINARY:
                prompt_parts.append("Scale: 0=No/Absent, 1=Yes/Present")

            prompt_parts.append("")

        # Output format
        prompt_parts.extend(
            [
                "INSTRUCTIONS:",
                "Analyze the video and provide your evaluation in the following JSON format:",
                "",
                "```json",
                "{",
                '  "rubric": "' + self.name + '",',
                '  "overall_assessment": "<brief summary of the video content>",',
                '  "dimensions": {',
            ]
        )

        # Add dimension output format
        for i, dim in enumerate(self.dimensions):
            comma = "," if i < len(self.dimensions) - 1 else ""
            prompt_parts.extend(
                [
                    f'    "{dim.name}": {{',
                    '      "score": <numeric_score>,',
                    '      "confidence": <0.0-1.0>,',
                    '      "reasoning": "<brief explanation>",',
                    '      "key_moments": ["<MM:SS timestamps if applicable>"]',
                    f"    }}{comma}",
                ]
            )

        prompt_parts.extend(
            [
                "  },",
                '  "highlight_recommendation": <true/false>,',
                '  "highlight_reasoning": "<why this is/isn\'t a highlight>"',
                "}",
                "```",
            ]
        )

        return "\n".join(prompt_parts)

    def build_structured_output_prompt(self) -> str:
        """Build a prompt specifically designed for structured output with precise highlight boundaries.
        
        This prompt explicitly requests start/end timestamps for highlights within the video segment,
        enabling creation of targeted clips between 15-90 seconds instead of full 2-minute segments.
        """
        prompt_parts = [
            f"Task: Analyze this video segment to identify precise highlight moments using the '{self.name}' rubric",
            f"Description: {self.description}",
            "",
            "CRITICAL INSTRUCTIONS:",
            "- You must identify the EXACT start and end timestamps for any highlight within this video segment",
            "- Highlights must be between 15 and 90 seconds long",
            "- Use MM:SS format for all timestamps (e.g., 01:23 for 1 minute 23 seconds)",
            "- If multiple highlights exist, identify the BEST one with precise boundaries",
            "- Timestamps should mark the actual beginning and end of the interesting content",
            "",
            "ANALYSIS DIMENSIONS:",
            "",
        ]
        
        # Add each dimension's details  
        for i, dim in enumerate(self.dimensions, 1):
            prompt_parts.extend([
                f"### {i}. {dim.name}",
                f"Description: {dim.description}",
                "",
            ])
            
            if dim.evaluation_criteria:
                prompt_parts.append("Evaluation criteria:")
                for criterion in dim.evaluation_criteria:
                    prompt_parts.append(f"- {criterion}")
                prompt_parts.append("")
            
            # Add scoring scale based on type
            if dim.type == DimensionType.SCALE_1_4:
                prompt_parts.append("Scale: 1=Poor, 2=Fair, 3=Good, 4=Excellent")
            elif dim.type == DimensionType.NUMERIC:
                prompt_parts.append(f"Scale: {dim.min_score} to {dim.max_score}")
            elif dim.type == DimensionType.BINARY:
                prompt_parts.append("Scale: 0=No/Absent, 1=Yes/Present")
            
            prompt_parts.append("")
        
        # Add final instructions
        prompt_parts.extend([
            "RESPONSE REQUIREMENTS:",
            "1. Watch the entire video segment carefully",
            "2. Evaluate each dimension with specific scores and confidence",
            "3. If you identify a highlight, provide EXACT start and end timestamps",
            "4. Ensure highlight duration is between 15-90 seconds", 
            "5. Provide clear reasoning for your highlight boundaries",
            "",
            "Your response will be automatically parsed, so follow the JSON structure exactly.",
        ])
        
        return "\n".join(prompt_parts)


@dataclass
class HighlightBoundary:
    """Represents precise start/end boundaries for a highlight within a video segment."""
    
    start_timestamp: str  # MM:SS format (e.g., "01:23")
    end_timestamp: str    # MM:SS format (e.g., "02:45") 
    confidence: float     # 0.0-1.0 confidence in these boundaries
    reasoning: str        # Explanation of why these boundaries were chosen
    
    def to_seconds(self) -> tuple[float, float]:
        """Convert MM:SS timestamps to seconds.
        
        Returns:
            Tuple of (start_seconds, end_seconds)
            
        Raises:
            ValueError: If timestamp format is invalid
        """
        def parse_timestamp(ts: str) -> float:
            match = re.match(r'^(\d{1,2}):(\d{2})$', ts.strip())
            if not match:
                raise ValueError(f"Invalid timestamp format: {ts}. Expected MM:SS")
            minutes, seconds = map(int, match.groups())
            return minutes * 60 + seconds
        
        start_secs = parse_timestamp(self.start_timestamp)
        end_secs = parse_timestamp(self.end_timestamp)
        
        if start_secs >= end_secs:
            raise ValueError(f"Start time {self.start_timestamp} must be before end time {self.end_timestamp}")
        
        return start_secs, end_secs
    
    def validate_duration(self, min_duration: float = 15.0, max_duration: float = 90.0) -> bool:
        """Validate that highlight duration is within acceptable bounds."""
        try:
            start_secs, end_secs = self.to_seconds()
            duration = end_secs - start_secs
            return min_duration <= duration <= max_duration
        except ValueError:
            return False


def create_gemini_response_schema(rubric: ScoringRubric) -> Dict[str, Any]:
    """Create a JSON Schema for structured Gemini API responses.
    
    This schema ensures reliable parsing of Gemini responses and enforces
    the structure needed for precise highlight boundary extraction.
    
    Args:
        rubric: The scoring rubric being used for evaluation
        
    Returns:
        JSON Schema dictionary compatible with Gemini's response_schema parameter
    """
    # Build dimension properties dynamically
    dimension_properties = {}
    for dim in rubric.dimensions:
        dim_schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "minLength": 5},
                "key_moments": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "MM:SS timestamps of key moments for this dimension"
                }
            },
            "required": ["score", "confidence", "reasoning"]
        }
        
        # Add score constraints based on dimension type
        if dim.type == DimensionType.SCALE_1_4:
            dim_schema["properties"]["score"]["minimum"] = 1
            dim_schema["properties"]["score"]["maximum"] = 4
        elif dim.type == DimensionType.BINARY:
            dim_schema["properties"]["score"]["minimum"] = 0
            dim_schema["properties"]["score"]["maximum"] = 1
        elif dim.type == DimensionType.NUMERIC:
            dim_schema["properties"]["score"]["minimum"] = dim.min_score
            dim_schema["properties"]["score"]["maximum"] = dim.max_score
            
        dimension_properties[dim.name] = dim_schema
    
    # Main schema structure
    schema = {
        "type": "object",
        "properties": {
            "rubric_name": {
                "type": "string",
                "enum": [rubric.name]
            },
            "overall_assessment": {
                "type": "string", 
                "minLength": 10,
                "description": "Brief summary of video content and analysis"
            },
            "dimensions": {
                "type": "object",
                "properties": dimension_properties,
                "required": list(dimension_properties.keys())
            },
            "highlight_detected": {
                "type": "boolean",
                "description": "Whether a highlight was identified in this segment"
            },
            "highlight_boundary": {
                "type": "object",
                "properties": {
                    "start_timestamp": {
                        "type": "string",
                        "description": "Start time in MM:SS format"
                    },
                    "end_timestamp": {
                        "type": "string", 
                        "description": "End time in MM:SS format"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in these precise boundaries"
                    },
                    "reasoning": {
                        "type": "string",
                        "minLength": 10,
                        "description": "Explanation of why these boundaries were chosen"
                    }
                },
                "required": ["start_timestamp", "end_timestamp", "confidence", "reasoning"],
                "description": "Precise highlight boundaries within the video segment"
            },
            "overall_highlight_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Overall score indicating highlight quality"
            }
        },
        "required": [
            "rubric_name", 
            "overall_assessment", 
            "dimensions", 
            "highlight_detected",
            "overall_highlight_score"
        ]
    }
    
    return schema


class ScoringStrategy(Protocol):
    """Protocol for scoring strategies."""

    async def score(
        self, content: Any, rubric: ScoringRubric, context: Optional[list[Any]] = None
    ) -> dict[str, tuple[float, float]]:
        """Score content against rubric dimensions.
        Returns dict of dimension_name -> (score, confidence)
        """
        ...


@dataclass
class ScoringContext:
    """Context for maintaining state across segments."""

    segment_history: list[Any] = field(default_factory=list)
    dimension_scores: defaultdict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_segment_scores(self, scores: dict[str, float]) -> None:
        """Add scores from a segment."""
        for dimension, score in scores.items():
            self.dimension_scores[dimension].append(score)

    def get_aggregated_scores(self, rubric: ScoringRubric) -> dict[str, float]:
        """Get aggregated scores based on rubric definitions."""
        aggregated = {}

        for dimension in rubric.dimensions:
            scores = self.dimension_scores.get(dimension.name, [])
            if not scores:
                continue

            if dimension.aggregation_method == AggregationMethod.MEAN:
                aggregated[dimension.name] = sum(scores) / len(scores)
            elif dimension.aggregation_method == AggregationMethod.MAX:
                aggregated[dimension.name] = max(scores)
            elif dimension.aggregation_method == AggregationMethod.MIN:
                aggregated[dimension.name] = min(scores)
            elif dimension.aggregation_method == AggregationMethod.WEIGHTED_MEAN:
                # Apply temporal decay
                weights = [
                    (1 - dimension.temporal_weight_decay) ** i
                    for i in range(len(scores))
                ]
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                total_weight = sum(weights)
                aggregated[dimension.name] = (
                    weighted_sum / total_weight if total_weight > 0 else 0
                )

        return aggregated


# Pre-built dimension templates for common use cases
class DimensionTemplates:
    """Common dimension templates."""

    @staticmethod
    def action_intensity() -> DimensionDefinition:
        """Template for action intensity scoring."""
        return DimensionDefinition(
            name="action_intensity",
            description="Level of action, movement, and visual excitement in the video",
            type=DimensionType.SCALE_1_4,
            weight=1.0,
            evaluation_criteria=[
                "Fast-paced camera movements or quick cuts",
                "Multiple simultaneous actions or events",
                "Visual effects, explosions, or particle effects",
                "Rapid character/object movement or combat",
                "High-energy moments with intensity peaks",
            ],
            examples=[
                DimensionExample(
                    input_description="Calm dialogue scene with static camera",
                    expected_score=1,
                    reasoning="No action, minimal movement, static composition",
                ),
                DimensionExample(
                    input_description="Intense boss fight with explosions and rapid combat",
                    expected_score=4,
                    reasoning="Multiple simultaneous actions, effects, fast movement",
                ),
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    @staticmethod
    def educational_value() -> DimensionDefinition:
        """Template for educational content scoring."""
        return DimensionDefinition(
            name="educational_value",
            description="How much the content teaches or explains concepts",
            type=DimensionType.NUMERIC,
            weight=1.2,
            scoring_prompt="Rate the educational value from 0 (no learning) to 1 (highly educational)",
            evaluation_criteria=[
                "Clear explanation of concepts",
                "Use of examples or demonstrations",
                "Structured presentation of information",
                "Depth of knowledge shared",
                "Practical applicability",
            ],
            aggregation_method=AggregationMethod.WEIGHTED_MEAN,
            temporal_weight_decay=0.1,
        )

    @staticmethod
    def humor() -> DimensionDefinition:
        """Template for humor/comedy scoring."""
        return DimensionDefinition(
            name="humor",
            description="How funny or comedic the content is",
            type=DimensionType.NUMERIC,
            weight=0.8,
            scoring_prompt="Rate the humor level from 0 (not funny) to 1 (extremely funny)",
            evaluation_criteria=[
                "Jokes and punchlines",
                "Comedic timing",
                "Unexpected or absurd situations",
                "Witty dialogue or commentary",
                "Physical comedy",
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    # Gaming-specific dimensions
    @staticmethod
    def skill_display() -> DimensionDefinition:
        """Template for gaming skill showcase moments."""
        return DimensionDefinition(
            name="skill_display",
            description="Demonstration of exceptional gaming skill or technique",
            type=DimensionType.SCALE_1_4,
            weight=1.5,
            evaluation_criteria=[
                "Precision and accuracy of gameplay execution",
                "Difficulty of maneuver or technique performed",
                "Clutch plays under pressure situations",
                "Creative or unexpected strategies",
                "Outplaying opponents through superior mechanics",
            ],
            examples=[
                DimensionExample(
                    input_description="Player misses easy shot and dies",
                    expected_score=1,
                    reasoning="Poor execution, no skill demonstration",
                ),
                DimensionExample(
                    input_description="360 no-scope headshot to win the round",
                    expected_score=4,
                    reasoning="Extremely difficult technique executed perfectly",
                ),
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    @staticmethod
    def emotional_moment() -> DimensionDefinition:
        """Template for emotionally impactful moments."""
        return DimensionDefinition(
            name="emotional_moment",
            description="Moments that evoke strong emotional responses",
            type=DimensionType.SCALE_1_4,
            weight=1.2,
            evaluation_criteria=[
                "Player's emotional reaction (excitement, frustration, joy)",
                "Narrative or story developments",
                "Tense or suspenseful situations",
                "Victory or defeat moments",
                "Community/chat reaction intensity",
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    # Sports-specific dimensions
    @staticmethod
    def scoring_play() -> DimensionDefinition:
        """Template for sports scoring moments."""
        return DimensionDefinition(
            name="scoring_play",
            description="Plays that result in points or goals",
            type=DimensionType.BINARY,
            weight=2.0,
            evaluation_criteria=[
                "Goal, touchdown, or point scored",
                "Successful conversion or penalty",
                "Game-changing score",
                "Record-breaking achievement",
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    @staticmethod
    def momentum_shift() -> DimensionDefinition:
        """Template for momentum-changing moments in sports."""
        return DimensionDefinition(
            name="momentum_shift",
            description="Moments that significantly change game dynamics",
            type=DimensionType.SCALE_1_4,
            weight=1.3,
            evaluation_criteria=[
                "Defensive stops or turnovers",
                "Comeback sequences",
                "Key player substitutions or injuries",
                "Crowd energy changes",
                "Strategic timeout or play calls",
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    # Education-specific dimensions
    @staticmethod
    def concept_clarity() -> DimensionDefinition:
        """Template for educational concept explanation."""
        return DimensionDefinition(
            name="concept_clarity",
            description="How clearly a concept is explained or demonstrated",
            type=DimensionType.SCALE_1_4,
            weight=1.5,
            evaluation_criteria=[
                "Step-by-step breakdown of complex ideas",
                "Use of visual aids or diagrams",
                "Real-world examples and applications",
                "Addressing common misconceptions",
                "Summary or key takeaways provided",
            ],
            aggregation_method=AggregationMethod.WEIGHTED_MEAN,
            temporal_weight_decay=0.05,
        )

    @staticmethod
    def engagement_level() -> DimensionDefinition:
        """Template for audience engagement in educational content."""
        return DimensionDefinition(
            name="engagement_level",
            description="How engaging and attention-holding the content is",
            type=DimensionType.SCALE_1_4,
            weight=1.0,
            evaluation_criteria=[
                "Interactive elements or questions",
                "Varied presentation style",
                "Enthusiasm and energy of presenter",
                "Pacing and rhythm of content delivery",
                "Use of humor or relatable examples",
            ],
            aggregation_method=AggregationMethod.MEAN,
        )

    # Corporate/Business dimensions
    @staticmethod
    def key_decision() -> DimensionDefinition:
        """Template for important business decisions or announcements."""
        return DimensionDefinition(
            name="key_decision",
            description="Important decisions, announcements, or strategic points",
            type=DimensionType.BINARY,
            weight=2.0,
            evaluation_criteria=[
                "Major business announcements",
                "Strategic decisions explained",
                "Financial results or projections",
                "Product launches or demos",
                "Policy changes or initiatives",
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    @staticmethod
    def technical_demo() -> DimensionDefinition:
        """Template for technical demonstrations."""
        return DimensionDefinition(
            name="technical_demo",
            description="Live demonstration of technical features or capabilities",
            type=DimensionType.SCALE_1_4,
            weight=1.3,
            evaluation_criteria=[
                "Live coding or system demonstration",
                "Feature walkthrough with examples",
                "Problem-solving in real-time",
                "Integration or workflow demonstrations",
                "Performance benchmarks shown",
            ],
            aggregation_method=AggregationMethod.MAX,
        )

    # General purpose dimensions
    @staticmethod
    def visual_interest() -> DimensionDefinition:
        """Template for visually interesting or appealing content."""
        return DimensionDefinition(
            name="visual_interest",
            description="Visual appeal and production quality",
            type=DimensionType.SCALE_1_4,
            weight=0.8,
            evaluation_criteria=[
                "High-quality cinematography or graphics",
                "Interesting visual compositions",
                "Special effects or animations",
                "Unique or beautiful locations",
                "Professional editing and transitions",
            ],
            aggregation_method=AggregationMethod.MEAN,
        )

    @staticmethod
    def narrative_importance() -> DimensionDefinition:
        """Template for story or narrative significance."""
        return DimensionDefinition(
            name="narrative_importance",
            description="Importance to overall story or narrative arc",
            type=DimensionType.SCALE_1_4,
            weight=1.1,
            evaluation_criteria=[
                "Major plot developments",
                "Character revelations or growth",
                "Story climax or resolution",
                "Important dialogue or monologue",
                "Foreshadowing or callbacks",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
