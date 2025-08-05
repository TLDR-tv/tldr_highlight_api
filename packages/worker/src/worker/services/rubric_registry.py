"""Registry of available scoring rubrics."""

from typing import Dict, Optional
from worker.services.dimension_framework import (
    AggregationMethod,
    DimensionDefinition,
    DimensionExample,
    DimensionTemplates,
    DimensionType,
    ScoringRubric,
)


class RubricRegistry:
    """Registry of all available scoring rubrics."""
    
    _rubrics: Dict[str, ScoringRubric] = {}
    _initialized: bool = False
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry with all available rubrics."""
        if cls._initialized:
            return
            
        # Register all rubrics
        cls.register_general_rubric()
        cls.register_action_content_rubric()
        cls.register_educational_rubric()
        cls.register_product_showcase_rubric()
        cls.register_dyli_rubric()
        
        cls._initialized = True
    
    @classmethod
    def get_rubric(cls, name: str) -> Optional[ScoringRubric]:
        """Get a rubric by name."""
        cls.initialize()
        return cls._rubrics.get(name.lower())
    
    @classmethod
    def list_rubrics(cls) -> Dict[str, str]:
        """List all available rubrics with descriptions."""
        cls.initialize()
        return {
            name: rubric.description 
            for name, rubric in cls._rubrics.items()
        }
    
    @classmethod
    def _register(cls, name: str, rubric: ScoringRubric) -> None:
        """Register a rubric."""
        cls._rubrics[name.lower()] = rubric
    
    @classmethod
    def register_general_rubric(cls) -> None:
        """Register a general-purpose rubric that works for most content."""
        rubric = ScoringRubric(
            name="General Content",
            description="General-purpose rubric for diverse content types",
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.6,
        )
        
        # Mix of dimensions that work across content types
        rubric.add_dimension(DimensionTemplates.action_intensity())
        rubric.add_dimension(DimensionTemplates.emotional_moment())
        rubric.add_dimension(DimensionTemplates.visual_interest())
        rubric.add_dimension(DimensionTemplates.narrative_importance())
        
        # Add engagement dimension
        engagement = DimensionDefinition(
            name="audience_engagement",
            description="Moments likely to engage or captivate audience",
            type=DimensionType.SCALE_1_4,
            weight=1.5,
            evaluation_criteria=[
                "Surprising or unexpected moments",
                "High energy or excitement",
                "Relatable or shareable content",
                "Clear focal point or subject",
                "Memorable or quotable moments",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(engagement)
        
        cls._register("general", rubric)
    
    @classmethod
    def register_action_content_rubric(cls) -> None:
        """Register rubric for action-heavy content (gaming, sports, etc)."""
        rubric = ScoringRubric(
            name="Action Content",
            description="Optimized for action-heavy content like gaming and sports",
            highlight_threshold=0.65,
            highlight_confidence_threshold=0.6,
        )
        
        # Action-focused dimensions
        rubric.add_dimension(DimensionTemplates.action_intensity())
        rubric.add_dimension(DimensionTemplates.skill_display())
        
        # Peak moments
        peak_moment = DimensionDefinition(
            name="peak_moment",
            description="Climactic or peak moments in the action",
            type=DimensionType.BINARY,
            weight=2.5,
            evaluation_criteria=[
                "Game-winning or decisive plays",
                "Exceptional skill demonstrations",
                "Comebacks or turnarounds",
                "Record-breaking moments",
                "High-stakes situations resolved",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(peak_moment)
        
        rubric.add_dimension(DimensionTemplates.emotional_moment())
        rubric.add_dimension(DimensionTemplates.momentum_shift())
        
        cls._register("action", rubric)
    
    @classmethod
    def register_educational_rubric(cls) -> None:
        """Register rubric for educational and informational content."""
        rubric = ScoringRubric(
            name="Educational Content",
            description="Optimized for educational, tutorial, and informational content",
            highlight_threshold=0.75,
            highlight_confidence_threshold=0.65,
        )
        
        # Educational dimensions
        rubric.add_dimension(DimensionTemplates.concept_clarity())
        rubric.add_dimension(DimensionTemplates.educational_value())
        
        # Key insight moments
        key_insight = DimensionDefinition(
            name="key_insight",
            description="Moments containing key insights or 'aha' moments",
            type=DimensionType.SCALE_1_4,
            weight=2.0,
            evaluation_criteria=[
                "Core concept explained clearly",
                "Common misconception corrected",
                "Practical tip or trick revealed",
                "Complex idea simplified",
                "Important conclusion reached",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(key_insight)
        
        rubric.add_dimension(DimensionTemplates.engagement_level())
        rubric.add_dimension(DimensionTemplates.visual_interest())
        
        cls._register("educational", rubric)
    
    @classmethod
    def register_product_showcase_rubric(cls) -> None:
        """Register rubric for product demos and showcases."""
        rubric = ScoringRubric(
            name="Product Showcase",
            description="Optimized for product demonstrations, reviews, and showcases",
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.65,
        )
        
        # Product visibility
        product_focus = DimensionDefinition(
            name="product_focus",
            description="Clear focus on product features and details",
            type=DimensionType.SCALE_1_4,
            weight=2.5,
            evaluation_criteria=[
                "Product shown clearly and prominently",
                "Key features demonstrated",
                "Product benefits explained",
                "Comparison with alternatives",
                "Usage scenarios shown",
            ],
            aggregation_method=AggregationMethod.MEAN,
        )
        rubric.add_dimension(product_focus)
        
        rubric.add_dimension(DimensionTemplates.technical_demo())
        
        # Value proposition
        value_prop = DimensionDefinition(
            name="value_proposition",
            description="Communication of product value and benefits",
            type=DimensionType.NUMERIC,
            weight=1.8,
            min_score=0.0,
            max_score=1.0,
            evaluation_criteria=[
                "Price or value mentioned",
                "Unique selling points highlighted",
                "Problem-solution fit demonstrated",
                "Customer testimonials or reactions",
                "ROI or benefits quantified",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(value_prop)
        
        rubric.add_dimension(DimensionTemplates.visual_interest())
        rubric.add_dimension(DimensionTemplates.engagement_level())
        
        cls._register("product", rubric)
    
    @classmethod
    def register_dyli_rubric(cls) -> None:
        """Register DYLI's custom rubric for collectibles content."""
        rubric = ScoringRubric(
            name="DYLI Trading Card & Collectibles",
            description="Optimized for trading card pack openings and collectibles reveals",
            highlight_threshold=0.65,
            highlight_confidence_threshold=0.7,
        )
        
        # Rare pull moment - highest weight
        rare_pull = DimensionDefinition(
            name="rare_pull_moment",
            description="Moments when rare, valuable, or chase cards are revealed",
            type=DimensionType.BINARY,
            weight=3.0,
            scoring_prompt="Is a rare, valuable, or highly sought-after card being revealed in this moment?",
            evaluation_criteria=[
                "Card is being pulled from a pack or revealed for the first time",
                "Card has special rarity indicators (holographic, numbered, special edition)",
                "Creator mentions rarity, value, or that it's a 'hit' or 'chase card'",
                "Visible excitement or surprise reaction to the card",
                "Card is held up prominently to camera or zoomed in on",
            ],
            examples=[
                DimensionExample(
                    input_description="Creator pulls a common base card and quickly moves to next",
                    expected_score=0,
                    reasoning="Common card with no special reaction or emphasis",
                ),
                DimensionExample(
                    input_description="Creator pulls holographic card, shouts 'NO WAY!', shows it to camera",
                    expected_score=1,
                    reasoning="Rare card reveal with strong reaction and clear presentation",
                ),
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(rare_pull)
        
        # Product showcase - critical for marketplace
        showcase = DimensionDefinition(
            name="product_showcase",
            description="Clear visibility and presentation of products/cards",
            type=DimensionType.SCALE_1_4,
            weight=2.5,
            scoring_prompt="How well are the products/cards showcased visually?",
            evaluation_criteria=[
                "Cards held steady and clearly visible to camera",
                "Good lighting on the product",
                "Close-up shots or zoom-ins on card details",
                "Card features (artwork, text, rarity markers) are readable",
                "Multiple angles or thorough examination of product",
            ],
            aggregation_method=AggregationMethod.MEAN,
        )
        rubric.add_dimension(showcase)
        
        # Excitement level
        excitement = DimensionDefinition(
            name="excitement_level",
            description="Creator's excitement and energy level during reveals",
            type=DimensionType.SCALE_1_4,
            weight=2.0,
            evaluation_criteria=[
                "Vocal excitement (shouting, exclamations, surprise)",
                "Physical reactions (jumping, gesturing, celebrating)",
                "Verbal expressions of excitement or disbelief",
                "Building anticipation before reveals",
                "Sustained energy throughout the segment",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(excitement)
        
        # Collector reaction
        reaction = DimensionDefinition(
            name="collector_reaction",
            description="Authentic collector emotions and reactions",
            type=DimensionType.SCALE_1_4,
            weight=1.8,
            evaluation_criteria=[
                "Genuine surprise or disappointment",
                "Nostalgic reactions or personal connections to cards",
                "Sharing collecting stories or experiences",
                "Reactions that other collectors would relate to",
                "Emotional investment in the pulls",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(reaction)
        
        # Value discussion
        value = DimensionDefinition(
            name="value_discussion",
            description="Discussion of card value, rarity, or market price",
            type=DimensionType.NUMERIC,
            weight=1.5,
            min_score=0.0,
            max_score=1.0,
            evaluation_criteria=[
                "Mentions specific market value or price ranges",
                "Discusses card rarity (1 in X packs, print run)",
                "Compares to other cards in set or market",
                "Mentions grading potential or condition",
                "Shares collecting tips or market insights",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(value)
        
        cls._register("dyli", rubric)