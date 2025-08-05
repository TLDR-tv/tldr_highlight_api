"""Factory for creating scoring rubrics and configurations."""

from .dimension_framework import (
    AggregationMethod,
    DimensionDefinition,
    DimensionExample,
    DimensionTemplates,
    DimensionType,
    ScoringRubric,
)


class ScoringRubricFactory:
    """Factory for creating pre-configured scoring rubrics."""

    @staticmethod
    def create_gaming_rubric(
        name: str = "Gaming Highlights",
        description: str = "Scoring rubric for gaming content highlights",
    ) -> ScoringRubric:
        """Create a rubric optimized for gaming content.

        Returns:
            ScoringRubric configured for gaming highlights

        """
        rubric = ScoringRubric(
            name=name,
            description=description,
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.6,
        )

        # Add gaming-specific dimensions
        rubric.add_dimension(DimensionTemplates.action_intensity())
        rubric.add_dimension(DimensionTemplates.skill_display())
        rubric.add_dimension(DimensionTemplates.emotional_moment())
        rubric.add_dimension(DimensionTemplates.humor())

        return rubric

    @staticmethod
    def create_sports_rubric(
        name: str = "Sports Highlights",
        description: str = "Scoring rubric for sports content highlights",
    ) -> ScoringRubric:
        """Create a rubric optimized for sports content.

        Returns:
            ScoringRubric configured for sports highlights

        """
        rubric = ScoringRubric(
            name=name,
            description=description,
            highlight_threshold=0.65,
            highlight_confidence_threshold=0.7,
        )

        # Add sports-specific dimensions
        rubric.add_dimension(DimensionTemplates.scoring_play())
        rubric.add_dimension(DimensionTemplates.momentum_shift())
        rubric.add_dimension(DimensionTemplates.action_intensity())
        rubric.add_dimension(DimensionTemplates.emotional_moment())

        return rubric

    @staticmethod
    def create_education_rubric(
        name: str = "Educational Highlights",
        description: str = "Scoring rubric for educational content highlights",
    ) -> ScoringRubric:
        """Create a rubric optimized for educational content.

        Returns:
            ScoringRubric configured for educational highlights

        """
        rubric = ScoringRubric(
            name=name,
            description=description,
            highlight_threshold=0.75,
            highlight_confidence_threshold=0.65,
        )

        # Add education-specific dimensions
        rubric.add_dimension(DimensionTemplates.concept_clarity())
        rubric.add_dimension(DimensionTemplates.educational_value())
        rubric.add_dimension(DimensionTemplates.engagement_level())
        rubric.add_dimension(DimensionTemplates.visual_interest())

        return rubric

    @staticmethod
    def create_corporate_rubric(
        name: str = "Corporate Highlights",
        description: str = "Scoring rubric for corporate/business content highlights",
    ) -> ScoringRubric:
        """Create a rubric optimized for corporate content.

        Returns:
            ScoringRubric configured for corporate highlights

        """
        rubric = ScoringRubric(
            name=name,
            description=description,
            highlight_threshold=0.8,
            highlight_confidence_threshold=0.7,
        )

        # Add corporate-specific dimensions
        rubric.add_dimension(DimensionTemplates.key_decision())
        rubric.add_dimension(DimensionTemplates.technical_demo())
        rubric.add_dimension(DimensionTemplates.concept_clarity())
        rubric.add_dimension(DimensionTemplates.engagement_level())

        return rubric

    @staticmethod
    def create_general_rubric(
        name: str = "General Highlights",
        description: str = "General-purpose scoring rubric for diverse content",
    ) -> ScoringRubric:
        """Create a general-purpose rubric.

        Returns:
            ScoringRubric configured for general content

        """
        rubric = ScoringRubric(
            name=name,
            description=description,
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.6,
        )

        # Add general dimensions
        rubric.add_dimension(DimensionTemplates.action_intensity())
        rubric.add_dimension(DimensionTemplates.emotional_moment())
        rubric.add_dimension(DimensionTemplates.visual_interest())
        rubric.add_dimension(DimensionTemplates.narrative_importance())
        rubric.add_dimension(DimensionTemplates.humor())

        return rubric

    @staticmethod
    def create_dyli_rubric(
        name: str = "DYLI Trading Card & Collectibles",
        description: str = "Optimized for trading card pack openings, collectibles reveals, and merchandise showcases",
    ) -> ScoringRubric:
        """Create a rubric optimized for DYLI's collectibles content.
        
        Returns:
            ScoringRubric configured for trading card and collectibles highlights
            
        """
        rubric = ScoringRubric(
            name=name,
            description=description,
            highlight_threshold=0.65,  # Slightly lower to catch good moments
            highlight_confidence_threshold=0.7,
        )
        
        # Primary dimensions for trading card/collectibles content
        
        # Rare pull moment - highest weight for the money shots
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
        
        # Product showcase - critical for marketplace visibility
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
        
        # Excitement level - drives engagement
        excitement = DimensionDefinition(
            name="excitement_level",
            description="Creator's excitement and energy level during reveals",
            type=DimensionType.SCALE_1_4,
            weight=2.0,
            scoring_prompt="Rate the creator's excitement and energy in this moment",
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
        
        # Collector reaction - authenticity matters
        reaction = DimensionDefinition(
            name="collector_reaction",
            description="Authentic collector emotions and reactions",
            type=DimensionType.SCALE_1_4,
            weight=1.8,
            scoring_prompt="Rate the authenticity and relatability of collector reactions",
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
        
        # Value discussion - educational for buyers
        value = DimensionDefinition(
            name="value_discussion",
            description="Discussion of card value, rarity, market price, or collectibility",
            type=DimensionType.NUMERIC,
            weight=1.5,
            min_score=0.0,
            max_score=1.0,
            scoring_prompt="Rate how much valuable information about pricing/rarity is shared (0-1)",
            evaluation_criteria=[
                "Mentions specific market value or price ranges",
                "Discusses card rarity (1 in X packs, print run numbers)",
                "Compares to other cards in set or market",
                "Mentions grading potential or condition",
                "Shares collecting tips or market insights",
            ],
            aggregation_method=AggregationMethod.MAX,
        )
        rubric.add_dimension(value)
        
        # Add some general dimensions with lower weights
        rubric.add_dimension(DimensionTemplates.visual_interest())
        
        # Community engagement - important for social commerce
        community = DimensionDefinition(
            name="community_engagement",
            description="Interaction with viewers and collector community",
            type=DimensionType.SCALE_1_4,
            weight=1.2,
            evaluation_criteria=[
                "Acknowledges viewers or community",
                "Responds to chat or comments",
                "Creates interactive moments (polls, questions)",
                "Shares community inside jokes or references",
                "Encourages viewer participation",
            ],
            aggregation_method=AggregationMethod.MEAN,
        )
        rubric.add_dimension(community)
        
        return rubric

    @staticmethod
    def create_custom_rubric(
        name: str,
        description: str,
        dimensions: list[DimensionDefinition],
        highlight_threshold: float = 0.7,
        highlight_confidence_threshold: float = 0.6,
        requires_all_dimensions: bool = False,
        normalization_enabled: bool = True,
    ) -> ScoringRubric:
        """Create a custom rubric with specified dimensions.

        Args:
            name: Rubric name
            description: Rubric description
            dimensions: List of dimension definitions
            highlight_threshold: Score threshold for highlights
            highlight_confidence_threshold: Confidence threshold
            requires_all_dimensions: Whether all dimensions must be scored
            normalization_enabled: Whether to normalize weights

        Returns:
            Custom ScoringRubric

        """
        rubric = ScoringRubric(
            name=name,
            description=description,
            dimensions=dimensions,
            highlight_threshold=highlight_threshold,
            highlight_confidence_threshold=highlight_confidence_threshold,
            requires_all_dimensions=requires_all_dimensions,
            normalization_enabled=normalization_enabled,
        )

        return rubric

    @staticmethod
    def create_from_template_names(
        name: str, description: str, template_names: list[str], **kwargs
    ) -> ScoringRubric:
        """Create a rubric using dimension template names.

        Args:
            name: Rubric name
            description: Rubric description
            template_names: List of template names from DimensionTemplates
            **kwargs: Additional arguments for create_custom_rubric

        Returns:
            ScoringRubric with specified templates

        Raises:
            ValueError: If template name not found

        """
        dimensions = []

        for template_name in template_names:
            # Get template method from DimensionTemplates
            if hasattr(DimensionTemplates, template_name):
                template_method = getattr(DimensionTemplates, template_name)
                dimensions.append(template_method())
            else:
                available = [
                    name
                    for name in dir(DimensionTemplates)
                    if not name.startswith("_")
                    and callable(getattr(DimensionTemplates, name))
                ]
                raise ValueError(
                    f"Unknown template: {template_name}. "
                    f"Available templates: {', '.join(available)}"
                )

        return ScoringRubricFactory.create_custom_rubric(
            name=name, description=description, dimensions=dimensions, **kwargs
        )
