"""Factory for creating scoring rubrics and configurations."""

from .dimension_framework import (
    DimensionDefinition,
    DimensionTemplates,
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
