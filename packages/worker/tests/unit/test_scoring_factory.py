"""Unit tests for scoring factory."""

import pytest
from worker.services.scoring_factory import ScoringRubricFactory
from worker.services.dimension_framework import (
    DimensionDefinition,
    DimensionTemplates,
    ScoringRubric,
    DimensionType,
    AggregationMethod,
)


class TestScoringRubricFactory:
    """Test ScoringRubricFactory methods."""

    def test_create_gaming_rubric_defaults(self):
        """Test creating gaming rubric with default values."""
        rubric = ScoringRubricFactory.create_gaming_rubric()
        
        assert rubric.name == "Gaming Highlights"
        assert rubric.description == "Scoring rubric for gaming content highlights"
        assert rubric.highlight_threshold == 0.7
        assert rubric.highlight_confidence_threshold == 0.6
        assert len(rubric.dimensions) == 4
        
        # Check dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "action_intensity" in dimension_names
        assert "skill_display" in dimension_names
        assert "emotional_moment" in dimension_names
        assert "humor" in dimension_names

    def test_create_gaming_rubric_custom_params(self):
        """Test creating gaming rubric with custom parameters."""
        rubric = ScoringRubricFactory.create_gaming_rubric(
            name="Epic Gaming Moments",
            description="Find the most epic gaming clips"
        )
        
        assert rubric.name == "Epic Gaming Moments"
        assert rubric.description == "Find the most epic gaming clips"
        assert len(rubric.dimensions) == 4

    def test_create_sports_rubric_defaults(self):
        """Test creating sports rubric with default values."""
        rubric = ScoringRubricFactory.create_sports_rubric()
        
        assert rubric.name == "Sports Highlights"
        assert rubric.description == "Scoring rubric for sports content highlights"
        assert rubric.highlight_threshold == 0.65
        assert rubric.highlight_confidence_threshold == 0.7
        assert len(rubric.dimensions) == 4
        
        # Check dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "scoring_play" in dimension_names
        assert "momentum_shift" in dimension_names
        assert "action_intensity" in dimension_names
        assert "emotional_moment" in dimension_names

    def test_create_sports_rubric_custom_params(self):
        """Test creating sports rubric with custom parameters."""
        rubric = ScoringRubricFactory.create_sports_rubric(
            name="Championship Moments",
            description="Key moments from championship games"
        )
        
        assert rubric.name == "Championship Moments"
        assert rubric.description == "Key moments from championship games"
        assert len(rubric.dimensions) == 4

    def test_create_education_rubric_defaults(self):
        """Test creating education rubric with default values."""
        rubric = ScoringRubricFactory.create_education_rubric()
        
        assert rubric.name == "Educational Highlights"
        assert rubric.description == "Scoring rubric for educational content highlights"
        assert rubric.highlight_threshold == 0.75
        assert rubric.highlight_confidence_threshold == 0.65
        assert len(rubric.dimensions) == 4
        
        # Check dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "concept_clarity" in dimension_names
        assert "educational_value" in dimension_names
        assert "engagement_level" in dimension_names
        assert "visual_interest" in dimension_names

    def test_create_education_rubric_custom_params(self):
        """Test creating education rubric with custom parameters."""
        rubric = ScoringRubricFactory.create_education_rubric(
            name="Key Learning Moments",
            description="Critical teaching points"
        )
        
        assert rubric.name == "Key Learning Moments"
        assert rubric.description == "Critical teaching points"
        assert len(rubric.dimensions) == 4

    def test_create_corporate_rubric_defaults(self):
        """Test creating corporate rubric with default values."""
        rubric = ScoringRubricFactory.create_corporate_rubric()
        
        assert rubric.name == "Corporate Highlights"
        assert rubric.description == "Scoring rubric for corporate/business content highlights"
        assert rubric.highlight_threshold == 0.8
        assert rubric.highlight_confidence_threshold == 0.7
        assert len(rubric.dimensions) == 4
        
        # Check dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "key_decision" in dimension_names
        assert "technical_demo" in dimension_names
        assert "concept_clarity" in dimension_names
        assert "engagement_level" in dimension_names

    def test_create_corporate_rubric_custom_params(self):
        """Test creating corporate rubric with custom parameters."""
        rubric = ScoringRubricFactory.create_corporate_rubric(
            name="Board Meeting Highlights",
            description="Important decisions and announcements"
        )
        
        assert rubric.name == "Board Meeting Highlights"
        assert rubric.description == "Important decisions and announcements"
        assert len(rubric.dimensions) == 4

    def test_create_general_rubric_defaults(self):
        """Test creating general rubric with default values."""
        rubric = ScoringRubricFactory.create_general_rubric()
        
        assert rubric.name == "General Highlights"
        assert rubric.description == "General-purpose scoring rubric for diverse content"
        assert rubric.highlight_threshold == 0.7
        assert rubric.highlight_confidence_threshold == 0.6
        assert len(rubric.dimensions) == 5
        
        # Check dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "action_intensity" in dimension_names
        assert "emotional_moment" in dimension_names
        assert "visual_interest" in dimension_names
        assert "narrative_importance" in dimension_names
        assert "humor" in dimension_names

    def test_create_general_rubric_custom_params(self):
        """Test creating general rubric with custom parameters."""
        rubric = ScoringRubricFactory.create_general_rubric(
            name="Universal Highlights",
            description="Works for any content type"
        )
        
        assert rubric.name == "Universal Highlights"
        assert rubric.description == "Works for any content type"
        assert len(rubric.dimensions) == 5

    def test_create_custom_rubric_minimal(self):
        """Test creating custom rubric with minimal parameters."""
        dimensions = [
            DimensionDefinition(name="test1", description="Test dimension 1"),
            DimensionDefinition(name="test2", description="Test dimension 2"),
        ]
        
        rubric = ScoringRubricFactory.create_custom_rubric(
            name="Custom Test",
            description="Test custom rubric",
            dimensions=dimensions
        )
        
        assert rubric.name == "Custom Test"
        assert rubric.description == "Test custom rubric"
        assert len(rubric.dimensions) == 2
        assert rubric.highlight_threshold == 0.7  # Default
        assert rubric.highlight_confidence_threshold == 0.6  # Default
        assert rubric.requires_all_dimensions is False  # Default
        assert rubric.normalization_enabled is True  # Default

    def test_create_custom_rubric_all_params(self):
        """Test creating custom rubric with all parameters."""
        dimensions = [
            DimensionDefinition(
                name="excitement",
                description="Level of excitement",
                type=DimensionType.SCALE_1_4,
                weight=2.0
            ),
            DimensionDefinition(
                name="importance",
                description="Importance of moment",
                type=DimensionType.NUMERIC,
                weight=1.5
            ),
        ]
        
        rubric = ScoringRubricFactory.create_custom_rubric(
            name="Custom Advanced",
            description="Advanced custom rubric",
            dimensions=dimensions,
            highlight_threshold=0.85,
            highlight_confidence_threshold=0.75,
            requires_all_dimensions=True,
            normalization_enabled=False
        )
        
        assert rubric.name == "Custom Advanced"
        assert rubric.description == "Advanced custom rubric"
        assert len(rubric.dimensions) == 2
        assert rubric.highlight_threshold == 0.85
        assert rubric.highlight_confidence_threshold == 0.75
        assert rubric.requires_all_dimensions is True
        assert rubric.normalization_enabled is False
        
        # Check dimensions were added correctly
        assert rubric.dimensions[0].name == "excitement"
        assert rubric.dimensions[0].weight == 2.0
        assert rubric.dimensions[1].name == "importance"
        assert rubric.dimensions[1].weight == 1.5

    def test_create_from_template_names_valid(self):
        """Test creating rubric from valid template names."""
        rubric = ScoringRubricFactory.create_from_template_names(
            name="Mixed Templates",
            description="Rubric from various templates",
            template_names=["action_intensity", "educational_value", "humor"]
        )
        
        assert rubric.name == "Mixed Templates"
        assert rubric.description == "Rubric from various templates"
        assert len(rubric.dimensions) == 3
        
        dimension_names = [d.name for d in rubric.dimensions]
        assert "action_intensity" in dimension_names
        assert "educational_value" in dimension_names
        assert "humor" in dimension_names

    def test_create_from_template_names_with_kwargs(self):
        """Test creating rubric from template names with additional kwargs."""
        rubric = ScoringRubricFactory.create_from_template_names(
            name="Custom Templates",
            description="Templates with custom settings",
            template_names=["skill_display", "emotional_moment"],
            highlight_threshold=0.9,
            highlight_confidence_threshold=0.8,
            requires_all_dimensions=True
        )
        
        assert rubric.name == "Custom Templates"
        assert len(rubric.dimensions) == 2
        assert rubric.highlight_threshold == 0.9
        assert rubric.highlight_confidence_threshold == 0.8
        assert rubric.requires_all_dimensions is True

    def test_create_from_template_names_invalid_template(self):
        """Test creating rubric with invalid template name."""
        with pytest.raises(ValueError) as exc_info:
            ScoringRubricFactory.create_from_template_names(
                name="Invalid",
                description="Invalid template test",
                template_names=["action_intensity", "invalid_template"]
            )
        
        assert "Unknown template: invalid_template" in str(exc_info.value)
        assert "Available templates:" in str(exc_info.value)

    def test_create_from_template_names_empty_list(self):
        """Test creating rubric with empty template list."""
        rubric = ScoringRubricFactory.create_from_template_names(
            name="Empty",
            description="No templates",
            template_names=[]
        )
        
        assert rubric.name == "Empty"
        assert rubric.description == "No templates"
        assert len(rubric.dimensions) == 0

    def test_create_from_template_names_all_available(self):
        """Test creating rubric with all available templates."""
        # Get all available template names
        available_templates = [
            name for name in dir(DimensionTemplates)
            if not name.startswith("_") and callable(getattr(DimensionTemplates, name))
        ]
        
        rubric = ScoringRubricFactory.create_from_template_names(
            name="All Templates",
            description="Every available template",
            template_names=available_templates
        )
        
        assert rubric.name == "All Templates"
        assert len(rubric.dimensions) == len(available_templates)
        
        # Verify all templates were added
        dimension_names = [d.name for d in rubric.dimensions]
        for template_name in available_templates:
            template_method = getattr(DimensionTemplates, template_name)
            dimension = template_method()
            assert dimension.name in dimension_names

    def test_gaming_rubric_dimension_properties(self):
        """Test specific properties of gaming rubric dimensions."""
        rubric = ScoringRubricFactory.create_gaming_rubric()
        
        # Find action_intensity dimension
        action_dim = next(d for d in rubric.dimensions if d.name == "action_intensity")
        assert action_dim.type == DimensionType.SCALE_1_4
        assert action_dim.aggregation_method == AggregationMethod.MAX
        
        # Find skill_display dimension
        skill_dim = next(d for d in rubric.dimensions if d.name == "skill_display")
        assert skill_dim.weight == 1.5
        assert skill_dim.type == DimensionType.SCALE_1_4

    def test_sports_rubric_dimension_properties(self):
        """Test specific properties of sports rubric dimensions."""
        rubric = ScoringRubricFactory.create_sports_rubric()
        
        # Find scoring_play dimension
        scoring_dim = next(d for d in rubric.dimensions if d.name == "scoring_play")
        assert scoring_dim.type == DimensionType.BINARY
        assert scoring_dim.weight == 2.0
        
        # Find momentum_shift dimension
        momentum_dim = next(d for d in rubric.dimensions if d.name == "momentum_shift")
        assert momentum_dim.type == DimensionType.SCALE_1_4
        assert momentum_dim.weight == 1.3

    def test_education_rubric_dimension_properties(self):
        """Test specific properties of education rubric dimensions."""
        rubric = ScoringRubricFactory.create_education_rubric()
        
        # Find concept_clarity dimension
        clarity_dim = next(d for d in rubric.dimensions if d.name == "concept_clarity")
        assert clarity_dim.weight == 1.5
        assert clarity_dim.aggregation_method == AggregationMethod.WEIGHTED_MEAN
        
        # Find educational_value dimension
        edu_dim = next(d for d in rubric.dimensions if d.name == "educational_value")
        assert edu_dim.type == DimensionType.NUMERIC
        assert edu_dim.temporal_weight_decay == 0.1

    def test_dimension_uniqueness_in_rubrics(self):
        """Test that dimensions are not shared between rubrics (new instances)."""
        rubric1 = ScoringRubricFactory.create_gaming_rubric()
        rubric2 = ScoringRubricFactory.create_gaming_rubric()
        
        # Modify a dimension in rubric1
        rubric1.dimensions[0].weight = 5.0
        
        # Verify rubric2 is not affected
        assert rubric2.dimensions[0].weight != 5.0

    def test_factory_methods_are_static(self):
        """Test that all factory methods are static."""
        import inspect
        
        for name, method in inspect.getmembers(ScoringRubricFactory, predicate=inspect.ismethod):
            if not name.startswith("_"):
                # Should be static method
                assert isinstance(inspect.getattr_static(ScoringRubricFactory, name), staticmethod)