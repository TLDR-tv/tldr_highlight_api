"""Comprehensive tests for rubric registry."""

import pytest
from unittest.mock import patch
from worker.services.rubric_registry import RubricRegistry
from worker.services.dimension_framework import ScoringRubric


class TestRubricRegistry:
    """Test RubricRegistry class comprehensively."""

    def setup_method(self):
        """Reset registry state before each test."""
        RubricRegistry._rubrics = {}
        RubricRegistry._initialized = False

    def teardown_method(self):
        """Reset registry state after each test."""
        RubricRegistry._rubrics = {}
        RubricRegistry._initialized = False

    def test_initialize_registry(self):
        """Test registry initialization."""
        assert not RubricRegistry._initialized
        assert len(RubricRegistry._rubrics) == 0
        
        RubricRegistry.initialize()
        
        assert RubricRegistry._initialized
        assert len(RubricRegistry._rubrics) > 0

    def test_initialize_registry_idempotent(self):
        """Test that multiple initializations don't cause issues."""
        RubricRegistry.initialize()
        initial_count = len(RubricRegistry._rubrics)
        initial_initialized = RubricRegistry._initialized
        
        # Initialize again
        RubricRegistry.initialize()
        
        # Should not change state
        assert len(RubricRegistry._rubrics) == initial_count
        assert RubricRegistry._initialized == initial_initialized

    @patch('worker.services.rubric_registry.logger')
    def test_initialize_logs_debug_when_already_initialized(self, mock_logger):
        """Test that re-initialization logs debug message."""
        RubricRegistry.initialize()
        mock_logger.reset_mock()
        
        # Initialize again
        RubricRegistry.initialize()
        
        mock_logger.debug.assert_called_once_with("RubricRegistry already initialized, skipping")

    @patch('worker.services.rubric_registry.logger')
    def test_initialize_logs_info_messages(self, mock_logger):
        """Test that initialization logs appropriate info messages."""
        RubricRegistry.initialize()
        
        # Check that info messages were logged
        info_calls = [call for call in mock_logger.info.call_args_list if call]
        assert len(info_calls) >= 1
        assert any("Initializing RubricRegistry" in str(call) for call in info_calls)
        assert any("RubricRegistry initialized with" in str(call) for call in info_calls)

    def test_get_rubric_success(self):
        """Test getting an existing rubric."""
        RubricRegistry.initialize()
        
        # Should have at least 'general' rubric
        rubric = RubricRegistry.get_rubric("general")
        assert rubric is not None
        assert isinstance(rubric, ScoringRubric)
        assert rubric.name is not None

    def test_get_rubric_case_insensitive(self):
        """Test that rubric lookup is case insensitive."""
        RubricRegistry.initialize()
        
        # Try different cases
        rubric1 = RubricRegistry.get_rubric("GENERAL")
        rubric2 = RubricRegistry.get_rubric("General")
        rubric3 = RubricRegistry.get_rubric("general")
        
        assert rubric1 is not None
        assert rubric1 == rubric2
        assert rubric2 == rubric3

    def test_get_rubric_not_found(self):
        """Test getting a non-existent rubric."""
        RubricRegistry.initialize()
        
        rubric = RubricRegistry.get_rubric("nonexistent")
        assert rubric is None

    @patch('worker.services.rubric_registry.logger')
    def test_get_rubric_logs_debug_when_found(self, mock_logger):
        """Test that finding a rubric logs debug message."""
        RubricRegistry.initialize()
        mock_logger.reset_mock()
        
        RubricRegistry.get_rubric("general")
        
        mock_logger.debug.assert_called_once()
        debug_call = mock_logger.debug.call_args[0][0]
        assert "Found rubric" in debug_call

    @patch('worker.services.rubric_registry.logger')
    def test_get_rubric_logs_warning_when_not_found(self, mock_logger):
        """Test that missing rubric logs warning message."""
        RubricRegistry.initialize()
        mock_logger.reset_mock()
        
        RubricRegistry.get_rubric("nonexistent")
        
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "not found" in warning_call
        assert "Available rubrics:" in warning_call

    def test_list_rubrics(self):
        """Test listing all available rubrics."""
        RubricRegistry.initialize()
        
        rubrics = RubricRegistry.list_rubrics()
        
        assert isinstance(rubrics, dict)
        assert len(rubrics) > 0
        
        # Check that all values are strings (descriptions)
        for name, description in rubrics.items():
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_list_rubrics_initializes_if_needed(self):
        """Test that list_rubrics initializes registry if needed."""
        assert not RubricRegistry._initialized
        
        rubrics = RubricRegistry.list_rubrics()
        
        assert RubricRegistry._initialized
        assert len(rubrics) > 0

    def test_register_method(self):
        """Test the _register method."""
        test_rubric = ScoringRubric(
            name="Test Rubric",
            description="A test rubric",
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.6
        )
        
        RubricRegistry._register("test", test_rubric)
        
        assert "test" in RubricRegistry._rubrics
        assert RubricRegistry._rubrics["test"] == test_rubric

    def test_register_method_case_insensitive_key(self):
        """Test that _register uses lowercase keys."""
        test_rubric = ScoringRubric(
            name="Test Rubric",
            description="A test rubric",
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.6
        )
        
        RubricRegistry._register("TEST_RUBRIC", test_rubric)
        
        assert "test_rubric" in RubricRegistry._rubrics
        assert RubricRegistry._rubrics["test_rubric"] == test_rubric

    @patch('worker.services.rubric_registry.logger')
    def test_register_logs_debug(self, mock_logger):
        """Test that _register logs debug message."""
        test_rubric = ScoringRubric(
            name="Test Rubric",
            description="A test rubric",
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.6
        )
        
        RubricRegistry._register("test", test_rubric)
        
        mock_logger.debug.assert_called_once()
        debug_call = mock_logger.debug.call_args[0][0]
        assert "Registered rubric:" in debug_call

    def test_register_general_rubric(self):
        """Test registering the general rubric."""
        RubricRegistry.register_general_rubric()
        
        assert "general content" in RubricRegistry._rubrics
        rubric = RubricRegistry._rubrics["general content"]
        assert rubric.name == "General Content"
        assert "general-purpose rubric" in rubric.description.lower()
        assert len(rubric.dimensions) > 0

    def test_register_action_content_rubric(self):
        """Test registering the action content rubric."""
        RubricRegistry.register_action_content_rubric()
        
        # Find the rubric (key should be lowercase)
        action_rubric = None
        for key, rubric in RubricRegistry._rubrics.items():
            if "action" in key or "gaming" in key:
                action_rubric = rubric
                break
        
        assert action_rubric is not None
        assert len(action_rubric.dimensions) > 0

    def test_register_educational_rubric(self):
        """Test registering the educational rubric."""
        RubricRegistry.register_educational_rubric()
        
        # Find the rubric
        edu_rubric = None
        for key, rubric in RubricRegistry._rubrics.items():
            if "education" in key:
                edu_rubric = rubric
                break
        
        assert edu_rubric is not None
        assert len(edu_rubric.dimensions) > 0

    def test_register_product_showcase_rubric(self):
        """Test registering the product showcase rubric."""
        RubricRegistry.register_product_showcase_rubric()
        
        # Find the rubric
        product_rubric = None
        for key, rubric in RubricRegistry._rubrics.items():
            if "product" in key:
                product_rubric = rubric
                break
        
        assert product_rubric is not None
        assert len(product_rubric.dimensions) > 0

    def test_register_dyli_rubric(self):
        """Test registering the DYLI rubric."""
        RubricRegistry.register_dyli_rubric()
        
        # Find the rubric
        dyli_rubric = None
        for key, rubric in RubricRegistry._rubrics.items():
            if "dyli" in key or "trading" in key:
                dyli_rubric = rubric
                break
        
        assert dyli_rubric is not None
        assert len(dyli_rubric.dimensions) > 0

    def test_all_registered_rubrics_are_valid(self):
        """Test that all registered rubrics are valid."""
        RubricRegistry.initialize()
        
        for name, rubric in RubricRegistry._rubrics.items():
            assert isinstance(rubric, ScoringRubric)
            assert len(rubric.name) > 0
            assert len(rubric.description) > 0
            assert rubric.highlight_threshold > 0
            assert rubric.highlight_confidence_threshold > 0
            assert len(rubric.dimensions) > 0

    def test_registry_contains_expected_rubrics(self):
        """Test that registry contains expected rubrics after initialization."""
        RubricRegistry.initialize()
        
        rubric_names = list(RubricRegistry._rubrics.keys())
        
        # Should have at least these rubrics
        expected_patterns = ["general", "action", "education", "product", "dyli"]
        
        for pattern in expected_patterns:
            assert any(pattern in name for name in rubric_names), \
                f"Expected rubric matching '{pattern}' not found in {rubric_names}"

    def test_get_rubric_initializes_if_needed(self):
        """Test that get_rubric initializes registry if not already done."""
        assert not RubricRegistry._initialized
        
        rubric = RubricRegistry.get_rubric("general")
        
        assert RubricRegistry._initialized
        # Should find a rubric since initialization happened
        assert rubric is not None or len(RubricRegistry._rubrics) > 0

    def test_rubric_registry_thread_safety_basic(self):
        """Test basic thread safety of registry initialization."""
        # This is a basic test - true thread safety would require more complex testing
        import threading
        
        results = []
        
        def init_and_get():
            RubricRegistry.initialize()
            rubric = RubricRegistry.get_rubric("general")
            results.append(rubric)
        
        threads = [threading.Thread(target=init_and_get) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should have completed successfully
        assert len(results) == 5
        # Registry should only be initialized once
        assert RubricRegistry._initialized

    def test_registry_state_consistency(self):
        """Test that registry state remains consistent across operations."""
        RubricRegistry.initialize()
        initial_rubrics = dict(RubricRegistry._rubrics)
        initial_count = len(initial_rubrics)
        
        # Multiple operations
        RubricRegistry.get_rubric("general")
        RubricRegistry.list_rubrics()
        RubricRegistry.get_rubric("nonexistent")
        RubricRegistry.initialize()  # Should be idempotent
        
        # State should be unchanged
        assert len(RubricRegistry._rubrics) == initial_count
        assert RubricRegistry._rubrics == initial_rubrics

    def test_rubric_dimensions_have_required_properties(self):
        """Test that all rubric dimensions have required properties."""
        RubricRegistry.initialize()
        
        for rubric_name, rubric in RubricRegistry._rubrics.items():
            for dimension in rubric.dimensions:
                # All dimensions should have these basic properties
                assert hasattr(dimension, 'name')
                assert hasattr(dimension, 'description')
                assert hasattr(dimension, 'type')
                assert hasattr(dimension, 'weight')
                
                # Values should be valid
                assert len(dimension.name) > 0
                assert len(dimension.description) > 0
                assert 0 <= dimension.weight <= 10