"""Simple prompt template value object for AI-powered highlight detection."""

from dataclasses import dataclass
from typing import Any
from string import Template


@dataclass(frozen=True)
class PromptTemplate:
    """A simple prompt template for AI analysis.

    Uses Python's string.Template for safe variable substitution.
    """

    template: str

    def __post_init__(self) -> None:
        """Validate template."""
        if not self.template or not self.template.strip():
            raise ValueError("Template cannot be empty")

        # Validate it's a valid template
        try:
            Template(self.template)
        except Exception as e:
            raise ValueError(f"Invalid template format: {e}")

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Rendered prompt string
        """
        template = Template(self.template)
        return template.safe_substitute(**kwargs)

    @classmethod
    def for_dimension_scoring(cls) -> "PromptTemplate":
        """Create a template for dimension scoring."""
        return cls(
            template="""Analyze this $content_type content for the following dimensions:

$dimensions_list

Context: $context

For each dimension, provide:
1. A score between 0.0 and 1.0
2. A confidence level (high/medium/low)
3. Brief evidence for your scoring

Focus on objective analysis based on the content provided."""
        )

    @classmethod
    def for_highlight_detection(cls) -> "PromptTemplate":
        """Create a template for highlight detection."""
        return cls(
            template="""Identify highlight-worthy moments in this $content_type.

Duration: $duration seconds
Context: $context

Look for moments that are:
- Exceptional or skillful
- Emotionally impactful
- Visually impressive
- Narratively important

Return timestamps and descriptions of potential highlights."""
        )
