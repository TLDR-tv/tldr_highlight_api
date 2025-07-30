"""Configurable prompt template value object for AI-powered highlight detection.

This value object encapsulates prompt templates that can be customized by B2B consumers
to tailor highlight detection to their specific content and requirements.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Optional
from string import Template


@dataclass(frozen=True)
class ConfigurablePromptTemplate:
    """A configurable prompt template for AI analysis.
    
    Allows B2B consumers to customize how AI models analyze content
    while maintaining safety and effectiveness.
    """
    
    # Main analysis prompt
    analysis_prompt: str
    
    # System instructions
    system_instructions: str = ""
    
    # Scoring instructions
    scoring_instructions: str = ""
    
    # Context instructions  
    context_instructions: str = ""
    
    # Safety constraints
    max_length: int = 4000
    required_variables: Optional[Set[str]] = None
    
    def __post_init__(self) -> None:
        """Validate the prompt template."""
        if self.required_variables is None:
            # Set default required variables
            object.__setattr__(self, 'required_variables', {'context', 'content_type'})
        
        # Validate length
        total_length = len(self.analysis_prompt) + len(self.system_instructions) + \
                      len(self.scoring_instructions) + len(self.context_instructions)
        
        if total_length > self.max_length:
            raise ValueError(f"Total prompt length {total_length} exceeds maximum {self.max_length}")
        
        # Check for required variables
        full_template = self.get_full_template()
        template_vars = self._extract_template_variables(full_template)
        
        missing_vars = self.required_variables - template_vars
        if missing_vars:
            raise ValueError(f"Template is missing required variables: {missing_vars}")
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render the template with provided context.
        
        Args:
            context: Dictionary of variables to substitute
            
        Returns:
            Rendered prompt string
        """
        # Ensure required variables are present
        missing_vars = self.required_variables - set(context.keys())
        if missing_vars:
            raise ValueError(f"Context is missing required variables: {missing_vars}")
        
        # Create template and substitute
        template = Template(self.get_full_template())
        
        try:
            return template.substitute(context)
        except KeyError as e:
            raise ValueError(f"Template variable {e} not found in context")
    
    def get_full_template(self) -> str:
        """Get the complete prompt template."""
        parts = []
        
        if self.system_instructions:
            parts.append(f"SYSTEM: {self.system_instructions}")
        
        if self.context_instructions:
            parts.append(f"CONTEXT: {self.context_instructions}")
        
        parts.append(f"ANALYSIS: {self.analysis_prompt}")
        
        if self.scoring_instructions:
            parts.append(f"SCORING: {self.scoring_instructions}")
        
        return "\n\n".join(parts)
    
    def _extract_template_variables(self, template_str: str) -> Set[str]:
        """Extract variable names from template string."""
        # Find all ${variable} patterns
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, template_str)
        
        # Also find $variable patterns (without braces)
        pattern2 = r'\$([a-zA-Z_][a-zA-Z0-9_]*)'
        matches2 = re.findall(pattern2, template_str)
        
        return set(matches + matches2)
    
    def validate_context(self, context: Dict[str, Any]) -> List[str]:
        """Validate a context dictionary against this template.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required variables
        missing_vars = self.required_variables - set(context.keys())
        if missing_vars:
            errors.append(f"Missing required variables: {missing_vars}")
        
        # Check for potentially unsafe content
        for key, value in context.items():
            if isinstance(value, str):
                if len(value) > 1000:  # Limit individual variable length
                    errors.append(f"Variable '{key}' too long ({len(value)} chars)")
                
                # Check for potential injection attempts
                suspicious_patterns = [
                    r'<script',
                    r'javascript:',
                    r'data:',
                    r'vbscript:',
                    r'onload=',
                    r'onerror=',
                ]
                
                for pattern in suspicious_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Variable '{key}' contains suspicious content")
                        break
        
        return errors
    
    @classmethod
    def default(cls) -> 'ConfigurablePromptTemplate':
        """Create a default general-purpose prompt template."""
        return cls(
            analysis_prompt="""
Analyze this ${content_type} content segment for highlight-worthy moments.

Context: ${context}

Focus on identifying:
1. Moments of exceptional skill or performance
2. High-impact plays that affect the outcome
3. Rare or unexpected events
4. Visually impressive sequences
5. Emotionally intense moments
6. Moments with strong narrative value

For each potential highlight, consider:
- Technical skill level displayed
- Impact on the overall experience
- Rarity and uniqueness
- Visual appeal and clarity
- Emotional intensity
- Story/narrative importance
- Timing and context

Rate each dimension from 0.0 to 1.0 and provide specific descriptions.
            """.strip(),
            
            system_instructions="""
You are a professional content analyst specializing in identifying highlight-worthy moments.
Analyze content objectively and provide detailed scoring across multiple dimensions.
Focus on moments that would be valuable for clips and sharing.
            """.strip(),
            
            scoring_instructions="""
Provide scores for each dimension (0.0-1.0):
- skill_execution: Technical skill and precision displayed
- game_impact: Effect on the outcome or game state
- rarity: How uncommon or special this event is
- visual_spectacle: Visual appeal and production value
- emotional_intensity: Emotional reaction potential
- narrative_value: Importance to the overall story
- timing_importance: Significance of when this occurred
- momentum_shift: Impact on momentum or flow

Include confidence level (0.0-1.0) and detailed description.
            """.strip(),
            
            context_instructions="""
Consider the provided context including:
- Previous highlights from this content
- Overall content type and setting
- Audience preferences and expectations
Be contextually aware but focus on universally appealing moments.
            """.strip()
        )
    
    @classmethod
    def gaming_focused(cls) -> 'ConfigurablePromptTemplate':
        """Create a gaming-focused prompt template."""
        return cls(
            analysis_prompt="""
Analyze this gaming content for highlight-worthy moments.

Game: ${game_name}
Context: ${context}

Look for gaming highlights such as:
1. Clutch plays and comeback moments
2. Exceptional skill displays (aim, movement, timing)
3. Rare achievements (aces, perfect games, etc.)
4. High-impact plays that change the game
5. Impressive team coordination
6. Funny or unexpected moments
7. Perfect execution of strategies

Consider gaming-specific factors:
- Skill ceiling and difficulty of the play
- Game knowledge and tactical awareness
- Mechanical precision and reaction time
- Risk vs reward of the play
- Context within the match/game

Rate based on what would impress other gamers and create shareable moments.
            """.strip(),
            
            system_instructions="""
You are a gaming content expert who understands what makes compelling gaming highlights.
Focus on plays that demonstrate skill, create excitement, or tell a story.
Consider both casual and competitive gaming contexts.
            """.strip(),
            
            scoring_instructions="""
For gaming content, weight these dimensions appropriately:
- skill_execution: Mechanical skill, aim, movement, timing
- game_impact: Effect on match outcome, objective control
- rarity: How often this type of play happens
- visual_spectacle: Clear action, good angles, exciting visuals
- emotional_intensity: Hype potential, tension, excitement
- narrative_value: Comeback story, underdog moments
- timing_importance: Clutch timing, pressure situations
- momentum_shift: Game-changing moments

Consider the game's skill ceiling and competitive context.
            """.strip()
        )
    
    @classmethod
    def sports_focused(cls) -> 'ConfigurablePromptTemplate':
        """Create a sports-focused prompt template."""
        return cls(
            analysis_prompt="""
Analyze this sports content for highlight-worthy moments.

Sport: ${sport_name}
Context: ${context}

Identify sports highlights including:
1. Exceptional athletic performances
2. Game-changing plays and moments
3. Rare achievements and records
4. Dramatic finishes and clutch performances
5. Outstanding individual skill displays
6. Strategic plays and team coordination
7. Emotional moments and celebrations

Consider sports-specific factors:
- Athletic difficulty and skill level
- Game situation and stakes
- Historical context and significance
- Visual impact and cinematography
- Crowd and commentator reactions
- Statistical significance

Focus on moments that showcase the best of human athletic achievement.
            """.strip(),
            
            system_instructions="""
You are a sports content analyst with deep understanding of athletic competition.
Identify moments that showcase peak performance, drama, and sporting excellence.
Consider both the athletic achievement and entertainment value.
            """.strip()
        )
    
    @classmethod
    def from_b2c_config(cls, b2c_config: Dict[str, Any]) -> 'ConfigurablePromptTemplate':
        """Create a B2B template from B2C game configuration.
        
        Args:
            b2c_config: Configuration dictionary from B2C system
            
        Returns:
            Configured prompt template
        """
        # Extract prompt from B2C config
        analysis_prompt = b2c_config.get('analysis_prompt', '') or \
                         b2c_config.get('prompt_template', '')
        
        game_name = b2c_config.get('game_name', 'Unknown Game')
        
        # Build comprehensive prompt incorporating B2C elements
        if not analysis_prompt:
            analysis_prompt = f"""
Analyze this {game_name} gameplay for highlight-worthy moments.

Context: ${{context}}

Focus on {game_name}-specific highlights and moments that would appeal to viewers.
            """.strip()
        
        # Add dimension-specific scoring if available
        scoring_instructions = """
Provide scores for each dimension (0.0-1.0):
- skill_execution: Technical skill and precision displayed
- game_impact: Effect on the game outcome
- rarity: How uncommon this moment is
- visual_spectacle: Visual appeal and excitement
- emotional_intensity: Emotional impact potential
- narrative_value: Story and context importance
- timing_importance: Significance of timing
- momentum_shift: Impact on game momentum

Include confidence level and detailed description.
        """.strip()
        
        # Incorporate keywords if available
        keywords = b2c_config.get('keywords', {})
        if keywords:
            keyword_examples = []
            for priority, words in keywords.items():
                if words:
                    keyword_examples.append(f"{priority}: {', '.join(words[:5])}")
            
            if keyword_examples:
                analysis_prompt += f"\n\nPay attention to these types of moments:\n" + \
                                 "\n".join(keyword_examples)
        
        return cls(
            analysis_prompt=analysis_prompt,
            system_instructions=f"""
You are analyzing {game_name} content for highlights.
Focus on moments that demonstrate skill, create excitement, or have entertainment value.
Consider the game's specific mechanics and what makes compelling content.
            """.strip(),
            scoring_instructions=scoring_instructions,
            context_instructions="""
Consider the provided context about recent highlights and stream state.
Avoid repetitive content while identifying genuinely noteworthy moments.
            """.strip()
        )