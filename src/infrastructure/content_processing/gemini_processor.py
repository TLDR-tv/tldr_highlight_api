"""Gemini AI processing infrastructure implementation.

This module provides AI-powered content analysis using Google's Gemini API
as an infrastructure component using the new google-genai library.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google GenAI library not available. Using mock responses.")


@dataclass
class GeminiProcessorConfig:
    """Configuration for Gemini processor."""

    api_key: Optional[str] = None
    model_name: str = "gemini-2.0-flash-001"  # Updated to latest model
    temperature: float = 0.7
    max_tokens: int = 1000
    enable_vision: bool = True
    enable_multimodal: bool = True
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    use_vertex_ai: bool = False
    project_id: Optional[str] = None
    location: str = "us-central1"


@dataclass
class GeminiAnalysisRequest:
    """Request for Gemini analysis."""

    content_type: str  # "image", "text", "multimodal"
    content: Union[bytes, str, Dict[str, Any]]
    prompt: str
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GeminiAnalysisResult:
    """Result from Gemini analysis."""

    request_id: str
    content_type: str
    analysis_text: str
    highlight_score: float
    detected_elements: List[str]
    suggested_tags: List[str]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class GeminiProcessor:
    """Infrastructure component for AI-powered content analysis.

    Handles integration with Google's Gemini API for advanced
    content understanding and highlight detection.
    """

    def __init__(self, config: GeminiProcessorConfig):
        """Initialize Gemini processor.

        Args:
            config: Gemini processor configuration
        """
        self.config = config
        self._request_count = 0
        self._total_processing_time = 0.0
        self._client: Optional[genai.Client] = None
        
        # Initialize Gemini client if available
        if GEMINI_AVAILABLE:
            try:
                if config.use_vertex_ai:
                    # Initialize with Vertex AI
                    if config.project_id:
                        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", config.project_id)
                        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", config.location)
                        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
                    
                    self._client = genai.Client(
                        vertexai=True,
                        project=config.project_id,
                        location=config.location
                    )
                    logger.info(f"Initialized Gemini processor with Vertex AI: {config.model_name}")
                    
                elif config.api_key:
                    # Initialize with API key
                    self._client = genai.Client(api_key=config.api_key)
                    logger.info(f"Initialized Gemini processor with API key: {config.model_name}")
                    
                else:
                    # Try to get API key from environment
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if api_key:
                        self._client = genai.Client(api_key=api_key)
                        logger.info(f"Initialized Gemini processor with env API key: {config.model_name}")
                    else:
                        logger.warning("No API key or Vertex AI configuration provided")
                        self._client = None
                        
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self._client = None
        else:
            logger.info("Google GenAI library not available, using mock responses")

    async def analyze_image(
        self, image_data: bytes, prompt: Optional[str] = None
    ) -> GeminiAnalysisResult:
        """Analyze an image using Gemini vision capabilities.

        Args:
            image_data: Raw image data
            prompt: Optional analysis prompt

        Returns:
            Analysis result
        """
        if not self.config.enable_vision:
            raise ValueError("Vision analysis is not enabled")

        default_prompt = (
            "Analyze this image for exciting or noteworthy moments. "
            "Describe what's happening and rate how likely this is "
            "to be a highlight moment on a scale of 0 to 1."
        )

        request = GeminiAnalysisRequest(
            content_type="image", content=image_data, prompt=prompt or default_prompt
        )

        return await self._process_request(request)

    async def analyze_text(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> GeminiAnalysisResult:
        """Analyze text content using Gemini.

        Args:
            text: Text to analyze
            context: Optional context information

        Returns:
            Analysis result
        """
        prompt = (
            "Analyze this text for sentiment, excitement level, and "
            "whether it indicates a highlight moment. Consider the context "
            "of live streaming and viewer reactions."
        )

        request = GeminiAnalysisRequest(
            content_type="text", content=text, prompt=prompt, context=context
        )

        return await self._process_request(request)

    async def analyze_multimodal(
        self,
        video_frame: bytes,
        audio_transcript: str,
        chat_messages: List[str],
        timestamp: float,
    ) -> GeminiAnalysisResult:
        """Analyze multiple content types together.

        Args:
            video_frame: Video frame data
            audio_transcript: Audio transcription
            chat_messages: Recent chat messages
            timestamp: Content timestamp

        Returns:
            Analysis result
        """
        if not self.config.enable_multimodal:
            raise ValueError("Multimodal analysis is not enabled")

        content = {
            "video_frame": video_frame,
            "audio_transcript": audio_transcript,
            "chat_messages": chat_messages,
            "timestamp": timestamp,
        }

        prompt = (
            "Analyze this multimodal content from a live stream. "
            "Consider the visual content, audio commentary, and viewer chat reactions. "
            "Determine if this represents a highlight moment and explain why. "
            "Rate the highlight potential from 0 to 1."
        )

        request = GeminiAnalysisRequest(
            content_type="multimodal", content=content, prompt=prompt
        )

        return await self._process_request(request)

    async def _process_request(
        self, request: GeminiAnalysisRequest
    ) -> GeminiAnalysisResult:
        """Process a Gemini analysis request.

        Args:
            request: Analysis request

        Returns:
            Analysis result
        """
        start_time = asyncio.get_event_loop().time()
        self._request_count += 1
        request_id = f"gemini_{self._request_count}"

        # Use real Gemini API if client is available
        if self._client and GEMINI_AVAILABLE:
            try:
                return await self._process_with_api(request, request_id, start_time)
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}")
                # Fall back to mock response
                return await self._process_mock_request(request, request_id, start_time)
        else:
            # Use mock response
            return await self._process_mock_request(request, request_id, start_time)

    async def _process_with_api(
        self, request: GeminiAnalysisRequest, request_id: str, start_time: float
    ) -> GeminiAnalysisResult:
        """Process request using real Gemini API."""
        
        # Prepare content based on request type
        contents = []
        
        if request.content_type == "image":
            # Add image content
            image_part = types.Part.from_bytes(
                data=request.content, 
                mime_type="image/jpeg"  # Assume JPEG, could be made configurable
            )
            contents = [request.prompt, image_part]
            
        elif request.content_type == "text":
            contents = [f"{request.prompt}\n\nText to analyze: {request.content}"]
            
        elif request.content_type == "multimodal":
            # Handle multimodal content
            content_dict = request.content
            prompt_parts = [request.prompt]
            
            # Add video frame if available
            if "video_frame" in content_dict:
                image_part = types.Part.from_bytes(
                    data=content_dict["video_frame"],
                    mime_type="image/jpeg"
                )
                prompt_parts.append(image_part)
            
            # Add audio transcript and chat
            if "audio_transcript" in content_dict:
                prompt_parts.append(f"Audio transcript: {content_dict['audio_transcript']}")
            
            if "chat_messages" in content_dict:
                chat_text = "\n".join(content_dict["chat_messages"])
                prompt_parts.append(f"Chat messages: {chat_text}")
                
            contents = prompt_parts
        
        # Configure generation
        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        
        # Make API call
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=config
            )
        )
        
        # Parse response
        analysis_text = response.text
        
        # Extract highlight score from response (simplified)
        highlight_score = self._extract_highlight_score(analysis_text)
        
        # Extract detected elements and tags (simplified)
        detected_elements = self._extract_detected_elements(analysis_text)
        suggested_tags = self._extract_suggested_tags(analysis_text)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        self._total_processing_time += processing_time

        return GeminiAnalysisResult(
            request_id=request_id,
            content_type=request.content_type,
            analysis_text=analysis_text,
            highlight_score=highlight_score,
            detected_elements=detected_elements,
            suggested_tags=suggested_tags,
            confidence=0.85,  # Could be extracted from response
            processing_time=processing_time,
            metadata={
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "real_api": True,
            },
        )

    async def _process_mock_request(
        self, request: GeminiAnalysisRequest, request_id: str, start_time: float
    ) -> GeminiAnalysisResult:
        """Process request using mock responses."""
        
        await asyncio.sleep(0.5)  # Simulate API latency

        # Mock response based on content type
        if request.content_type == "image":
            analysis_text = (
                "The image shows an intense gaming moment with high action. "
                "Players are engaged in a critical play that could determine "
                "the outcome of the match."
            )
            highlight_score = 0.85
            detected_elements = ["player_action", "crowd_reaction", "score_change"]
            suggested_tags = ["clutch", "gameplay", "intense"]

        elif request.content_type == "text":
            analysis_text = (
                "The text analysis reveals high excitement and positive sentiment. "
                "Multiple exclamation marks and enthusiasm indicators suggest "
                "this is a significant moment."
            )
            highlight_score = 0.75
            detected_elements = ["excitement", "positive_sentiment"]
            suggested_tags = ["hype", "reaction"]

        else:  # multimodal
            analysis_text = (
                "Multimodal analysis shows strong correlation between visual action, "
                "excited commentary, and spike in chat activity. This appears to be "
                "a significant highlight moment with high viewer engagement."
            )
            highlight_score = 0.92
            detected_elements = ["visual_climax", "audio_excitement", "chat_spike"]
            suggested_tags = ["highlight", "epic_moment", "viewer_favorite"]

        processing_time = asyncio.get_event_loop().time() - start_time
        self._total_processing_time += processing_time

        return GeminiAnalysisResult(
            request_id=request_id,
            content_type=request.content_type,
            analysis_text=analysis_text,
            highlight_score=highlight_score,
            detected_elements=detected_elements,
            suggested_tags=suggested_tags,
            confidence=0.85,
            processing_time=processing_time,
            metadata={
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "real_api": False,
            },
        )

    def _extract_highlight_score(self, text: str) -> float:
        """Extract highlight score from analysis text."""
        # Simple pattern matching for scores
        import re
        
        # Look for patterns like "score: 0.85" or "8.5/10" or "85%"
        patterns = [
            r"score[:\s]+([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)/10",
            r"([0-9]+)%",
            r"highlight.*?([0-9]*\.?[0-9]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-1 range
                    if score > 1:
                        score = score / 100 if score <= 100 else score / 10
                    return min(max(score, 0.0), 1.0)
                except ValueError:
                    continue
        
        # Default fallback score
        return 0.5

    def _extract_detected_elements(self, text: str) -> List[str]:
        """Extract detected elements from analysis text."""
        elements = []
        
        # Look for common highlight indicators
        indicators = {
            "action": ["action", "movement", "play", "shot", "goal"],
            "excitement": ["exciting", "intense", "dramatic", "thrilling"],
            "crowd_reaction": ["crowd", "audience", "cheer", "reaction"],
            "score_change": ["score", "point", "lead", "win"],
            "visual_climax": ["visual", "spectacular", "amazing", "incredible"],
            "audio_excitement": ["loud", "shouting", "commentary", "announcer"],
        }
        
        text_lower = text.lower()
        for element, keywords in indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                elements.append(element)
        
        return elements[:5]  # Limit to top 5

    def _extract_suggested_tags(self, text: str) -> List[str]:
        """Extract suggested tags from analysis text."""
        tags = []
        
        # Common gaming/streaming tags
        tag_keywords = {
            "clutch": ["clutch", "critical", "crucial"],
            "epic": ["epic", "amazing", "incredible", "spectacular"],
            "hype": ["hype", "exciting", "thrilling"],
            "gameplay": ["gameplay", "play", "game"],
            "reaction": ["reaction", "response", "crowd"],
            "highlight": ["highlight", "moment", "peak"],
        }
        
        text_lower = text.lower()
        for tag, keywords in tag_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags[:3]  # Limit to top 3

    async def batch_analyze(
        self, requests: List[GeminiAnalysisRequest]
    ) -> List[GeminiAnalysisResult]:
        """Process multiple analysis requests in batch.

        Args:
            requests: List of analysis requests

        Returns:
            List of analysis results
        """
        # Process requests concurrently with rate limiting
        tasks = []
        for i, request in enumerate(requests):
            # Add delay to avoid rate limits
            delay = i * 0.1
            task = self._delayed_process(request, delay)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    async def _delayed_process(
        self, request: GeminiAnalysisRequest, delay: float
    ) -> GeminiAnalysisResult:
        """Process request with delay.

        Args:
            request: Analysis request
            delay: Delay before processing

        Returns:
            Analysis result
        """
        await asyncio.sleep(delay)
        return await self._process_request(request)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get Gemini processing statistics.

        Returns:
            Dictionary of processing statistics
        """
        avg_time = (
            self._total_processing_time / self._request_count
            if self._request_count > 0
            else 0
        )

        return {
            "requests_processed": self._request_count,
            "average_processing_time": avg_time,
            "total_processing_time": self._total_processing_time,
            "config": {
                "model": self.config.model_name,
                "vision_enabled": self.config.enable_vision,
                "multimodal_enabled": self.config.enable_multimodal,
            },
        }

    async def cleanup(self) -> None:
        """Clean up Gemini processor resources."""
        self._request_count = 0
        self._total_processing_time = 0.0
        logger.info("Gemini processor cleanup completed")
