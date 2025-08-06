"""Gemini-based video scoring implementation with File API support."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from google import genai
from google.genai import types

from .dimension_framework import (
    ScoringRubric, 
    ScoringStrategy, 
    HighlightBoundary,
    create_gemini_response_schema
)

logger = logging.getLogger(__name__)


@dataclass
class VideoFile:
    """Represents an uploaded video file in Gemini."""

    uri: str
    name: str
    mime_type: str
    size_bytes: int

    @property
    def file_data(self) -> types.FileData:
        """Convert to Gemini FileData type."""
        return types.FileData(file_uri=self.uri, mime_type=self.mime_type)


class GeminiFileManager:
    """Manages video file uploads to Gemini File API with context manager support."""

    def __init__(self, client: genai.Client):
        """Initialize with Gemini client."""
        self.client = client
        self._uploaded_files: list[VideoFile] = []

    async def upload_video(self, video_path: Path) -> VideoFile:
        """Upload a video file to Gemini File API.

        Args:
            video_path: Path to the video file

        Returns:
            VideoFile object with upload details

        Raises:
            ValueError: If file doesn't exist or is too large
            RuntimeError: If upload fails

        """
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        # Check file size (2GB limit)
        size_bytes = video_path.stat().st_size
        if size_bytes > 2 * 1024 * 1024 * 1024:  # 2GB
            raise ValueError(f"Video file too large: {size_bytes} bytes (max 2GB)")

        # Determine MIME type
        mime_type = self._get_video_mime_type(video_path)

        try:
            # Upload file
            logger.info(f"Uploading video: {video_path.name} ({size_bytes} bytes)")

            # Upload using the File API
            response = await asyncio.to_thread(
                self.client.files.upload,
                file=str(video_path),
                config=types.UploadFileConfig(mime_type=mime_type),
            )

            video_file = VideoFile(
                uri=response.uri or "",
                name=response.name or "",
                mime_type=mime_type,
                size_bytes=size_bytes,
            )

            self._uploaded_files.append(video_file)
            logger.info(f"Video uploaded successfully: {video_file.uri}")

            return video_file

        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            raise RuntimeError(f"Video upload failed: {e}") from e

    async def poll_until_active(
        self, 
        video_file: VideoFile, 
        max_wait_time: int = 600,  # 10 minutes
        initial_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> None:
        """Poll file status until it becomes ACTIVE using exponential backoff.
        
        Args:
            video_file: VideoFile to check
            max_wait_time: Maximum time to wait in seconds (default: 10 minutes)
            initial_delay: Initial delay between checks in seconds
            max_delay: Maximum delay between checks in seconds
            
        Raises:
            RuntimeError: If file fails to become ACTIVE within max_wait_time
            ValueError: If file enters FAILED state
        """
        start_time = time.time()
        delay = initial_delay
        
        logger.info(f"Polling file status for {video_file.name} until ACTIVE")
        
        while time.time() - start_time < max_wait_time:
            try:
                # Get current file status
                file_info = await asyncio.to_thread(
                    self.client.files.get, 
                    name=video_file.name
                )
                
                # Check file state
                if hasattr(file_info, 'state') and file_info.state:
                    state_name = file_info.state.name if hasattr(file_info.state, 'name') else str(file_info.state)
                    logger.debug(f"File {video_file.name} state: {state_name}")
                    
                    if state_name == "ACTIVE":
                        logger.info(f"File {video_file.name} is now ACTIVE")
                        return
                    elif state_name == "FAILED":
                        raise ValueError(f"File {video_file.name} failed to process: {getattr(file_info.state, 'error_message', 'Unknown error')}")
                    elif state_name in ["PROCESSING", "STATE_UNSPECIFIED"]:
                        # Continue polling
                        logger.debug(f"File {video_file.name} still processing, waiting {delay}s...")
                    else:
                        logger.warning(f"Unknown file state: {state_name}")
                else:
                    # No state information available, assume still processing
                    logger.debug(f"No state information for {video_file.name}, assuming still processing")
                
                # Wait before next check with exponential backoff
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)  # Double delay up to max_delay
                
            except Exception as e:
                logger.error(f"Error checking file status: {e}")
                # Wait before retrying
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
        
        # Timeout reached
        elapsed = time.time() - start_time
        raise RuntimeError(f"File {video_file.name} did not become ACTIVE within {elapsed:.1f} seconds")

    async def delete_file(self, video_file: VideoFile) -> None:
        """Delete an uploaded file from Gemini.

        Args:
            video_file: VideoFile to delete

        """
        try:
            await asyncio.to_thread(self.client.files.delete, name=video_file.name)
            logger.info(f"Deleted video file: {video_file.name}")
            self._uploaded_files.remove(video_file)
        except Exception as e:
            logger.error(f"Failed to delete video file: {e}")

    async def cleanup(self) -> None:
        """Delete all uploaded files."""
        for video_file in self._uploaded_files[
            :
        ]:  # Copy list to avoid modification during iteration
            await self.delete_file(video_file)

    @staticmethod
    def _get_video_mime_type(video_path: Path) -> str:
        """Determine MIME type from file extension."""
        ext = video_path.suffix.lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".mpeg": "video/mpeg",
            ".mpg": "video/mpg",
            ".webm": "video/webm",
            ".avi": "video/x-msvideo",
            ".flv": "video/x-flv",
            ".wmv": "video/x-ms-wmv",
            ".mov": "video/quicktime",
            ".3gp": "video/3gpp",
        }
        return mime_types.get(ext, "video/mp4")


@asynccontextmanager
async def gemini_video_file(
    client: genai.Client, video_path: Path
) -> AsyncIterator[VideoFile]:
    """Context manager for handling video file uploads with automatic cleanup and status polling.

    Args:
        client: Gemini client
        video_path: Path to video file

    Yields:
        VideoFile object that is guaranteed to be in ACTIVE state

    Example:
        async with gemini_video_file(client, Path("video.mp4")) as video:
            # Use video.file_data in Gemini API calls
            response = await client.models.generate_content(...)

    """
    manager = GeminiFileManager(client)
    video_file = await manager.upload_video(video_path)

    try:
        # Poll until file is ACTIVE before yielding
        await manager.poll_until_active(video_file)
        yield video_file
    finally:
        # Cleanup: delete the uploaded file
        await manager.delete_file(video_file)


class GeminiVideoScorer(ScoringStrategy):
    """Video scoring strategy using Google Gemini 2.0 Flash."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        """Initialize Gemini scorer.

        Args:
            api_key: Gemini API key
            model_name: Model to use (default: gemini-2.0-flash)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum output tokens

        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.file_manager = GeminiFileManager(self.client)

    async def score(
        self, content: Any, rubric: ScoringRubric, context: Optional[list[Any]] = None
    ) -> dict[str, tuple[float, float]]:
        """Score video content against rubric dimensions.

        Args:
            content: Video file path (Path or str)
            rubric: Scoring rubric with dimensions
            context: Optional context (previous segments)

        Returns:
            Dictionary mapping dimension_name -> (score, confidence)

        """
        video_path = Path(content) if isinstance(content, str) else content

        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        # Use context manager for automatic cleanup
        async with gemini_video_file(self.client, video_path) as video_file:
            # Build multi-dimensional prompt
            prompt = rubric.build_multi_dimensional_prompt()

            # Create content parts
            contents = types.Content(
                parts=[
                    types.Part(file_data=video_file.file_data),
                    types.Part(text=prompt),
                ]
            )

            # Generate response
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        response_mime_type="application/json",
                    ),
                )

                # Log the raw Gemini response for debugging
                logger.info(f"Gemini response received (model={self.model_name}, temp={self.temperature}): {response.text}")

                # Parse response
                return self._parse_response(response, rubric)

            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                # Return low confidence scores on error
                return {dim.name: (0.0, 0.0) for dim in rubric.dimensions}

    async def score_with_boundaries(
        self, content: Any, rubric: ScoringRubric, context: Optional[list[Any]] = None
    ) -> tuple[dict[str, tuple[float, float]], Optional[HighlightBoundary]]:
        """Score video content and extract precise highlight boundaries using structured output.
        
        This method uses Gemini's structured output capabilities to reliably extract
        both dimensional scores and precise highlight timestamps, enabling creation
        of targeted clips instead of full segment clips.

        Args:
            content: Video file path (Path or str)
            rubric: Scoring rubric with dimensions  
            context: Optional context (previous segments)

        Returns:
            Tuple of (dimension_scores, highlight_boundary)
            - dimension_scores: Dict mapping dimension_name -> (score, confidence)
            - highlight_boundary: HighlightBoundary object if highlight detected, None otherwise

        """
        video_path = Path(content) if isinstance(content, str) else content

        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        # Use context manager for automatic cleanup
        async with gemini_video_file(self.client, video_path) as video_file:
            # Build structured prompt for precise boundary detection
            prompt = rubric.build_structured_output_prompt()
            
            # Create JSON schema for structured output
            response_schema = create_gemini_response_schema(rubric)

            # Create content parts
            contents = types.Content(
                parts=[
                    types.Part(file_data=video_file.file_data),
                    types.Part(text=prompt),
                ]
            )

            # Generate response with structured output
            try:
                logger.info(f"Attempting structured output with Gemini API (model={self.model_name})")
                
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                    ),
                )

                # Parse structured response
                try:
                    return self._parse_structured_response(response, rubric)
                except (json.JSONDecodeError, KeyError, ValueError) as parse_error:
                    # JSON parsing failed - likely truncated response, trigger fallback
                    raise RuntimeError(f"Structured response parsing failed: {parse_error}")

            except Exception as e:
                logger.error(f"Structured output failed, attempting fallback: {e}")
                
                # Fallback to basic JSON parsing
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                            response_mime_type="application/json",
                        ),
                    )
                    
                    # Use existing parsing method
                    scores = self._parse_response(response, rubric)
                    logger.warning("Using fallback parsing - no precise boundaries available")
                    return scores, None
                    
                except Exception as fallback_error:
                    logger.error(f"Both structured and fallback parsing failed: {fallback_error}")
                    # Return low confidence scores with no boundaries
                    return {dim.name: (0.0, 0.0) for dim in rubric.dimensions}, None

    def _parse_structured_response(
        self, response: Any, rubric: ScoringRubric
    ) -> tuple[dict[str, tuple[float, float]], Optional[HighlightBoundary]]:
        """Parse structured Gemini response with highlight boundaries.

        Args:
            response: Gemini API response with structured output
            rubric: Scoring rubric for validation

        Returns:
            Tuple of (dimension_scores, highlight_boundary)

        """
        try:
            # Extract JSON from response
            result_text = response.text
            result = json.loads(result_text)
            
            logger.info(f"Parsed structured Gemini response - highlight_detected: {result.get('highlight_detected')}")

            # Extract dimension scores
            dimension_scores = result.get("dimensions", {})
            scores = {}
            
            for dim in rubric.dimensions:
                if dim.name in dimension_scores:
                    dim_data = dimension_scores[dim.name]
                    score = float(dim_data.get("score", 0.0))
                    confidence = float(dim_data.get("confidence", 0.5))

                    # Normalize score based on dimension type
                    if dim.type.value == "scale_1_4":
                        score = (score - 1) / 3  # Convert 1-4 to 0-1
                    elif dim.type.value == "binary":
                        score = float(score)  # Ensure 0.0 or 1.0

                    scores[dim.name] = (score, confidence)
                else:
                    # Missing dimension - low confidence
                    scores[dim.name] = (0.0, 0.0)

            # Extract highlight boundary if detected
            highlight_boundary = None
            if result.get("highlight_detected", False) and "highlight_boundary" in result:
                boundary_data = result["highlight_boundary"]
                
                try:
                    highlight_boundary = HighlightBoundary(
                        start_timestamp=boundary_data["start_timestamp"],
                        end_timestamp=boundary_data["end_timestamp"], 
                        confidence=float(boundary_data["confidence"]),
                        reasoning=boundary_data["reasoning"]
                    )
                    
                    # Validate duration constraints  
                    if not highlight_boundary.validate_duration():
                        logger.warning(
                            f"Highlight boundary duration invalid: "
                            f"{highlight_boundary.start_timestamp} to {highlight_boundary.end_timestamp}"
                        )
                        highlight_boundary = None
                    else:
                        logger.info(
                            f"Extracted valid highlight boundary: "
                            f"{highlight_boundary.start_timestamp} to {highlight_boundary.end_timestamp}"
                        )
                        
                except (KeyError, ValueError) as e:
                    logger.error(f"Failed to parse highlight boundary: {e}")
                    highlight_boundary = None

            return scores, highlight_boundary

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse structured response: {e}")
            logger.debug(f"Raw response: {response.text}")

            # Return default low-confidence scores with no boundary
            return {dim.name: (0.0, 0.0) for dim in rubric.dimensions}, None

    def _parse_response(
        self, response: Any, rubric: ScoringRubric
    ) -> dict[str, tuple[float, float]]:
        """Parse Gemini response into dimension scores.

        Args:
            response: Gemini API response
            rubric: Scoring rubric for validation

        Returns:
            Dictionary mapping dimension_name -> (score, confidence)

        """
        try:
            # Extract JSON from response
            result_text = response.text

            # Parse JSON
            result = json.loads(result_text)
            
            # Log the parsed JSON structure
            logger.info(f"Parsed Gemini JSON response - keys: {list(result.keys()) if isinstance(result, dict) else 'not_dict'}, content: {result}")

            # Extract dimension scores
            dimension_scores = result.get("dimensions", {})

            # Convert to expected format
            scores = {}
            for dim in rubric.dimensions:
                if dim.name in dimension_scores:
                    dim_data = dimension_scores[dim.name]
                    score = float(dim_data.get("score", 0.0))
                    confidence = float(dim_data.get("confidence", 0.5))

                    # Normalize score based on dimension type
                    if dim.type.value == "scale_1_4":
                        score = (score - 1) / 3  # Convert 1-4 to 0-1
                    elif dim.type.value == "binary":
                        score = float(score)  # Ensure 0.0 or 1.0

                    scores[dim.name] = (score, confidence)
                else:
                    # Missing dimension - low confidence
                    scores[dim.name] = (0.0, 0.0)

            return scores

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {response.text}")

            # Return default low-confidence scores
            return {dim.name: (0.0, 0.0) for dim in rubric.dimensions}

    async def score_with_timestamp(
        self, content: Any, dimension: Any, timestamp: str
    ) -> tuple[float, float, str]:
        """Score a specific timestamp in the video.

        Args:
            content: Video file path
            dimension: Single dimension to evaluate
            timestamp: Timestamp in MM:SS format

        Returns:
            Tuple of (score, confidence, reasoning)

        """
        video_path = Path(content) if isinstance(content, str) else content

        async with gemini_video_file(self.client, video_path) as video_file:
            # Build single dimension prompt with timestamp
            prompt = dimension.build_evaluation_prompt(video_timestamp=timestamp)

            contents = types.Content(
                parts=[
                    types.Part(file_data=video_file.file_data),
                    types.Part(text=prompt),
                ]
            )

            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        response_mime_type="application/json",
                    ),
                )

                result = json.loads(response.text or "{}")
                score = float(result.get("score", 0.0))
                confidence = float(result.get("confidence", 0.5))
                reasoning = result.get("reasoning", "")

                # Normalize score
                if dimension.type.value == "scale_1_4":
                    score = (score - 1) / 3

                return score, confidence, reasoning

            except Exception as e:
                logger.error(f"Failed to score timestamp {timestamp}: {e}")
                return 0.0, 0.0, f"Error: {str(e)}"
