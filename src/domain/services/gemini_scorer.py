"""Gemini-based video scoring implementation with File API support."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from google import genai
from google.genai import types

from .dimension_framework import ScoringRubric, ScoringStrategy

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
        return types.FileData(
            file_uri=self.uri,
            mime_type=self.mime_type
        )


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
                config=types.FileConfig(mime_type=mime_type)
            )
            
            video_file = VideoFile(
                uri=response.uri,
                name=response.name,
                mime_type=mime_type,
                size_bytes=size_bytes
            )
            
            self._uploaded_files.append(video_file)
            logger.info(f"Video uploaded successfully: {video_file.uri}")
            
            return video_file
            
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            raise RuntimeError(f"Video upload failed: {e}") from e
    
    async def delete_file(self, video_file: VideoFile) -> None:
        """Delete an uploaded file from Gemini.
        
        Args:
            video_file: VideoFile to delete
        """
        try:
            await asyncio.to_thread(
                self.client.files.delete,
                name=video_file.name
            )
            logger.info(f"Deleted video file: {video_file.name}")
            self._uploaded_files.remove(video_file)
        except Exception as e:
            logger.error(f"Failed to delete video file: {e}")
    
    async def cleanup(self) -> None:
        """Delete all uploaded files."""
        for video_file in self._uploaded_files[:]:  # Copy list to avoid modification during iteration
            await self.delete_file(video_file)
    
    @staticmethod
    def _get_video_mime_type(video_path: Path) -> str:
        """Determine MIME type from file extension."""
        ext = video_path.suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.mpeg': 'video/mpeg',
            '.mpg': 'video/mpg',
            '.webm': 'video/webm',
            '.avi': 'video/x-msvideo',
            '.flv': 'video/x-flv',
            '.wmv': 'video/x-ms-wmv',
            '.mov': 'video/quicktime',
            '.3gp': 'video/3gpp'
        }
        return mime_types.get(ext, 'video/mp4')


@asynccontextmanager
async def gemini_video_file(client: genai.Client, video_path: Path) -> AsyncIterator[VideoFile]:
    """Context manager for handling video file uploads with automatic cleanup.
    
    Args:
        client: Gemini client
        video_path: Path to video file
        
    Yields:
        VideoFile object
        
    Example:
        async with gemini_video_file(client, Path("video.mp4")) as video:
            # Use video.file_data in Gemini API calls
            response = await client.models.generate_content(...)
    """
    manager = GeminiFileManager(client)
    video_file = await manager.upload_video(video_path)
    
    try:
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
        max_tokens: int = 2048
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
        self,
        content: Any,
        rubric: ScoringRubric,
        context: Optional[list[Any]] = None
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
                    types.Part(
                        file_data=video_file.file_data
                    ),
                    types.Part(
                        text=prompt
                    )
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
                        response_mime_type="application/json"
                    )
                )
                
                # Parse response
                return self._parse_response(response, rubric)
                
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                # Return low confidence scores on error
                return {
                    dim.name: (0.0, 0.0) 
                    for dim in rubric.dimensions
                }
    
    def _parse_response(
        self, 
        response: Any, 
        rubric: ScoringRubric
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
            return {
                dim.name: (0.0, 0.0) 
                for dim in rubric.dimensions
            }
    
    async def score_with_timestamp(
        self,
        content: Any,
        dimension: Any,
        timestamp: str
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
                    types.Part(text=prompt)
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
                        response_mime_type="application/json"
                    )
                )
                
                result = json.loads(response.text)
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