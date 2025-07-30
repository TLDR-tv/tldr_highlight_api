"""Streamlined Celery tasks for FFmpeg stream ingestion and AI highlight detection.

This module provides the core Celery tasks for the B2B highlight detection pipeline:
1. FFmpeg-based stream ingestion and chunking
2. AI-powered highlight detection using B2BStreamAgent
"""

import asyncio
import logging
import tempfile
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

import structlog
from celery import Task

from src.core.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream, StreamStatus
from src.infrastructure.async_processing.celery_app import celery_app
from src.infrastructure.media.ffmpeg_integration import (
    FFmpegProcessor, 
    FFmpegProbe,
    VideoFrameExtractor,
    TranscodeOptions,
    VideoCodec,
    AudioCodec,
    ContainerFormat
)
from src.domain.services.stream_processing_service import StreamProcessingService
from src.domain.services.b2b_stream_agent import B2BStreamAgent
from src.domain.repositories.highlight_agent_config_repository import HighlightAgentConfigRepository
from src.infrastructure.async_processing.error_handler import ErrorHandler
from src.infrastructure.async_processing.progress_tracker import ProgressTracker, ProgressEvent
from src.infrastructure.async_processing.webhook_dispatcher import WebhookDispatcher, WebhookEvent

logger = structlog.get_logger(__name__)


class StreamProcessingTask(Task):
    """Base task class for stream processing with B2B agent integration."""

    autoretry_for = (Exception,)
    max_retries = 3
    default_retry_delay = 60
    retry_backoff = True
    retry_jitter = True

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.progress_tracker = ProgressTracker()
        self.webhook_dispatcher = WebhookDispatcher()
        self.ffmpeg_processor = FFmpegProcessor(hardware_acceleration=True)
        self.frame_extractor = VideoFrameExtractor(use_hardware_acceleration=True)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with B2B agent cleanup."""
        logger.error(
            "Stream processing task failed",
            task_id=task_id,
            task_name=self.name,
            exception=str(exc),
            args=args,
            kwargs=kwargs,
        )

        if args and len(args) > 0:
            stream_id = args[0]
            try:
                # Update progress with failure
                self.progress_tracker.update_progress(
                    stream_id=stream_id,
                    progress_percentage=0,
                    status="failed",
                    event_type=ProgressEvent.ERROR,
                    details={"error": str(exc), "task": self.name},
                )

                # Send failure webhook
                asyncio.create_task(
                    self.webhook_dispatcher.dispatch_webhook(
                        stream_id=stream_id,
                        event=WebhookEvent.ERROR_OCCURRED,
                        data={
                            "error": str(exc),
                            "task": self.name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
            except Exception as e:
                logger.error("Failed to handle task failure", error=str(e))


@celery_app.task(bind=True, base=StreamProcessingTask, name="ingest_stream_with_ffmpeg")
def ingest_stream_with_ffmpeg(
    self, 
    stream_id: int, 
    chunk_duration: int = 30,
    agent_config_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ingest stream using FFmpeg and create segments for AI analysis.
    
    This task:
    1. Uses FFmpeg to connect to the stream source
    2. Creates video/audio segments for processing
    3. Extracts keyframes for visual analysis
    4. Prepares data for AI highlight detection
    
    Args:
        stream_id: Stream ID to process
        chunk_duration: Duration of each chunk in seconds
        agent_config_id: Optional B2B agent configuration ID
        
    Returns:
        Dict with ingestion results and segment information
    """
    logger.info("Starting FFmpeg stream ingestion", stream_id=stream_id, chunk_duration=chunk_duration)
    
    try:
        with get_db_session() as db:
            # Get stream
            stream = db.query(Stream).filter(Stream.id == stream_id).first()
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Update status to processing
            stream.status = StreamStatus.PROCESSING
            db.commit()
            
            # Update progress
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=10,
                status="processing",
                event_type=ProgressEvent.STARTED,
                details={"task": "ffmpeg_stream_ingestion", "chunk_duration": chunk_duration},
            )
            
            # Probe stream first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                media_info = loop.run_until_complete(
                    FFmpegProbe.probe_stream(stream.source_url, timeout=15)
                )
                
                logger.info("Stream probed successfully", 
                           format=media_info.format_name,
                           video_streams=len(media_info.video_streams),
                           audio_streams=len(media_info.audio_streams))
                
            except Exception as e:
                logger.error(f"Failed to probe stream: {e}")
                raise ValueError(f"Cannot access stream: {e}")
            
            # Create temporary working directory
            temp_dir = tempfile.mkdtemp(prefix=f"stream_{stream_id}_")
            
            try:
                # Start stream ingestion with chunking
                segments = loop.run_until_complete(
                    self._ingest_stream_segments(
                        stream.source_url, 
                        temp_dir, 
                        chunk_duration,
                        media_info
                    )
                )
                
                # Extract keyframes for visual analysis
                keyframes = loop.run_until_complete(
                    self._extract_keyframes_for_analysis(
                        stream.source_url,
                        temp_dir,
                        max_duration=300  # 5 minutes max for initial analysis
                    )
                )
                
                # Update progress
                self.progress_tracker.update_progress(
                    stream_id=stream_id,
                    progress_percentage=40,
                    status="processing", 
                    event_type=ProgressEvent.PROGRESS_UPDATE,
                    details={
                        "task": "stream_ingestion_complete",
                        "segments_created": len(segments),
                        "keyframes_extracted": len(keyframes),
                        "temp_dir": temp_dir
                    },
                )
                
                # Trigger AI highlight detection task
                detect_highlights_with_ai.delay(
                    stream_id=stream_id,
                    ingestion_data={
                        "segments": segments,
                        "keyframes": keyframes,
                        "temp_dir": temp_dir,
                        "media_info": {
                            "format": media_info.format_name,
                            "duration": media_info.duration,
                            "video_streams": len(media_info.video_streams),
                            "audio_streams": len(media_info.audio_streams)
                        }
                    },
                    agent_config_id=agent_config_id
                )
                
                return {
                    "stream_id": stream_id,
                    "status": "ingestion_complete",
                    "segments_created": len(segments),
                    "keyframes_extracted": len(keyframes),
                    "media_info": {
                        "format": media_info.format_name,
                        "duration": media_info.duration,
                        "video_streams": len(media_info.video_streams),
                        "audio_streams": len(media_info.audio_streams)
                    },
                    "temp_dir": temp_dir,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                # Clean up temp directory on failure
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise e
            finally:
                loop.close()
                
    except Exception as exc:
        logger.error("FFmpeg stream ingestion failed", stream_id=stream_id, error=str(exc))
        
        # Update stream status to failed
        try:
            with get_db_session() as db:
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if stream:
                    stream.status = StreamStatus.FAILED
                    stream.error_message = str(exc)
                    db.commit()
        except Exception as db_exc:
            logger.error("Failed to update stream status", error=str(db_exc))
        
        raise exc

    async def _ingest_stream_segments(
        self, 
        stream_url: str, 
        temp_dir: str, 
        chunk_duration: int,
        media_info
    ) -> List[Dict[str, Any]]:
        """Ingest stream and create segments using FFmpeg."""
        segments = []
        
        try:
            # Create segments directory
            segments_dir = os.path.join(temp_dir, "segments")
            os.makedirs(segments_dir, exist_ok=True)
            
            # Determine optimal transcoding options based on stream
            video_info = media_info.video_streams[0] if media_info.video_streams else None
            audio_info = media_info.audio_streams[0] if media_info.audio_streams else None
            
            transcode_options = TranscodeOptions(
                video_codec=VideoCodec.H264,
                audio_codec=AudioCodec.AAC,
                video_bitrate=2000,  # 2Mbps
                audio_bitrate=128,   # 128kbps
                quality="fast",
                container=ContainerFormat.MP4
            )
            
            # For now, create one segment for the entire stream
            # In production, you'd implement proper chunking
            segment_path = os.path.join(segments_dir, f"segment_001.mp4")
            
            success = await self.ffmpeg_processor.transcode_stream(
                input_source=stream_url,
                output_path=segment_path,
                options=transcode_options
            )
            
            if success and os.path.exists(segment_path):
                segment_info = {
                    "id": "segment_001",
                    "path": segment_path,
                    "start_time": 0,
                    "duration": chunk_duration,
                    "video_info": {
                        "width": video_info.width if video_info else 0,
                        "height": video_info.height if video_info else 0,
                        "fps": video_info.fps if video_info else 0,
                        "codec": video_info.codec if video_info else "unknown"
                    },
                    "audio_info": {
                        "sample_rate": audio_info.sample_rate if audio_info else 0,
                        "channels": audio_info.channels if audio_info else 0,
                        "codec": audio_info.codec if audio_info else "unknown"
                    }
                }
                segments.append(segment_info)
                
                logger.info(f"Created segment: {segment_path}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to create stream segments: {e}")
            raise

    async def _extract_keyframes_for_analysis(
        self, 
        stream_url: str, 
        temp_dir: str,
        max_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Extract keyframes for visual analysis."""
        try:
            keyframes_dir = os.path.join(temp_dir, "keyframes")
            os.makedirs(keyframes_dir, exist_ok=True)
            
            # Extract keyframes
            keyframe_files = await self.frame_extractor.extract_keyframes(
                stream_url=stream_url,
                output_dir=keyframes_dir,
                duration=max_duration
            )
            
            keyframes = []
            for frame_path, timestamp in keyframe_files:
                keyframe_info = {
                    "path": frame_path,
                    "timestamp": timestamp,
                    "filename": os.path.basename(frame_path)
                }
                keyframes.append(keyframe_info)
            
            logger.info(f"Extracted {len(keyframes)} keyframes for analysis")
            return keyframes
            
        except Exception as e:
            logger.error(f"Failed to extract keyframes: {e}")
            # Don't fail the entire task if keyframe extraction fails
            return []


@celery_app.task(bind=True, base=StreamProcessingTask, name="detect_highlights_with_ai")
def detect_highlights_with_ai(
    self,
    stream_id: int,
    ingestion_data: Dict[str, Any],
    agent_config_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect highlights using B2B AI agent with consumer-specific configuration.
    
    This task:
    1. Initializes the B2BStreamAgent with appropriate configuration
    2. Processes each segment using the agent's AI analysis
    3. Creates highlights based on agent's recommendations
    4. Stores results and triggers completion notifications
    
    Args:
        stream_id: Stream ID to process
        ingestion_data: Results from FFmpeg ingestion task
        agent_config_id: Optional B2B agent configuration ID
        
    Returns:
        Dict with highlight detection results
    """
    logger.info("Starting AI highlight detection", stream_id=stream_id, agent_config_id=agent_config_id)
    
    try:
        with get_db_session() as db:
            # Get stream
            stream = db.query(Stream).filter(Stream.id == stream_id).first()
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Update progress
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=50,
                status="processing",
                event_type=ProgressEvent.PROGRESS_UPDATE,
                details={"task": "ai_highlight_detection_started"},
            )
            
            # Initialize B2B agent (this would be injected in real implementation)
            # For now, simulate the agent creation
            from src.domain.entities.highlight_agent_config import HighlightAgentConfig
            
            # Use default config if none specified
            if agent_config_id:
                # In real implementation, fetch from repository
                agent_config = HighlightAgentConfig.create_default_gaming_config(
                    organization_id=1,  # Would get from stream/user
                    user_id=stream.user_id
                )
            else:
                agent_config = HighlightAgentConfig.create_default_gaming_config(
                    organization_id=1,
                    user_id=stream.user_id
                )
            
            # Create B2B agent
            b2b_agent = B2BStreamAgent(
                stream=stream,
                agent_config=agent_config,
                content_analyzer=None  # Would inject real analyzer
            )
            
            # Start the agent
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(b2b_agent.start())
                
                # Process each segment
                all_highlights = []
                segments = ingestion_data.get("segments", [])
                keyframes = ingestion_data.get("keyframes", [])
                
                for i, segment in enumerate(segments):
                    # Create segment data for analysis
                    segment_data = {
                        "id": segment["id"],
                        "path": segment["path"],
                        "start_time": segment["start_time"],
                        "duration": segment["duration"],
                        "video_info": segment["video_info"],
                        "audio_info": segment["audio_info"],
                        "keyframes": [kf for kf in keyframes 
                                     if segment["start_time"] <= kf["timestamp"] < segment["start_time"] + segment["duration"]]
                    }
                    
                    # Process segment with B2B agent
                    candidates = loop.run_until_complete(
                        b2b_agent.analyze_content_segment(segment_data)
                    )
                    
                    # Create highlights from candidates
                    for candidate in candidates:
                        should_create = loop.run_until_complete(
                            b2b_agent.should_create_highlight(candidate)
                        )
                        
                        if should_create:
                            highlight = loop.run_until_complete(
                                b2b_agent.create_highlight(candidate)
                            )
                            if highlight:
                                all_highlights.append(highlight)
                    
                    # Update progress
                    progress = 50 + (i + 1) * 30 / len(segments)
                    self.progress_tracker.update_progress(
                        stream_id=stream_id,
                        progress_percentage=progress,
                        status="processing",
                        event_type=ProgressEvent.PROGRESS_UPDATE,
                        details={
                            "task": "segment_analysis_complete",
                            "segment": i + 1,
                            "total_segments": len(segments),
                            "highlights_so_far": len(all_highlights)
                        },
                    )
                
                # Stop the agent
                loop.run_until_complete(b2b_agent.stop())
                
                # Update stream status to completed
                stream.status = StreamStatus.COMPLETED
                stream.completed_at = datetime.now(timezone.utc)
                db.commit()
                
                # Clean up temporary files
                temp_dir = ingestion_data.get("temp_dir")
                if temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                
                # Final progress update
                self.progress_tracker.update_progress(
                    stream_id=stream_id,
                    progress_percentage=100,
                    status="completed",
                    event_type=ProgressEvent.COMPLETED,
                    details={
                        "task": "ai_highlight_detection_complete",
                        "total_highlights": len(all_highlights),
                        "agent_metrics": b2b_agent.get_performance_metrics()
                    },
                )
                
                # Send completion webhook
                asyncio.create_task(
                    self.webhook_dispatcher.dispatch_webhook(
                        stream_id=stream_id,
                        event=WebhookEvent.PROCESSING_COMPLETE,
                        data={
                            "stream_id": stream_id,
                            "status": "completed",
                            "highlights_count": len(all_highlights),
                            "agent_metrics": b2b_agent.get_performance_metrics(),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
                
                return {
                    "stream_id": stream_id,
                    "status": "completed",
                    "highlights_created": len(all_highlights),
                    "agent_config_used": agent_config.name,
                    "agent_metrics": b2b_agent.get_performance_metrics(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            finally:
                loop.close()
                
    except Exception as exc:
        logger.error("AI highlight detection failed", stream_id=stream_id, error=str(exc))
        
        # Update stream status to failed
        try:
            with get_db_session() as db:
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if stream:
                    stream.status = StreamStatus.FAILED
                    stream.error_message = str(exc)
                    db.commit()
        except Exception as db_exc:
            logger.error("Failed to update stream status", error=str(db_exc))
        
        raise exc


@celery_app.task(bind=True, name="cleanup_stream_resources")
def cleanup_stream_resources(self, max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up temporary files and resources from stream processing.
    
    Args:
        max_age_hours: Maximum age in hours for resources to keep
        
    Returns:
        Dict with cleanup results
    """
    logger.info("Starting stream resource cleanup", max_age_hours=max_age_hours)
    
    try:
        import tempfile
        import shutil
        from datetime import timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        temp_files_cleaned = 0
        
        # Clean up temporary directories
        temp_root = tempfile.gettempdir()
        for item in os.listdir(temp_root):
            if item.startswith("stream_"):
                item_path = os.path.join(temp_root, item)
                if os.path.isdir(item_path):
                    try:
                        # Check if directory is old enough
                        mod_time = datetime.fromtimestamp(os.path.getmtime(item_path), tz=timezone.utc)
                        if mod_time < cutoff_time:
                            shutil.rmtree(item_path)
                            temp_files_cleaned += 1
                            logger.debug(f"Cleaned up temp directory: {item_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {item_path}: {e}")
        
        # Clean up completed streams from database
        with get_db_session() as db:
            old_streams = (
                db.query(Stream)
                .filter(
                    Stream.status == StreamStatus.COMPLETED,
                    Stream.completed_at < cutoff_time,
                )
                .count()
            )
            
            # Archive rather than delete (or implement retention policy)
            # For now, just log the count
            logger.info(f"Found {old_streams} completed streams older than {max_age_hours} hours")
        
        cleanup_result = {
            "temp_directories_cleaned": temp_files_cleaned,
            "old_streams_found": old_streams,
            "cutoff_time": cutoff_time.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info("Stream resource cleanup completed", **cleanup_result)
        return cleanup_result
        
    except Exception as exc:
        logger.error("Failed to cleanup stream resources", error=str(exc))
        raise exc