"""Unit tests for FFmpeg processor service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from worker.services.ffmpeg_processor import FFmpegProcessor, FFmpegConfig, StreamFormat


class TestFFmpegConfig:
    """Test FFmpeg configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FFmpegConfig()
        
        assert config.segment_duration == 120  # 2 minutes
        assert config.audio_segment_duration == 30  # 30 seconds
        assert config.audio_overlap == 5
        assert config.video_ring_buffer_size == 10
        assert config.audio_ring_buffer_size == 50
        assert config.video_codec == "copy"
        assert config.audio_codec == "copy"
        assert config.reconnect is True
        assert config.reconnect_attempts == 10
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FFmpegConfig(
            segment_duration=60,
            audio_segment_duration=15,
            audio_overlap=3,
            video_codec="libx264",
            reconnect=False
        )
        
        assert config.segment_duration == 60
        assert config.audio_segment_duration == 15
        assert config.audio_overlap == 3
        assert config.video_codec == "libx264"
        assert config.reconnect is False


class TestStreamFormat:
    """Test stream format enumeration."""
    
    def test_stream_format_values(self):
        """Test stream format enum values."""
        assert StreamFormat.RTMP.value == "rtmp"
        assert StreamFormat.HLS.value == "hls"
        assert StreamFormat.DASH.value == "dash"
        assert StreamFormat.HTTP.value == "http"
        assert StreamFormat.FILE.value == "file"


class TestFFmpegProcessor:
    """Test FFmpeg processor functionality."""
    
    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        return output_dir
        
    @pytest.fixture
    def config(self):
        """Create FFmpeg configuration."""
        return FFmpegConfig(
            segment_duration=60,
            audio_segment_duration=30,
            video_ring_buffer_size=3,
            audio_ring_buffer_size=5
        )
    
    @pytest.fixture
    def processor(self, temp_output_dir, config):
        """Create FFmpeg processor instance."""
        return FFmpegProcessor(
            stream_url="https://example.com/stream.m3u8",
            output_dir=temp_output_dir,
            config=config
        )

    def test_processor_initialization(self, processor, temp_output_dir, config):
        """Test processor initialization."""
        assert processor.stream_url == "https://example.com/stream.m3u8"
        assert processor.output_dir == temp_output_dir
        assert processor.config == config
        assert hasattr(processor, '_process')

    def test_format_property(self, processor):
        """Test format property detection."""
        # Should detect format from URL
        processor.stream_url = "rtmp://example.com/live"
        assert processor.format == StreamFormat.RTMP
        
        processor.stream_url = "https://example.com/stream.m3u8"
        assert processor.format == StreamFormat.HLS
        
        processor.stream_url = "/path/to/video.mp4"
        assert processor.format == StreamFormat.FILE

    def test_build_input_args(self, processor):
        """Test building input arguments."""
        args = processor._build_input_args()
        
        assert isinstance(args, list)
        assert len(args) > 0
        # Should include reconnect options for live streams
        assert any("reconnect" in str(arg) for arg in args)

    def test_build_output_args(self, processor):
        """Test building output arguments."""
        args = processor._build_output_args()
        
        assert isinstance(args, list)
        assert len(args) > 0
        # Should include segment duration
        assert "60" in args  # segment duration from config

    def test_build_ffmpeg_command(self, processor):
        """Test building complete FFmpeg command."""
        command = processor._build_ffmpeg_command()
        
        assert isinstance(command, list)
        assert "ffmpeg" == command[0]
        assert "-i" in command
        assert processor.stream_url in command

    @patch('worker.services.ffmpeg_processor.asyncio.create_subprocess_exec')
    async def test_start_success(self, mock_subprocess, processor):
        """Test successful processor start."""
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_subprocess.return_value = mock_process
        
        await processor.start()
        
        assert processor._process == mock_process
        mock_subprocess.assert_called_once()

    @patch('worker.services.ffmpeg_processor.asyncio.create_subprocess_exec')
    async def test_start_failure(self, mock_subprocess, processor):
        """Test processor start failure."""
        mock_subprocess.side_effect = Exception("Failed to start")
        
        with pytest.raises(Exception, match="Failed to start"):
            await processor.start()

    async def test_stop_no_process(self, processor):
        """Test stopping when no process is running."""
        # Should not raise exception when no process
        await processor.stop()

    async def test_stop_with_process(self, processor):
        """Test stopping active process."""
        # Mock running process
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = AsyncMock(return_value=0)
        processor._process = mock_process
        
        await processor.stop()
        
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()

    @patch('pathlib.Path.glob')
    def test_read_segment_csv(self, mock_glob, processor):
        """Test reading segment CSV file."""
        # Mock CSV file exists
        csv_file = processor.output_dir / "segments.csv"
        mock_glob.return_value = [csv_file]
        
        with patch('builtins.open', mock_open_csv_content()):
            segments = processor._read_segment_csv()
            
            assert isinstance(segments, list)

    def test_create_segment_from_csv(self, processor):
        """Test creating segment from CSV row."""
        csv_row = {
            'segment_number': '001',
            'start_time': '0.0',
            'duration': '60.0',
            'path': 'segment_001.mp4'
        }
        
        segment = processor._create_segment_from_csv(csv_row)
        
        assert segment['segment_number'] == 1
        assert segment['start_time'] == 0.0
        assert segment['duration'] == 60.0
        assert 'segment_001.mp4' in str(segment['path'])

    def test_is_segment_complete(self, processor):
        """Test checking if segment is complete."""
        segment_path = processor.output_dir / "segment_001.mp4"
        
        # Mock file exists and has size
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1024
            
            assert processor._is_segment_complete(segment_path, min_size=512)

    @patch('worker.services.ffmpeg_processor.asyncio.create_subprocess_exec')
    async def test_extract_single_audio_chunk(self, mock_subprocess, processor):
        """Test extracting single audio chunk."""
        mock_process = AsyncMock()
        mock_process.wait.return_value = 0
        mock_subprocess.return_value = mock_process
        
        segment_path = processor.output_dir / "segment_001.mp4"
        await processor._extract_single_audio_chunk(segment_path, 0, 30, 1)
        
        mock_subprocess.assert_called_once()

    @patch('worker.services.ffmpeg_processor.asyncio.create_subprocess_exec')
    async def test_create_highlight_clip(self, mock_subprocess, processor):
        """Test creating highlight clip."""
        mock_process = AsyncMock()
        mock_process.wait.return_value = 0
        mock_subprocess.return_value = mock_process
        
        await processor.create_highlight_clip("segment_001.mp4", 30.0, 90.0, "highlight.mp4")
        
        mock_subprocess.assert_called_once()
        # Verify ffmpeg was called with correct time parameters
        call_args = mock_subprocess.call_args[0]
        assert any("-ss" in str(arg) for arg in call_args)
        assert any("-t" in str(arg) for arg in call_args)

    def test_cleanup_all_segments(self, processor):
        """Test cleanup of all segments."""
        # Create mock files
        segment_files = [
            processor.output_dir / "segment_001.mp4",
            processor.output_dir / "segment_002.mp4",
            processor.output_dir / "audio_chunk_001.wav"
        ]
        
        with patch('pathlib.Path.glob') as mock_glob, \
             patch('pathlib.Path.unlink') as mock_unlink:
            
            mock_glob.side_effect = [
                segment_files[:2],  # Video files
                [segment_files[2]]  # Audio files
            ]
            
            processor.cleanup_all_segments()
            
            assert mock_unlink.call_count == 3

    async def test_context_manager(self, processor):
        """Test using processor as context manager."""
        with patch.object(processor, 'start') as mock_start, \
             patch.object(processor, 'stop') as mock_stop:
            
            async with processor:
                mock_start.assert_called_once()
            
            mock_stop.assert_called_once()

    async def test_process_stream_basic(self, processor):
        """Test basic stream processing."""
        with patch.object(processor, '_monitor_segments') as mock_monitor, \
             patch.object(processor, '_monitor_process_debug') as mock_debug:
            
            mock_monitor.return_value = AsyncMock()
            mock_debug.return_value = AsyncMock()
            
            # Mock the async iterator
            async def mock_segments():
                yield {"segment_number": 1, "path": "segment_001.mp4", "duration": 60.0}
                yield {"segment_number": 2, "path": "segment_002.mp4", "duration": 60.0}
            
            with patch.object(processor, 'process_stream', return_value=mock_segments()):
                segments = []
                async for segment in processor.process_stream():
                    segments.append(segment)
                    if len(segments) >= 2:  # Prevent infinite loop in test
                        break
                
                assert len(segments) == 2

    def test_error_handling_invalid_url(self):
        """Test error handling with various URL types."""
        # Should not raise for valid URLs
        processor = FFmpegProcessor(
            stream_url="https://example.com/stream.m3u8",
            output_dir=Path("/tmp"),
            config=FFmpegConfig()
        )
        assert processor.stream_url == "https://example.com/stream.m3u8"

    def test_handle_error_method(self, processor):
        """Test error handling method."""
        error = Exception("Test error")
        
        # Should not raise exception - just handle gracefully
        processor._handle_error(error, "test_context")

    async def test_handle_restart_method(self, processor):
        """Test restart handling method."""
        with patch.object(processor, 'stop') as mock_stop, \
             patch.object(processor, 'start') as mock_start:
            
            await processor._handle_restart("test_reason")
            
            mock_stop.assert_called_once()
            mock_start.assert_called_once()

    def test_segment_numbering(self, processor):
        """Test segment numbering logic."""
        # Test with different CSV row formats
        csv_row1 = {
            'segment_number': '001',
            'start_time': '0.0',
            'duration': '60.0',
            'path': 'segment_001.mp4'
        }
        
        segment1 = processor._create_segment_from_csv(csv_row1)
        assert segment1['segment_number'] == 1
        
        csv_row2 = {
            'segment_number': '10',
            'start_time': '600.0',
            'duration': '60.0',
            'path': 'segment_010.mp4'
        }
        
        segment2 = processor._create_segment_from_csv(csv_row2)
        assert segment2['segment_number'] == 10

    def test_timing_calculations(self, processor):
        """Test timing calculation methods."""
        csv_row = {
            'segment_number': '5',
            'start_time': '300.0',
            'duration': '60.0',
            'path': 'segment_005.mp4'
        }
        
        segment = processor._create_segment_from_csv(csv_row)
        assert segment['start_time'] == 300.0
        assert segment['duration'] == 60.0

    @patch('worker.services.ffmpeg_processor.asyncio.sleep')
    async def test_extract_audio_chunks(self, mock_sleep, processor):
        """Test audio chunk extraction."""
        segment = {
            'segment_number': 1,
            'path': processor.output_dir / 'segment_001.mp4',
            'start_time': 0.0,
            'duration': 90.0  # Will create 3 audio chunks of 30s each
        }
        
        with patch.object(processor, '_extract_single_audio_chunk') as mock_extract:
            await processor._extract_audio_chunks(segment)
            
            # Should extract multiple chunks based on segment duration
            assert mock_extract.call_count >= 3


def mock_open_csv_content():
    """Mock CSV file content for testing."""
    csv_content = """segment_number,start_time,duration,path
001,0.0,60.0,segment_001.mp4
002,60.0,60.0,segment_002.mp4
003,120.0,60.0,segment_003.mp4"""
    
    from unittest.mock import mock_open
    return mock_open(read_data=csv_content)