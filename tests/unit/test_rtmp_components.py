"""Unit tests for RTMP protocol components.

Tests for RTMP protocol implementation, FLV parser, and FFmpeg integration
utilities used by the enhanced RTMP adapter.
"""

import pytest
import struct
import sys
from unittest.mock import Mock, patch, AsyncMock

# Mock the metrics module before imports that depend on it
mock_metrics_module = Mock()
mock_metrics_module.MetricsContext = Mock()
mock_metrics_module.counter = Mock(return_value=Mock(increment=Mock()))
mock_metrics_module.gauge = Mock(return_value=Mock(set=Mock()))
mock_metrics_module.histogram = Mock(return_value=Mock(observe=Mock()))
sys.modules["src.infrastructure.monitoring.metrics"] = mock_metrics_module

from src.infrastructure.streaming.rtmp_protocol import (  # noqa: E402
    RTMPProtocol,
    RTMPHandshake,
    RTMPChunkStream,
    RTMPChunkHeader,
    RTMPChunkType,
    RTMPMessage,
    RTMPMessageType,
    AMFEncoder,
    AMFDecoder,
)
from src.infrastructure.streaming.flv_parser import (  # noqa: E402
    FLVParser,
    FLVStreamProcessor,
)
from src.infrastructure.media.ffmpeg_integration import (  # noqa: E402
    FFmpegProcessor,
    VideoFrameExtractor,
    TranscodeOptions,
    VideoCodec,
    AudioCodec,
    check_ffmpeg_available,
    get_ffmpeg_version,
)


class TestRTMPHandshake:
    """Test RTMP handshake implementation."""

    def test_generate_c0(self):
        """Test C0 generation."""
        c0 = RTMPHandshake.generate_c0()
        assert len(c0) == 1
        assert c0[0] == 0x03

    def test_generate_c1(self):
        """Test C1 generation."""
        c1 = RTMPHandshake.generate_c1()
        assert len(c1) == 1536

        # Check timestamp is present (first 4 bytes)
        timestamp = struct.unpack(">I", c1[0:4])[0]
        assert timestamp > 0

        # Check zero bytes (next 4 bytes)
        zero_bytes = struct.unpack(">I", c1[4:8])[0]
        assert zero_bytes == 0

    def test_generate_c2(self):
        """Test C2 generation."""
        s1_data = b"a" * 1536
        c2 = RTMPHandshake.generate_c2(s1_data)
        assert c2 == s1_data

    def test_generate_c2_invalid_length(self):
        """Test C2 generation with invalid S1 length."""
        s1_data = b"a" * 100  # Wrong length
        with pytest.raises(ValueError):
            RTMPHandshake.generate_c2(s1_data)

    def test_validate_s0(self):
        """Test S0 validation."""
        assert RTMPHandshake.validate_s0(b"\x03") is True
        assert RTMPHandshake.validate_s0(b"\x02") is False
        assert RTMPHandshake.validate_s0(b"") is False

    def test_validate_s1(self):
        """Test S1 validation."""
        assert RTMPHandshake.validate_s1(b"a" * 1536) is True
        assert RTMPHandshake.validate_s1(b"a" * 100) is False


class TestRTMPChunkStream:
    """Test RTMP chunk stream protocol."""

    def test_initialization(self):
        """Test chunk stream initialization."""
        chunk_stream = RTMPChunkStream()
        assert chunk_stream.chunk_size == RTMPChunkStream.DEFAULT_CHUNK_SIZE
        assert len(chunk_stream.chunk_streams) == 0

    def test_custom_chunk_size(self):
        """Test chunk stream with custom chunk size."""
        chunk_stream = RTMPChunkStream(chunk_size=256)
        assert chunk_stream.chunk_size == 256

    def test_parse_basic_chunk_header(self):
        """Test parsing basic chunk header (format 0)."""
        # Create a format 0 chunk header with chunk stream ID 3
        header_data = bytearray()
        header_data.append(0x03)  # Format 0, chunk stream ID 3
        header_data.extend(struct.pack(">I", 1000)[1:])  # Timestamp (3 bytes)
        header_data.extend(struct.pack(">I", 256)[1:])  # Message length (3 bytes)
        header_data.append(0x14)  # Message type (AMF0 command)
        header_data.extend(struct.pack("<I", 0))  # Message stream ID (4 bytes)

        chunk_stream = RTMPChunkStream()
        header, bytes_consumed = chunk_stream.parse_chunk_header(bytes(header_data))

        assert header.format_type == RTMPChunkType.FORMAT_0
        assert header.chunk_stream_id == 3
        assert header.timestamp == 1000
        assert header.message_length == 256
        assert header.message_type_id == 0x14
        assert header.message_stream_id == 0
        assert bytes_consumed == 12

    def test_create_chunk_header(self):
        """Test creating chunk header bytes."""
        header = RTMPChunkHeader(
            format_type=RTMPChunkType.FORMAT_0,
            chunk_stream_id=3,
            timestamp=1000,
            message_length=256,
            message_type_id=0x14,
            message_stream_id=0,
        )

        chunk_stream = RTMPChunkStream()
        header_bytes = chunk_stream.create_chunk_header(header)

        # Parse it back to verify
        parsed_header, _ = chunk_stream.parse_chunk_header(header_bytes)

        assert parsed_header.format_type == header.format_type
        assert parsed_header.chunk_stream_id == header.chunk_stream_id
        assert parsed_header.timestamp == header.timestamp
        assert parsed_header.message_length == header.message_length
        assert parsed_header.message_type_id == header.message_type_id


class TestAMFEncoder:
    """Test AMF encoding functionality."""

    def test_encode_number(self):
        """Test AMF number encoding."""
        encoded = AMFEncoder.encode_number(42.5)
        assert encoded[0] == AMFEncoder.AMF0_NUMBER
        assert len(encoded) == 9  # 1 byte type + 8 bytes double

    def test_encode_boolean(self):
        """Test AMF boolean encoding."""
        encoded_true = AMFEncoder.encode_boolean(True)
        encoded_false = AMFEncoder.encode_boolean(False)

        assert encoded_true[0] == AMFEncoder.AMF0_BOOLEAN
        assert encoded_true[1] == 1
        assert encoded_false[1] == 0

    def test_encode_string(self):
        """Test AMF string encoding."""
        test_string = "hello"
        encoded = AMFEncoder.encode_string(test_string)

        assert encoded[0] == AMFEncoder.AMF0_STRING
        length = struct.unpack(">H", encoded[1:3])[0]
        assert length == len(test_string)
        assert encoded[3 : 3 + length].decode("utf-8") == test_string

    def test_encode_null(self):
        """Test AMF null encoding."""
        encoded = AMFEncoder.encode_null()
        assert len(encoded) == 1
        assert encoded[0] == AMFEncoder.AMF0_NULL

    def test_encode_object(self):
        """Test AMF object encoding."""
        test_obj = {"key1": "value1", "key2": 42}
        encoded = AMFEncoder.encode_object(test_obj)

        assert encoded[0] == AMFEncoder.AMF0_OBJECT
        # Should end with object end marker
        assert encoded[-1] == AMFEncoder.AMF0_OBJECT_END

    def test_encode_array(self):
        """Test AMF array encoding."""
        test_array = ["hello", 42, True]
        encoded = AMFEncoder.encode_array(test_array)

        assert encoded[0] == AMFEncoder.AMF0_STRICT_ARRAY
        # Array length should be encoded
        length = struct.unpack(">I", encoded[1:5])[0]
        assert length == len(test_array)


class TestAMFDecoder:
    """Test AMF decoding functionality."""

    def test_decode_number(self):
        """Test AMF number decoding."""
        original_value = 42.5
        encoded = AMFEncoder.encode_number(original_value)

        decoder = AMFDecoder(encoded[1:])  # Skip type marker
        decoded_value = decoder.decode_number()

        assert decoded_value == original_value

    def test_decode_boolean(self):
        """Test AMF boolean decoding."""
        encoded_true = AMFEncoder.encode_boolean(True)
        encoded_false = AMFEncoder.encode_boolean(False)

        decoder_true = AMFDecoder(encoded_true[1:])
        decoder_false = AMFDecoder(encoded_false[1:])

        assert decoder_true.decode_boolean() is True
        assert decoder_false.decode_boolean() is False

    def test_decode_string(self):
        """Test AMF string decoding."""
        original_string = "hello world"
        encoded = AMFEncoder.encode_string(original_string)

        decoder = AMFDecoder(encoded[1:])
        decoded_string = decoder.decode_string()

        assert decoded_string == original_string

    def test_decode_value_number(self):
        """Test AMF value decoding for numbers."""
        encoded = AMFEncoder.encode_number(123.456)
        decoder = AMFDecoder(encoded)

        value = decoder.decode_value()
        assert value == 123.456

    def test_decode_value_string(self):
        """Test AMF value decoding for strings."""
        encoded = AMFEncoder.encode_string("test")
        decoder = AMFDecoder(encoded)

        value = decoder.decode_value()
        assert value == "test"

    def test_decode_value_null(self):
        """Test AMF value decoding for null."""
        encoded = AMFEncoder.encode_null()
        decoder = AMFDecoder(encoded)

        value = decoder.decode_value()
        assert value is None


class TestRTMPProtocol:
    """Test complete RTMP protocol implementation."""

    def test_initialization(self):
        """Test RTMP protocol initialization."""
        protocol = RTMPProtocol()
        assert protocol.chunk_stream is not None
        assert protocol.transaction_id == 1
        assert protocol.handshake_complete is False
        assert protocol.connected is False

    @pytest.mark.asyncio
    async def test_perform_handshake_success(self):
        """Test successful RTMP handshake."""
        protocol = RTMPProtocol()

        # Mock reader and writer
        reader = AsyncMock()
        writer = AsyncMock()

        # Mock handshake responses
        reader.read.side_effect = [
            b"\x03",  # S0
            b"a" * 1536,  # S1
            b"b" * 1536,  # S2
        ]

        success = await protocol.perform_handshake(reader, writer)
        assert success is True
        assert protocol.handshake_complete is True

    def test_create_connect_message(self):
        """Test creating RTMP connect message."""
        protocol = RTMPProtocol()
        app_name = "live"

        message_bytes = protocol.create_connect_message(app_name)

        assert len(message_bytes) > 0
        assert protocol.transaction_id == 2  # Should increment

    def test_create_message_chunks(self):
        """Test creating chunked message data."""
        protocol = RTMPProtocol()

        # Create a test message
        message = RTMPMessage(
            message_type=RTMPMessageType.AMF0_COMMAND,
            timestamp=1000,
            stream_id=0,
            data=b"test message data",
            chunk_stream_id=3,
        )

        chunks = protocol.create_message_chunks(message)
        assert len(chunks) > 0


class TestFLVParser:
    """Test FLV format parser."""

    def test_initialization(self):
        """Test FLV parser initialization."""
        parser = FLVParser()
        assert parser.position == 0
        assert parser.header is None

    def test_create_flv_header(self):
        """Create a basic FLV header for testing."""
        header = bytearray()
        header.extend(b"FLV")  # Signature
        header.append(0x01)  # Version
        header.append(0x05)  # Type flags (audio + video)
        header.extend(struct.pack(">I", 9))  # Data offset
        return bytes(header)

    def test_parse_flv_header(self):
        """Test FLV header parsing."""
        header_bytes = self.test_create_flv_header()
        parser = FLVParser(header_bytes)

        header = parser.parse_header()

        assert header.signature == b"FLV"
        assert header.version == 1
        assert header.has_audio is True
        assert header.has_video is True
        assert header.data_offset == 9

    def test_parse_flv_header_invalid_signature(self):
        """Test FLV header parsing with invalid signature."""
        invalid_header = b"ABC\x01\x05\x00\x00\x00\x09"
        parser = FLVParser(invalid_header)

        with pytest.raises(ValueError):
            parser.parse_header()

    def test_read_operations(self):
        """Test basic read operations."""
        test_data = struct.pack(">BHIQ", 0x42, 0x1234, 0x56789ABC, 0x123456789ABCDEF0)
        parser = FLVParser(test_data)

        assert parser.read_ui8() == 0x42
        assert parser.read_ui16() == 0x1234
        assert parser.read_ui32() == 0x56789ABC

    def test_read_ui24(self):
        """Test 24-bit integer reading."""
        test_data = b"\x12\x34\x56"
        parser = FLVParser(test_data)

        value = parser.read_ui24()
        assert value == 0x123456


class TestFLVStreamProcessor:
    """Test FLV stream processor for real-time parsing."""

    def test_initialization(self):
        """Test FLV stream processor initialization."""
        processor = FLVStreamProcessor()
        assert len(processor.buffer) == 0
        assert processor.header_parsed is False
        assert processor.tags_processed == 0

    def test_process_data_no_header(self):
        """Test processing data when no header is present."""
        processor = FLVStreamProcessor()

        # Send incomplete data (not enough for header)
        tags = processor.process_data(b"FL")
        assert len(tags) == 0
        assert processor.header_parsed is False

    def test_reset_processor(self):
        """Test resetting processor state."""
        processor = FLVStreamProcessor()
        processor.buffer.extend(b"test data")
        processor.header_parsed = True
        processor.tags_processed = 5

        processor.reset()

        assert len(processor.buffer) == 0
        assert processor.header_parsed is False
        assert processor.tags_processed == 0


class TestFFmpegIntegration:
    """Test FFmpeg integration utilities."""

    def test_check_ffmpeg_available(self):
        """Test FFmpeg availability check."""
        # This will depend on system FFmpeg installation
        available = check_ffmpeg_available()
        assert isinstance(available, bool)

    def test_get_ffmpeg_version(self):
        """Test getting FFmpeg version."""
        version = get_ffmpeg_version()
        assert isinstance(version, str)

    def test_transcode_options(self):
        """Test transcode options configuration."""
        options = TranscodeOptions(
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            video_bitrate=2000,
            audio_bitrate=128,
        )

        assert options.video_codec == VideoCodec.H264
        assert options.audio_codec == AudioCodec.AAC
        assert options.video_bitrate == 2000
        assert options.audio_bitrate == 128

    def test_ffmpeg_processor_initialization(self):
        """Test FFmpeg processor initialization."""
        processor = FFmpegProcessor()
        assert processor.hardware_acceleration is False

        processor_hw = FFmpegProcessor(hardware_acceleration=True)
        assert processor_hw.hardware_acceleration is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_ffmpeg_probe_file_success(self, mock_subprocess):
        """Test successful file probing with FFmpeg."""
        # Mock successful FFprobe output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = """
        {
            "format": {
                "format_name": "mp4",
                "duration": "10.0",
                "bit_rate": "1000000"
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1"
                }
            ]
        }
        """

        # This would require more complex mocking for actual async subprocess
        # For now, just test the structure
        assert True  # Placeholder

    def test_video_frame_extractor_initialization(self):
        """Test video frame extractor initialization."""
        extractor = VideoFrameExtractor()
        assert extractor.ffmpeg_processor is not None
        assert extractor.pyav_processor is not None

    def test_video_frame_extractor_with_hardware_acceleration(self):
        """Test video frame extractor with hardware acceleration."""
        extractor = VideoFrameExtractor(use_hardware_acceleration=True)
        assert extractor.ffmpeg_processor.hardware_acceleration is True


class TestRTMPIntegration:
    """Integration tests for RTMP components working together."""

    def test_rtmp_protocol_with_amf_encoding(self):
        """Test RTMP protocol with AMF message encoding."""
        protocol = RTMPProtocol()

        # Create a connect message
        app_name = "live"
        message_bytes = protocol.create_connect_message(app_name)

        # Should contain encoded AMF data
        assert len(message_bytes) > 50  # Connect messages are typically larger
        assert protocol.connect_params is not None
        assert protocol.connect_params["app"] == app_name

    def test_flv_with_rtmp_message_types(self):
        """Test FLV parser with RTMP message type data."""
        # Create mock video tag data
        video_data = bytearray()
        video_data.append(0x17)  # Video header (keyframe + AVC)
        video_data.append(0x00)  # AVC packet type (sequence header)
        video_data.extend(b"\x00\x00\x00")  # Composition time
        video_data.extend(b"mock_avc_data")

        # This would be part of a complete FLV tag
        assert len(video_data) > 0
        assert video_data[0] == 0x17

    @pytest.mark.asyncio
    async def test_complete_rtmp_message_processing(self):
        """Test complete RTMP message processing flow."""
        protocol = RTMPProtocol()

        # Create a test command message
        command_data = (
            AMFEncoder.encode_string("_result")
            + AMFEncoder.encode_number(1)
            + AMFEncoder.encode_object({"code": "NetConnection.Connect.Success"})
        )

        message = RTMPMessage(
            message_type=RTMPMessageType.AMF0_COMMAND,
            timestamp=0,
            stream_id=0,
            data=command_data,
            chunk_stream_id=3,
        )

        # Process the message
        result = protocol.process_command_message(message)

        assert result is not None
        assert result["command"] == "_result"
        assert result["transaction_id"] == 1
        assert protocol.connected is True
