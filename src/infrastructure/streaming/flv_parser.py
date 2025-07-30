"""FLV (Flash Video) format parser for RTMP streams.

This module provides comprehensive FLV format parsing capabilities for extracting
video frames, audio samples, and metadata from RTMP streams in the TL;DR Highlight API.

FLV Specification: Adobe Flash Video File Format Specification v10.1
"""

import logging
import struct
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from dataclasses import dataclass
from enum import Enum
import io

logger = logging.getLogger(__name__)


class FLVTagType(Enum):
    """FLV tag types."""

    AUDIO = 8
    VIDEO = 9
    SCRIPT_DATA = 18


class FLVSoundFormat(Enum):
    """FLV audio codec formats."""

    LINEAR_PCM_PLATFORM_ENDIAN = 0
    ADPCM = 1
    MP3 = 2
    LINEAR_PCM_LITTLE_ENDIAN = 3
    NELLYMOSER_16_MONO = 4
    NELLYMOSER_8_MONO = 5
    NELLYMOSER = 6
    G711_A_LAW = 7
    G711_MU_LAW = 8
    RESERVED = 9
    AAC = 10
    SPEEX = 11
    MP3_8_KHZ = 14
    DEVICE_SPECIFIC = 15


class FLVSoundRate(Enum):
    """FLV audio sample rates."""

    RATE_5_5_KHZ = 0
    RATE_11_KHZ = 1
    RATE_22_KHZ = 2
    RATE_44_KHZ = 3


class FLVSoundSize(Enum):
    """FLV audio sample sizes."""

    BITS_8 = 0
    BITS_16 = 1


class FLVSoundType(Enum):
    """FLV audio channel configuration."""

    MONO = 0
    STEREO = 1


class FLVVideoCodec(Enum):
    """FLV video codec formats."""

    JPEG = 1
    SORENSON_H263 = 2
    SCREEN_VIDEO = 3
    ON2_VP6 = 4
    ON2_VP6_ALPHA = 5
    SCREEN_VIDEO_V2 = 6
    AVC = 7  # H.264


class FLVVideoFrameType(Enum):
    """FLV video frame types."""

    KEY_FRAME = 1
    INTER_FRAME = 2
    DISPOSABLE_INTER_FRAME = 3
    GENERATED_KEY_FRAME = 4
    VIDEO_INFO_COMMAND = 5


class FLVAVCPacketType(Enum):
    """FLV AVC packet types (for H.264)."""

    SEQUENCE_HEADER = 0
    NALU = 1
    END_OF_SEQUENCE = 2


@dataclass
class FLVHeader:
    """FLV file header information."""

    signature: bytes
    version: int
    type_flags: int
    data_offset: int
    has_audio: bool
    has_video: bool


@dataclass
class FLVTagHeader:
    """FLV tag header information."""

    tag_type: FLVTagType
    data_size: int
    timestamp: int
    timestamp_extended: int
    stream_id: int

    @property
    def full_timestamp(self) -> int:
        """Get full 32-bit timestamp."""
        return (self.timestamp_extended << 24) | self.timestamp


@dataclass
class FLVAudioTag:
    """FLV audio tag data."""

    sound_format: FLVSoundFormat
    sound_rate: FLVSoundRate
    sound_size: FLVSoundSize
    sound_type: FLVSoundType
    aac_packet_type: Optional[int] = None
    data: bytes = b""


@dataclass
class FLVVideoTag:
    """FLV video tag data."""

    frame_type: FLVVideoFrameType
    codec_id: FLVVideoCodec
    avc_packet_type: Optional[FLVAVCPacketType] = None
    composition_time: int = 0
    data: bytes = b""


@dataclass
class FLVScriptTag:
    """FLV script/metadata tag data."""

    name: str
    value: Any
    raw_data: bytes = b""


@dataclass
class FLVTag:
    """Complete FLV tag."""

    header: FLVTagHeader
    data: Union[FLVAudioTag, FLVVideoTag, FLVScriptTag]
    previous_tag_size: int


class FLVStreamInfo:
    """Information about an FLV stream."""

    def __init__(self) -> None:
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.framerate: Optional[float] = None
        self.videodatarate: Optional[float] = None
        self.audiodatarate: Optional[float] = None
        self.audiosamplerate: Optional[float] = None
        self.audiosamplesize: Optional[int] = None
        self.stereo: Optional[bool] = None
        self.videocodecid: Optional[int] = None
        self.audiocodecid: Optional[int] = None
        self.duration: Optional[float] = None
        self.filesize: Optional[int] = None
        self.lasttimestamp: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def update_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update stream info from onMetaData."""
        for key, value in metadata.items():
            if hasattr(self, key.lower()):
                setattr(self, key.lower(), value)
            else:
                self.metadata[key] = value


class AMFDataType(Enum):
    """AMF data types for script data."""

    NUMBER = 0
    BOOLEAN = 1
    STRING = 2
    OBJECT = 3
    MOVIECLIP = 4
    NULL = 5
    UNDEFINED = 6
    REFERENCE = 7
    ECMA_ARRAY = 8
    OBJECT_END_MARKER = 9
    STRICT_ARRAY = 10
    DATE = 11
    LONG_STRING = 12


class FLVParser:
    """Complete FLV format parser for RTMP streams."""

    FLV_SIGNATURE = b"FLV"

    def __init__(self, data: bytes = b""):
        self.data = io.BytesIO(data)
        self.position = 0
        self.header: Optional[FLVHeader] = None
        self.stream_info = FLVStreamInfo()
        self.sequence_headers: Dict[str, bytes] = {}  # Store AVC/AAC sequence headers

    def reset(self, data: bytes) -> None:
        """Reset parser with new data."""
        self.data = io.BytesIO(data)
        self.position = 0

    def append_data(self, data: bytes) -> None:
        """Append more data to the parser."""
        current_data = self.data.getvalue()
        self.data = io.BytesIO(current_data + data)

    def read_bytes(self, count: int) -> bytes:
        """Read specified number of bytes."""
        data = self.data.read(count)
        if len(data) < count:
            raise EOFError(f"Expected {count} bytes, got {len(data)}")
        self.position += count
        return data

    def read_ui8(self) -> int:
        """Read unsigned 8-bit integer."""
        result: tuple[int] = struct.unpack("B", self.read_bytes(1))
        return result[0]

    def read_ui16(self) -> int:
        """Read unsigned 16-bit integer (big-endian)."""
        result: tuple[int] = struct.unpack(">H", self.read_bytes(2))
        return result[0]

    def read_ui24(self) -> int:
        """Read unsigned 24-bit integer (big-endian)."""
        data = self.read_bytes(3)
        result: tuple[int] = struct.unpack(">I", b"\\x00" + data)
        return result[0]

    def read_ui32(self) -> int:
        """Read unsigned 32-bit integer (big-endian)."""
        result: tuple[int] = struct.unpack(">I", self.read_bytes(4))
        return result[0]

    def read_double(self) -> float:
        """Read 64-bit double (big-endian)."""
        result: tuple[float] = struct.unpack(">d", self.read_bytes(8))
        return result[0]

    def parse_header(self) -> FLVHeader:
        """Parse FLV file header."""
        try:
            signature = self.read_bytes(3)
            if signature != self.FLV_SIGNATURE:
                raise ValueError(f"Invalid FLV signature: {signature!r}")

            version = self.read_ui8()
            type_flags = self.read_ui8()
            data_offset = self.read_ui32()

            has_audio = bool(type_flags & 0x04)
            has_video = bool(type_flags & 0x01)

            self.header = FLVHeader(
                signature=signature,
                version=version,
                type_flags=type_flags,
                data_offset=data_offset,
                has_audio=has_audio,
                has_video=has_video,
            )

            # Skip to data offset
            if data_offset > 9:
                self.read_bytes(data_offset - 9)

            logger.debug(
                f"Parsed FLV header: version={version}, audio={has_audio}, video={has_video}"
            )
            return self.header

        except Exception as e:
            logger.error(f"Error parsing FLV header: {e}")
            raise

    def parse_tag_header(self) -> FLVTagHeader:
        """Parse FLV tag header."""
        try:
            # Read previous tag size (first tag will be 0)
            _ = self.read_ui32()

            # Read tag header
            tag_type_byte = self.read_ui8()
            tag_type = FLVTagType(tag_type_byte)

            data_size = self.read_ui24()
            timestamp = self.read_ui24()
            timestamp_extended = self.read_ui8()
            stream_id = self.read_ui24()

            return FLVTagHeader(
                tag_type=tag_type,
                data_size=data_size,
                timestamp=timestamp,
                timestamp_extended=timestamp_extended,
                stream_id=stream_id,
            )

        except Exception as e:
            logger.error(f"Error parsing FLV tag header: {e}")
            raise

    def parse_audio_tag(self, data_size: int) -> FLVAudioTag:
        """Parse FLV audio tag."""
        try:
            if data_size == 0:
                return FLVAudioTag(
                    sound_format=FLVSoundFormat.LINEAR_PCM_PLATFORM_ENDIAN,
                    sound_rate=FLVSoundRate.RATE_44_KHZ,
                    sound_size=FLVSoundSize.BITS_16,
                    sound_type=FLVSoundType.STEREO,
                )

            # Read audio header
            audio_header = self.read_ui8()

            sound_format = FLVSoundFormat((audio_header >> 4) & 0x0F)
            sound_rate = FLVSoundRate((audio_header >> 2) & 0x03)
            sound_size = FLVSoundSize((audio_header >> 1) & 0x01)
            sound_type = FLVSoundType(audio_header & 0x01)

            aac_packet_type = None
            remaining_size = data_size - 1

            # Handle AAC sequence header
            if sound_format == FLVSoundFormat.AAC and remaining_size > 0:
                aac_packet_type = self.read_ui8()
                remaining_size -= 1

                if aac_packet_type == 0:  # AAC sequence header
                    sequence_data = self.read_bytes(remaining_size)
                    self.sequence_headers["aac"] = sequence_data
                    logger.debug(
                        f"Stored AAC sequence header: {len(sequence_data)} bytes"
                    )

                    return FLVAudioTag(
                        sound_format=sound_format,
                        sound_rate=sound_rate,
                        sound_size=sound_size,
                        sound_type=sound_type,
                        aac_packet_type=aac_packet_type,
                        data=sequence_data,
                    )

            # Read audio data
            audio_data = self.read_bytes(remaining_size) if remaining_size > 0 else b""

            return FLVAudioTag(
                sound_format=sound_format,
                sound_rate=sound_rate,
                sound_size=sound_size,
                sound_type=sound_type,
                aac_packet_type=aac_packet_type,
                data=audio_data,
            )

        except Exception as e:
            logger.error(f"Error parsing audio tag: {e}")
            raise

    def parse_video_tag(self, data_size: int) -> FLVVideoTag:
        """Parse FLV video tag."""
        try:
            if data_size == 0:
                return FLVVideoTag(
                    frame_type=FLVVideoFrameType.KEY_FRAME, codec_id=FLVVideoCodec.AVC
                )

            # Read video header
            video_header = self.read_ui8()

            frame_type = FLVVideoFrameType((video_header >> 4) & 0x0F)
            codec_id = FLVVideoCodec(video_header & 0x0F)

            avc_packet_type = None
            composition_time = 0
            remaining_size = data_size - 1

            # Handle AVC (H.264) video
            if codec_id == FLVVideoCodec.AVC and remaining_size >= 4:
                avc_packet_type = FLVAVCPacketType(self.read_ui8())
                composition_time = struct.unpack(">I", b"\\x00" + self.read_bytes(3))[0]
                # Handle signed 24-bit composition time
                if composition_time >= 0x800000:
                    composition_time -= 0x1000000

                remaining_size -= 4

                if avc_packet_type == FLVAVCPacketType.SEQUENCE_HEADER:
                    # AVC sequence header (SPS/PPS)
                    sequence_data = self.read_bytes(remaining_size)
                    self.sequence_headers["avc"] = sequence_data
                    logger.debug(
                        f"Stored AVC sequence header: {len(sequence_data)} bytes"
                    )

                    return FLVVideoTag(
                        frame_type=frame_type,
                        codec_id=codec_id,
                        avc_packet_type=avc_packet_type,
                        composition_time=composition_time,
                        data=sequence_data,
                    )

            # Read video data
            video_data = self.read_bytes(remaining_size) if remaining_size > 0 else b""

            return FLVVideoTag(
                frame_type=frame_type,
                codec_id=codec_id,
                avc_packet_type=avc_packet_type,
                composition_time=composition_time,
                data=video_data,
            )

        except Exception as e:
            logger.error(f"Error parsing video tag: {e}")
            raise

    def parse_amf_value(self) -> Tuple[Any, int]:
        """Parse AMF value and return (value, bytes_consumed)."""
        start_pos = self.data.tell()

        try:
            value_type = self.read_ui8()
            value: Any

            if value_type == AMFDataType.NUMBER.value:
                value = self.read_double()
            elif value_type == AMFDataType.BOOLEAN.value:
                value = bool(self.read_ui8())
            elif value_type == AMFDataType.STRING.value:
                string_length = self.read_ui16()
                value = self.read_bytes(string_length).decode("utf-8")
            elif value_type == AMFDataType.LONG_STRING.value:
                string_length = self.read_ui32()
                value = self.read_bytes(string_length).decode("utf-8")
            elif value_type == AMFDataType.NULL.value:
                value = None
            elif value_type == AMFDataType.UNDEFINED.value:
                value = None
            elif value_type == AMFDataType.ECMA_ARRAY.value:
                array_length = self.read_ui32()
                value = {}

                while True:
                    # Read property name
                    name_length = self.read_ui16()
                    if name_length == 0:
                        # Check for object end marker
                        marker = self.read_ui8()
                        if marker == AMFDataType.OBJECT_END_MARKER.value:
                            break
                        else:
                            # Put back the byte
                            self.data.seek(self.data.tell() - 1)

                    name = self.read_bytes(name_length).decode("utf-8")
                    prop_value, _ = self.parse_amf_value()
                    value[name] = prop_value

            elif value_type == AMFDataType.STRICT_ARRAY.value:
                array_length = self.read_ui32()
                value = []
                for _ in range(array_length):
                    item_value, _ = self.parse_amf_value()
                    value.append(item_value)
            else:
                logger.warning(f"Unsupported AMF data type: {value_type}")
                value = None

            bytes_consumed = self.data.tell() - start_pos
            return value, bytes_consumed

        except Exception as e:
            logger.error(f"Error parsing AMF value: {e}")
            return None, self.data.tell() - start_pos

    def parse_script_tag(self, data_size: int) -> FLVScriptTag:
        """Parse FLV script/metadata tag."""
        try:
            start_pos = self.data.tell()

            # Parse script name
            name, name_bytes = self.parse_amf_value()
            if not isinstance(name, str):
                name = "unknown"

            # Parse script value
            remaining = data_size - name_bytes
            if remaining > 0:
                value, _ = self.parse_amf_value()
            else:
                value = None

            # Read any remaining data
            end_pos = start_pos + data_size
            current_pos = self.data.tell()
            if current_pos < end_pos:
                remaining_data = self.read_bytes(end_pos - current_pos)
            else:
                remaining_data = b""

            # Update stream info if this is onMetaData
            if name == "onMetaData" and isinstance(value, dict):
                self.stream_info.update_from_metadata(value)
                logger.debug(f"Updated stream info from onMetaData: {value}")

            return FLVScriptTag(name=name, value=value, raw_data=remaining_data)

        except Exception as e:
            logger.error(f"Error parsing script tag: {e}")
            # Skip remaining data
            end_pos = self.data.tell() + data_size
            self.data.seek(end_pos)
            self.position = end_pos

            return FLVScriptTag(name="error", value=None)

    def parse_tag(self) -> Optional[FLVTag]:
        """Parse complete FLV tag."""
        try:
            # Check if we have enough data for a tag header
            current_pos = self.data.tell()
            remaining = len(self.data.getvalue()) - current_pos

            if remaining < 15:  # Minimum tag size (4 + 11)
                return None

            header = self.parse_tag_header()

            # Parse tag data based on type
            data: Union[FLVAudioTag, FLVVideoTag, FLVScriptTag]
            if header.tag_type == FLVTagType.AUDIO:
                data = self.parse_audio_tag(header.data_size)
            elif header.tag_type == FLVTagType.VIDEO:
                data = self.parse_video_tag(header.data_size)
            elif header.tag_type == FLVTagType.SCRIPT_DATA:
                data = self.parse_script_tag(header.data_size)
            else:
                logger.warning(f"Unknown tag type: {header.tag_type}")
                # Skip unknown tag data
                self.read_bytes(header.data_size)
                return None

            return FLVTag(
                header=header,
                data=data,
                previous_tag_size=0,  # Will be set when reading next tag
            )

        except EOFError:
            # Not enough data available
            return None
        except Exception as e:
            logger.error(f"Error parsing FLV tag: {e}")
            return None

    def parse_tags(self) -> Generator[FLVTag, None, None]:
        """Parse all available FLV tags."""
        try:
            while True:
                tag = self.parse_tag()
                if tag is None:
                    break
                yield tag

        except Exception as e:
            logger.error(f"Error parsing FLV tags: {e}")

    def extract_video_frames(self, tags: List[FLVTag]) -> List[Tuple[int, bytes, bool]]:
        """Extract video frames from FLV tags.

        Returns:
            List of (timestamp, frame_data, is_keyframe) tuples
        """
        frames = []

        for tag in tags:
            if (
                isinstance(tag.data, FLVVideoTag)
                and tag.data.codec_id == FLVVideoCodec.AVC
                and tag.data.avc_packet_type == FLVAVCPacketType.NALU
            ):
                timestamp = tag.header.full_timestamp + tag.data.composition_time
                is_keyframe = tag.data.frame_type == FLVVideoFrameType.KEY_FRAME

                frames.append((timestamp, tag.data.data, is_keyframe))

        return frames

    def extract_audio_samples(self, tags: List[FLVTag]) -> List[Tuple[int, bytes]]:
        """Extract audio samples from FLV tags.

        Returns:
            List of (timestamp, audio_data) tuples
        """
        samples = []

        for tag in tags:
            if (
                isinstance(tag.data, FLVAudioTag)
                and tag.data.sound_format == FLVSoundFormat.AAC
                and tag.data.aac_packet_type == 1
            ):  # AAC raw data
                timestamp = tag.header.full_timestamp
                samples.append((timestamp, tag.data.data))

        return samples

    def get_metadata(self) -> Dict[str, Any]:
        """Get stream metadata."""
        metadata = {
            "width": self.stream_info.width,
            "height": self.stream_info.height,
            "framerate": self.stream_info.framerate,
            "duration": self.stream_info.duration,
            "video_codec": self.stream_info.videocodecid,
            "audio_codec": self.stream_info.audiocodecid,
            "video_bitrate": self.stream_info.videodatarate,
            "audio_bitrate": self.stream_info.audiodatarate,
            "audio_samplerate": self.stream_info.audiosamplerate,
            "audio_channels": 2 if self.stream_info.stereo else 1,
            "has_sequence_headers": {
                "avc": "avc" in self.sequence_headers,
                "aac": "aac" in self.sequence_headers,
            },
        }

        # Add custom metadata
        metadata.update(self.stream_info.metadata)

        return metadata

    def get_sequence_headers(self) -> Dict[str, bytes]:
        """Get stored sequence headers (AVC/AAC configuration)."""
        return self.sequence_headers.copy()


class FLVStreamProcessor:
    """Higher-level FLV stream processor for real-time parsing."""

    def __init__(self, buffer_size: int = 65536):
        self.parser = FLVParser()
        self.buffer = bytearray()
        self.buffer_size = buffer_size
        self.header_parsed = False
        self.tags_processed = 0
        self.last_timestamp = 0

    def process_data(self, data: bytes) -> List[FLVTag]:
        """Process incoming FLV data and return parsed tags."""
        self.buffer.extend(data)
        tags = []

        try:
            # Update parser with current buffer
            self.parser.reset(bytes(self.buffer))

            # Parse header if not done yet
            if not self.header_parsed:
                try:
                    header = self.parser.parse_header()
                    self.header_parsed = True
                    logger.info(
                        f"FLV stream: audio={header.has_audio}, video={header.has_video}"
                    )
                except Exception as e:
                    logger.debug(f"Header not ready yet: {e}")
                    return []

            # Parse tags
            parsed_tags = list(self.parser.parse_tags())
            tags.extend(parsed_tags)

            # Update statistics
            self.tags_processed += len(parsed_tags)
            if parsed_tags:
                self.last_timestamp = parsed_tags[-1].header.full_timestamp

            # Keep only unprocessed data in buffer
            processed_bytes = self.parser.data.tell()
            self.buffer = self.buffer[processed_bytes:]

            # Limit buffer size to prevent memory issues
            if len(self.buffer) > self.buffer_size:
                logger.warning(
                    f"FLV buffer overflow, discarding {len(self.buffer) - self.buffer_size} bytes"
                )
                self.buffer = self.buffer[-self.buffer_size :]

        except Exception as e:
            logger.error(f"Error processing FLV data: {e}")

        return tags

    def get_stream_info(self) -> Dict[str, Any]:
        """Get current stream information."""
        metadata = self.parser.get_metadata()
        metadata.update(
            {
                "tags_processed": self.tags_processed,
                "last_timestamp": self.last_timestamp,
                "buffer_size": len(self.buffer),
                "header_parsed": self.header_parsed,
            }
        )
        return metadata

    def reset(self) -> None:
        """Reset the processor state."""
        self.parser = FLVParser()
        self.buffer.clear()
        self.header_parsed = False
        self.tags_processed = 0
        self.last_timestamp = 0
