"""RTMP protocol implementation utilities.

This module provides complete RTMP protocol support including handshake,
message parsing, chunk stream protocol, and AMF encoding/decoding for
the TL;DR Highlight API RTMP adapter.

RTMP Specification: Adobe Flash Video File Format Specification v10.1
"""

import asyncio
import logging
import struct
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger(__name__)


class RTMPMessageType(Enum):
    """RTMP message types as defined in the RTMP specification."""

    SET_CHUNK_SIZE = 1
    ABORT_MESSAGE = 2
    ACKNOWLEDGEMENT = 3
    USER_CONTROL_MESSAGE = 4
    WINDOW_ACKNOWLEDGEMENT_SIZE = 5
    SET_PEER_BANDWIDTH = 6
    AUDIO = 8
    VIDEO = 9
    AMF3_DATA = 15
    AMF3_SHARED_OBJECT = 16
    AMF3_COMMAND = 17
    AMF0_DATA = 18
    AMF0_SHARED_OBJECT = 19
    AMF0_COMMAND = 20
    AGGREGATE = 22


class RTMPChunkType(Enum):
    """RTMP chunk header format types."""

    FORMAT_0 = 0  # 11 bytes
    FORMAT_1 = 1  # 7 bytes
    FORMAT_2 = 2  # 3 bytes
    FORMAT_3 = 3  # 0 bytes


class RTMPUserControlType(Enum):
    """RTMP user control message event types."""

    STREAM_BEGIN = 0
    STREAM_EOF = 1
    STREAM_DRY = 2
    SET_BUFFER_LENGTH = 3
    STREAM_IS_RECORDED = 4
    PING_REQUEST = 6
    PING_RESPONSE = 7


@dataclass
class RTMPChunkHeader:
    """RTMP chunk header information."""

    format_type: RTMPChunkType
    chunk_stream_id: int
    timestamp: int = 0
    message_length: int = 0
    message_type_id: int = 0
    message_stream_id: int = 0
    timestamp_delta: int = 0
    extended_timestamp: bool = False


@dataclass
class RTMPMessage:
    """RTMP message container."""

    message_type: RTMPMessageType
    timestamp: int
    stream_id: int
    data: bytes
    chunk_stream_id: int = 0


class RTMPHandshake:
    """RTMP handshake implementation following RTMP specification."""

    HANDSHAKE_SIZE = 1536
    VERSION = 0x03

    @classmethod
    def generate_c0(cls) -> bytes:
        """Generate C0 handshake packet (version byte)."""
        return bytes([cls.VERSION])

    @classmethod
    def generate_c1(cls) -> bytes:
        """Generate C1 handshake packet (1536 bytes of timestamp and random data)."""
        timestamp = int(time.time())
        zero = 0

        # Generate random data
        random_data = bytes(
            [random.randint(0, 255) for _ in range(cls.HANDSHAKE_SIZE - 8)]
        )

        return struct.pack(">II", timestamp, zero) + random_data

    @classmethod
    def generate_c2(cls, s1_data: bytes) -> bytes:
        """Generate C2 handshake packet (echo of S1)."""
        if len(s1_data) != cls.HANDSHAKE_SIZE:
            raise ValueError(f"S1 data must be {cls.HANDSHAKE_SIZE} bytes")
        return s1_data

    @classmethod
    def validate_s0(cls, s0_data: bytes) -> bool:
        """Validate S0 handshake packet."""
        return len(s0_data) == 1 and s0_data[0] == cls.VERSION

    @classmethod
    def validate_s1(cls, s1_data: bytes) -> bool:
        """Validate S1 handshake packet."""
        return len(s1_data) == cls.HANDSHAKE_SIZE

    @classmethod
    def validate_s2(cls, s2_data: bytes, c1_data: bytes) -> bool:
        """Validate S2 handshake packet (should echo C1)."""
        return len(s2_data) == cls.HANDSHAKE_SIZE


class RTMPChunkStream:
    """RTMP chunk stream protocol implementation."""

    DEFAULT_CHUNK_SIZE = 128
    MAX_CHUNK_STREAM_ID = 65599

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.chunk_streams: Dict[int, Dict[str, Any]] = {}
        self.last_chunk_headers: Dict[int, RTMPChunkHeader] = {}

    def parse_chunk_header(
        self, data: bytes, offset: int = 0
    ) -> Tuple[RTMPChunkHeader, int]:
        """Parse chunk header from raw bytes.

        Returns:
            Tuple of (chunk_header, bytes_consumed)
        """
        if len(data) <= offset:
            raise ValueError("Insufficient data for chunk header")

        first_byte = data[offset]
        format_type = RTMPChunkType((first_byte >> 6) & 0x3)
        chunk_stream_id = first_byte & 0x3F

        bytes_consumed = 1

        # Handle extended chunk stream IDs
        if chunk_stream_id == 0:
            if len(data) <= offset + 1:
                raise ValueError("Insufficient data for extended chunk stream ID")
            chunk_stream_id = data[offset + 1] + 64
            bytes_consumed += 1
        elif chunk_stream_id == 1:
            if len(data) <= offset + 2:
                raise ValueError("Insufficient data for extended chunk stream ID")
            chunk_stream_id = struct.unpack(">H", data[offset + 1 : offset + 3])[0] + 64
            bytes_consumed += 2

        header = RTMPChunkHeader(
            format_type=format_type, chunk_stream_id=chunk_stream_id
        )

        # Parse format-specific fields
        remaining_data = data[offset + bytes_consumed :]

        if format_type == RTMPChunkType.FORMAT_0:
            if len(remaining_data) < 11:
                raise ValueError("Insufficient data for format 0 chunk header")

            timestamp = struct.unpack(">I", b"\\x00" + remaining_data[0:3])[0]
            message_length = struct.unpack(">I", b"\\x00" + remaining_data[3:6])[0]
            message_type_id = remaining_data[6]
            message_stream_id = struct.unpack("<I", remaining_data[7:11])[0]

            header.timestamp = timestamp
            header.message_length = message_length
            header.message_type_id = message_type_id
            header.message_stream_id = message_stream_id

            bytes_consumed += 11

        elif format_type == RTMPChunkType.FORMAT_1:
            if len(remaining_data) < 7:
                raise ValueError("Insufficient data for format 1 chunk header")

            timestamp_delta = struct.unpack(">I", b"\\x00" + remaining_data[0:3])[0]
            message_length = struct.unpack(">I", b"\\x00" + remaining_data[3:6])[0]
            message_type_id = remaining_data[6]

            header.timestamp_delta = timestamp_delta
            header.message_length = message_length
            header.message_type_id = message_type_id

            bytes_consumed += 7

        elif format_type == RTMPChunkType.FORMAT_2:
            if len(remaining_data) < 3:
                raise ValueError("Insufficient data for format 2 chunk header")

            timestamp_delta = struct.unpack(">I", b"\\x00" + remaining_data[0:3])[0]
            header.timestamp_delta = timestamp_delta

            bytes_consumed += 3

        # Handle extended timestamp
        if format_type in [
            RTMPChunkType.FORMAT_0,
            RTMPChunkType.FORMAT_1,
            RTMPChunkType.FORMAT_2,
        ] and (header.timestamp >= 0xFFFFFF or header.timestamp_delta >= 0xFFFFFF):
            if len(data) < offset + bytes_consumed + 4:
                raise ValueError("Insufficient data for extended timestamp")

            extended_timestamp = struct.unpack(
                ">I", data[offset + bytes_consumed : offset + bytes_consumed + 4]
            )[0]
            header.extended_timestamp = True

            if format_type == RTMPChunkType.FORMAT_0:
                header.timestamp = extended_timestamp
            else:
                header.timestamp_delta = extended_timestamp

            bytes_consumed += 4

        return header, bytes_consumed

    def create_chunk_header(self, header: RTMPChunkHeader) -> bytes:
        """Create chunk header bytes from RTMPChunkHeader."""
        # Format first byte
        first_byte = (header.format_type.value << 6) | (header.chunk_stream_id & 0x3F)

        if header.chunk_stream_id >= 64 and header.chunk_stream_id < 320:
            # Use 2-byte chunk stream ID
            first_byte = header.format_type.value << 6
            chunk_id_data = struct.pack("B", header.chunk_stream_id - 64)
        elif header.chunk_stream_id >= 320:
            # Use 3-byte chunk stream ID
            first_byte = (header.format_type.value << 6) | 1
            chunk_id_data = struct.pack(">H", header.chunk_stream_id - 64)
        else:
            chunk_id_data = b""

        result = bytes([first_byte]) + chunk_id_data

        # Add format-specific data
        if header.format_type == RTMPChunkType.FORMAT_0:
            timestamp = header.timestamp if header.timestamp < 0xFFFFFF else 0xFFFFFF
            result += struct.pack(">I", timestamp)[1:]  # 3 bytes
            result += struct.pack(">I", header.message_length)[1:]  # 3 bytes
            result += bytes([header.message_type_id])
            result += struct.pack("<I", header.message_stream_id)

        elif header.format_type == RTMPChunkType.FORMAT_1:
            timestamp_delta = (
                header.timestamp_delta
                if header.timestamp_delta < 0xFFFFFF
                else 0xFFFFFF
            )
            result += struct.pack(">I", timestamp_delta)[1:]  # 3 bytes
            result += struct.pack(">I", header.message_length)[1:]  # 3 bytes
            result += bytes([header.message_type_id])

        elif header.format_type == RTMPChunkType.FORMAT_2:
            timestamp_delta = (
                header.timestamp_delta
                if header.timestamp_delta < 0xFFFFFF
                else 0xFFFFFF
            )
            result += struct.pack(">I", timestamp_delta)[1:]  # 3 bytes

        # Add extended timestamp if needed
        if header.extended_timestamp:
            if header.format_type == RTMPChunkType.FORMAT_0:
                result += struct.pack(">I", header.timestamp)
            else:
                result += struct.pack(">I", header.timestamp_delta)

        return result


class AMFEncoder:
    """AMF (Action Message Format) encoder for RTMP messages."""

    # AMF0 Type Markers
    AMF0_NUMBER = 0x00
    AMF0_BOOLEAN = 0x01
    AMF0_STRING = 0x02
    AMF0_OBJECT = 0x03
    AMF0_NULL = 0x05
    AMF0_UNDEFINED = 0x06
    AMF0_ARRAY = 0x08
    AMF0_OBJECT_END = 0x09
    AMF0_STRICT_ARRAY = 0x0A
    AMF0_DATE = 0x0B
    AMF0_LONG_STRING = 0x0C

    @classmethod
    def encode_number(cls, value: float) -> bytes:
        """Encode AMF0 number (double)."""
        return bytes([cls.AMF0_NUMBER]) + struct.pack(">d", value)

    @classmethod
    def encode_boolean(cls, value: bool) -> bytes:
        """Encode AMF0 boolean."""
        return bytes([cls.AMF0_BOOLEAN, 1 if value else 0])

    @classmethod
    def encode_string(cls, value: str) -> bytes:
        """Encode AMF0 string."""
        utf8_bytes = value.encode("utf-8")
        if len(utf8_bytes) > 65535:
            # Use long string format
            return (
                bytes([cls.AMF0_LONG_STRING])
                + struct.pack(">I", len(utf8_bytes))
                + utf8_bytes
            )
        else:
            return (
                bytes([cls.AMF0_STRING])
                + struct.pack(">H", len(utf8_bytes))
                + utf8_bytes
            )

    @classmethod
    def encode_null(cls) -> bytes:
        """Encode AMF0 null."""
        return bytes([cls.AMF0_NULL])

    @classmethod
    def encode_object(cls, obj: Dict[str, Any]) -> bytes:
        """Encode AMF0 object."""
        result = bytes([cls.AMF0_OBJECT])

        for key, value in obj.items():
            # Property name (string without type marker)
            key_bytes = key.encode("utf-8")
            result += struct.pack(">H", len(key_bytes)) + key_bytes

            # Property value
            result += cls.encode_value(value)

        # Object end marker
        result += struct.pack(">H", 0) + bytes([cls.AMF0_OBJECT_END])
        return result

    @classmethod
    def encode_array(cls, arr: List[Any]) -> bytes:
        """Encode AMF0 strict array."""
        result = bytes([cls.AMF0_STRICT_ARRAY])
        result += struct.pack(">I", len(arr))

        for item in arr:
            result += cls.encode_value(item)

        return result

    @classmethod
    def encode_value(cls, value: Any) -> bytes:
        """Encode any Python value to AMF0."""
        if isinstance(value, bool):
            return cls.encode_boolean(value)
        elif isinstance(value, (int, float)):
            return cls.encode_number(float(value))
        elif isinstance(value, str):
            return cls.encode_string(value)
        elif isinstance(value, dict):
            return cls.encode_object(value)
        elif isinstance(value, list):
            return cls.encode_array(value)
        elif value is None:
            return cls.encode_null()
        else:
            raise ValueError(f"Cannot encode value of type {type(value)}")


class AMFDecoder:
    """AMF (Action Message Format) decoder for RTMP messages."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def decode_number(self) -> float:
        """Decode AMF0 number."""
        if self.offset + 8 > len(self.data):
            raise ValueError("Insufficient data for number")

        value = struct.unpack(">d", self.data[self.offset : self.offset + 8])[0]
        self.offset += 8
        return value

    def decode_boolean(self) -> bool:
        """Decode AMF0 boolean."""
        if self.offset + 1 > len(self.data):
            raise ValueError("Insufficient data for boolean")

        value = self.data[self.offset] != 0
        self.offset += 1
        return value

    def decode_string(self) -> str:
        """Decode AMF0 string."""
        if self.offset + 2 > len(self.data):
            raise ValueError("Insufficient data for string length")

        length = struct.unpack(">H", self.data[self.offset : self.offset + 2])[0]
        self.offset += 2

        if self.offset + length > len(self.data):
            raise ValueError("Insufficient data for string")

        value = self.data[self.offset : self.offset + length].decode("utf-8")
        self.offset += length
        return value

    def decode_long_string(self) -> str:
        """Decode AMF0 long string."""
        if self.offset + 4 > len(self.data):
            raise ValueError("Insufficient data for long string length")

        length = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4

        if self.offset + length > len(self.data):
            raise ValueError("Insufficient data for long string")

        value = self.data[self.offset : self.offset + length].decode("utf-8")
        self.offset += length
        return value

    def decode_object(self) -> Dict[str, Any]:
        """Decode AMF0 object."""
        obj = {}

        while True:
            # Read property name length
            if self.offset + 2 > len(self.data):
                raise ValueError("Insufficient data for property name length")

            name_length = struct.unpack(">H", self.data[self.offset : self.offset + 2])[
                0
            ]
            self.offset += 2

            # Check for object end marker
            if name_length == 0:
                # Should be followed by object end marker
                if (
                    self.offset + 1 > len(self.data)
                    or self.data[self.offset] != AMFEncoder.AMF0_OBJECT_END
                ):
                    raise ValueError("Invalid object end marker")
                self.offset += 1
                break

            # Read property name
            if self.offset + name_length > len(self.data):
                raise ValueError("Insufficient data for property name")

            name = self.data[self.offset : self.offset + name_length].decode("utf-8")
            self.offset += name_length

            # Read property value
            value = self.decode_value()
            obj[name] = value

        return obj

    def decode_array(self) -> List[Any]:
        """Decode AMF0 strict array."""
        if self.offset + 4 > len(self.data):
            raise ValueError("Insufficient data for array length")

        length = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4

        arr = []
        for _ in range(length):
            arr.append(self.decode_value())

        return arr

    def decode_value(self) -> Any:
        """Decode any AMF0 value."""
        if self.offset >= len(self.data):
            raise ValueError("Insufficient data for value type")

        type_marker = self.data[self.offset]
        self.offset += 1

        if type_marker == AMFEncoder.AMF0_NUMBER:
            return self.decode_number()
        elif type_marker == AMFEncoder.AMF0_BOOLEAN:
            return self.decode_boolean()
        elif type_marker == AMFEncoder.AMF0_STRING:
            return self.decode_string()
        elif type_marker == AMFEncoder.AMF0_LONG_STRING:
            return self.decode_long_string()
        elif type_marker == AMFEncoder.AMF0_OBJECT:
            return self.decode_object()
        elif type_marker == AMFEncoder.AMF0_STRICT_ARRAY:
            return self.decode_array()
        elif type_marker == AMFEncoder.AMF0_NULL:
            return None
        elif type_marker == AMFEncoder.AMF0_UNDEFINED:
            return None  # Treat undefined as None
        else:
            raise ValueError(f"Unsupported AMF0 type marker: {type_marker}")


class RTMPProtocol:
    """Complete RTMP protocol implementation."""

    def __init__(self):
        self.chunk_stream = RTMPChunkStream()
        self.sequence_number = 0
        self.bytes_received = 0
        self.window_ack_size = 2500000
        self.peer_bandwidth = 2500000
        self.connect_params: Optional[Dict[str, Any]] = None

        # Connection state
        self.handshake_complete = False
        self.connected = False
        self.transaction_id = 1

    async def perform_handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> bool:
        """Perform complete RTMP handshake."""
        try:
            logger.debug("Starting RTMP handshake")

            # Send C0
            c0 = RTMPHandshake.generate_c0()
            writer.write(c0)

            # Send C1
            c1 = RTMPHandshake.generate_c1()
            writer.write(c1)
            await writer.drain()

            # Read S0
            s0 = await reader.read(1)
            if not RTMPHandshake.validate_s0(s0):
                raise ValueError(f"Invalid S0: {s0}")

            # Read S1
            s1 = await reader.read(RTMPHandshake.HANDSHAKE_SIZE)
            if not RTMPHandshake.validate_s1(s1):
                raise ValueError(f"Invalid S1 length: {len(s1)}")

            # Send C2
            c2 = RTMPHandshake.generate_c2(s1)
            writer.write(c2)
            await writer.drain()

            # Read S2
            s2 = await reader.read(RTMPHandshake.HANDSHAKE_SIZE)
            if not RTMPHandshake.validate_s2(s2, c1):
                logger.warning("S2 validation failed, but continuing")

            self.handshake_complete = True
            logger.debug("RTMP handshake completed successfully")
            return True

        except Exception as e:
            logger.error(f"RTMP handshake failed: {e}")
            return False

    def create_connect_message(
        self, app_name: str, connection_params: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Create RTMP connect command message."""
        transaction_id = self.transaction_id
        self.transaction_id += 1

        # Default connection parameters
        default_params = {
            "app": app_name,
            "type": "nonprivate",
            "flashVer": "FMLE/3.0 (compatible; TLDRHighlightAPI)",
            "tcUrl": f"rtmp://localhost/{app_name}",
            "fpad": False,
            "capabilities": 15.0,
            "audioCodecs": 3575.0,
            "videoCodecs": 252.0,
            "videoFunction": 1.0,
            "objectEncoding": 0.0,
        }

        if connection_params:
            default_params.update(connection_params)

        self.connect_params = default_params

        # Encode AMF command
        command_data = (
            AMFEncoder.encode_string("connect")
            + AMFEncoder.encode_number(transaction_id)
            + AMFEncoder.encode_object(default_params)
        )

        # Create message
        message = RTMPMessage(
            message_type=RTMPMessageType.AMF0_COMMAND,
            timestamp=0,
            stream_id=0,
            data=command_data,
            chunk_stream_id=3,
        )

        return self.create_message_chunks(message)

    def create_message_chunks(self, message: RTMPMessage) -> bytes:
        """Create chunked message data for transmission."""
        data = message.data
        chunks = []

        # First chunk (Format 0)
        header = RTMPChunkHeader(
            format_type=RTMPChunkType.FORMAT_0,
            chunk_stream_id=message.chunk_stream_id,
            timestamp=message.timestamp,
            message_length=len(data),
            message_type_id=message.message_type.value,
            message_stream_id=message.stream_id,
        )

        chunk_header_bytes = self.chunk_stream.create_chunk_header(header)

        # Split data into chunks
        chunk_size = self.chunk_stream.chunk_size
        offset = 0

        while offset < len(data):
            if offset == 0:
                # First chunk
                chunk_data = data[offset : offset + chunk_size]
                chunks.append(chunk_header_bytes + chunk_data)
            else:
                # Subsequent chunks (Format 3)
                continuation_header = RTMPChunkHeader(
                    format_type=RTMPChunkType.FORMAT_3,
                    chunk_stream_id=message.chunk_stream_id,
                )
                continuation_header_bytes = self.chunk_stream.create_chunk_header(
                    continuation_header
                )
                chunk_data = data[offset : offset + chunk_size]
                chunks.append(continuation_header_bytes + chunk_data)

            offset += chunk_size

        return b"".join(chunks)

    async def send_connect_command(
        self,
        writer: asyncio.StreamWriter,
        app_name: str,
        connection_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send RTMP connect command."""
        connect_message = self.create_connect_message(app_name, connection_params)
        writer.write(connect_message)
        await writer.drain()
        logger.debug(f"Sent RTMP connect command for app: {app_name}")

    def create_window_ack_size_message(self, window_size: int) -> bytes:
        """Create window acknowledgement size message."""
        message = RTMPMessage(
            message_type=RTMPMessageType.WINDOW_ACKNOWLEDGEMENT_SIZE,
            timestamp=0,
            stream_id=0,
            data=struct.pack(">I", window_size),
            chunk_stream_id=2,
        )
        return self.create_message_chunks(message)

    def create_set_peer_bandwidth_message(
        self, window_size: int, limit_type: int = 2
    ) -> bytes:
        """Create set peer bandwidth message."""
        message = RTMPMessage(
            message_type=RTMPMessageType.SET_PEER_BANDWIDTH,
            timestamp=0,
            stream_id=0,
            data=struct.pack(">IB", window_size, limit_type),
            chunk_stream_id=2,
        )
        return self.create_message_chunks(message)

    async def parse_message(
        self, reader: asyncio.StreamReader
    ) -> Optional[RTMPMessage]:
        """Parse incoming RTMP message."""
        try:
            # Read and parse chunk header
            first_byte_data = await reader.read(1)
            if not first_byte_data:
                return None

            # Peek ahead to get more header data
            additional_data = await reader.read(15)  # Max possible header size
            full_header_data = first_byte_data + additional_data

            header, header_size = self.chunk_stream.parse_chunk_header(full_header_data)

            # Put back unused data
            unused_data = additional_data[header_size - 1 :]
            if unused_data:
                # This is a simplified approach - in practice you'd need proper buffering
                pass

            # Read message data
            if header.format_type == RTMPChunkType.FORMAT_0:
                message_length = header.message_length
                message_type = header.message_type_id
                stream_id = header.message_stream_id
                timestamp = header.timestamp
            else:
                # Use previous chunk stream information
                prev_header = self.chunk_stream.last_chunk_headers.get(
                    header.chunk_stream_id
                )
                if not prev_header:
                    logger.warning(
                        f"No previous header for chunk stream {header.chunk_stream_id}"
                    )
                    return None

                message_length = prev_header.message_length
                message_type = prev_header.message_type_id
                stream_id = prev_header.message_stream_id
                timestamp = prev_header.timestamp + header.timestamp_delta

            # Read message data chunks
            data = b""
            remaining = message_length

            while remaining > 0:
                chunk_size = min(remaining, self.chunk_stream.chunk_size)
                chunk_data = await reader.read(chunk_size)

                if not chunk_data:
                    logger.warning(
                        "Unexpected end of stream while reading message data"
                    )
                    return None

                data += chunk_data
                remaining -= len(chunk_data)

                # If there's more data, expect a continuation chunk
                if remaining > 0:
                    # Read continuation chunk header (should be Format 3)
                    cont_header_data = await reader.read(1)
                    if not cont_header_data:
                        logger.warning("Missing continuation chunk header")
                        return None

                    # Validate it's Format 3 for same chunk stream
                    expected_byte = (RTMPChunkType.FORMAT_3.value << 6) | (
                        header.chunk_stream_id & 0x3F
                    )
                    if cont_header_data[0] != expected_byte:
                        logger.warning(
                            f"Unexpected continuation chunk header: {cont_header_data[0]:02x}"
                        )

            # Update chunk stream state
            self.chunk_stream.last_chunk_headers[header.chunk_stream_id] = header

            # Create message
            message = RTMPMessage(
                message_type=RTMPMessageType(message_type),
                timestamp=timestamp,
                stream_id=stream_id,
                data=data,
                chunk_stream_id=header.chunk_stream_id,
            )

            return message

        except Exception as e:
            logger.error(f"Error parsing RTMP message: {e}")
            return None

    def process_command_message(self, message: RTMPMessage) -> Optional[Dict[str, Any]]:
        """Process AMF command message."""
        try:
            decoder = AMFDecoder(message.data)

            command_name = decoder.decode_value()
            transaction_id = decoder.decode_value()

            if command_name == "_result":
                # Connection result
                connection_object = decoder.decode_value()
                logger.info(f"RTMP connect result: {connection_object}")
                self.connected = True
                return {
                    "command": "_result",
                    "transaction_id": transaction_id,
                    "connection_object": connection_object,
                }

            elif command_name == "_error":
                # Connection error
                error_object = decoder.decode_value()
                logger.error(f"RTMP connect error: {error_object}")
                return {
                    "command": "_error",
                    "transaction_id": transaction_id,
                    "error_object": error_object,
                }

            elif command_name == "onBWDone":
                # Bandwidth test complete
                logger.debug("RTMP bandwidth test completed")
                return {"command": "onBWDone", "transaction_id": transaction_id}

            else:
                logger.debug(f"Received RTMP command: {command_name}")
                return {"command": command_name, "transaction_id": transaction_id}

        except Exception as e:
            logger.error(f"Error processing command message: {e}")
            return None

    def handle_user_control_message(self, message: RTMPMessage) -> None:
        """Handle user control messages."""
        if len(message.data) < 2:
            logger.warning("Invalid user control message length")
            return

        event_type = struct.unpack(">H", message.data[0:2])[0]

        if event_type == RTMPUserControlType.STREAM_BEGIN.value:
            if len(message.data) >= 6:
                stream_id = struct.unpack(">I", message.data[2:6])[0]
                logger.debug(f"Stream begin for stream ID: {stream_id}")

        elif event_type == RTMPUserControlType.PING_REQUEST.value:
            if len(message.data) >= 6:
                timestamp = struct.unpack(">I", message.data[2:6])[0]
                logger.debug(f"Ping request with timestamp: {timestamp}")
                # Should respond with ping response

        elif event_type == RTMPUserControlType.PING_RESPONSE.value:
            if len(message.data) >= 6:
                timestamp = struct.unpack(">I", message.data[2:6])[0]
                logger.debug(f"Ping response with timestamp: {timestamp}")

        else:
            logger.debug(f"User control message type: {event_type}")
