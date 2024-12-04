import asyncio
import zlib
from typing import Optional

from loguru import logger
from websockets.asyncio.client import ClientConnection

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState
from vocode.streaming.telephony.constants import DEFAULT_AUDIO_ENCODING, DEFAULT_SAMPLING_RATE


class HMEOutputDevice(AbstractOutputDevice):
    """Handles audio output for HME/NEXEO system integration.

    Formats and sends audio messages according to the NEXEO protocol specification:
    - 4 bytes CRC32 of audio data
    - 1 byte target lane
    - 11 bytes header padding
    - Audio data in LINEAR16 PCM format
    """

    def __init__(
        self,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        audio_encoding: AudioEncoding = DEFAULT_AUDIO_ENCODING,
        audio_mode: str = "Fixed",  # "Fixed" or "Variable"
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.audio_websocket: Optional[ClientConnection] = None
        self.audio_mode = audio_mode
        # Calculate bytes per 30ms chunk for Fixed mode
        self.fixed_chunk_size = int((sampling_rate * 2 * 0.03))  # 2 bytes per sample for LINEAR16
        logger.info(
            f"Initialized HME output device - Mode: {audio_mode}, Sampling Rate: {sampling_rate}, "
            f"Encoding: {audio_encoding}, Chunk Size: {self.fixed_chunk_size} bytes"
        )

    async def _run_loop(self):
        """Process audio chunks from the input queue and send them to the NEXEO system."""
        try:
            while True:
                try:
                    item = await self._input_queue.get()
                except asyncio.CancelledError:
                    logger.info("[HME] Audio output loop cancelled")
                    return

                self.interruptible_event = item
                audio_chunk = item.payload

                if item.is_interrupted():
                    logger.info(f"[HME] Audio chunk interrupted - ID: {id(audio_chunk)}")
                    audio_chunk.on_interrupt()
                    audio_chunk.state = ChunkState.INTERRUPTED
                    continue

                logger.debug(
                    f"[HME] Processing chunk: {id(audio_chunk)}, {len(audio_chunk.data)} bytes"
                )
                await self.play(audio_chunk.data)

                if hasattr(audio_chunk, "on_play"):
                    try:
                        audio_chunk.on_play()
                    except Exception as e:
                        logger.error(
                            f"[HME] Error calling on_play for chunk {id(audio_chunk)}: {e}"
                        )

                audio_chunk.state = ChunkState.PLAYED
                self.interruptible_event.is_interruptible = False

        except Exception as e:
            logger.error(f"[HME] Audio loop failed: {str(e)}")

    def initialize_audio(self, websocket: ClientConnection):
        """Initialize the WebSocket connection for audio streaming."""
        logger.info(f"Initializing audio WebSocket connection - Remote: {websocket.remote_address}")
        self.audio_websocket = websocket

    async def play(self, audio_data: bytes):
        """Send an audio message to the NEXEO system.

        Format:
        - 4 bytes: CRC32 of audio data
        - 1 byte: target lane (1 for single lane)
        - 11 bytes: header padding
        - Remaining bytes: LINEAR16 PCM audio data

        Args:
            audio_data: Raw PCM audio bytes to send
        """
        if not self.audio_websocket:
            logger.error("[HME] Cannot send audio - WebSocket not initialized")
            return

        try:
            if self.audio_mode == "Fixed":
                # Split audio into 30ms chunks
                for i in range(0, len(audio_data), self.fixed_chunk_size):
                    chunk = audio_data[i : i + self.fixed_chunk_size]
                    if len(chunk) < self.fixed_chunk_size:
                        # Pad last chunk if needed
                        chunk = chunk.ljust(self.fixed_chunk_size, b"\x00")

                    await self._send_chunk(chunk)
                    # Wait 30ms before sending next chunk
                    await asyncio.sleep(0.03)
            else:
                # Variable mode - send as is
                await self._send_chunk(audio_data)

        except Exception as e:
            logger.error(
                f"[HME] Error sending audio message: {str(e)}, Audio size: {len(audio_data)} bytes"
            )
            raise

    async def _send_chunk(self, audio_data: bytes):
        """Helper method to send a single chunk with proper HME formatting

        Format:
        - 4 bytes: CRC32 of audio data (big endian)
        - 1 byte: target lane (1 for single lane)
        - 11 bytes: header padding
        - Remaining bytes: LINEAR16 PCM audio data
        """
        # Calculate CRC32 of audio data only
        crc = zlib.crc32(audio_data) & 0xFFFFFFFF
        crc_bytes = crc.to_bytes(4, byteorder="big")  # Match incoming big-endian format

        # Single lane (1) for now per minimal implementation
        target_lane = 1
        lane_byte = target_lane.to_bytes(1, byteorder="big")

        # 11 bytes of zero padding
        padding = b"\x00" * 11

        # Assemble message in same format as incoming
        message = crc_bytes + lane_byte + padding + audio_data

        logger.debug(
            f"[HME] Sending audio chunk: Size={len(audio_data)} bytes, "
            f"Lane={target_lane}, CRC={hex(crc)}"
        )
        await self.audio_websocket.send(message)

    async def terminate(self):
        """Clean up resources and close the WebSocket connection."""
        logger.info("Terminating HME output device")
        if self.audio_websocket:
            try:
                logger.info(
                    f"Closing audio WebSocket connection to {self.audio_websocket.remote_address}"
                )
                await self.audio_websocket.close()
            except Exception as e:
                logger.error(f"Error closing audio WebSocket: {str(e)}")
        await super().terminate()

    def interrupt(self):
        pass
