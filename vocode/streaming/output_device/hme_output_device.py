import asyncio
from typing import Optional

from loguru import logger
from websockets.asyncio.client import ClientConnection

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_SAMPLING_RATE,
    PCM_SILENCE_BYTE,
)
from vocode.streaming.utils.create_task import asyncio_create_task


class HMEError(Exception):
    """Base exception for HME-related errors"""

    pass


class HMEConnectionError(HMEError):
    """Raised when WebSocket connection fails"""

    pass


class HMEOutputDevice(AbstractOutputDevice):
    """Handles audio output for HME/NEXEO system integration."""

    def __init__(
        self,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        audio_encoding: AudioEncoding = DEFAULT_AUDIO_ENCODING,
        audio_mode: str = "Fixed",
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.audio_websocket: Optional[ClientConnection] = None
        self.hme_chunk_size = 1920  # Total size for single lane (interleaved stereo)
        self.chunk_interval = 0.03  # 30ms gap between chunks
        self._audio_task: Optional[asyncio.Task] = None

        logger.info(
            f"Initialized HME output device - Mode: {audio_mode}, Sampling Rate: {sampling_rate}, "
            f"Encoding: {audio_encoding}, Chunk Size: {self.hme_chunk_size} bytes"
        )

    async def __aenter__(self):
        """Initialize resources when entering context"""
        logger.info("Entering HME output device context")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources when exiting context"""
        await self.terminate()

    async def _convert_mono_to_stereo(self, mono_data: bytes) -> bytes:
        """Convert mono audio to stereo format"""
        stereo_chunk = bytearray()
        for i in range(0, len(mono_data), 2):
            sample = mono_data[i : i + 2]
            stereo_chunk.extend(sample * 2)  # Duplicate for both channels
        return bytes(stereo_chunk)

    async def _create_silence(self) -> bytes:
        """Create a stereo silence chunk"""
        return PCM_SILENCE_BYTE * self.hme_chunk_size

    async def _send_chunk(self, chunk: bytes):
        """Send audio chunk with proper timing"""
        if not self.audio_websocket:
            raise HMEConnectionError("No WebSocket connection available")

        try:
            await self.audio_websocket.send(chunk)
            await asyncio.sleep(self.chunk_interval)
        except Exception as e:
            raise HMEError(f"Failed to send audio chunk: {str(e)}") from e

    def _get_audio_data(self, item: any) -> bytes:
        """Extract audio data from various input types"""
        if hasattr(item, "payload"):
            audio_chunk = item.payload
            return audio_chunk.data if hasattr(audio_chunk, "data") else audio_chunk
        return item

    async def _run_loop(self):
        """Process audio chunks with proper timing."""
        logger.info("[HME] Starting audio output loop")
        try:
            while True:
                item = await self._input_queue.get()
                if item is None:
                    break

                if item:
                    # Process actual audio data
                    audio_data = self._get_audio_data(item)
                    stereo_audio = await self._convert_mono_to_stereo(audio_data)
                    if len(stereo_audio) == self.hme_chunk_size:
                        await self._send_chunk(stereo_audio)
                        logger.debug(f"[HME] Sent audio chunk of size: {len(stereo_audio)} bytes")
                    else:
                        logger.warning(f"[HME] Invalid chunk size: {len(stereo_audio)} bytes")
                else:
                    # Send silence when no audio data
                    await self._send_chunk(await self._create_silence())

        except asyncio.CancelledError:
            logger.info("[HME] Audio output loop cancelled")
            raise
        except Exception as e:
            logger.error(f"[HME] Error in output loop: {e}")
            raise

    def initialize_audio(self, websocket: ClientConnection):
        """Initialize the WebSocket connection for audio streaming."""
        logger.info(f"Initializing audio WebSocket connection - Remote: {websocket.remote_address}")
        self.audio_websocket = websocket
        self._audio_task = asyncio_create_task(self._run_loop())

    async def play(self, audio_data: bytes):
        """Queue audio data for processing."""
        await self._input_queue.put(audio_data)

    async def terminate(self):
        """Clean up resources and close the WebSocket connection."""
        logger.info("Terminating HME output device")
        if self._audio_task:
            self._audio_task.cancel()
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
        """Handle interruption of audio output"""
        if self._audio_task:
            self._audio_task.cancel()
