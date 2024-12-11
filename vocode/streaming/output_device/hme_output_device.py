import asyncio
import zlib
from typing import Optional

import sounddevice as sd
from loguru import logger
from websockets.asyncio.client import ClientConnection

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.output_device.blocking_speaker_output import BlockingSpeakerOutput
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_SAMPLING_RATE,
    PCM_SILENCE_BYTE,
)
from vocode.streaming.utils.create_task import asyncio_create_task
from vocode.streaming.utils.worker import InterruptibleEvent


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
        enable_local_playback: bool = True,
        conversation: Optional[StreamingConversation] = None,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.audio_websocket: Optional[ClientConnection] = None
        self.audio_websocket_lock = asyncio.Lock()
        self.hme_chunk_size = 1920
        self.chunk_interval = 0.03
        self._audio_task: Optional[asyncio.Task] = None
        self._ensure_audio_task_lock = asyncio.Lock()
        self.enable_local_playback = enable_local_playback
        self.speaker_output = None
        self._ready = asyncio.Event()
        self.conversation = conversation

        if enable_local_playback:
            try:
                device_info = sd.query_devices(kind="output")
                logger.info(f"Found audio output device: {device_info}")
                if device_info is None:
                    logger.warning("No output device found, disabling local playback")
                    self.enable_local_playback = False
                else:
                    self.speaker_output = BlockingSpeakerOutput(
                        device_info=device_info,
                        sampling_rate=24000,
                        audio_encoding=self.audio_encoding,
                    )
                    self.speaker_output.start()
                    logger.info(
                        f"Speaker output initialized and started successfully at {self.sampling_rate}Hz"
                    )
            except Exception as e:
                logger.error(f"Failed to initialize speaker output: {e}")
                self.enable_local_playback = False

        logger.info(
            f"Initialized HME output device - Mode: {audio_mode}, Sampling Rate: {sampling_rate}, "
            f"Encoding: {audio_encoding}, Chunk Size: {self.hme_chunk_size} bytes"
        )

    async def __aenter__(self) -> "HMEOutputDevice":
        """Initialize resources when entering context"""
        logger.info("Entering HME output device context")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Exception],
        exc_val: Optional[Exception],
        exc_tb: Optional[Exception],
    ) -> None:
        """Cleanup resources when exiting context"""
        await self.terminate()

    async def _convert_mono_to_stereo(self, mono_data: bytes) -> bytes:
        """Convert mono audio to stereo format."""
        try:
            if not mono_data:
                logger.warning("Received empty mono data")
                return b""

            stereo_chunk = bytearray()
            for i in range(0, len(mono_data), 2):
                sample = mono_data[i : i + 2]
                stereo_chunk.extend(sample * 2)  # Duplicate for both channels
            return bytes(stereo_chunk)
        except Exception as e:
            logger.error(f"Error converting mono to stereo: {e}")
            return b""

    async def _create_silence(self) -> bytes:
        """Create a stereo silence chunk"""
        return PCM_SILENCE_BYTE * self.hme_chunk_size

    async def _send_chunk(self, audio_data: bytes) -> None:
        """Send audio chunk with lock protection"""
        try:
            async with self.audio_websocket_lock:
                if self.audio_websocket:
                    await self.audio_websocket.send(audio_data)
                else:
                    logger.warning("No audio WebSocket connection, skipping send")
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
            if "code = 1000" not in str(e):
                raise

    def _get_audio_data(self, item: InterruptibleEvent[AudioChunk]) -> bytes:
        """Extract audio data from queue item and pad with silence if needed."""
        try:
            audio_data = b""
            if hasattr(item, "payload"):
                if hasattr(item.payload, "data"):
                    audio_data = item.payload.data
                else:
                    logger.debug(f"No data attribute on payload, type: {type(item.payload)}")
                    audio_data = item.payload
            else:
                logger.debug(f"No payload attribute found, using item directly, type: {type(item)}")
                audio_data = item

            # Pad with silence if needed
            if len(audio_data) < self.hme_chunk_size // 2:  # Divide by 2 since input is mono
                padding_size = (self.hme_chunk_size // 2) - len(audio_data)
                audio_data += PCM_SILENCE_BYTE * padding_size
                logger.debug(f"Padded {padding_size} bytes of silence to reach chunk size")

            return audio_data
        except Exception as e:
            logger.error(f"Error extracting audio data: {e}")
            return PCM_SILENCE_BYTE * (self.hme_chunk_size // 2)

    async def _run_loop(self) -> None:
        logger.info("[HME] Starting audio output loop")
        try:
            self._ready.set()
            self.is_speaking: bool = False
            self._pending_interrupt: bool = False
            while True:
                try:
                    # Attempt to get a chunk within the specified interval
                    item: InterruptibleEvent[AudioChunk] = await asyncio.wait_for(
                        self._input_queue.get(),
                        timeout=self.chunk_interval,
                    )

                    if item.is_interrupted():
                        self._interrupt_audio_chunk(item)
                        continue

                    mono_audio: bytes = self._get_audio_data(item)
                    is_real_chunk: bool = not self._is_silence(mono_audio)

                    # if we got a real chunk, we're speaking
                    if is_real_chunk:
                        self.is_speaking = True
                        self.conversation.mark_last_action_timestamp()

                    # Create tasks for concurrent audio output
                    tasks = []

                    # Add speaker output task if enabled
                    if (
                        self.enable_local_playback
                        and self.speaker_output
                        and not item.is_interrupted()
                    ):
                        tasks.append(self.speaker_output.play(mono_audio))

                    # Add HME output task if not interrupted
                    if not item.is_interrupted():
                        stereo_audio = await self._convert_mono_to_stereo(mono_audio)
                        tasks.append(self._send_chunk(stereo_audio))
                        item.payload.on_play()

                    # Run tasks concurrently if any exist
                    if tasks:
                        await asyncio.gather(*tasks)

                except asyncio.TimeoutError:
                    # No item retrieved within the chunk interval, means gap in speech
                    # Send a single silence chunk
                    stereo_audio = await self._create_silence()
                    await self._send_chunk(stereo_audio)
                    # We are now listening since no real chunk has arrived in time
                    self.is_speaking = False

                    if self._pending_interrupt:
                        logger.debug("[HME] Interrupting and draining queue because we're speaking")
                        self._drain_audio_chunk_queue(self._input_queue)
                        self._pending_interrupt = False

                await asyncio.sleep(self.chunk_interval)

        except asyncio.CancelledError:
            logger.info("[HME] Audio output loop cancelled")
            raise
        except Exception as e:
            logger.error(f"[HME] Error in output loop: {e}")
            raise

    def _interrupt_audio_chunk(self, item: InterruptibleEvent[AudioChunk]) -> None:
        """Handle interruption of a single audio chunk."""
        audio_chunk = item.payload
        audio_chunk.on_interrupt()
        audio_chunk.state = ChunkState.INTERRUPTED
        logger.debug("[HME] Interrupted audio chunk")

    def _drain_audio_chunk_queue(
        self, queue: asyncio.Queue[InterruptibleEvent[AudioChunk]]
    ) -> None:
        """Clear all audio chunks from queue, marking them as interrupted."""
        items_cleared = 0
        while True:
            try:
                item = queue.get_nowait()
                self._interrupt_audio_chunk(item)
                items_cleared += 1
            except asyncio.QueueEmpty:
                break
        logger.debug(f"[HME] Cleared {items_cleared} audio chunks from queue")

    def initialize_audio(self, websocket: ClientConnection) -> None:
        """Initialize the WebSocket connection for audio streaming."""
        logger.info(f"Initializing audio WebSocket connection - Remote: {websocket.remote_address}")
        self.audio_websocket = websocket
        asyncio_create_task(self._ensure_audio_task_running())
        self._audio_task = asyncio_create_task(self._run_loop())

    async def terminate(self) -> None:
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
        if self.speaker_output:
            await self.speaker_output.terminate()
        await super().terminate()

    def interrupt(self) -> None:
        logger.debug("[HME] Interrupt called")
        # Only interrupt (drain) if we're currently 'speaking' (has played at least one real chunk)
        # and have queued chunks to drain
        if self._input_queue.empty():
            logger.debug("[HME] Interrupt called but no queued chunks, ignoring")
            return

        if self.is_speaking:
            logger.debug("[HME] Interrupting and draining queue because we're speaking")
            self._drain_audio_chunk_queue(self._input_queue)
            self._pending_interrupt = True
        else:
            logger.debug("[HME] Currently silent, draining queue")
            # TODO (@orban) - do we need to drain the queue here?
            self._drain_audio_chunk_queue(self._input_queue)

    async def _ensure_audio_task_running(self) -> None:
        """Ensure the audio task is running, restart if needed."""
        async with self._ensure_audio_task_lock:
            if not self._audio_task or self._audio_task.done():
                logger.info("[HME] Starting/restarting audio output loop")
                self._audio_task = asyncio_create_task(self._run_loop())

    def _is_silence(self, mono_audio: bytes) -> bool:
        # If the data is entirely silence bytes, consider it silence
        return all(b == PCM_SILENCE_BYTE[0] for b in mono_audio)

    async def consume(self, item: AudioChunk) -> None:
        """Process incoming audio chunks."""
        logger.debug(
            f"[HME] Consuming audio chunk, is_speaking={self.is_speaking}, queue_size={self._input_queue.qsize()}"
        )
        await self._input_queue.put(item)

    async def _validate_audio_frame(self, message: bytes) -> Optional[bytes]:
        if len(message) < 17:
            logger.warning(f"Audio frame too small: {len(message)} bytes")
            return None

        # Extract frame components
        crc = message[:4]
        audio_bytes = message[16:]  # Skip 16 byte header

        # Validate CRC checksum
        computed_crc = zlib.crc32(audio_bytes) & 0xFFFFFFFF
        received_crc = int.from_bytes(crc, byteorder="big")

        if computed_crc != received_crc:
            logger.warning(
                "CRC validation failed",
                frame_size=len(message),
                computed_crc=hex(computed_crc),
                received_crc=hex(received_crc),
            )
            return None

        return audio_bytes

    async def wait_for_ready(self) -> None:
        await self._ready.wait()
