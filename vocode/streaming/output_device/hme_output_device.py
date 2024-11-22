import asyncio
from typing import Optional

import websockets
from loguru import logger
from regex import F

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState


class HMEOutputDevice(AbstractOutputDevice):
    def __init__(
        self,
        sampling_rate: int = 16000,
        audio_encoding: AudioEncoding = AudioEncoding.LINEAR16,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._is_terminated = False
        self._input_queue: asyncio.Queue = asyncio.Queue()
        self.interruptible_event = None

    async def initialize(self, websocket: websockets.WebSocketClientProtocol) -> None:
        """Initialize the output device with a websocket connection"""
        logger.info("HMEOutputDevice: Initializing")
        self.websocket = websocket
        self._is_terminated = False
        logger.info("HMEOutputDevice: Starting worker")
        await self.start()
        logger.info("HMEOutputDevice: Initialization complete")

    async def _run_loop(self):
        """Process audio chunks from the queue"""
        logger.info("HMEOutputDevice: Starting run loop")
        while not self._is_terminated:
            try:
                item = await self._input_queue.get()

                self.interruptible_event = item
                audio_chunk = item.payload

                if item.is_interrupted():
                    audio_chunk.on_interrupt()
                    audio_chunk.state = ChunkState.INTERRUPTED
                    continue

                if not self.websocket or not self.websocket.open:
                    # Log once and mark chunk as failed
                    logger.error("HMEOutputDevice: WebSocket is not open, dropping audio chunk")
                    audio_chunk.state = ChunkState.INTERRUPTED
                    self._is_terminated = True  # Stop processing more chunks
                    break  # Exit the loop

                try:
                    await self.websocket.send(audio_chunk.data)
                    if hasattr(audio_chunk, "on_play"):
                        audio_chunk.on_play()
                    audio_chunk.state = ChunkState.PLAYED
                except Exception as e:
                    logger.error(f"HMEOutputDevice: Error sending audio data: {e}")
                    audio_chunk.state = ChunkState.INTERRUPTED
                    self._is_terminated = True
                    break

                self.interruptible_event.is_interruptible = False

            except asyncio.CancelledError:
                logger.info("HMEOutputDevice: Run loop cancelled")
                break
            except Exception as e:
                logger.error(f"HMEOutputDevice: Error in run loop: {e}")
                self._is_terminated = True
                break

        logger.info("HMEOutputDevice: Run loop terminated")

    async def play(self, audio_chunk: AudioChunk) -> None:
        """Queue audio chunk for sending"""
        if not self._is_terminated:
            await self._input_queue.put(audio_chunk)

    async def terminate(self) -> None:
        """Clean up resources"""
        logger.info("HMEOutputDevice: Starting termination")
        
        # Set terminated flag to stop processing
        self._is_terminated = True
        
        # Clear the input queue
        while not self._input_queue.empty():
            try:
                item = self._input_queue.get_nowait()
                if hasattr(item.payload, "on_interrupt"):
                    item.payload.on_interrupt()
            except asyncio.QueueEmpty:
                break

        # Clear references
        self.websocket = None
        
        logger.info("HMEOutputDevice: Termination complete")

    def interrupt(self) -> None:
        """Handle interruption of audio output"""
        self._is_terminated = True
