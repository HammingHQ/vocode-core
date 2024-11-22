import asyncio
from typing import Optional
from asyncio import Task

import websockets
from loguru import logger

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.utils.worker import InterruptibleEvent


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
        self.websocket = websocket
        self._is_terminated = False
        await self.start()

    async def _run_loop(self):
        while not self._is_terminated:
            try:
                item = await self._input_queue.get()
                
                self.interruptible_event = item
                audio_chunk = item.payload

                if item.is_interrupted():
                    audio_chunk.on_interrupt()
                    audio_chunk.state = ChunkState.INTERRUPTED
                    continue

                if self.websocket and self.websocket.open:
                    await self.websocket.send(bytes(audio_chunk.data))
                    if hasattr(audio_chunk, "on_play"):
                        audio_chunk.on_play()
                    audio_chunk.state = ChunkState.PLAYED
                    self.interruptible_event.is_interruptible = False
                else:
                    audio_chunk.state = ChunkState.INTERRUPTED
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                break

    def interrupt(self):
        """Handle interruption by setting the current event as interrupted"""
        if self.interruptible_event and self.interruptible_event.is_interruptible:
            self.interruptible_event.interrupt()

    async def play(self, audio_chunk: AudioChunk) -> None:
        if not self._is_terminated and audio_chunk.data:
            event = InterruptibleEvent(payload=audio_chunk)
            await self._input_queue.put(event)

    async def terminate(self) -> None:
        self._is_terminated = True
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        await super().terminate()
