import asyncio
import logging
from io import BytesIO
from typing import Optional

from loguru import logger
from websockets.sync.client import ClientConnection

from vocode.streaming.hme.constants import AUDIO_ENCODING, DEFAULT_SAMPLING_RATE
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState

NUM_CHANNELS = 1


class HMEOutputDevice(AbstractOutputDevice):
    audio_websocket: ClientConnection

    def __init__(
        self,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        audio_encoding: AudioEncoding = AUDIO_ENCODING,
    ):
        super().__init__(sampling_rate, audio_encoding)

    async def _run_loop(self):
        while True:
            try:
                item = await self._input_queue.get()
            except asyncio.CancelledError:
                return

            self.interruptible_event = item
            audio_chunk = item.payload

            if item.is_interrupted():
                audio_chunk.on_interrupt()
                audio_chunk.state = ChunkState.INTERRUPTED
                continue

            await self.play(audio_chunk.data)
            if hasattr(audio_chunk, "on_play"):
                try:
                    audio_chunk.on_play()
                except Exception as e:
                    logging.error(f"Error calling on_play: {e}")
            audio_chunk.state = ChunkState.PLAYED
            self.interruptible_event.is_interruptible = False

    def initialize_audio(self, websocket: ClientConnection):
        self.audio_websocket = websocket

    async def play(self, item: bytes):
        with open("original.pcm", "ab") as f:
            f.write(item)

        buffer = BytesIO()
        for i in range(0, len(item), 2):
            buffer.write(item[i:i+2])
            buffer.write(b'\x00\x00')
        interleaved_bytes = buffer.getvalue()
        
        logger.info(f"Sending audio message of length {len(interleaved_bytes)} bytes")
        with open("inter.pcm", "ab") as f:
            f.write(interleaved_bytes)
        await self.audio_websocket.send(interleaved_bytes)

    def interrupt(self):
        pass
