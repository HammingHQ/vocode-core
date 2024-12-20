import asyncio
import logging
from io import BytesIO
from typing import Optional

import sounddevice as sd
from loguru import logger
from websockets.sync.client import ClientConnection

from vocode.streaming.hme.constants import AUDIO_ENCODING, DEFAULT_SAMPLING_RATE
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState
from vocode.streaming.output_device.blocking_speaker_output import BlockingSpeakerOutput
from vocode.streaming.telephony.constants import PCM_SILENCE_BYTE

NUM_CHANNELS = 1
OUTPUT_CHUNK_SIZE = 1920


class HMEOutputDevice(AbstractOutputDevice):
    audio_websocket: ClientConnection

    def __init__(
        self,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        audio_encoding: AudioEncoding = AUDIO_ENCODING,
    ):
        super().__init__(sampling_rate, audio_encoding)
        device_info = sd.query_devices(kind="output")
        self.speaker_output = BlockingSpeakerOutput(
            device_info=device_info,
            sampling_rate=DEFAULT_SAMPLING_RATE,
            audio_encoding=self.audio_encoding,
        )

    def start(self) -> asyncio.Task:
        base_task = super().start()
        local_audio_task = self.speaker_output.start()
        return asyncio.gather(base_task, local_audio_task)

    async def terminate(self):
        await self.speaker_output.terminate()
        await super().terminate()

    async def _run_loop(self):
        logger.info("Starting HME output device loop...")
        while True:
            try:
                item = await asyncio.wait_for(self._input_queue.get(), 0.03)
            except asyncio.exceptions.TimeoutError:
                logger.info("Timeout error, playing silence")
                await self.play_silence()
                continue
            except asyncio.CancelledError:
                return

            self.interruptible_event = item
            audio_chunk = item.payload

            if item.is_interrupted():
                audio_chunk.on_interrupt()
                audio_chunk.state = ChunkState.INTERRUPTED
                continue

            await self.play(audio_chunk.data)
            asyncio.create_task(self.speaker_output.play(audio_chunk.data))

            if hasattr(audio_chunk, "on_play"):
                try:
                    audio_chunk.on_play()
                except Exception as e:
                    logging.error(f"Error calling on_play: {e}")
            audio_chunk.state = ChunkState.PLAYED
            self.interruptible_event.is_interruptible = False

    def initialize_audio(self, websocket: ClientConnection):
        self.audio_websocket = websocket

    async def play_silence(self):
        white_noise_bytes = bytes([
            0x3A, 0xD4,
            0x9F, 0x10,
            0x72, 0xFB,
            0xC6, 0x03,
            0xA7, 0x5E,
            0x01, 0x9C,
            0xBB, 0x77,
            0x40, 0xC1,
            0xF5, 0x2D,
            0x0E, 0xA2
        ])
        # silence_bytes = PCM_SILENCE_BYTE * OUTPUT_CHUNK_SIZE
        await self.audio_websocket.send(white_noise_bytes * 96)

    async def play(self, item: bytes):
        buffer = BytesIO()
        for i in range(0, len(item), 2):
            buffer.write(item[i : i + 2])
            buffer.write(b"\x00\x00")
        if len(buffer.getvalue()) < OUTPUT_CHUNK_SIZE:
            padding_size = OUTPUT_CHUNK_SIZE - len(buffer.getvalue())
            buffer.write(b"\x00" * padding_size)
        interleaved_bytes = buffer.getvalue()
        await self.audio_websocket.send(interleaved_bytes)

    def interrupt(self):
        pass
