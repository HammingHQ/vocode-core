import asyncio
import queue
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger


from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.rate_limit_interruptions_output_device import (
    RateLimitInterruptionsOutputDevice,
)
from vocode.streaming.utils.worker import ThreadAsyncWorker

DEFAULT_SAMPLING_RATE = 44100


class _PlaybackWorker(ThreadAsyncWorker[bytes]):

    def __init__(self, *, device_info: dict, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.device_info = device_info
        super().__init__()
        logger.info(
            f"Initializing audio output stream with: sampling_rate={sampling_rate}, device={device_info['name']}"
        )
        try:
            self.stream = sd.OutputStream(
                channels=1,
                samplerate=self.sampling_rate,
                dtype=np.int16,
                device=int(self.device_info["index"]),
            )
            self._ended = False
            # Send initial silence to verify stream is working
            silence = self.sampling_rate * b"\x00"
            self.consume_nonblocking(silence)
            self.stream.start()
            logger.info("Audio output stream started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio stream: {e}")
            raise

    def _run_loop(self):
        while not self._ended:
            try:
                chunk = self.input_janus_queue.sync_q.get(timeout=1)
                try:
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    self.stream.write(audio_data)
                except Exception as e:
                    logger.error(f"Error writing to audio stream: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in playback loop: {e}")

    async def terminate(self):
        logger.info("Terminating audio playback worker")
        self._ended = True
        await super().terminate()
        self.stream.close()


class BlockingSpeakerOutput(RateLimitInterruptionsOutputDevice):

    def __init__(
        self,
        device_info: dict,
        sampling_rate: Optional[int] = None,
        audio_encoding: AudioEncoding = AudioEncoding.LINEAR16,
    ):
        sampling_rate = sampling_rate or int(
            device_info.get("default_samplerate", DEFAULT_SAMPLING_RATE)
        )
        super().__init__(sampling_rate=sampling_rate, audio_encoding=audio_encoding)
        self.playback_worker = _PlaybackWorker(device_info=device_info, sampling_rate=sampling_rate)

    async def play(self, chunk):
        self.playback_worker.consume_nonblocking(chunk)

    def start(self) -> asyncio.Task:
        self.playback_worker.start()
        return super().start()

    async def terminate(self):
        await self.playback_worker.terminate()
        await super().terminate()

    @classmethod
    def from_default_device(
        cls,
        **kwargs,
    ):
        return cls(sd.query_devices(kind="output"), **kwargs)
