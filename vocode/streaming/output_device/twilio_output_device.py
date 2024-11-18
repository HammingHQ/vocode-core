from __future__ import annotations

import asyncio
import audioop
import base64
import io
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from loguru import logger
from pydantic import BaseModel
from pydub import AudioSegment

from vocode.streaming.constants import BackgroundNoiseType
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.telephony.constants import DEFAULT_AUDIO_ENCODING, DEFAULT_SAMPLING_RATE
from vocode.streaming.utils.create_task import asyncio_create_task
from vocode.streaming.utils.dtmf_utils import DTMFToneGenerator, KeypadEntry
from vocode.streaming.utils.worker import InterruptibleEvent

BACKGROUND_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "background_audio")

AUDIO_FRAME_RATE = 8000
AUDIO_SAMPLE_WIDTH = 1
AUDIO_CHANNELS = 1


class ChunkFinishedMarkMessage(BaseModel):
    chunk_id: str


MarkMessage = Union[ChunkFinishedMarkMessage]  # space for more mark messages


@dataclass
class AudioItem:
    chunk: bytes
    chunk_id: str


class TwilioOutputDevice(AbstractOutputDevice):
    def __init__(
        self,
        ws: Optional[WebSocket] = None,
        stream_sid: Optional[str] = None,
        background_noise: Optional[BackgroundNoiseType] = None,
    ):
        super().__init__(sampling_rate=DEFAULT_SAMPLING_RATE, audio_encoding=DEFAULT_AUDIO_ENCODING)
        self.is_stopping = asyncio.Event()
        self.is_stopped = asyncio.Event()
        self.ws = ws
        self.stream_sid = stream_sid
        self.background_noise = background_noise

        self._twilio_events_queue: asyncio.Queue[str] = asyncio.Queue()
        self._mark_message_queue: asyncio.Queue[MarkMessage] = asyncio.Queue()
        self._unprocessed_audio_chunks_queue: asyncio.Queue[InterruptibleEvent[AudioChunk]] = (
            asyncio.Queue()
        )
        self._audio_queue: asyncio.Queue[AudioItem] = asyncio.Queue()

    def start(self):
        logger.debug(f"TwilioOutputDevice: starting with stream_sid={self.stream_sid}")
        queues: dict[str, asyncio.Queue] = {
            "twilio_events_queue": self._twilio_events_queue,
            "mark_message_queue": self._mark_message_queue,
            "unprocessed_audio_chunks_queue": self._unprocessed_audio_chunks_queue,
            "audio_queue": self._audio_queue,
        }

        for queue_name, queue in queues.items():
            if queue.qsize() > 0:
                logger.warning(
                    f"TwilioOutputDevice: starting with {queue.qsize()} items in the {queue_name} queue"
                )
        self.is_stopping.clear()
        self.is_stopped.clear()
        super().start()

    def _interrupt_audio_chunk(self, item: InterruptibleEvent[AudioChunk]):
        audio_chunk = item.payload
        audio_chunk.on_interrupt()
        audio_chunk.state = ChunkState.INTERRUPTED

    def _drain_audio_chunk_queue(self, queue: asyncio.Queue[InterruptibleEvent[AudioChunk]]):
        while True:
            try:
                item = queue.get_nowait()
                self._interrupt_audio_chunk(item)
            except asyncio.QueueEmpty:
                break

    def _drain_queue(self, queue: asyncio.Queue):
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stop(self):
        logger.debug(f"TwilioOutputDevice: stopping...")
        self.is_stopping.set()
        await super().terminate()
        logger.debug(f"TwilioOutputDevice: waiting worker task to finish...")
        await self.is_stopped.wait()
        logger.debug(f"TwilioOutputDevice: draining queues...")
        self._drain_audio_chunk_queue(self._input_queue)
        self._drain_queue(self._unprocessed_audio_chunks_queue)
        self._drain_queue(self._mark_message_queue)
        self._drain_queue(self._twilio_events_queue)
        self._drain_queue(self._audio_queue)
        logger.debug(f"TwilioOutputDevice: stopped.")

    def _can_enqueue(self) -> bool:
        return not self.is_stopping.is_set() and not self.is_stopped.is_set()

    def consume_nonblocking(self, item: InterruptibleEvent[AudioChunk]):
        if self._can_enqueue() and not item.is_interrupted():
            if self.background_noise:
                self._audio_queue.put_nowait(
                    AudioItem(chunk=item.payload.data, chunk_id=str(item.payload.chunk_id))
                )
            else:
                self._send_audio_chunk_and_mark(
                    chunk=item.payload.data, chunk_id=str(item.payload.chunk_id)
                )
            self._unprocessed_audio_chunks_queue.put_nowait(item)
        else:
            self._interrupt_audio_chunk(item)

    def interrupt(self):
        self._send_clear_message()

    def enqueue_mark_message(self, mark_message: MarkMessage):
        if self._can_enqueue():
            self._mark_message_queue.put_nowait(mark_message)

    def send_dtmf_tones(self, keypad_entries: List[KeypadEntry]):
        tone_generator = DTMFToneGenerator()
        for keypad_entry in keypad_entries:
            logger.info(f"Sending DTMF tone {keypad_entry.value}")
            dtmf_tone = tone_generator.generate(
                keypad_entry, sampling_rate=self.sampling_rate, audio_encoding=self.audio_encoding
            )
            dtmf_message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(dtmf_tone).decode("utf-8")},
            }
            self._twilio_events_queue.put_nowait(json.dumps(dtmf_message))

    async def _send_twilio_messages(self):
        while True:
            try:
                twilio_event = await self._twilio_events_queue.get()
            except asyncio.CancelledError:
                return
            if self.ws.application_state == WebSocketState.DISCONNECTED:
                break
            try:
                await self.ws.send_text(twilio_event)
            except Exception as e:
                logger.error(f"TwilioOutputDevice: error sending twilio event: {e}")
                break

    async def _process_mark_messages(self):
        while True:
            try:
                # mark messages are tagged with the chunk ID that is attached to the audio chunk
                # but they are guaranteed to come in the same order as the audio chunks, and we
                # don't need to build resiliency there
                mark_message = await self._mark_message_queue.get()
                item = await self._unprocessed_audio_chunks_queue.get()
            except asyncio.CancelledError:
                return

            self.interruptible_event = item
            audio_chunk = item.payload

            if mark_message.chunk_id != str(audio_chunk.chunk_id):
                logger.error(
                    f"Received a mark message out of order with chunk ID {mark_message.chunk_id}"
                )

            if item.is_interrupted():
                audio_chunk.on_interrupt()
                audio_chunk.state = ChunkState.INTERRUPTED
                continue

            audio_chunk.on_play()
            audio_chunk.state = ChunkState.PLAYED

            self.interruptible_event.is_interruptible = False

    async def _run_loop(self):
        try:
            send_twilio_messages_task = asyncio_create_task(self._send_twilio_messages())
            process_mark_messages_task = asyncio_create_task(self._process_mark_messages())

            if self.background_noise:
                logger.info(f"Setting background noise task for {self.background_noise}")
                send_background_noise_task = asyncio_create_task(self._send_background_noise())
                await asyncio.gather(
                    send_twilio_messages_task,
                    process_mark_messages_task,
                    send_background_noise_task,
                )
            else:
                logger.info("No background noise task")
                await asyncio.gather(send_twilio_messages_task, process_mark_messages_task)
        finally:
            self.is_stopped.set()

    def _send_audio_chunk_and_mark(self, chunk: bytes, chunk_id: str):
        media_message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode("utf-8")},
        }
        self._twilio_events_queue.put_nowait(json.dumps(media_message))
        mark_message = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": chunk_id,
            },
        }
        self._twilio_events_queue.put_nowait(json.dumps(mark_message))

    def _send_clear_message(self):
        clear_message = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        self._twilio_events_queue.put_nowait(json.dumps(clear_message))

    def _get_background_noise_path(self) -> Optional[str]:
        if self.background_noise is None:
            return None
        return f"{BACKGROUND_AUDIO_PATH}/{self.background_noise.value}.wav"

    async def _send_background_noise(self):
        background_noise_path = self._get_background_noise_path()
        if not background_noise_path:
            logger.error("Could not find background noise file")
            return
        sound = AudioSegment.from_file(background_noise_path)
        sound = sound.set_channels(AUDIO_CHANNELS)
        sound = sound.set_frame_rate(AUDIO_FRAME_RATE)
        sound = sound.set_sample_width(AUDIO_SAMPLE_WIDTH)
        # sound = sound + 2  # 2dB louder
        output_bytes_io = io.BytesIO()
        sound.export(output_bytes_io, format="raw")
        raw_bytes = output_bytes_io.getvalue()
        raw_data = audioop.lin2ulaw(raw_bytes, 1)
        default_noise_chunk_size = 800  # 100ms
        current_position = 0

        while True:
            current_time = time.perf_counter()
            # TODO: Handle the wrap-around case better.
            if current_position + default_noise_chunk_size >= len(raw_data):
                current_position = 0

            try:
                audio_item = self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                audio_item = None

            if audio_item is None:
                chunk_size = default_noise_chunk_size
                chunk = raw_data[current_position : current_position + chunk_size]
                self._send_noise_message(chunk)
            else:
                audio_chunk = audio_item.chunk
                chunk_size = len(audio_chunk)
                noise_chunk = raw_data[current_position : current_position + chunk_size]
                chunk = overlap_with_noise(audio_chunk, noise_chunk)
                self._send_audio_chunk_and_mark(chunk, audio_item.chunk_id)

            current_position += chunk_size
            elapsed_time = time.perf_counter() - current_time
            duration = chunk_size / AUDIO_FRAME_RATE
            sleep_seconds = max(duration - elapsed_time, 0)
            await asyncio.sleep(sleep_seconds)

    def _send_noise_message(self, chunk: bytes):
        media_message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode("utf-8")},
        }
        self._twilio_events_queue.put_nowait(json.dumps(media_message))


def overlap_with_noise(chunk: bytes, noise_chunk: bytes) -> bytes:
    chunk_linear = audioop.ulaw2lin(chunk, 1)
    noise_linear = audioop.ulaw2lin(noise_chunk, 1)

    chunk_sound = AudioSegment.from_raw(
        io.BytesIO(chunk_linear),
        frame_rate=AUDIO_FRAME_RATE,
        sample_width=AUDIO_SAMPLE_WIDTH,
        channels=AUDIO_CHANNELS,
    )

    noise_sound = AudioSegment.from_raw(
        io.BytesIO(noise_linear),
        frame_rate=AUDIO_FRAME_RATE,
        sample_width=AUDIO_SAMPLE_WIDTH,
        channels=AUDIO_CHANNELS,
    )
    mixed = chunk_sound.overlay(noise_sound)
    mixed = mixed.set_channels(AUDIO_CHANNELS)
    mixed = mixed.set_frame_rate(AUDIO_FRAME_RATE)
    mixed = mixed.set_sample_width(AUDIO_SAMPLE_WIDTH)

    output_bytes_io = io.BytesIO()
    mixed.export(output_bytes_io, format="raw")
    raw_bytes = output_bytes_io.getvalue()

    return audioop.lin2ulaw(raw_bytes, 1)
