import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Final, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from vocode.streaming.hme.events_manager import HMEEventsManager
from vocode.streaming.hme.schema import (
    BaseMessage,
    BotAudioPayload,
    MsgType,
    Topic,
    TopicTemplate,
)
from vocode.streaming.hme.websocket_connection import StreamType, WebSocketConnection
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk, ChunkState
from vocode.streaming.utils.create_task import asyncio_create_task
from vocode.streaming.utils.worker import InterruptibleEvent

load_dotenv()


class AudioConfig:
    CHUNK_INTERVAL: Final[float] = 0.03
    SINGLE_LANE_CHUNK_SIZE: Final[int] = 1920
    DUAL_LANE_CHUNK_SIZE: Final[int] = 3840
    DEFAULT_SAMPLE_RATE: Final[int] = 16000


class AudioMode(str, Enum):
    FIXED = "Fixed"
    VARIABLE = "Variable"


class HMEMarkMessage(BaseModel):
    chunk_id: str
    timestamp: float


@dataclass
class AudioItem:
    chunk: bytes
    chunk_id: str


class HMEOutputDevice(AbstractOutputDevice):
    def __init__(
        self,
        aot_provider_url: str,
        client_id: str,
        store_id: str,
        sampling_rate: int = AudioConfig.DEFAULT_SAMPLE_RATE,
        audio_encoding: AudioEncoding = AudioEncoding.LINEAR16,
        audio_mode: str = AudioMode.FIXED,
        is_dual_lane: bool = False,
    ):
        logger.info(f"Initializing HMEOutputDevice with client_id={client_id}, store_id={store_id}")
        logger.debug(
            f"Audio config: mode={audio_mode}, sampling_rate={sampling_rate}, encoding={audio_encoding}"
        )

        super().__init__(sampling_rate=sampling_rate, audio_encoding=audio_encoding)

        # State management (from Twilio pattern)
        self.is_stopping = asyncio.Event()
        self.is_stopped = asyncio.Event()

        # Connection info
        self.aot_provider_url = aot_provider_url
        self.client_id = client_id
        self.store_id = store_id
        self.audio_mode = AudioMode(audio_mode)
        self.chunk_size = (
            AudioConfig.DUAL_LANE_CHUNK_SIZE if is_dual_lane else AudioConfig.SINGLE_LANE_CHUNK_SIZE
        )

        # Queues
        self._audio_queue: asyncio.Queue[AudioItem] = asyncio.Queue()
        self._mark_message_queue: asyncio.Queue[HMEMarkMessage] = asyncio.Queue()
        self._unprocessed_audio_chunks_queue: asyncio.Queue[InterruptibleEvent[AudioChunk]] = (
            asyncio.Queue()
        )
        self._control_message_queue: asyncio.Queue[dict] = asyncio.Queue()

        # State tracking
        self._last_chunk_time = 0
        self.interruptible_event: Optional[InterruptibleEvent] = None
        self._events = HMEEventsManager()
        self._terminated = asyncio.Event()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._connections: Dict[StreamType, WebSocketConnection] = {}

    def _can_enqueue(self) -> bool:
        return not self.is_stopping.is_set() and not self.is_stopped.is_set()

    def interrupt(self):
        # Stop all currently playing audio
        if self.interruptible_event:
            self._interrupt_audio_chunk(self.interruptible_event)
            self.interruptible_event = None

    def consume_nonblocking(self, item: InterruptibleEvent[AudioChunk]):
        if self._can_enqueue() and not item.is_interrupted():
            self._audio_queue.put_nowait(
                AudioItem(chunk=item.payload.data, chunk_id=str(id(item.payload)))
            )
            self._unprocessed_audio_chunks_queue.put_nowait(item)
        else:
            self._interrupt_audio_chunk(item)

    def _interrupt_audio_chunk(self, item: InterruptibleEvent[AudioChunk]):
        audio_chunk = item.payload
        audio_chunk.on_interrupt()
        audio_chunk.state = ChunkState.INTERRUPTED

    async def start(self):
        """Initialize and start WebSocket connections"""
        logger.info("Starting HMEOutputDevice")

        # Setup connections
        for stream_type in [StreamType.AUDIO, StreamType.MESSAGE]:
            url = f"{self.aot_provider_url}/{stream_type.value}"
            self._connections[stream_type] = WebSocketConnection(
                url=url,
                stream_type=stream_type,
                client_id=self.client_id,
                store_id=self.store_id,
            )

        # Start main processing tasks
        self._tasks["run_loop"] = asyncio_create_task(self._run_loop())

        logger.info("HMEOutputDevice started successfully")

    async def _run_loop(self):
        try:
            send_audio_task = asyncio_create_task(self._send_audio())
            process_mark_messages_task = asyncio_create_task(self._process_mark_messages())
            send_control_messages_task = asyncio_create_task(self._send_control_messages())

            await asyncio.gather(
                send_audio_task, process_mark_messages_task, send_control_messages_task
            )
        except Exception as e:
            logger.error(f"Error in run loop: {e}")
        finally:
            self.is_stopped.set()

    async def stop(self):
        """Clean shutdown following Twilio pattern"""
        logger.debug("HMEOutputDevice: stopping...")
        self.is_stopping.set()
        await super().terminate()
        logger.debug("HMEOutputDevice: waiting for tasks...")
        await self.is_stopped.wait()
        logger.debug("HMEOutputDevice: draining queues...")
        await self._drain_audio_chunk_queue(self._unprocessed_audio_chunks_queue)
        await self._drain_queue(self._mark_message_queue)
        await self._drain_queue(self._control_message_queue)
        await self._drain_queue(self._audio_queue)
        logger.debug("HMEOutputDevice: stopped.")

    async def _send_audio(self):
        """Send audio chunks with proper timing"""
        while not self.is_stopping.is_set():
            try:
                audio_item = await self._audio_queue.get()

                if self.audio_mode == AudioMode.FIXED:
                    current_time = time.time()
                    time_since_last = current_time - self._last_chunk_time
                    if time_since_last < AudioConfig.CHUNK_INTERVAL:
                        await asyncio.sleep(AudioConfig.CHUNK_INTERVAL - time_since_last)
                    self._last_chunk_time = time.time()

                if StreamType.AUDIO in self._connections:
                    await self._connections[StreamType.AUDIO].send(audio_item.chunk)
                    # Queue mark message after sending
                    await self._mark_message_queue.put(
                        HMEMarkMessage(chunk_id=audio_item.chunk_id, timestamp=time.time())
                    )
                    logger.debug(f"Sent audio chunk {audio_item.chunk_id}")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

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

    async def _send_control_messages(self):
        """Send control messages through message connection"""
        while not self.is_stopping.is_set():
            try:
                message = await self._control_message_queue.get()
                if StreamType.MESSAGE in self._connections:
                    await self._connections[StreamType.MESSAGE].send(json.dumps(message))
                    logger.debug(f"Sent control message: {message}")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Error sending control message: {e}")

    async def _drain_audio_chunk_queue(self, queue: asyncio.Queue[InterruptibleEvent[AudioChunk]]):
        """Drain audio chunk queue and interrupt chunks"""
        while True:
            try:
                item = queue.get_nowait()
                self._interrupt_audio_chunk(item)
            except asyncio.QueueEmpty:
                break

    async def _drain_queue(self, queue: asyncio.Queue):
        """Drain generic queue"""
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _handle_audio(self, audio_chunk: bytes):
        """Handle incoming audio chunks"""
        if len(audio_chunk) != self.chunk_size:
            logger.warning(
                f"Received audio chunk of incorrect size: {len(audio_chunk)}, expected {self.chunk_size}"
            )
            return

        try:
            chunk = AudioChunk(data=audio_chunk)
            await self._unprocessed_audio_chunks_queue.put(
                InterruptibleEvent(payload=chunk, is_interruptible=True)
            )
            logger.debug("Audio chunk queued successfully")
        except Exception as e:
            logger.error(f"Error handling audio chunk: {e}")

    async def _handle_message(self, stream_type: StreamType, message: bytes):
        """Handle incoming messages"""
        try:
            msg_data = json.loads(message)
            base_message = BaseMessage.model_validate(msg_data)

            match base_message.meta.msg_type:
                case MsgType.REQUEST:
                    await self._handle_request(base_message)
                case MsgType.ALERT:
                    await self._handle_alert(base_message)
                case MsgType.BOT:
                    await self._handle_bot(base_message)
                case MsgType.STATUS:
                    await self._handle_status(base_message)
                case _:
                    logger.debug(f"Unhandled message type: {base_message.meta.msg_type}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_request(self, message: BaseMessage):
        """Handle request messages (arrive/depart)"""
        try:
            topic = TopicTemplate(message.payload.topic)
            logger.info(f"Processing request message with topic: {topic}")

            match topic:
                case TopicTemplate.ARRIVE:
                    logger.info(f"Vehicle arrive event for lane {message.payload.lane}")
                    await self._events.emit("vehicle_arrive", lane=message.payload.lane)
                case TopicTemplate.DEPART:
                    logger.info(f"Vehicle depart event for lane {message.payload.lane}")
                    await self._events.emit("vehicle_depart", lane=message.payload.lane)
                case TopicTemplate.AUTO_ESCALATION:
                    logger.info(f"Auto escalation event for lane {message.payload.lane}")
                    await self._events.emit("auto_escalation", lane=message.payload.lane)
                case _:
                    logger.debug(f"Unhandled request topic: {topic}")
        except Exception as e:
            logger.error(f"Error handling request message: {e}")

    async def _handle_alert(self, message: BaseMessage):
        """Handle alert messages"""
        try:
            topic = TopicTemplate(message.payload.topic)
            logger.info(f"Processing alert message with topic: {topic}")

            match topic:
                case TopicTemplate.CREW_ESCALATION:
                    logger.info(f"Crew escalation alert for lane {message.payload.lane}")
                    await self._events.emit("crew_escalation", lane=message.payload.lane)
                case TopicTemplate.AUDIO_ALERT:
                    logger.info("Audio alert received, interrupting current audio")
                    if self.interruptible_event:
                        self.interruptible_event.interrupt()
                        self.interruptible_event = None
                case _:
                    logger.debug(f"Unhandled alert topic: {topic}")
        except Exception as e:
            logger.error(f"Error handling alert message: {e}")

    async def _handle_bot(self, message: BaseMessage):
        """Handle bot messages"""
        logger.debug("Processing bot message")
        if isinstance(message.payload, BotAudioPayload):
            logger.debug(f"Bot audio payload received: {len(message.payload.audio_data)} bytes")
            if len(message.payload.audio_data) == message.payload.chunk_size:
                chunk = AudioChunk(data=message.payload.audio_data)
                await self._unprocessed_audio_chunks_queue.put(
                    InterruptibleEvent(payload=chunk, is_interruptible=True)
                )
                logger.debug("Bot audio chunk queued successfully")
            else:
                logger.warning(f"Invalid bot audio chunk size: {len(message.payload.audio_data)}")

    async def _handle_status(self, message: BaseMessage):
        """Handle status messages"""
        topic = Topic(template=TopicTemplate(message.payload.topic))
        logger.info(f"Processing status message with topic: {topic.template}")
        match topic.template:
            case TopicTemplate.BOT_AVAILABLE:
                logger.info(f"Bot status update: {message.payload} bots available")
                await self._events.emit("bot_status", count=message.payload)
            case TopicTemplate.AUDIO_INTERRUPTION:
                logger.info("Audio interruption status received")
                if self.interruptible_event:
                    self.interruptible_event.interrupt()
                    self.interruptible_event = None
            case _:
                logger.debug(f"Unhandled status topic: {topic.value}")

    async def play(self, audio_chunk: bytes):
        """Send audio through audio connection"""
        logger.debug(f"Playing audio chunk of size {len(audio_chunk)} bytes")
        if StreamType.AUDIO in self._connections:
            await self._audio_queue.put(AudioItem(chunk=audio_chunk, chunk_id=str(id(audio_chunk))))
            logger.debug("Audio chunk queued successfully")
        else:
            logger.warning("No audio connection available")

    async def send_message(self, stream_type: StreamType, message: dict):
        """Send control message through specified connection"""
        if stream_type in self._connections:
            await self._connections[stream_type].send(json.dumps(message))

    async def terminate(self):
        """Clean shutdown of all connections"""
        if self._terminated.is_set():
            return

        self._terminated.set()

        # Cancel all tasks
        for task in self._tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for connection in self._connections.values():
            await connection.disconnect()
