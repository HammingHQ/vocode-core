import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from loguru import logger
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.utils.create_task import asyncio_create_task
from vocode.utils.gen_encrypted_token import generate_encrypted_token


class StreamType(str, Enum):
    AUDIO = "audio"
    MESSAGE = "message"
    BOT = "bot"


class HMEConversation(StreamingConversation[HMEOutputDevice]):
    """Manages HME audio and message WebSocket connections and streaming conversation.

    Attributes:
        audio_connected (asyncio.Event): Event set when audio WebSocket connects
        message_connected (asyncio.Event): Event set when message WebSocket connects
        audio_websocket (Optional[ClientConnection]): Audio WebSocket connection
        message_websocket (Optional[ClientConnection]): Message WebSocket connection
        auth_token (Optional[str]): JWT auth token for WebSocket connections
    """

    def __init__(
        self,
        aot_provider_url: str,
        client_id: str,
        store_id: str,
        idle_timeout_seconds: int = 60,
        *args,
        **kwargs,
    ):
        logger.bind(client_id=client_id, store_id=store_id).info("Starting HME conversation")
        super().__init__(*args, **kwargs)

        # Connection settings
        self.aot_provider_url = aot_provider_url
        self.client_id = client_id
        self.store_id = store_id

        # Connection state
        self.audio_connected = asyncio.Event()
        self.message_connected = asyncio.Event()
        self.audio_websocket: Optional[ClientConnection] = None
        self.message_websocket: Optional[ClientConnection] = None
        self.auth_token: Optional[str] = None
        self.arrival_response_received = asyncio.Event()
        self.audio_frame_received = asyncio.Event()
        self.idle_time_threshold = idle_timeout_seconds

    # Core lifecycle methods

    async def start(self):
        """Start audio and message WebSocket connections and conversation."""
        logger.info("Starting HME conversation")

        # Pause idle check during startup
        self.set_check_for_idle_paused(True)

        # Start base conversation components first
        await super().start()

        # Start WebSocket tasks
        self.audio_task = asyncio_create_task(self.run_audio_task())
        self.message_task = asyncio_create_task(self.run_message_task())

        # Wait for both connections and initial setup
        await asyncio.gather(
            self.audio_connected.wait(),
            self.message_connected.wait(),
            self.output_device.wait_for_ready(),
        )

        await asyncio.sleep(1.0)
        await self.send_arrived_message()

    async def terminate(self):
        """Clean up resources and send depart message."""
        logger.info("Terminating HME conversation")
        try:
            await self.send_depart_message()
        except Exception as e:
            logger.error(f"Error sending depart message: {e}")

        if self.audio_websocket:
            await self.audio_websocket.wait_closed()
        if self.message_websocket:
            await self.message_websocket.wait_closed()

        await super().terminate()

    async def wait_for_termination(self):
        """Wait for audio and message tasks to complete."""
        logger.info("Waiting for HME conversation termination")
        await asyncio.gather(
            self.audio_task,
            self.message_task,
        )
        await super().wait_for_termination()

    # WebSocket connection handlers

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

    async def run_audio_task(self):
        """Maintain audio WebSocket connection and handle incoming audio frames."""
        url = f"{self.aot_provider_url}/audio"
        headers = await self.base_headers(url, StreamType.AUDIO)
        logger.info(f"[HME] Connecting to audio WebSocket: {url}")
        safe_headers = {**headers, "auth-token": "[REDACTED]"}
        logger.debug(f"[HME] Audio headers: {safe_headers}")

        # Open file for writing audio data
        audio_file = open(f"audio_{self.id}.raw", "wb")
        logger.info(f"[HME] Writing audio to file: audio_{self.id}.raw")

        try:
            async with connect(url, additional_headers=headers) as websocket:
                logger.info("[HME] Audio WebSocket connected")
                self.output_device.audio_websocket = websocket
                self.output_device.initialize_audio(websocket)
                self.audio_connected.set()

                async for message in websocket:
                    # Let output device handle all audio frame processing

                    # Skip header bytes (16 bytes: 4 for CRC, 1 for lane, 11 padding)
                    audio_data = message[16:]
                    self.receive_audio(audio_data)
                    logger.debug(f"[HME] Audio frame received: {len(audio_data)} bytes")

                    # Write audio data to file
                    audio_file.write(audio_data)
                    audio_file.flush()

                    # Play incoming audio through local speaker if enabled
                    if (
                        self.output_device.enable_local_playback
                        and self.output_device.speaker_output
                    ):

                        await self.output_device.speaker_output.play(audio_data)
                        logger.debug("[HME] Played incoming audio through speaker")

                    if not self.audio_frame_received.is_set():
                        self.audio_frame_received.set()

        except Exception as e:
            self._audio_error = e
            self.audio_connected.set()  # Unblock start() but with error
            raise
        finally:
            audio_file.close()
            logger.info(f"[HME] Closed audio file: audio_{self.id}.raw")

    async def run_message_task(self):
        """Maintain message WebSocket connection and handle incoming messages."""
        url = f"{self.aot_provider_url}/message"
        headers = await self.base_headers(url, StreamType.MESSAGE)
        logger.info(f"[HME] Connecting to message WebSocket - URL: {url}")
        safe_headers = {**headers, "auth-token": "[REDACTED]"}
        logger.debug(f"[HME] Message headers: {safe_headers}")

        try:
            async with connect(url, additional_headers=headers) as websocket:
                logger.info("Successfully connected to message WebSocket")
                self.message_websocket = websocket
                self.message_connected.set()

                async for message in websocket:
                    logger.info(f"Received message: {message}")
                    try:
                        parsed_message = json.loads(message)
                        logger.debug(f"Parsed message: {json.dumps(parsed_message, indent=2)}")
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse message as JSON: {message}")
                        continue

                    topic = parsed_message.get("topic")
                    if topic == "aot/request/audio-interruption":
                        logger.info("[HME] Received audio interruption request")

        except WebSocketException as e:
            logger.error(f"Message WebSocket error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in message task: {str(e)}")
            raise

    # Message handling methods

    async def send_message(self, message: str):
        """Send message through message WebSocket."""
        if not self.message_websocket:
            logger.error("Attempted to send message but WebSocket not connected")
            return
        logger.info(f"[HME] Sending message: {message[:100]}...")
        await self.message_websocket.send(message)

    async def send_arrived_message(self):
        """Send arrival message to initiate conversation."""
        logger.info(f"Sending vehicle arrival message - Car ID: {self.id}")
        arrive_message = {
            "topic": "NEXEO/request/lane1/arrive",
            "payload": {"event": "arrive"},
            "meta": {
                "deviceID": self.client_id,
                "timestamp": datetime.now().isoformat(),
                "msgId": str(uuid.uuid4()),
                "storeId": self.store_id,
                "carID": self.id,
                "msgType": "request",
            },
        }
        logger.debug(f"Arrival message: {json.dumps(arrive_message, indent=2)}")
        await self.send_message(json.dumps(arrive_message))

    async def send_depart_message(self):
        """Send departure message to end conversation."""
        logger.info(f"Sending vehicle departure message - Car ID: {self.id}")
        depart_message = {
            "topic": "NEXEO/request/lane1/depart",
            "payload": {"event": "depart"},
            "meta": {
                "deviceID": self.client_id,
                "timestamp": datetime.now().isoformat(),
                "msgId": str(uuid.uuid4()),
                "storeId": self.store_id,
                "carID": self.id,
                "msgType": "request",
            },
        }
        logger.debug(f"Departure message: {json.dumps(depart_message, indent=2)}")
        await self.send_message(json.dumps(depart_message))

    # Authentication methods

    async def base_headers(self, url: str, stream_type: StreamType) -> Dict[str, str]:
        """Generate base headers required for all HME connections."""
        if not self.auth_token:
            claims = {
                "url": url,
                "stream_type": stream_type.value,
                "client_id": self.client_id,
                "store_id": self.store_id,
            }
            logger.debug(f"Generating auth token with claims: {claims}")
            self.auth_token = await generate_encrypted_token(
                subject="dev@hamming.ai",
                expiry_hours=48,
                additional_claims=claims,
            )
            logger.debug("Auth token generated successfully")

        return {
            "base-sn": self.client_id,
            "store-id": self.store_id,
            "client-version": "1.0.0",
            "protocol-version": "10.3",
            "Content-Type": "application/json",
            "stream-type": stream_type.value,
            "auth-token": self.auth_token,
        }

    async def reconnect_audio(self):
        """Attempt to reconnect audio websocket if disconnected"""
        try:
            if self.output_device.audio_websocket:
                await self.output_device.audio_websocket.close()
            await self.run_audio_task()
        except Exception as e:
            logger.error(f"Failed to reconnect audio: {e}")
