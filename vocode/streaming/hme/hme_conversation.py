import asyncio
import json
import uuid
import zlib
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

    def __init__(self, aot_provider_url: str, client_id: str, store_id: str, *args, **kwargs):
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

    # Core lifecycle methods

    async def start(self):
        """Start audio and message WebSocket connections and conversation."""
        logger.info("Starting HME conversation")

        # Start WebSocket tasks
        self.audio_task = asyncio_create_task(self.run_audio_task())
        self.message_task = asyncio_create_task(self.run_message_task())

        logger.debug("Waiting for WebSocket connections...")
        await self.audio_connected.wait()
        await self.message_connected.wait()
        logger.info("WebSocket connections established")

        # Add delay after connections established
        logger.debug("Waiting 1 second before sending arrival message...")
        await asyncio.sleep(1.0)

        # Send arrival message and wait for first audio frame
        await self.send_arrived_message()
        try:
            # Wait up to 5 seconds for first audio frame
            await asyncio.wait_for(self.audio_frame_received.wait(), timeout=5.0)
            logger.info("Received initial audio frame")

            # Start the base conversation and let it manage its own lifecycle
            await super().start()

        except asyncio.TimeoutError:
            logger.error("No audio received after arrival message, terminating")
            await self.terminate()

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

    async def run_audio_task(self):
        """Maintain audio WebSocket connection and handle incoming audio frames."""
        url = f"{self.aot_provider_url}/audio"
        headers = await self.base_headers(url, StreamType.AUDIO)
        logger.info(f"[HME] Connecting to audio WebSocket: {url}")
        safe_headers = {**headers, "auth-token": "[REDACTED]"}
        logger.debug(f"[HME] Audio headers: {safe_headers}")

        # Track seen CRCs to deduplicate messages
        seen_crcs = set()

        try:
            async with connect(url, additional_headers=headers) as websocket:
                logger.info("[HME] Audio WebSocket connected")
                self.audio_websocket = websocket
                self.output_device.initialize_audio(websocket)
                self.audio_connected.set()

                async for message in websocket:
                    frame_size = len(message)
                    logger.debug(f"[HME] Audio frame received: {frame_size} bytes")

                    # Parse message components
                    crc = message[:4]
                    target_lane = message[4]
                    audio_bytes = message[16:]

                    # Validate CRC
                    computed_crc = zlib.crc32(audio_bytes) & 0xFFFFFFFF
                    crc_bytes = int.from_bytes(crc, byteorder="big")

                    if computed_crc != crc_bytes:
                        logger.warning(
                            "CRC validation failed",
                            computed_crc=computed_crc,
                            received_crc=crc_bytes,
                        )
                        continue

                    # Skip if we've seen this CRC before
                    if crc_bytes in seen_crcs:
                        logger.debug(f"Skipping duplicate audio frame with CRC: {crc_bytes}")
                        continue

                    # Add CRC to seen set
                    seen_crcs.add(crc_bytes)

                    # Limit size of seen_crcs to prevent memory growth
                    if len(seen_crcs) > 1000:
                        seen_crcs.clear()

                    logger.debug(
                        f"Processing audio frame - Lane: {target_lane}, Size: {len(audio_bytes)} bytes, CRC: {computed_crc}"
                    )
                    # Set event on first audio frame
                    self.audio_frame_received.set()

                    # Process audio
                    self.receive_audio(audio_bytes)

        except WebSocketException as e:
            logger.error(f"[HME] Audio task failed: {str(e)}")
            raise

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
