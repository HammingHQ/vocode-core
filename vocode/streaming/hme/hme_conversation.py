import asyncio
import json
import uuid
import zlib
from datetime import datetime
from enum import Enum
from typing import Dict

from loguru import logger
from websockets.asyncio.client import connect
from websockets.sync.client import ClientConnection

from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.utils.create_task import asyncio_create_task
from vocode.utils.gen_encrypted_token import generate_encrypted_token

KEY_PATH = "/Users/marius/ham/vocode-core/apps/hme/pubkey.pem"


class StreamType(str, Enum):
    AUDIO = "audio"
    MESSAGE = "message"
    BOT = "bot"


class HMEConversation(StreamingConversation[HMEOutputDevice]):
    audio_connected: asyncio.Event
    message_connected: asyncio.Event
    audio_websocket: ClientConnection | None = None
    message_websocket: ClientConnection | None = None
    auth_token: str | None = None

    def __init__(self, aot_provider_url: str, client_id: str, store_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aot_provider_url = aot_provider_url
        self.client_id = client_id
        self.store_id = store_id
        self.audio_connected = asyncio.Event()
        self.message_connected = asyncio.Event()

    async def start(self):
        self.audio_task = asyncio_create_task(self.run_audio_task())
        self.message_task = asyncio_create_task(self.run_message_task())

        await self.audio_connected.wait()
        await self.message_connected.wait()

        await super().start()
        await self.send_arrived_message()

    async def wait_for_termination(self):
        await asyncio.gather(
            self.audio_task,
            self.message_task,
        )
        await super().wait_for_termination()

    async def run_audio_task(self):
        url = f"{self.aot_provider_url}/audio"
        headers = await self.base_headers(url, StreamType.AUDIO)
        logger.info(f"Connecting to audio url={url} with headers={headers}")
        async with connect(url, additional_headers=headers) as websocket:
            logger.info("Connected to audio websocket!")
            self.audio_websocket = websocket
            self.output_device.initialize_audio(websocket)
            self.audio_connected.set()
            async for message in websocket:
                logger.info(f"Received audio message: {len(message)} bytes")
                crc = message[0:4]
                target_lane = message[4]
                logger.info(f"Received message for target lane {target_lane}")
                audio_bytes = message[16:]
                computed_crc = zlib.crc32(audio_bytes) & 0xFFFFFFFF
                crc_bytes = int.from_bytes(crc, byteorder='big')
                if computed_crc != crc_bytes:
                    logger.warning(f"CRC mismatch - computed: {computed_crc}, received: {crc_bytes}")
                else:
                    logger.info(f"Received audio message with valid CRC={computed_crc}")
                self.receive_audio(audio_bytes)
                asyncio.create_task(self.output_device.speaker_output.play(audio_bytes))


    async def run_message_task(self):
        url = f"{self.aot_provider_url}/message"
        headers = await self.base_headers(url, StreamType.MESSAGE)
        logger.info(f"Connecting to message url={url} with headers={headers}")
        async with connect(url, additional_headers=headers) as websocket:
            logger.info("Connected to message websocket!")
            self.message_websocket = websocket
            self.message_connected.set()
            async for message in websocket:
                logger.info(f"Received message: {message}")

    async def send_arrived_message(self): 
        logger.info(f"Vehicle arrived: carID={self.id}")
        arrive_message = {
            "topic": "NEXEO/request/lane1/arrive",
            "payload": {
                "event": "arrive"
            },
            "meta": {
                "deviceID": self.client_id,
                "timestamp": datetime.now().isoformat(),
                "msgId": str(uuid.uuid4()),
                "storeId": self.store_id,
                "carID": self.id,
                "msgType": "request",
            },
        }
        logger.info(f"sending: {arrive_message}")
        await asyncio.sleep(1)
        await self.send_message(json.dumps(arrive_message))

    async def send_message(self, message: str):
        await self.message_websocket.send(message)

    async def base_headers(self, url: str, stream_type: StreamType) -> Dict[str, str]:
        """Generate base headers required for all HME connections"""
        if not self.auth_token:
            claims = {
                "url": url,
                "stream_type": stream_type.value,
                "client_id": self.client_id,
                "store_id": self.store_id,
            }
            self.auth_token = await generate_encrypted_token(
                public_key_path=KEY_PATH,
                subject="dev@hamming.ai",
                expiry_hours=48,
                additional_claims=claims,
            )
        headers = {
            "base-sn": self.client_id,
            "store-id": self.store_id,
            "client-version": "1.0.0",
            "protocol-version": "10.3",
            "Content-Type": "application/json",
            "stream-type": stream_type.value,
            "auth-token": self.auth_token,
        }
        return headers
