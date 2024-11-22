import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import websockets
from loguru import logger

from vocode.utils.gen_encrypted_token import generate_encrypted_token


class StreamType(str, Enum):
    AUDIO = "audio"
    MESSAGE = "message"
    BOT = "bot"


@dataclass
class WebSocketConfig:
    # Connection settings
    CONNECTION_TIMEOUT: float = 5.0
    RECONNECT_ATTEMPTS: int = 3
    RECONNECT_DELAY: float = 1.0
    PING_INTERVAL: float = 5.0
    PING_TIMEOUT: float = 20.0

    # Authentication settings
    AUTH_REQUIRED: bool = True
    AUTH_HEADER_NAME: str = "auth-token"
    PUBLIC_KEY_PATH: str = "pubkey.pem"
    TOKEN_SUBJECT: str = "dev@hamming.ai"
    TOKEN_EXPIRY_HOURS: int = 1

    # HME-specific settings
    PROTOCOL_VERSION: str = "10.3"
    CLIENT_VERSION: str = "1.0.0"


class WebSocketConnection:
    def __init__(
        self,
        url: str,
        stream_type: StreamType,
        client_id: str,
        store_id: str,
        lane: Optional[int] = None,
        config: Optional[WebSocketConfig] = None,
    ):
        self.url = url
        self.stream_type = stream_type
        self.client_id = client_id
        self.store_id = store_id
        self.lane = lane
        self.config = config or WebSocketConfig()
        self.websocket = None
        self._connect_lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self.closed = False
        self._token: Optional[str] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.disconnect()

    async def connect(self) -> None:
        """Establish WebSocket connection with proper handshake"""
        async with self._connect_lock:
            if self.websocket and not self.websocket.closed:
                return

            try:
                logger.debug(f"Connecting to {self.url} ({self.stream_type.value})")
                headers = await self.base_headers()

                # Create connection with proper error handling and retry logic
                for attempt in range(self.config.RECONNECT_ATTEMPTS):
                    try:
                        self.websocket = await websockets.connect(
                            self.url,
                            extra_headers=headers,
                            ping_interval=self.config.PING_INTERVAL,
                            ping_timeout=self.config.PING_TIMEOUT,
                            close_timeout=self.config.CONNECTION_TIMEOUT,
                        )

                        # Verify connection is established with a ping/pong
                        try:
                            pong_waiter = await self.websocket.ping()
                            await asyncio.wait_for(
                                pong_waiter, timeout=self.config.CONNECTION_TIMEOUT
                            )
                            self._connected.set()
                            self.closed = False
                            logger.info(
                                f"{self.stream_type.value} WebSocket connected to {self.url}"
                            )
                            return  # Successfully connected
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Connection verification timeout for {self.stream_type.value}, attempt {attempt + 1}"
                            )
                            await self.disconnect(force=True)
                            if attempt < self.config.RECONNECT_ATTEMPTS - 1:
                                await asyncio.sleep(self.config.RECONNECT_DELAY * (attempt + 1))
                                continue
                            raise ConnectionError(
                                f"Failed to verify {self.stream_type.value} connection after {attempt + 1} attempts"
                            )

                    except websockets.InvalidStatusCode as e:
                        logger.error(f"Server rejected connection: {e}")
                        if attempt < self.config.RECONNECT_ATTEMPTS - 1:
                            await asyncio.sleep(self.config.RECONNECT_DELAY * (attempt + 1))
                            continue
                        raise ConnectionError(f"Failed to connect to {self.stream_type.value}: {e}")
                    except websockets.InvalidMessage as e:
                        logger.error(f"Invalid WebSocket handshake: {e}")
                        if attempt < self.config.RECONNECT_ATTEMPTS - 1:
                            await asyncio.sleep(self.config.RECONNECT_DELAY * (attempt + 1))
                            continue
                        raise ConnectionError(
                            f"Invalid WebSocket handshake for {self.stream_type.value}"
                        )
                    except Exception as e:
                        logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                        if attempt < self.config.RECONNECT_ATTEMPTS - 1:
                            await asyncio.sleep(self.config.RECONNECT_DELAY * (attempt + 1))
                            continue
                        raise ConnectionError(f"Failed to connect to {self.stream_type.value}: {e}")

            except Exception as e:
                self.websocket = None
                self._connected.clear()
                self.closed = True
                raise

    async def wait_for_connect(self) -> None:
        await self._connected.wait()

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and ready"""
        return (
            self.websocket is not None
            and not self.closed
            and self._connected.is_set()
            and not self.websocket.closed
        )

    async def send(self, message: Union[str, bytes, Dict]) -> None:
        """Send message with connection check and retry"""
        if not self.is_connected:
            error_msg = f"{self.stream_type.value} WebSocket is not connected"
            logger.error(error_msg)
            try:
                await self.connect()
                # Wait for connection to be fully established
                try:
                    await asyncio.wait_for(
                        self._connected.wait(), timeout=self.config.CONNECTION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise ConnectionError(
                        f"Timeout waiting for {self.stream_type.value} connection"
                    )
            except Exception as e:
                raise ConnectionError(
                    f"Failed to establish {self.stream_type.value} connection: {e}"
                )

        try:
            if isinstance(message, dict):
                message = json.dumps(message)
            await self.websocket.send(message)
            logger.debug(f"Sent {self.stream_type.value} message: {message}")
        except websockets.ConnectionClosed as e:
            logger.error(f"Connection closed while sending: {e}")
            self._connected.clear()
            self.closed = True
            raise ConnectionError(
                f"Connection closed while sending to {self.stream_type.value}: {e}"
            )
        except Exception as e:
            logger.error(f"Send error: {e}")
            self._connected.clear()
            raise

    async def disconnect(self, force: bool = False) -> None:
        """Close WebSocket connection with proper handshake"""
        if self.closed or not self.websocket:
            return

        try:
            if force:
                self.websocket.transport.close()
            else:
                try:
                    await asyncio.wait_for(
                        self.websocket.close(code=1000, reason="Normal closure"),
                        timeout=self.config.CONNECTION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Clean disconnect timed out for {self.stream_type.value}, forcing close"
                    )
                    self.websocket.transport.close()
        except Exception as e:
            logger.error(f"Error during WebSocket closure: {e}")
        finally:
            self.websocket = None
            self._connected.clear()
            self.closed = True

    async def receive(self) -> Optional[Union[str, bytes]]:
        """Receive message from WebSocket with proper error handling"""
        if not self.websocket:
            return None

        try:
            message = await self.websocket.recv()
            if isinstance(message, str):
                try:
                    # Try to parse as JSON for prettier logging
                    parsed = json.loads(message)
                    logger.debug(
                        f"< {self.stream_type.value} Received JSON: {json.dumps(parsed, indent=2)}"
                    )
                except Exception:
                    logger.debug(f"< {self.stream_type.value} Received text: {message}")
            else:
                logger.debug(f"< {self.stream_type.value} Received {len(message)} bytes")
            return message
        except websockets.ConnectionClosed as e:
            logger.info(f"= {self.stream_type.value} Connection closed: {e.code} {e.reason}")
            return None
        except Exception as e:
            logger.error(f"! {self.stream_type.value} Receive error: {e}")
            raise

    async def base_headers(self) -> Dict[str, str]:
        """Generate base headers required for all HME connections"""
        headers = {
            "base-sn": self.client_id,
            "store-id": self.store_id,
            "client-version": self.config.CLIENT_VERSION,
            "protocol-version": self.config.PROTOCOL_VERSION,
            "Content-Type": "application/json",
            "stream-type": self.stream_type.value,
        }

        if self.lane is not None:
            headers["lane"] = str(self.lane)

        if self.config.AUTH_REQUIRED:
            # Generate token asynchronously if not already generated
            if not self._token:
                claims = {
                    "url": self.url,
                    "client_id": self.client_id,
                    "store_id": self.store_id,
                    "stream_type": self.stream_type.value,
                }
                if self.lane is not None:
                    claims["lane"] = str(self.lane)

                try:
                    self._token = await generate_encrypted_token(
                        public_key_path=self.config.PUBLIC_KEY_PATH,
                        subject=self.config.TOKEN_SUBJECT,
                        expiry_hours=self.config.TOKEN_EXPIRY_HOURS,
                        additional_claims=claims,
                    )
                except Exception as e:
                    logger.error(f"Failed to generate auth token: {e}")
                    raise

            headers[self.config.AUTH_HEADER_NAME] = self._token

        return headers

    async def on_close(self):
        """Handle WebSocket closure"""
        logger.info(f"{self.stream_type.value} WebSocket connection closed")
        await self.disconnect()

    async def on_message(self, message: bytes):
        """Handle incoming messages"""
        # Implement message handling logic or emit events
        logger.debug(f"Received message on {self.stream_type.value}: {message}")


class ConnectionPool:
    def __init__(self, max_size: int = 10):
        self._pool: Dict[StreamType, List[WebSocketConnection]] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def acquire(self, stream_type: StreamType) -> WebSocketConnection:
        async with self._lock:
            if not self._pool.get(stream_type):
                raise ConnectionError(f"No connection available for {stream_type}")
            return self._pool[stream_type].pop()

    async def release(self, connection: WebSocketConnection) -> None:
        async with self._lock:
            stream_type = connection.stream_type
            if stream_type not in self._pool:
                self._pool[stream_type] = []
            if len(self._pool[stream_type]) < self._max_size:
                self._pool[stream_type].append(connection)
            else:
                await connection.disconnect()
