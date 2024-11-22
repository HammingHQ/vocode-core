import asyncio
from typing import Optional

import websockets
from loguru import logger

from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.utils.gen_encrypted_token import generate_encrypted_token


class HMEConversation(StreamingConversation[HMEOutputDevice]):
    def __init__(
        self,
        aot_provider_url: str,
        client_id: str = "hamming-1",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.ws_url = f"{aot_provider_url}/message"
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.receive_frames_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize WebSocket connection and start conversation"""
        try:
            encrypted_jwt_token = await generate_encrypted_token()
            headers = {
                "auth-token": encrypted_jwt_token,
                "base-sn": self.client_id,
            }
            
            logger.info(f"HME: Connecting to WebSocket at {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=20,
            )
            logger.info("HME: WebSocket connected successfully")
            
            # Initialize output device before starting conversation
            logger.info("HME: Initializing output device")
            await self.output_device.initialize(self.websocket)
            
            # Start receiving frames
            logger.info("HME: Starting receive frames task")
            self.receive_frames_task = asyncio.create_task(self._receive_frames())
            
            # Start conversation after everything is initialized
            logger.info("HME: Starting conversation")
            await super().start()
            
        except Exception as e:
            logger.error(f"HME: Failed to establish WebSocket connection: {str(e)}")
            raise

    async def _receive_frames(self) -> None:
        """Receive audio frames from the WebSocket and pipe them to receive_audio"""
        if not self.websocket:
            logger.error("HME: WebSocket is not connected.")
            return

        try:
            logger.info("HME: Starting to receive frames")
            async for message in self.websocket:
                if isinstance(message, bytes):
                    logger.debug(f"HME: Received audio frame of size {len(message)} bytes")
                    self.receive_audio(message)
                else:
                    logger.debug(f"HME: Received non-audio message: {message}")
        except websockets.ConnectionClosed:
            logger.info("HME: WebSocket connection closed normally")
        except Exception as e:
            logger.error(f"HME: Error in receive frames: {e}")

    async def terminate(self) -> None:
        """Handle depart event - clean up WebSocket connection and stop audio streaming"""
        logger.info("HME: Starting conversation termination")
        
        # Set terminated flag first to stop processing new data
        self._is_terminated = True
        
        # Cancel receive frames task
        if self.receive_frames_task:
            logger.debug("HME: Cancelling receive frames task")
            self.receive_frames_task.cancel()
            try:
                await self.receive_frames_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.websocket:
            logger.debug("HME: Closing WebSocket connection")
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"HME: Error closing websocket: {e}")
            self.websocket = None

        # Terminate output device
        logger.debug("HME: Terminating output device")
        await self.output_device.terminate()
        
        # Call parent termination
        logger.debug("HME: Calling parent terminate")
        await super().terminate()
        
        logger.info("HME: Conversation terminated")
