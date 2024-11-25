from unittest.mock import AsyncMock, patch

import pytest

from vocode.streaming.hme.websocket_connection import StreamType, WebSocketConnection
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.hme_output_device import HMEOutputDevice


@pytest.mark.asyncio
class TestHMEOutputDevice:
    @pytest.fixture(autouse=True)
    def setup_test(self, mocker):
        self.aot_provider_url = "ws://test.com"
        self.client_id = "test-client"
        self.store_id = "test-store"

        # Create mock connections
        self.mock_connections = {}
        for stream_type in StreamType:
            mock_connection = AsyncMock(spec=WebSocketConnection)
            mock_connection.websocket = AsyncMock()
            mock_connection.stream_type = stream_type
            mock_connection.wait_for_connect = AsyncMock()
            mock_connection.disconnect = AsyncMock()
            mock_connection.send = AsyncMock()
            self.mock_connections[stream_type] = mock_connection

    @patch("vocode.streaming.hme.websocket_connection.WebSocketConnection")
    async def test_initialization_and_start(self, mock_websocket_connection):
        mock_websocket_connection.side_effect = lambda **kwargs: self.mock_connections[
            kwargs["stream_type"]
        ]

        output_device = HMEOutputDevice(
            aot_provider_url=self.aot_provider_url,
            client_id=self.client_id,
            store_id=self.store_id,
            sampling_rate=16000,
            audio_encoding=AudioEncoding.LINEAR16,
        )

        await output_device.start()

        # Verify connections were created for all stream types
        assert len(output_device._connections) == len(StreamType)
        assert len(output_device._tasks) == len(StreamType)

        # Verify each connection was started
        for connection in self.mock_connections.values():
            connection.wait_for_connect.assert_awaited_once()

    @patch("vocode.streaming.hme.websocket_connection.WebSocketConnection")
    async def test_send_message(self, mock_websocket_connection):
        mock_websocket_connection.side_effect = lambda **kwargs: self.mock_connections[
            kwargs["stream_type"]
        ]

        output_device = HMEOutputDevice(
            aot_provider_url=self.aot_provider_url, client_id=self.client_id, store_id=self.store_id
        )
        await output_device.start()

        test_message = {"type": "test", "data": "test_data"}
        await output_device.send_message(StreamType.MESSAGE, test_message)

        # Verify message was sent through message connection
        message_connection = output_device._connections[StreamType.MESSAGE]
        message_connection.send.assert_called_once_with(test_message)

    @patch("vocode.streaming.hme.websocket_connection.WebSocketConnection")
    async def test_handle_message(self, mock_websocket_connection):
        mock_websocket_connection.side_effect = lambda **kwargs: self.mock_connections[
            kwargs["stream_type"]
        ]

        output_device = HMEOutputDevice(
            aot_provider_url=self.aot_provider_url, client_id=self.client_id, store_id=self.store_id
        )
        await output_device.start()

        # Test audio message handling
        test_audio = bytes([0] * 1920)
        await output_device._handle_message(StreamType.AUDIO, test_audio)

        # Verify message was queued
        queued_message = await output_device._input_queue.get()
        assert queued_message == test_audio

    @patch("vocode.streaming.hme.websocket_connection.WebSocketConnection")
    async def test_terminate(self, mock_websocket_connection):
        mock_websocket_connection.side_effect = lambda **kwargs: self.mock_connections[
            kwargs["stream_type"]
        ]

        output_device = HMEOutputDevice(
            aot_provider_url=self.aot_provider_url, client_id=self.client_id, store_id=self.store_id
        )
        await output_device.start()

        await output_device.terminate()

        # Verify all connections were closed
        for connection in self.mock_connections.values():
            connection.disconnect.assert_awaited_once()

        # Verify terminated flag is set
        assert output_device._terminated.is_set()
