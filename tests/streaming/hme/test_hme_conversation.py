import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from vocode.streaming.hme.hme_conversation import HMEConversation
from vocode.streaming.models.audio import AudioEncoding

class TestHMEConversation(unittest.IsolatedAsyncioTestCase):
    @patch('vocode.streaming.hme.hme_conversation.generate_encrypted_token', new_callable=AsyncMock)
    @patch('vocode.streaming.hme.hme_conversation.websockets.connect', new_callable=AsyncMock)
    async def test_initialize(self, mock_websockets_connect, mock_generate_token):
        # Arrange
        mock_generate_token.return_value = 'mock_token'
        mock_websocket = AsyncMock()
        mock_websockets_connect.return_value = mock_websocket
        output_device = AsyncMock()
        mock_synthesizer = AsyncMock()
        mock_synthesizer.get_synthesizer_config = MagicMock(return_value=MagicMock(
            audio_encoding=AudioEncoding.LINEAR16,
            sampling_rate=16000
        ))

        conversation = HMEConversation(
            aot_provider_url='ws://mockserver',
            client_id='test-client',
            output_device=output_device,
            transcriber=AsyncMock(),
            agent=AsyncMock(),
            synthesizer=mock_synthesizer,
        )

        # Act
        await conversation.initialize()

        # Assert
        mock_generate_token.assert_awaited_once()
        mock_websockets_connect.assert_awaited_once_with(
            'ws://mockserver/message',
            extra_headers={
                'auth-token': 'mock_token',
                'base-sn': 'test-client',
            },
            ping_interval=20,
        )
        output_device.initialize.assert_awaited_once_with(mock_websocket)
        self.assertIsNotNone(conversation.receive_frames_task)

    @patch('vocode.streaming.hme.hme_conversation.asyncio.create_task')
    async def test_terminate(self, mock_create_task):
        # Arrange
        mock_task = AsyncMock()
        mock_create_task.return_value = mock_task
        conversation = HMEConversation(
            aot_provider_url='ws://mockserver',
            client_id='test-client',
            output_device=AsyncMock(),
            transcriber=AsyncMock(),
            agent=AsyncMock(),
            synthesizer=AsyncMock(),
        )
        conversation.websocket = AsyncMock()
        conversation.receive_frames_task = mock_task

        # Act
        await conversation.terminate()

        # Assert
        mock_task.cancel.assert_called_once()
        conversation.websocket.close.assert_awaited_once()
        conversation.output_device.terminate.assert_awaited_once()
        self.assertTrue(conversation._is_terminated) 