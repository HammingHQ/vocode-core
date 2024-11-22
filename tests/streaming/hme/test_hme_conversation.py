from unittest.mock import AsyncMock, MagicMock

import pytest

from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.hme.conversation import ConversationState, HMEConversation
from vocode.streaming.hme.websocket_connection import StreamType
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.utils.worker import InterruptibleEvent


@pytest.fixture
def mock_output_device(mocker):
    output_device = AsyncMock()
    output_device.start = AsyncMock()
    output_device.terminate = AsyncMock()
    output_device.send_message = AsyncMock()
    output_device.set_interruptible_event_factory = MagicMock()
    output_device.sampling_rate = 16000
    output_device.audio_encoding = AudioEncoding.LINEAR16
    return output_device


@pytest.fixture
def mock_transcriber(mocker):
    transcriber = AsyncMock()
    transcriber.get_transcriber_config = MagicMock(
        return_value=MagicMock(
            sampling_rate=16000, audio_encoding=AudioEncoding.LINEAR16, chunk_size=1920
        )
    )
    transcriber.start = AsyncMock()
    transcriber.terminate = AsyncMock()
    transcriber.send_audio = AsyncMock()
    return transcriber


@pytest.fixture
def mock_agent(mocker):
    agent = AsyncMock(spec=BaseAgent)
    agent.start = AsyncMock()
    agent.stop = AsyncMock()
    return agent


@pytest.fixture
def mock_synthesizer(mocker):
    synthesizer = AsyncMock(spec=BaseSynthesizer)
    synthesizer.create_interruptible_event_factory = MagicMock(
        return_value=lambda: InterruptibleEvent(payload=None)
    )
    synthesizer.get_synthesizer_config = MagicMock(
        return_value=MagicMock(sampling_rate=16000, audio_encoding=AudioEncoding.LINEAR16)
    )
    return synthesizer


@pytest.fixture
def hme_conversation(mock_output_device, mock_transcriber, mock_agent, mock_synthesizer):
    conversation = HMEConversation(
        aot_provider_url="ws://test.com",
        client_id="test-client",
        store_id="test-store",
        output_device=mock_output_device,
        transcriber=mock_transcriber,
        agent=mock_agent,
        synthesizer=mock_synthesizer,
        audio_mode="Fixed",
    )
    return conversation


@pytest.mark.asyncio
class TestHMEConversation:
    async def test_start(self, hme_conversation, mock_output_device):
        await hme_conversation.start()

        mock_output_device.start.assert_awaited_once()
        mock_output_device.send_message.assert_awaited_once_with(
            StreamType.MESSAGE,
            {
                "type": "arrive",
                "client_id": "test-client",
                "store_id": "test-store",
            },
        )
        assert hme_conversation._state.state == ConversationState.ACTIVE

    async def test_state_change_handling(self, hme_conversation, mock_transcriber):
        # Test ACTIVE state
        await hme_conversation._handle_state_change(ConversationState.ACTIVE)
        mock_transcriber.start.assert_awaited_once()
        assert hme_conversation._state.state == ConversationState.ACTIVE

        # Test TERMINATED state
        await hme_conversation._handle_state_change(ConversationState.TERMINATED)
        mock_transcriber.terminate.assert_awaited_once()
        assert hme_conversation._state.state == ConversationState.TERMINATED

    async def test_audio_handling(self, hme_conversation, mock_transcriber):
        test_audio = bytes([0] * 1920)
        await hme_conversation._handle_audio_received(test_audio)

        mock_transcriber.send_audio.assert_awaited_once_with(test_audio)
        assert hme_conversation._metrics.audio_metrics.bytes_received == 1920
        assert hme_conversation._metrics.audio_metrics.chunks_processed == 1

    async def test_error_handling(self, hme_conversation, mock_output_device):
        # Test non-critical error
        await hme_conversation._handle_error("Test error", critical=False)
        assert not hme_conversation._terminated.is_set()

        # Test critical error
        await hme_conversation._handle_error("Critical error", critical=True)
        assert hme_conversation._terminated.is_set()
        mock_output_device.terminate.assert_awaited_once()

    async def test_terminate(self, hme_conversation, mock_output_device):
        await hme_conversation.terminate()

        assert hme_conversation._state.state == ConversationState.TERMINATED
        assert hme_conversation._terminated.is_set()
        mock_output_device.terminate.assert_awaited_once()

    async def test_terminate_idempotent(self, hme_conversation, mock_output_device):
        await hme_conversation.terminate()
        await hme_conversation.terminate()

        mock_output_device.terminate.assert_awaited_once()
