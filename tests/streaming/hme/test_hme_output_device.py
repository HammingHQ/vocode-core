import pytest
from unittest.mock import AsyncMock
from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.output_device.audio_chunk import AudioChunk
@pytest.mark.asyncio
class TestHMEOutputDevice:
    async def test_play_and_terminate(self):
        # Arrange
        output_device = HMEOutputDevice()
        websocket = AsyncMock()
        await output_device.initialize(websocket)
        
        # Create test audio chunk
        audio_chunk = AudioChunk(data=b'test audio data')
        
        try:
            # Act - play audio
            await output_device.play(audio_chunk)
            
            # Act - terminate
            await output_device.terminate()
            
            # Assert
            assert output_device._is_terminated
            assert output_device.websocket is None
            
            # Verify websocket was called correctly
            websocket.send_bytes.assert_called_once_with(audio_chunk.data)
            websocket.close.assert_called_once()
            
        finally:
            # Cleanup
            if not output_device._is_terminated:
                await output_device.terminate()