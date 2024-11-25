import asyncio
import os
from contextlib import asynccontextmanager
from typing import Final

from dotenv import load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.hme.conversation import HMEConversation
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    AudioEncoding,
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

load_dotenv()


class Settings(BaseSettings):
    aot_provider_url: str = os.getenv("AOT_PROVIDER_URL", "ws://localhost:8080")
    client_id: str = os.getenv("CLIENT_ID", "<client_id>")
    store_id: str = os.getenv("STORE_ID", "default-store")
    audio_mode: str = os.getenv("AUDIO_MODE", "Fixed")
    public_key_path: str = os.getenv("PUBLIC_KEY_PATH", "pubkey.pem")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


async def wait_for_termination(conversation: HMEConversation):
    await conversation.wait_for_termination()
    await conversation.terminate()


class AppConfig:
    DEFAULT_SAMPLE_RATE: Final[int] = 16000
    CHUNK_SIZE: Final[int] = 1920
    VOICE_NAME: Final[str] = "en-US-AriaNeural"


def create_output_device(settings: Settings) -> HMEOutputDevice:
    return HMEOutputDevice(
        aot_provider_url=settings.aot_provider_url,
        client_id=settings.client_id,
        store_id=settings.store_id,
        sampling_rate=AppConfig.DEFAULT_SAMPLE_RATE,
        audio_encoding=AudioEncoding.LINEAR16,
        audio_mode=settings.audio_mode,
    )


def create_transcriber() -> DeepgramTranscriber:
    return DeepgramTranscriber(
        DeepgramTranscriberConfig(
            sampling_rate=AppConfig.DEFAULT_SAMPLE_RATE,
            audio_encoding=AudioEncoding.LINEAR16,
            chunk_size=AppConfig.CHUNK_SIZE,
            endpointing_config=PunctuationEndpointingConfig(),
        )
    )


def create_agent() -> ChatGPTAgent:
    return ChatGPTAgent(
        ChatGPTAgentConfig(
            initial_message=BaseMessage(text="Hello! How can I help you today?"),
            prompt_preamble="You are a helpful AI assistant at a drive-thru.",
            model_name="gpt-3.5-turbo",
        )
    )


def create_synthesizer() -> AzureSynthesizer:
    return AzureSynthesizer(
        AzureSynthesizerConfig(
            voice_name=AppConfig.VOICE_NAME,
            sampling_rate=AppConfig.DEFAULT_SAMPLE_RATE,
            audio_encoding=AudioEncoding.LINEAR16,
        )
    )


def create_conversation(settings: Settings) -> HMEConversation:
    return HMEConversation(
        aot_provider_url=settings.aot_provider_url,
        client_id=settings.client_id,
        store_id=settings.store_id,
        output_device=create_output_device(settings),
        transcriber=create_transcriber(),
        agent=create_agent(),
        synthesizer=create_synthesizer(),
        audio_mode=settings.audio_mode,
    )


@asynccontextmanager
async def manage_conversation(conversation: HMEConversation):
    try:
        await conversation.start()
        yield conversation
    finally:
        if not conversation._terminated.is_set():
            await conversation.terminate()


async def main() -> None:
    configure_pretty_logging()
    settings = Settings()

    conversation = create_conversation(settings)

    try:
        async with manage_conversation(conversation) as active_conversation:
            await wait_for_termination(active_conversation)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
