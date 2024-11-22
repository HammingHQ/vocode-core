import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.hme.hme_conversation import HMEConversation
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


async def wait_for_termination(conversation: HMEConversation):
    await conversation.wait_for_termination()
    await conversation.terminate()


async def main() -> None:
    configure_pretty_logging()
    settings = Settings()

    output_device = HMEOutputDevice()
    conversation = HMEConversation(
        aot_provider_url=settings.aot_provider_url,
        client_id=settings.client_id,
        output_device=output_device,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig(
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                sampling_rate=16000,
                audio_encoding=AudioEncoding.LINEAR16,
                chunk_size=16000,
            ),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                initial_message=BaseMessage(text="Hello! How can I assist you today?"),
                prompt_preamble="The AI is having a pleasant conversation about life.",
            )
        ),
        synthesizer=AzureSynthesizer(
            synthesizer_config=AzureSynthesizerConfig(
                voice_name="en-US-AriaNeural",
                sampling_rate=16000,
                audio_encoding=AudioEncoding.LINEAR16,
            )
        ),
    )

    try:
        await conversation.start()
        logger.info("Conversation started. Waiting for termination...")
        asyncio.create_task(wait_for_termination(conversation))
    except KeyboardInterrupt:
        logger.info("Received CTRL+C, initiating shutdown...")
    finally:
        # Ensure cleanup happens
        logger.info("Cleaning up resources...")
        await conversation.terminate()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
