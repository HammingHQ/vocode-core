import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.logging import configure_pretty_logging
from vocode.streaming.action.end_conversation import EndConversationVocodeActionConfig
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.agent.default_factory import DefaultAgentFactory
from vocode.streaming.hme.constants import AUDIO_ENCODING, DEFAULT_CHUNK_SIZE, DEFAULT_SAMPLING_RATE
from vocode.streaming.hme.hme_conversation import HMEConversation
from vocode.streaming.models.actions import (
    PhraseBasedActionTrigger,
    PhraseBasedActionTriggerConfig,
    PhraseTrigger,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.client_backend import InputAudioConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory

load_dotenv()
configure_pretty_logging()

transcriber_factory = DefaultTranscriberFactory()
agent_factory = DefaultAgentFactory()
synthesizer_factory = DefaultSynthesizerFactory()


DRIVE_THROUGH_CUSTOMER_PROMPT = """
You are a drive-through customer at a fast food restaurant.
You want to order a classic burger with cheese and a side of fries.
You are very quick and concise in your responses.
If you hear a partial or cut-off response from the worker, ask for them to repeat or clarify.
Never respond to incomplete sentences or questions.
Say "Goodbye" after completing your order.
"""


class Settings(BaseSettings):

    aot_provider_url: str = os.getenv("AOT_PROVIDER_URL", "ws://localhost:8080")
    client_id: str = os.getenv("CLIENT_ID", "<client_id>")
    store_id: str = os.getenv("STORE_ID", "default-store-2")
    audio_mode: str = os.getenv("AUDIO_MODE", "Fixed")
    public_key_path: str = os.getenv("PUBLIC_KEY_PATH", "pubkey.pem")

    # This means a .env file can be used to overload these settings
    # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def create_conversation(settings: Settings) -> HMEConversation:
    output_device = create_output_device(settings)
    transcriber_config = DeepgramTranscriberConfig.from_input_audio_config(
        input_audio_config=InputAudioConfig(
            sampling_rate=DEFAULT_SAMPLING_RATE,
            audio_encoding=AUDIO_ENCODING,
            chunk_size=DEFAULT_CHUNK_SIZE,
        ),
        endpointing_config=PunctuationEndpointingConfig(),
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )
    agent_config = ChatGPTAgentConfig(
        model_name="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        prompt_preamble=DRIVE_THROUGH_CUSTOMER_PROMPT,
        end_conversation_on_goodbye=True,
        goodbye_phrases=["goodbye", "bye", "see you", "see ya", "see you later"],
        allowed_idle_time_seconds=5,
    )
    synthesizer_config = AzureSynthesizerConfig.from_output_device(
        output_device=output_device,
        voice_name="en-US-TonyNeural",
    )

    return HMEConversation(
        aot_provider_url=settings.aot_provider_url,
        client_id=settings.client_id,
        store_id=settings.store_id,
        output_device=output_device,
        transcriber=transcriber_factory.create_transcriber(transcriber_config),
        agent=agent_factory.create_agent(agent_config),
        synthesizer=synthesizer_factory.create_synthesizer(synthesizer_config),
    )


def create_output_device(settings: Settings) -> HMEOutputDevice:
    return HMEOutputDevice()


async def wait_for_termination(conversation: HMEConversation):
    await conversation.wait_for_termination()
    await conversation.terminate()


async def main():
    settings = Settings()
    conversation = create_conversation(settings)
    await conversation.start()
    await wait_for_termination(conversation)


if __name__ == "__main__":
    asyncio.run(main())
