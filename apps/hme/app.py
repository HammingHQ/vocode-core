import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.logging import configure_pretty_logging
from vocode.streaming.action.end_conversation import EndConversationVocodeActionConfig
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
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.synthesizer.default_factory import DefaultSynthesizerFactory
from vocode.streaming.transcriber.default_factory import DefaultTranscriberFactory

load_dotenv()
configure_pretty_logging()


class Settings(BaseSettings):
    aot_provider_url: str = os.getenv("AOT_PROVIDER_URL", "ws://localhost:8080")
    client_id: str = os.getenv("CLIENT_ID", "<client_id>")
    store_id: str = os.getenv("STORE_ID", "default-store")
    audio_mode: str = os.getenv("AUDIO_MODE", "Fixed")
    public_key_path: str = os.getenv("PUBLIC_KEY_PATH", "pubkey.pem")
    conversation_timeout: int = int(os.getenv("CONVERSATION_TIMEOUT", "600"))  # seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


async def wait_for_termination(conversation: HMEConversation, timeout: int):
    try:
        # Wait for either termination or timeout
        await asyncio.wait_for(conversation.wait_for_termination(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Conversation timed out after {timeout} seconds")
    finally:
        # Ensure conversation is properly terminated
        await conversation.terminate()


async def run_conversation(settings: Settings):
    try:
        conversation = create_conversation(settings)
        await conversation.start()
        await wait_for_termination(conversation, settings.conversation_timeout)
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        if conversation:
            await conversation.terminate()
        raise


def create_conversation(settings: Settings) -> HMEConversation:
    transcriber_factory = DefaultTranscriberFactory()
    agent_factory = DefaultAgentFactory()
    synthesizer_factory = DefaultSynthesizerFactory()
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
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        prompt_preamble="""You are a CUSTOMER at a drive-through.
        You want to order a classic burger with cheese and a side of fries.
        Respond naturally to the drive-through worker's questions, be concise and direct, preferring short YES/NO answers.
        Keep responses concise and direct, as if speaking through a drive-through speaker.
        If you hear a partial or cut-off response from the worker, wait for them to repeat or clarify rather than taking on their role.
        Never pretend to be the drive-through worker - you are always the customer.""",
        actions=[
            EndConversationVocodeActionConfig(
                action_trigger=PhraseBasedActionTrigger(
                    config=PhraseBasedActionTriggerConfig(
                        phrase_triggers=[
                            PhraseTrigger(
                                phrase="goodbye", conditions=["phrase_condition_type_contains"]
                            )
                        ]
                    )
                )
            )
        ],
    )
    synthesizer_config = AzureSynthesizerConfig.from_output_device(
        output_device=output_device,
        voice_name="en-US-SteffanNeural",
        trailing_silence_seconds=0.5,
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
    return HMEOutputDevice(
        sampling_rate=DEFAULT_SAMPLING_RATE,
        audio_encoding=AUDIO_ENCODING,
        audio_mode=settings.audio_mode,
        enable_local_playback=True,
    )


async def main():
    settings = Settings()
    await run_conversation(settings)


if __name__ == "__main__":
    asyncio.run(main())
