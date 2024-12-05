import asyncio
from signal import SIGINT, SIGTERM

from loguru import logger

from vocode.streaming.hme.hme_conversation import HMEConversation
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig
from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber


async def main():
    # Configuration
    aot_provider_url = "wss://dev.lilaclabs.ai/hme/ben-franks"
    client_id = "test-client"
    store_id = "test-store"

    # Create output device first
    output_device = HMEOutputDevice(
        aot_provider_url=aot_provider_url,
        client_id=client_id,
        store_id=store_id,
        sampling_rate=16000,
        audio_encoding=AudioEncoding.LINEAR16,
        audio_mode="Fixed",
    )

    # Create transcriber
    transcriber = DeepgramTranscriber(
        DeepgramTranscriberConfig(
            sampling_rate=16000,
            audio_encoding=AudioEncoding.LINEAR16,
            chunk_size=1920,
        )
    )

    # Create HME conversation
    conversation = HMEConversation(
        aot_provider_url=aot_provider_url,
        client_id=client_id,
        store_id=store_id,
        output_device=output_device,
        transcriber=transcriber,
        agent=None,
        synthesizer=None,
        audio_mode="Fixed",
    )

    try:
        # Start output device first to establish connections
        await output_device.start()
        logger.info("Output device started")
        # Then start conversation
        await conversation.start()
        logger.info("Conversation started")
        await conversation.terminate()
        await conversation.wait_for_termination()
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
    finally:
        await output_device.terminate()
        await conversation.terminate()


def handle_signals():
    loop = asyncio.get_event_loop()
    for signal in (SIGINT, SIGTERM):
        loop.add_signal_handler(
            signal, lambda s=signal: asyncio.create_task(shutdown(loop, signal=s))
        )


async def shutdown(loop, signal=None):
    if signal:
        logger.info(f"Received exit signal {signal.name}")

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


if __name__ == "__main__":
    try:
        handle_signals()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
