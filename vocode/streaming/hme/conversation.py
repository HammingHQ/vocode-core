import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Union

from loguru import logger

from vocode.streaming.hme.events_manager import HMEEventsManager
from vocode.streaming.hme.hme_metrics import ConversationMetrics
from vocode.streaming.hme.schema import BaseMessage as HMEBaseMessage
from vocode.streaming.hme.schema import Lane, MsgType, Topic, TopicTemplate
from vocode.streaming.hme.state import ConversationState, ConversationStateContext
from vocode.streaming.hme.websocket_connection import (
    ConnectionPool,
    StreamType,
    WebSocketConnection,
)
from vocode.streaming.models.message import BaseMessage as VocodeBaseMessage
from vocode.streaming.output_device.hme_output_device import HMEOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation


class HMEConversation(StreamingConversation[HMEOutputDevice]):
    def __init__(
        self,
        aot_provider_url: str,
        client_id: str = "hamming-1",
        store_id: str = "default-store",
        lanes: List[Lane] = [Lane.ONE],
        audio_mode: str = "Fixed",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lanes = lanes
        self.client_id = client_id
        self.store_id = store_id
        self.aot_provider_url = aot_provider_url
        self.audio_mode = audio_mode

        # State management
        self._events = HMEEventsManager()
        self._state = ConversationStateContext(state=ConversationState.IDLE)
        self._metrics = ConversationMetrics(start_time=datetime.now())
        self._message_router = MessageRouter(self)
        self._terminated = asyncio.Event()
        self._terminate_lock = asyncio.Lock()

    async def _register_event_handlers(self) -> None:
        """Register event handlers for the conversation"""
        await self._events.subscribe("state_change", self._handle_state_change)
        await self._events.subscribe("audio_received", self._handle_audio_received)
        await self._events.subscribe("error", self._handle_error)

    async def start(self) -> None:
        try:
            # Register event handlers first
            await self._register_event_handlers()

            # Start the output device which manages connections
            if not self.output_device:
                raise ValueError("Output device not set")

            await self.output_device.start()

            # Send arrive message through output device
            arrive_message = {
                "type": "arrive",
                "client_id": self.client_id,
                "store_id": self.store_id,
            }

            # Change order: First emit state change, then send arrive message
            await self._events.emit("state_change", new_state=ConversationState.ACTIVE)
            await self.output_device.send_message(StreamType.MESSAGE, arrive_message)

            await super().start()

        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            await self.terminate()
            raise

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._terminated.is_set():
                return

            self._terminated.set()
            await self._events.emit("state_change", new_state=ConversationState.TERMINATED)
            await self.output_device.terminate()
            await super().terminate()

    async def _handle_audio_received(self, audio_data: bytes) -> None:
        """Handle received audio data"""
        try:
            if not self._terminated.is_set():
                self.transcriber.send_audio(audio_data)
                if hasattr(self._metrics, "audio_metrics"):
                    self._metrics.audio_metrics.bytes_received += len(audio_data)
                    self._metrics.audio_metrics.chunks_processed += 1
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            await self._events.emit("error", error_message=error_msg)

    async def _handle_error(self, error_message: str, critical: bool = False, **kwargs) -> None:
        """Handle error events"""
        logger.error(f"Conversation error: {error_message}")
        if critical:
            logger.critical("Critical error - terminating conversation")
            await self.terminate()

    async def _handle_state_change(self, new_state: ConversationState, **kwargs) -> None:
        """Handle state changes in the conversation.

        Args:
            new_state: The new state to transition to
            **kwargs: Additional state change metadata
        """
        try:
            old_state = self._state.state
            self._state.state = new_state

            # Update metrics
            self._metrics.state_changes[new_state.value] = datetime.now()

            # Log state change
            logger.info(f"State change: {old_state.value} -> {new_state.value}")

            # Handle specific state transitions
            if new_state == ConversationState.ACTIVE:
                # Start transcriber when conversation becomes active
                if self.transcriber:
                    await self.transcriber.start()

            elif new_state == ConversationState.TERMINATED:
                # Stop transcriber when conversation terminates
                if self.transcriber:
                    await self.transcriber.terminate()

            # Emit state change event for other handlers
            await self._events.emit(
                "state_changed", old_state=old_state, new_state=new_state, **kwargs
            )

        except Exception as e:
            error_msg = f"Error handling state change: {str(e)}"
            logger.error(error_msg)
            await self._events.emit("error", error_message=error_msg, critical=True)


class MessageRouter:
    def __init__(self, conversation: "HMEConversation"):
        self.conversation = conversation
        self.routes = {
            MsgType.REQUEST: self._handle_request,
            MsgType.ALERT: self._handle_alert,
            MsgType.STATUS: self._handle_status,
            MsgType.BOT: self._handle_bot,
        }

    async def _handle_request(self, message: HMEBaseMessage) -> None:
        topic = Topic(template=TopicTemplate(message.payload.topic))
        match topic.template:
            case TopicTemplate.ARRIVE:
                await self._handle_arrive(message)
            case TopicTemplate.DEPART:
                await self._handle_depart(message)
            case TopicTemplate.AUTO_ESCALATION:
                await self._handle_auto_escalation(message)
            case TopicTemplate.RECONNECT:
                await self._handle_reconnect(message)
            case TopicTemplate.AUDIO_INTERRUPTION:
                await self._handle_audio_interruption(message)

    async def _handle_alert(self, message: HMEBaseMessage) -> None:
        topic = Topic(template=TopicTemplate(message.payload.topic))
        if topic.template == TopicTemplate.CREW_ESCALATION:
            await self._handle_crew_escalation(message)
        elif topic.template == TopicTemplate.AUDIO_ALERT:
            await self._handle_audio_alert(message)

    async def _handle_status(self, message: HMEBaseMessage) -> None:
        topic = Topic(template=TopicTemplate(message.payload.topic))
        if topic.template == TopicTemplate.BOT_AVAILABLE:
            await self._handle_bot_status(message)
        else:
            logger.info(f"Received status update for {topic.value}")

    async def _handle_bot(self, message: HMEBaseMessage) -> None:
        topic = Topic(template=TopicTemplate(message.payload.topic))
        if topic.template == TopicTemplate.BOT_AUDIO:
            await self._handle_bot_audio(message)

    async def _handle_arrive(self, message: HMEBaseMessage) -> None:
        logger.info(f"Vehicle arrived at lane {message.payload.lane}")
        await self.conversation.transcriber.start()

        initial_message = VocodeBaseMessage(
            text="Hello! Welcome to our drive-thru. What can I get for you today?"
        )
        await self.conversation.send_initial_message(
            initial_message,
            self.conversation.initial_message_tracker,
        )

    async def _handle_depart(self, message: HMEBaseMessage) -> None:
        logger.info(f"Vehicle departed from lane {message.payload.lane}")
        await self.conversation.transcriber.terminate()

    async def _handle_auto_escalation(self, message: HMEBaseMessage) -> None:
        logger.info(f"Auto escalation requested for lane {message.payload.lane}")
        await self.conversation.transcriber.terminate()

    async def _handle_reconnect(self, message: HMEBaseMessage) -> None:
        logger.info("Reconnection requested")

    async def _handle_audio_interruption(self, message: HMEBaseMessage) -> None:
        logger.info(f"Audio interrupted on lane {message.payload.lane}")
        await self.conversation.broadcast_interrupt()

    async def _handle_crew_escalation(self, message: HMEBaseMessage) -> None:
        logger.info(f"Crew escalation on lane {message.payload.lane}")
        await self.conversation.transcriber.terminate()

    async def _handle_audio_alert(self, message: HMEBaseMessage) -> None:
        logger.warning(f"Audio alert received for lane {message.payload.lane}")
        await self.conversation.broadcast_interrupt()

    async def _handle_bot_status(self, message: HMEBaseMessage) -> None:
        logger.info(
            f"Bot status update received for lane " f"{message.payload.lane}: {message.payload}"
        )

    async def _handle_bot_audio(self, message: HMEBaseMessage) -> None:
        logger.info("Bot audio received")

    async def route(self, message: Union[str, bytes]) -> None:
        try:
            event = HMEBaseMessage.model_validate(message)
            handler = self.routes.get(event.meta.msg_type)
            if handler:
                await handler(event)
            else:
                logger.warning(f"Unknown message type: {event.meta.msg_type}")
        except Exception as e:
            logger.error(f"Error routing message: {str(e)}")
