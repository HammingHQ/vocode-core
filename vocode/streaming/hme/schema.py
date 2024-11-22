from enum import Enum
from typing import Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field


class Lane(int, Enum):
    ONE = 1
    TWO = 2


class TopicTemplate(str, Enum):
    # Request topics
    ARRIVE = "NEXEO/request/lane{}/arrive"
    DEPART = "NEXEO/request/lane{}/depart"
    AUTO_ESCALATION = "aot/request/auto-escalation/lane{}"
    RECONNECT = "aot/request/reconnect"  # Lane-independent
    AUDIO_INTERRUPTION = "aot/request/audio-interruption/lane{}"

    # Bot topics
    BOT_AUDIO = "aot/response/lane{}-audio"  # For bot audio broadcasting
    BOT_AVAILABLE = "NEXEO/aot/available_bots_count"

    # Alert topics
    CREW_ESCALATION = "NEXEO/alert/crew-escalation/lane{}"
    AUDIO_ALERT = "NEXEO/alert/audio/lane{}"

    def for_lane(self, lane: Lane) -> str:
        """Generate topic string for specific lane"""
        if self in {self.RECONNECT, self.BOT_AVAILABLE}:  # Lane-independent topics
            return self.value
        return self.value.format(lane.value)


class Topic(BaseModel):
    """Dynamic topic generator for multi-lane support"""

    template: TopicTemplate
    lane: Optional[Lane] = None

    @property
    def value(self) -> str:
        return self.template.for_lane(self.lane) if self.lane else self.template.value

    @classmethod
    def arrive(cls, lane: Lane) -> "Topic":
        return cls(template=TopicTemplate.ARRIVE, lane=lane)

    @classmethod
    def depart(cls, lane: Lane) -> "Topic":
        return cls(template=TopicTemplate.DEPART, lane=lane)

    @classmethod
    def crew_escalation(cls, lane: Lane) -> "Topic":
        return cls(template=TopicTemplate.CREW_ESCALATION, lane=lane)

    @classmethod
    def bot_audio(cls, lane: Lane) -> "Topic":
        return cls(template=TopicTemplate.BOT_AUDIO, lane=lane)


class MsgType(str, Enum):
    REQUEST = "request"
    ALERT = "alert"
    STATUS = "status"
    BOT = "bot"


class EscalationBy(str, Enum):
    CREW = "CREW"
    AUTO = "AUTO"


class BaseMetadata(BaseModel):
    device_id: str = Field(..., alias="deviceID")
    timestamp: str
    msg_id: str = Field(..., alias="msgId")
    store_id: str = Field(..., alias="storeId")
    msg_type: MsgType
    car_id: Optional[str] = Field(None, alias="carID")


class BasePayload(BaseModel):
    topic: str
    lane: Optional[Lane] = None


class ArrivePayload(BasePayload):
    lane: Optional[Lane] = Lane.ONE


class DepartPayload(BasePayload):
    lane: Optional[Lane] = Lane.ONE


class EscalationPayload(BasePayload):
    lane: Optional[Lane] = Lane.ONE
    escalation: Dict[str, Union[bool, EscalationBy]] = Field(
        default={
            "escalationBool": True,
            "escalationBy": EscalationBy.CREW,
        }
    )


class BotAudioPayload(BasePayload):
    lane: Optional[Lane] = Lane.ONE
    audio_data: bytes
    chunk_size: int = 960  # Employee-BOT audio chunk size


class StatusPayload(BasePayload):
    configuration: Optional[Dict] = None  # For NEXEO/status/configuration
    telemetry: Optional[Dict] = None  # For NEXEO/status/telemetry
    audio_telemetry: Optional[Dict] = None  # For NEXEO/audio/telemetry
    error: Optional[Dict] = None  # For NEXEO/status/error


class BaseMessage(BaseModel):
    meta: BaseMetadata
    payload: BasePayload


TOPIC_TO_MSGTYPE = {
    TopicTemplate.ARRIVE: MsgType.REQUEST,
    TopicTemplate.DEPART: MsgType.REQUEST,
    TopicTemplate.AUTO_ESCALATION: MsgType.REQUEST,
    TopicTemplate.CREW_ESCALATION: MsgType.ALERT,
    TopicTemplate.AUDIO_ALERT: MsgType.ALERT,
    TopicTemplate.RECONNECT: MsgType.STATUS,
    TopicTemplate.AUDIO_INTERRUPTION: MsgType.STATUS,
    TopicTemplate.BOT_AUDIO: MsgType.BOT,
    TopicTemplate.BOT_AVAILABLE: MsgType.STATUS,
}


def _topic_to_msgtype(topic: Topic) -> MsgType:
    return TOPIC_TO_MSGTYPE[topic.template]


def _msgtype_to_payloads(msg_type: MsgType) -> Tuple[Type[BasePayload]]:
    return MSGTYPE_TO_PAYLOADS[msg_type]


MSGTYPE_TO_PAYLOADS = {
    MsgType.REQUEST: (ArrivePayload, DepartPayload),
    MsgType.ALERT: (EscalationPayload,),
    MsgType.BOT: (BotAudioPayload,),
    MsgType.STATUS: (StatusPayload,),
}


class AudioStats(BaseModel):
    """Tracks audio streaming statistics"""

    bytes_sent: int = 0
    chunks_sent: int = 0
    errors: int = 0
    last_error: Optional[str] = None

    def update_sent(self, bytes_sent: int) -> None:
        """Update stats after sending audio chunk"""
        self.bytes_sent += bytes_sent
        self.chunks_sent += 1

    def increment_errors(self, error_msg: Optional[str] = None) -> None:
        """Track error occurrence"""
        self.errors += 1
        if error_msg:
            self.last_error = error_msg
