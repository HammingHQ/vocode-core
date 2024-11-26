from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.model import BaseModel, TypedModel
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig, SynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    TranscriberConfig,
)
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HOLD_DURATION,
    DEFAULT_HOLD_MESSAGE_DELAY,
    DEFAULT_IVR_HANDOFF_DELAY,
    DEFAULT_SAMPLING_RATE,
    VONAGE_AUDIO_ENCODING,
    VONAGE_CHUNK_SIZE,
    VONAGE_SAMPLING_RATE,
)


class TelephonyProviderConfig(BaseModel):
    record: bool = False


class TwilioConfig(TelephonyProviderConfig):
    account_sid: str
    auth_token: str
    extra_params: Optional[Dict[str, Any]] = {}
    account_supports_any_caller_id: bool = True


class VonageConfig(TelephonyProviderConfig):
    api_key: str
    api_secret: str
    application_id: str
    private_key: str


class CallEntity(BaseModel):
    phone_number: str


class CreateInboundCall(BaseModel):
    recipient: CallEntity
    caller: CallEntity
    transcriber_config: Optional[TranscriberConfig] = None
    agent_config: AgentConfig
    synthesizer_config: Optional[SynthesizerConfig] = None
    vonage_uuid: Optional[str] = None
    twilio_sid: Optional[str] = None
    conversation_id: Optional[str] = None
    twilio_config: Optional[TwilioConfig] = None
    vonage_config: Optional[VonageConfig] = None


class EndOutboundCall(BaseModel):
    call_id: str
    vonage_config: Optional[VonageConfig] = None
    twilio_config: Optional[TwilioConfig] = None


class CreateOutboundCall(BaseModel):
    recipient: CallEntity
    caller: CallEntity
    transcriber_config: Optional[TranscriberConfig] = None
    agent_config: AgentConfig
    synthesizer_config: Optional[SynthesizerConfig] = None
    conversation_id: Optional[str] = None
    vonage_config: Optional[VonageConfig] = None
    twilio_config: Optional[TwilioConfig] = None
    # TODO add IVR/etc.


class DialIntoZoomCall(BaseModel):
    recipient: CallEntity
    caller: CallEntity
    zoom_meeting_id: str
    zoom_meeting_password: Optional[str]
    transcriber_config: Optional[TranscriberConfig] = None
    agent_config: AgentConfig
    synthesizer_config: Optional[SynthesizerConfig] = None
    conversation_id: Optional[str] = None
    vonage_config: Optional[VonageConfig] = None
    twilio_config: Optional[TwilioConfig] = None


class CallConfigType(str, Enum):
    BASE = "call_config_base"
    TWILIO = "call_config_twilio"
    VONAGE = "call_config_vonage"


PhoneCallDirection = Literal["inbound", "outbound"]


class IvrConfig(BaseModel):
    ivr_message: Optional[str] = None
    wait_for_keypress: bool = False
    ivr_handoff_delay: Optional[float] = DEFAULT_IVR_HANDOFF_DELAY
    hold_message: Optional[str] = None
    hold_message_delay: Optional[float] = DEFAULT_HOLD_MESSAGE_DELAY
    hold_duration: Optional[float] = DEFAULT_HOLD_DURATION


class IvrLink(BaseModel):
    message: str
    next: str


class IvrNodeType(str, Enum):
    BASE = "ivr_node_base"
    MESSAGE = "ivr_node_message"
    HANDOFF = "ivr_node_handoff"
    HOLD = "ivr_node_hold"
    PLAY = "ivr_node_play"
    END = "ivr_node_end"

class IvrLinkType(str, Enum):
    COMMAND = "command"
    DTMF = "dtmf"

class IvrBaseNode(TypedModel, type=IvrNodeType.BASE.value):  # type: ignore
    id: str
    wait_delay: Optional[float] = 0
    is_final: Optional[bool] = False
    link_type: Optional[IvrLinkType] = IvrLinkType.COMMAND
    links: Optional[List[IvrLink]] = []


class IvrMessageNode(IvrBaseNode, type=IvrNodeType.MESSAGE.value):  # type: ignore
    message: str


class IvrHoldNode(IvrBaseNode, type=IvrNodeType.HOLD.value):  # type: ignore
    messages: List[str]
    delay: float
    duration: float

class IvrPlayNode(IvrBaseNode, type=IvrNodeType.PLAY.value):  # type: ignore
    sound: Literal["beep", "ring"]
    delay: float

class IvrEndNode(IvrBaseNode, type=IvrNodeType.END.value):  # type: ignore
    is_final: bool = True

IvrNode = Union[IvrMessageNode, IvrHoldNode, IvrPlayNode, IvrEndNode]


class IvrDagConfig(BaseModel):
    start: str
    nodes: Dict[str, IvrNode]
    fuzz_threshold: Optional[int] = 80


class BaseCallConfig(TypedModel, type=CallConfigType.BASE.value):  # type: ignore
    transcriber_config: TranscriberConfig
    agent_config: AgentConfig
    synthesizer_config: SynthesizerConfig
    from_phone: str
    to_phone: str
    sentry_tags: Dict[str, str] = {}
    conference: bool = False
    telephony_params: Optional[Dict[str, str]] = None
    direction: PhoneCallDirection
    ivr_config: Optional[IvrConfig] = None
    ivr_dag: Optional[IvrDagConfig] = None
    @staticmethod
    def default_transcriber_config():
        raise NotImplementedError

    @staticmethod
    def default_synthesizer_config():
        raise NotImplementedError


class TwilioCallConfig(BaseCallConfig, type=CallConfigType.TWILIO.value):  # type: ignore
    twilio_config: TwilioConfig
    twilio_sid: str

    @staticmethod
    def default_transcriber_config():
        return DeepgramTranscriberConfig(
            sampling_rate=DEFAULT_SAMPLING_RATE,
            audio_encoding=DEFAULT_AUDIO_ENCODING,
            chunk_size=DEFAULT_CHUNK_SIZE,
            model="phonecall",
            tier="nova",
            endpointing_config=PunctuationEndpointingConfig(),
        )

    @staticmethod
    def default_synthesizer_config():
        return AzureSynthesizerConfig(
            sampling_rate=DEFAULT_SAMPLING_RATE,
            audio_encoding=DEFAULT_AUDIO_ENCODING,
        )


class VonageCallConfig(BaseCallConfig, type=CallConfigType.VONAGE.value):  # type: ignore
    vonage_config: VonageConfig
    vonage_uuid: str
    output_to_speaker: bool = False

    @staticmethod
    def default_transcriber_config():
        return DeepgramTranscriberConfig(
            sampling_rate=VONAGE_SAMPLING_RATE,
            audio_encoding=VONAGE_AUDIO_ENCODING,
            chunk_size=VONAGE_CHUNK_SIZE,
            model="phonecall",
            tier="nova",
            endpointing_config=PunctuationEndpointingConfig(),
        )

    @staticmethod
    def default_synthesizer_config():
        return AzureSynthesizerConfig(
            sampling_rate=VONAGE_SAMPLING_RATE,
            audio_encoding=VONAGE_AUDIO_ENCODING,
        )


TelephonyConfig = Union[TwilioConfig, VonageConfig]
