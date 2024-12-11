from typing import Optional, Tuple

from vocode.streaming.models.telephony import AgentConfig, IvrDagConfig


class BaseDynamicCallManager:
    async def create_call(
        self, twilio_sid: str, from_number: str, to_number: str, call_id: str
    ) -> Tuple[Optional[AgentConfig], Optional[IvrDagConfig]]:
        raise NotImplementedError
