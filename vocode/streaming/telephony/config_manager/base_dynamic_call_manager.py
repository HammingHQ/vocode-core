from typing import Optional

from vocode.streaming.models.telephony import AgentConfig


class BaseDynamicCallManager:
    async def create_call(
        self, twilio_sid: str, from_number: str, to_number: str, call_id: str
    ) -> Optional[AgentConfig]:
        raise NotImplementedError
