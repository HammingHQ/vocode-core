from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ConversationState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ESCALATED = "escalated"
    TERMINATED = "terminated"


@dataclass
class ConversationStateContext:
    state: ConversationState
    lane: Optional[int] = None
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
