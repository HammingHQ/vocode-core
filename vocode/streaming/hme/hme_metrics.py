from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class AudioMetrics:
    bytes_sent: int = 0
    bytes_received: int = 0
    chunks_processed: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ConversationMetrics:
    start_time: datetime
    end_time: Optional[datetime] = None
    state_changes: Dict[str, datetime] = field(default_factory=lambda: defaultdict(datetime.now))
    audio_metrics: AudioMetrics = field(default_factory=AudioMetrics)
