from typing import Optional

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.model import BaseModel


class InputAudioConfig(BaseModel):
    sampling_rate: int
    audio_encoding: AudioEncoding
    chunk_size: int
    downsampling: Optional[int] = None
    mute_during_speech: bool = False


class OutputAudioConfig(BaseModel):
    sampling_rate: int
    audio_encoding: AudioEncoding
