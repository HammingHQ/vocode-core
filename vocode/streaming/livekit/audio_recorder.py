from typing import Optional, Dict, List
import wave
import time
import tempfile
import shutil

from loguru import logger

NUM_CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 44100
DELTA_THRESHOLD_MS = 10


class AudioTrack:
    track_id: int
    last_frame_time: float = 0
    wave_file_name: str
    wave_file: Optional[wave.Wave_write] = None

    def __init__(self, track_id: int, base_folder: str):
        self.track_id = track_id
        self.wave_file_name = self._generate_wave_file_name(base_folder)
        self.wave_file = None

    def start(self, start_time: float):
        self.last_frame_time = start_time
        self.wave_file = wave.open(self.wave_file_name, "wb")
        if self.wave_file is None:
            raise Exception(f"Failed to open wave file {self.wave_file_name}")
        self.wave_file.setnchannels(NUM_CHANNELS)
        self.wave_file.setsampwidth(SAMPLE_WIDTH)
        self.wave_file.setframerate(FRAME_RATE)

        logger.info(f"Started recording track {self.track_id} to {self.wave_file_name}")

    def close(self):
        if self.wave_file:
            self.wave_file.close()
            logger.info(f"Closed recording track {self.track_id} to {self.wave_file_name}")

    def record(self, frame: bytes, frame_time: float):
        delta_ms = frame_time - self.last_frame_time
        if delta_ms > DELTA_THRESHOLD_MS:
            silence = b"\x00\x00" * int((delta_ms / 1000) * FRAME_RATE)
            self.wave_file.writeframes(silence)

        self.wave_file.writeframes(frame)
        frame_duration_ms = (len(frame) / SAMPLE_WIDTH) / FRAME_RATE * 1000
        self.last_frame_time = frame_time + frame_duration_ms

    def _generate_wave_file_name(self, base_folder: str):
        return f"{base_folder}/recording_{self.track_id}.wav"


class AudioRecorder:
    start_time: float = 0
    tracks: Dict[int, AudioTrack]
    artifacts_folder: str
    is_recording: bool = False

    def __init__(self, track_count: int):
        self.tracks = {}
        self.artifacts_folder = tempfile.mkdtemp()
        logger.info(f"Created artifacts folder {self.artifacts_folder}")
        logger.info(f"Current tracks: {self.tracks}")
        for i in range(track_count):
            self.tracks[i] = AudioTrack(i, self.artifacts_folder)

    def start(self):
        self.start_time = self._current_time()
        for track in self.tracks.values():
            track.start(self.start_time)
        self.is_recording = True

    def close(self):
        if not self.is_recording:
            return
        self.is_recording = False
        for track in self.tracks.values():
            track.close()

    def record(self, track_id: int, frame: bytes):
        if not self.is_recording:
            return
        track = self.tracks[track_id]
        if not track:
            logger.error(f"No track found for track_id {track_id}")
            return
        track.record(frame, self._current_time())

    def get_recordings(self) -> List[str]:
        return [track.wave_file_name for track in self.tracks.values()]

    def cleanup(self):
        logger.info(f"Cleaning up artifacts folder {self.artifacts_folder}")
        shutil.rmtree(self.artifacts_folder)

    def _current_time(self):
        return time.time_ns() / 1e6  # convert to milliseconds
