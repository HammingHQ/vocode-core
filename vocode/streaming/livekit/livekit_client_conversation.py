import asyncio
from typing import List

from livekit import rtc
from loguru import logger

from vocode.streaming.livekit.livekit_events_manager import LiveKitEventsManager
from vocode.streaming.output_device.livekit_output_device import LiveKitOutputDevice
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.livekit.audio_recorder import AudioRecorder


TRACK_COUNT = 2


class LiveKitClientConversation(StreamingConversation[LiveKitOutputDevice]):
    room: rtc.Room
    user_track: rtc.Track
    user_participant: rtc.RemoteParticipant
    audio_recorder: AudioRecorder

    def __init__(self, *args, **kwargs):
        if kwargs.get("events_manager") is None:
            events_manager = LiveKitEventsManager()
            events_manager.attach_conversation(self)
            kwargs["events_manager"] = events_manager
        super().__init__(*args, **kwargs)
        self.receive_frames_task: asyncio.Task | None = None
        self.audio_recorder = AudioRecorder(TRACK_COUNT)

    async def start_room(self, ws_url: str, token: str):
        self.room = rtc.Room()
        self.audio_recorder.start()
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        await self.room.connect(ws_url, token)
        await self.output_device.initialize_source(self.room, self.audio_recorder)
        await super().start()

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"track subscribed: {publication.sid}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self.user_participant = participant
            self.user_track = track
            audio_stream = rtc.AudioStream(track)
            self.receive_frames_task = asyncio.create_task(self._receive_frames(audio_stream))

    async def _receive_frames(
        self,
        audio_stream: rtc.AudioStream,
    ):
        # this is where we will send the frames to transcription
        async for event in audio_stream:
            if self.is_active():
                frame = event.frame
                self.receive_audio(bytes(frame.data))
                if self.audio_recorder:
                    self.audio_recorder.record(1, bytes(frame.data))

    def _on_track_unsubscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.mark_terminated()

    def get_recordings(self) -> List[str]:
        return self.audio_recorder.get_recordings()

    async def terminate(self):
        if self.receive_frames_task:
            self.receive_frames_task.cancel()
        if self.audio_recorder:
            self.audio_recorder.close()
        return await super().terminate()
        # TODO: Enable this. It currently hangs.
        # await self.output_device.uninitialize_source()

    def cleanup(self):
        if self.audio_recorder:
            self.audio_recorder.cleanup()
