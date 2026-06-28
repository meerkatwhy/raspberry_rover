
from collections import deque
from dataclasses import dataclass
from threading import Condition

import pyaudio

from robot import config


@dataclass(frozen=True)
class AudioFrame:
    sequence: int
    data: bytes


class Microphone:
    """Owns the only PyAudio input stream and publishes ordered audio frames."""

    def __init__(self) -> None:
        frame_seconds = config.AUDIO_CHUNK_SAMPLES / config.AUDIO_RATE
        capacity = int(config.AUDIO_PREROLL_S / frame_seconds) + 8
        self._frames: deque[AudioFrame] = deque(maxlen=capacity)
        self._condition = Condition()
        self._sequence = 0
        self._audio = pyaudio.PyAudio()
        self._stream = None

    def start(self) -> None:
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=config.AUDIO_CHANNELS,
            rate=config.AUDIO_RATE,
            input=True,
            input_device_index=config.AUDIO_DEVICE_INDEX,
            frames_per_buffer=config.AUDIO_CHUNK_SAMPLES,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, data, frame_count, time_info, status):
        with self._condition:
            self._sequence += 1
            self._frames.append(AudioFrame(self._sequence, data))
            self._condition.notify_all()
        return None, pyaudio.paContinue

    def wait_after(self, sequence: int, timeout: float = 1.0) -> AudioFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: bool(self._frames) and self._frames[-1].sequence > sequence,
                timeout,
            )
            return next((frame for frame in self._frames if frame.sequence > sequence), None)

    def preroll_through(self, sequence: int) -> list[AudioFrame]:
        """Return each buffered frame at most once, ending at the trigger frame."""
        with self._condition:
            frames = [frame for frame in self._frames if frame.sequence <= sequence]
            count = round(
                config.AUDIO_PREROLL_S
                * config.AUDIO_RATE
                / config.AUDIO_CHUNK_SAMPLES
            )
            return frames[-count:]

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        self._audio.terminate()
