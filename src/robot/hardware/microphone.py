from dataclasses import dataclass
from queue import Empty, Queue

import pyaudio

from robot import config


@dataclass(frozen=True, slots=True)
class AudioChunk:
    sequence: int
    data: bytes


class Microphone:
    """Owns the only PyAudio input stream and publishes ordered audio frames."""

    def __init__(self) -> None:
        self._chunks = Queue()
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
        self._sequence += 1
        self._chunks.put_nowait(AudioChunk(self._sequence, data))
        return None, pyaudio.paContinue

    def get_chunk(self, timeout: float = 1.0) -> AudioChunk | None:
        try:
            return self._chunks.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._audio.terminate()
