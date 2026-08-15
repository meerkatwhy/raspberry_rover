from array import array
from pathlib import Path
import subprocess
from threading import Event, Lock
import wave

from robot import config
from robot.hardware.amp import Amp


def _scale_pcm16(data: bytes, volume: float) -> bytes:
    samples = array("h", data)
    return array("h", (int(sample * volume) for sample in samples)).tobytes()


class Speaker:
    def __init__(self, amp: Amp) -> None:
        self._amp = amp
        self._lock = Lock()
        self._piper: subprocess.Popen | None = None

    def speak(self, text: str, cancel: Event | None = None) -> None:
        if cancel is not None and cancel.is_set():
            return
        piper = subprocess.Popen(
            [config.PIPER_EXECUTABLE, "--model", str(config.PIPER_MODEL), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=False,
        )
        with self._lock:
            self._piper = piper
        try:
            piper.stdin.write((text + "\n").encode())
            piper.stdin.close()
            self._amp.play(self._volume_chunks(piper.stdout, cancel), config.PIPER_SAMPLE_RATE, 1, cancel)
            piper.wait()
        finally:
            piper.stdout.close()
            with self._lock:
                self._piper = None

    def play(self, path: str | Path, cancel: Event | None = None) -> None:
        with wave.open(str(path), "rb") as sound:
            self._amp.play(
                self._volume_chunks(self._wave_chunks(sound), cancel),
                sound.getframerate(),
                sound.getnchannels(),
                cancel,
            )

    def stop(self) -> None:
        with self._lock:
            piper = self._piper
        if piper is not None and piper.poll() is None:
            piper.terminate()
        self._amp.stop()

    @staticmethod
    def _wave_chunks(sound: wave.Wave_read):
        while data := sound.readframes(2048):
            yield data

    @staticmethod
    def _volume_chunks(chunks, cancel: Event | None):
        remainder = b""
        for data in chunks:
            if cancel is not None and cancel.is_set():
                return
            data = remainder + data
            remainder = data[len(data) & ~1 :]
            data = data[: len(data) & ~1]
            if data:
                yield _scale_pcm16(data, config.SPEAKER_VOLUME)
