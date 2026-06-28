
import io
import time
import wave
from collections.abc import Callable
from threading import Event, Thread

import numpy as np
import torch
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel
from silero_vad import load_silero_vad

from robot import config
from robot.hardware.microphone import Microphone


class Listener:
    """Alternates one microphone stream between wake and utterance modes."""

    def __init__(
        self,
        microphone: Microphone,
        on_wake: Callable[[], None],
        on_transcript: Callable[[str], None],
    ) -> None:
        self._microphone = microphone
        self._on_wake = on_wake
        self._on_transcript = on_transcript
        self._stop = Event()
        self._thread = Thread(target=self._run, name="listener", daemon=True)
        self._wakeword = WakeWordModel(wakeword_models=[config.WAKEWORD_MODEL])
        self._vad = load_silero_vad()
        self._whisper = WhisperModel(
            config.WHISPER_MODEL,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        sequence = 0
        while not self._stop.is_set():
            frame = self._microphone.wait_after(sequence)
            if frame is None:
                continue
            sequence = frame.sequence
            samples = np.frombuffer(frame.data, dtype=np.int16)
            predictions = self._wakeword.predict(samples)
            if max(predictions.values(), default=0.0) < config.WAKEWORD_THRESHOLD:
                continue

            self._on_wake()
            self._wakeword.reset()
            frames, sequence = self._record(frame.sequence)
            transcript = self._transcribe(b"".join(item.data for item in frames))
            if transcript:
                self._on_transcript(transcript)

    def _record(self, trigger_sequence: int):
        frames = self._microphone.preroll_through(trigger_sequence)
        sequence = trigger_sequence
        started = time.monotonic()
        speech_seen = False
        silent_samples = 0
        vad_buffer = np.empty(0, dtype=np.float32)
        self._vad.reset_states()

        while not self._stop.is_set():
            frame = self._microphone.wait_after(sequence)
            if frame is None:
                continue
            sequence = frame.sequence
            frames.append(frame)
            normalized = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
            vad_buffer = np.concatenate((vad_buffer, normalized / 32768.0))

            while len(vad_buffer) >= 512:
                window, vad_buffer = vad_buffer[:512], vad_buffer[512:]
                probability = self._vad(torch.from_numpy(window), config.AUDIO_RATE).item()
                if probability >= config.VAD_THRESHOLD:
                    speech_seen = True
                    silent_samples = 0
                elif speech_seen:
                    silent_samples += len(window)

            elapsed = time.monotonic() - started
            silence_ms = silent_samples * 1000 / config.AUDIO_RATE
            if (
                speech_seen
                and elapsed >= config.VAD_MIN_RECORDING_S
                and silence_ms >= config.VAD_SILENCE_MS
            ):
                break
            if not speech_seen and elapsed >= config.VAD_START_TIMEOUT_S:
                break
            if elapsed >= config.MAX_RECORDING_S:
                break
        return frames, sequence

    def _transcribe(self, pcm: bytes) -> str:
        wav = io.BytesIO()
        with wave.open(wav, "wb") as output:
            output.setnchannels(config.AUDIO_CHANNELS)
            output.setsampwidth(2)
            output.setframerate(config.AUDIO_RATE)
            output.writeframes(pcm)
        wav.seek(0)
        segments, _ = self._whisper.transcribe(
            wav,
            language="en",
            beam_size=1,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def wait(self) -> None:
        self._thread.join()
