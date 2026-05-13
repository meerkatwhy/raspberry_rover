"""
pip install onnxruntime numpy tqdm scipy
"""

import pyaudio
from openwakeword.model import Model
import numpy as np


class Microphone:
    def __init__(
        self,
        threshold: float = 0.5,
        rate: int = 16_000,
        chunk: int = 1_280,
    ) -> None:
        self.threshold = threshold
        self.chunk = chunk
        self.model = Model(
            wakeword_models=["hey_rhasspy"],
            inference_framework="onnx",
        )
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

    def listen(self) -> None:
        while True:
            raw_audio = self.stream.read(
                self.chunk,
                exception_on_overflow=False,
            )

            samples = np.frombuffer(raw_audio, dtype=np.int16)
            score = max(
                self.model.predict(samples).values(),
                default=0.0,
            )

            if score >= self.threshold:
                self.model.reset()
                return

    def close(self) -> None:
        self.stream.close()
        self.audio.terminate()