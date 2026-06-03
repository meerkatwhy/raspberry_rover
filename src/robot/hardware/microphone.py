import pyaudio

class Microphone:
    """Own a mono, 16-bit PyAudio input stream.

    The stream is opened lazily. Set ``device_index`` when the USB microphone
    is not the system's default input device.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        frames_per_buffer: int = 1_280,
        device_index: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.device_index = device_index

        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frames_per_buffer,
        )

    def read(self) -> bytes:
        """Read raw signed 16-bit PCM samples from the microphone."""
        return self.stream.read(
            num_frames = self.frames_per_buffer,
            exception_on_overflow=False,
        )

    def close(self) -> None:
        """Close the input stream and terminate PyAudio."""
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

