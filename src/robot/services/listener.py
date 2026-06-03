import asyncio
import queue
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel
from silero_vad import load_silero_vad

from robot.hardware.microphone import Microphone


@dataclass(frozen=True)
class AudioChunk:
    seq: int
    pcm: bytes


class Listener:
    SAMPLE_RATE = 16_000
    SAMPLE_WIDTH_BYTES = 2

    CAPTURE_CHUNK_SAMPLES = 1_280

    VAD_CHUNK_SAMPLES = 512

    RING_BUFFER_S = 2.0

    WAKEWORD_THRESHOLD = 0.5

    VAD_THRESHOLD = 0.5
    END_SILENCE_SECONDS = 0.8

    MAX_RECORDING_SECONDS = 30.0

    # 512 * 80 ms ~= 41 seconds.
    #
    # This queue does not need to hold the entire recording because
    # transcribe() consumes it continuously. The large bound mainly protects
    # us when the state machine is busy doing something else.
    AUDIO_QUEUE_MAX_CHUNKS = 512

    def __init__(self) -> None:
        # Initialize ML models
        self._wakeword_model = WakeWordModel(["hey_rhasspy"])
        self._vad_model = load_silero_vad()
        self._whisper_model = WhisperModel(
                    "base.en",
                    device="cpu",
                    compute_type="int8",
        )
        self.microphone = Microphone(
            sample_rate=self.SAMPLE_RATE,
            channels=1,
            frames_per_buffer=self.CAPTURE_CHUNK_SAMPLES,
        )

        # Continuous capture infrastructure.
        self._audio_queue: queue.Queue[AudioChunk] = queue.Queue(
            maxsize=self.AUDIO_QUEUE_MAX_CHUNKS
        )

        self._rolling_buffer = deque(
            maxlen=int(self.RING_BUFFER_S
                       * self.SAMPLE_RATE
                       / self.CAPTURE_CHUNK_SAMPLES
            )
        )

        self._rolling_buffer_lock = threading.Lock()

        self._capture_stop = threading.Event()

        self._capture_exception = None

        self._next_seq = 0

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="microphone-capture",
            daemon=True,
        )

        self._capture_thread.start()



    # ------------------------------------------------------------------
    # Microphone capture
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Continuously read the microphone.

        This thread is the ONLY code in Listener allowed to call
        Microphone.read().

        Therefore wake-word inference, VAD inference, state transitions,
        and Whisper inference can never delay the next PyAudio read.
        """

        try:
            while not self._capture_stop.is_set():
                pcm = self.microphone.read()

                chunk = AudioChunk(
                    seq=self._next_seq,
                    pcm=pcm,
                )

                self._next_seq += 1

                #
                # Always retain the newest two seconds.
                #
                with self._rolling_buffer_lock:
                    self._rolling_buffer.append(chunk)

                #
                # Publish to consumers.
                #
                # Never block microphone capture because a downstream
                # consumer is slow.
                #
                try:
                    self._audio_queue.put_nowait(chunk)

                except queue.Full:
                    #
                    # Drop the oldest queued chunk, not the newest.
                    #
                    # The ring buffer independently preserves the latest
                    # two seconds, so dropping stale queued data is safer
                    # than stopping microphone capture.
                    #
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        self._audio_queue.put_nowait(chunk)
                    except queue.Full:
                        pass

        except BaseException as exc:
            self._capture_exception = exc
            self._capture_stop.set()

    def _snapshot_rolling_buffer(self) -> tuple[list[AudioChunk], int]:
        """Atomically snapshot the current pre-roll audio."""
        with self._rolling_buffer_lock:
            chunks = list(self._rolling_buffer)

        if not chunks:
            return [], -1

        return chunks, chunks[-1].seq

    def _drain_audio_queue(self) -> None:
        """Discard queued audio that is no longer relevant."""
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    async def _next_audio_chunk(self) -> AudioChunk:
        """Wait asynchronously for the next captured chunk."""
        while True:
            self._raise_capture_error()

            try:
                return await asyncio.to_thread(
                    self._audio_queue.get,
                    True,
                    0.25,
                )

            except queue.Empty:
                continue

    def _raise_capture_error(self) -> None:
        if self._capture_exception is not None:
            raise RuntimeError(
                "Microphone capture thread failed"
            ) from self._capture_exception

    # ------------------------------------------------------------------
    # Wake word
    # ------------------------------------------------------------------

    async def wait_wakeword(self) -> None:
        """Wait until 'hey rhasspy' is detected."""

        # Importantly, this does NOT touch the rolling buffer. The capture
        # thread continues preserving the last two seconds independently.
        #
        self._drain_audio_queue()

        self._wakeword_model.reset()

        while True:
            chunk = await self._next_audio_chunk()

            pcm = np.frombuffer(
                chunk.pcm,
                dtype=np.int16,
            )

            #
            # Keep wake-word inference away from the capture thread.
            #
            prediction = await asyncio.to_thread(
                self._wakeword_model.predict,
                pcm,
            )

            score = float(
                prediction.get("hey_rhasspy", 0.0)
            )

            if score >= self.WAKEWORD_THRESHOLD:
                self._wakeword_model.reset()
                return

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(self) -> str:
        """Capture one utterance and transcribe it.

        The recording begins with an atomic snapshot of the latest two
        seconds of continuously captured microphone audio.

        Sequence numbers ensure that audio already contained in the
        snapshot is not duplicated when we begin consuming the live queue.
        """

        #
        # Snapshot the two-second rolling buffer.
        #
        # Since the capture thread never stopped, this snapshot includes
        # everything around the wake-word -> transcription transition.
        #
        pre_roll_chunks, snapshot_seq = (
            self._snapshot_rolling_buffer()
        )

        recorded = bytearray()

        for chunk in pre_roll_chunks:
            recorded.extend(chunk.pcm)

        #
        # Don't run VAD over the pre-roll.
        #
        # The wake word itself establishes that speech has occurred. The
        # pre-roll exists for Whisper, not endpoint detection.
        #
        self._reset_vad()

        vad_pending = bytearray()

        silence_samples = 0

        end_silence_samples = int(
            self.END_SILENCE_SECONDS
            * self.SAMPLE_RATE
        )

        max_recording_samples = int(
            self.MAX_RECORDING_SECONDS
            * self.SAMPLE_RATE
        )

        post_snapshot_samples = 0

        vad_frame_bytes = (
            self.VAD_CHUNK_SAMPLES
            * self.SAMPLE_WIDTH_BYTES
        )

        # We want the first chunk strictly AFTER the pre-roll snapshot.
        expected_seq = snapshot_seq + 1

        while True:
            chunk = await self._next_audio_chunk()

            #
            # The queue may contain chunks that were already captured in
            # our rolling-buffer snapshot.
            #
            # Skip them so audio is not duplicated.
            #
            if chunk.seq <= snapshot_seq:
                continue

            #
            # Detect an unexpected queue discontinuity.
            #
            # Normally this should never happen while transcription is
            # actively consuming audio.
            #
            if chunk.seq > expected_seq:
                print(
                    "Warning: microphone audio queue dropped "
                    f"{chunk.seq - expected_seq} chunk(s)"
                )

            expected_seq = chunk.seq + 1

            recorded.extend(chunk.pcm)
            vad_pending.extend(chunk.pcm)

            chunk_samples = (
                len(chunk.pcm)
                // self.SAMPLE_WIDTH_BYTES
            )

            post_snapshot_samples += chunk_samples

            #
            # Convert the 1280-sample capture chunks into the 512-sample
            # windows required by Silero.
            #
            while len(vad_pending) >= vad_frame_bytes:
                vad_bytes = bytes(vad_pending[:vad_frame_bytes])

                del vad_pending[:vad_frame_bytes]

                speech_probability = (
                    await asyncio.to_thread(
                        self._speech_probability,
                        vad_bytes,
                    )
                )

                if speech_probability >= self.VAD_THRESHOLD:
                    silence_samples = 0
                else:
                    silence_samples += self.VAD_CHUNK_SAMPLES

                if silence_samples >= end_silence_samples:
                    return await self._run_whisper(bytes(recorded))

            if post_snapshot_samples >= max_recording_samples:
                return await self._run_whisper(bytes(recorded))

    # ------------------------------------------------------------------
    # Silero
    # ------------------------------------------------------------------

    def _speech_probability(self, pcm_bytes: bytes) -> float:
        """Evaluate one 512-sample Silero VAD window."""
        pcm = np.frombuffer(
            pcm_bytes,
            dtype=np.int16,
        ).astype(np.float32)

        pcm /= 32768.0

        audio = torch.from_numpy(pcm)

        with torch.no_grad():
            probability = self._vad_model(
                audio,
                self.SAMPLE_RATE,
            )

        return float(probability.item())

    def _reset_vad(self) -> None:
        if self._vad_model is not None:
            self._vad_model.reset_states()

    # ------------------------------------------------------------------
    # Whisper
    # ------------------------------------------------------------------

    async def _run_whisper(self, pcm_bytes: bytes) -> str:
        audio = np.frombuffer(
            pcm_bytes,
            dtype=np.int16,
        ).astype(np.float32)

        audio /= 32768.0

        transcription = await asyncio.to_thread(self._transcribe_sync, audio)
        return transcription.strip()

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        # Segment generation is lazy, so both transcribe() and iteration
        # need to execute in this worker thread.
        segments, _ = self._whisper_model.transcribe(
            audio=audio,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        # Tell the capture thread to exit.
        self._capture_stop.set()

        if self._capture_thread is not None:
            await asyncio.to_thread(
                self._capture_thread.join,
                2.0,
            )
            self._capture_thread = None
        #
        # Only close PyAudio after the reader thread has stopped.
        #
        self.microphone.close()
        self._raise_capture_error()