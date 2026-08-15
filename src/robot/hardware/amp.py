import subprocess
from collections.abc import Iterable
from threading import Lock


class Amp:
    def __init__(self, device: str | None) -> None:
        self._device = device
        self._lock = Lock()
        self._player: subprocess.Popen | None = None

    def play(self, chunks: Iterable[bytes], sample_rate: int, channels: int, cancel=None) -> None:
        command = ["aplay", "-q", "-r", str(sample_rate), "-f", "S16_LE", "-t", "raw", "-c", str(channels)]
        if self._device is not None:
            command.extend(("-D", self._device))
        player = subprocess.Popen(command, stdin=subprocess.PIPE)
        with self._lock:
            self._player = player
        try:
            for chunk in chunks:
                if cancel is not None and cancel.is_set():
                    break
                player.stdin.write(chunk)
            player.stdin.close()
            player.wait()
        finally:
            with self._lock:
                self._player = None

    def stop(self) -> None:
        with self._lock:
            player = self._player
        if player is not None and player.poll() is None:
            player.terminate()
