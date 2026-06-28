
import subprocess
from threading import Event, Lock

from robot import config


class Speaker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._piper: subprocess.Popen | None = None
        self._player: subprocess.Popen | None = None

    def speak(self, text: str, cancel: Event | None = None) -> None:
        if cancel is not None and cancel.is_set():
            return
        piper_command = [
            config.PIPER_EXECUTABLE,
            "--model",
            str(config.PIPER_MODEL),
            "--output-raw",
        ]
        player_command = [
            "aplay",
            "-q",
            "-r",
            str(config.PIPER_SAMPLE_RATE),
            "-f",
            "S16_LE",
            "-t",
            "raw",
            "-c",
            "1",
        ]
        if config.APLAY_DEVICE is not None:
            player_command.extend(("-D", config.APLAY_DEVICE))

        piper = subprocess.Popen(
            piper_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=False,
        )
        player = subprocess.Popen(player_command, stdin=piper.stdout)
        piper.stdout.close()
        with self._lock:
            self._piper = piper
            self._player = player
        try:
            if cancel is not None and cancel.is_set():
                self.stop()
            try:
                piper.stdin.write((text + "\n").encode())
                piper.stdin.close()
            except BrokenPipeError:
                pass
            piper.wait()
            player.wait()
        finally:
            with self._lock:
                self._piper = None
                self._player = None

    def stop(self) -> None:
        with self._lock:
            processes = (self._player, self._piper)
        for process in processes:
            if process is not None and process.poll() is None:
                process.terminate()
