import time

from robot.hardware.camera import Camera
from robot.hardware.microphone import Microphone
from robot.services.commands import CommandService, parse_command
from robot.services.listener import Listener
from robot.services.vision import Vision


class PrintSpeaker:
    def speak(self, text, cancel=None) -> None:
        if cancel is None or not cancel.is_set():
            print(text, flush=True)

    def stop(self) -> None:
        pass


class MockDrive:
    SPEED_MPS = 0.4

    class Encoders:
        def __init__(self, drive) -> None:
            self._drive = drive
            self._distance = 0.0

        @property
        def distance_m(self) -> float:
            self._drive._update_distance()
            return self._distance

        def reset(self) -> None:
            self._drive._update_distance()
            self._distance = 0.0

    def __init__(self) -> None:
        self._left = 0.0
        self._right = 0.0
        self._last_update = time.monotonic()
        self.encoders = self.Encoders(self)

    def _update_distance(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        forward_speed = max(0.0, (self._left + self._right) / 2)
        self.encoders._distance += forward_speed * self.SPEED_MPS * elapsed

    def set(self, left: float, right: float) -> None:
        self._update_distance()
        self._left = left
        self._right = right

    def stop(self) -> None:
        self._update_distance()
        self._left = 0.0
        self._right = 0.0

    def inhibit(self, blocked: bool) -> None:
        pass

    def close(self) -> None:
        pass


class MockIMU:
    # At full opposing motor command, simulate a 200 degree/second turn.
    TURN_RATE_DPS_PER_SPEED_DIFFERENCE = 100.0

    def __init__(self, drive: MockDrive) -> None:
        self._drive = drive

    @property
    def yaw_dps(self) -> float:
        return (
            self._drive._right - self._drive._left
        ) * self.TURN_RATE_DPS_PER_SPEED_DIFFERENCE

    def close(self) -> None:
        pass


class MockUltrasonic:
    blocked = False

    def close(self) -> None:
        pass


def main() -> None:
    microphone = Microphone()
    camera = Camera()
    speaker = PrintSpeaker()
    commands = None
    listener = None

    try:
        camera.start()
        microphone.start()
        vision = Vision(camera)
        drive = MockDrive()
        commands = CommandService(
            drive,
            MockIMU(drive),
            MockUltrasonic(),
            vision,
            speaker,
        )
        def submit_text(text: str) -> None:
            print(text)
            command = parse_command(text)
            if command is not None:
                commands.submit_text(text)
            else:
                print("I didn't understand that command.", flush=True)

        listener = Listener(microphone, commands.cancel_all, submit_text)
        commands.start()
        listener.start()
        listener.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if listener is not None:
            listener.close()
        if commands is not None:
            commands.close()
        speaker.stop()
        microphone.close()
        camera.close()


if __name__ == "__main__":
    main()
