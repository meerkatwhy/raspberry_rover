
import re
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from threading import Condition, Event, Thread

from robot import config
from robot.hardware.imu import MPU6050
from robot.hardware.motors import Drive
from robot.hardware.ultrasonic import Ultrasonic
from robot.services.vision import Vision
from robot.services.speaker import Speaker


class CommandKind(Enum):
    GO = auto()
    TURN = auto()
    DESCRIBE = auto()
    FIND = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    amount: float = 0.0
    direction: int = 0
    target: str = ""


_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        value = 0
        for word in text.replace("-", " ").split():
            if word in _ONES:
                value += _ONES[word]
            elif word in _TENS:
                value += _TENS[word]
            elif word == "hundred":
                value = max(value, 1) * 100
            else:
                return None
        return float(value)


def parse_command(text: str) -> Command | None:
    text = re.sub(r"[^a-z0-9. ]", " ", text.lower())
    text = re.sub(r"\b(?:hey\s+)?rhasspy\b", "", text)
    text = " ".join(text.split())

    match = re.search(r"\bgo ([a-z0-9. -]+?) meters?\b", text)
    if match:
        amount = _number(match.group(1))
        if amount is not None:
            return Command(CommandKind.GO, amount=amount)
    match = re.search(r"\bturn (right|left) ([a-z0-9. -]+?) degrees?\b", text)
    if match:
        amount = _number(match.group(2))
        if amount is not None:
            direction = 1 if match.group(1) == "right" else -1
            return Command(CommandKind.TURN, amount=amount, direction=direction)
    if re.search(r"\bdescribe (?:the )?scene\b", text):
        return Command(CommandKind.DESCRIBE)
    match = re.search(r"\bfind (?:a |an |the )?(.+)$", text)
    if match:
        return Command(CommandKind.FIND, target=match.group(1).strip())
    return None


class CommandService:
    def __init__(
        self,
        drive: Drive,
        imu: MPU6050,
        ultrasonic: Ultrasonic,
        vision: Vision,
        speaker: Speaker,
    ) -> None:
        self._drive = drive
        self._imu = imu
        self._ultrasonic = ultrasonic
        self._vision = vision
        self._speaker = speaker
        self._condition = Condition()
        self._queue: deque[Command] = deque()
        self._current_cancel: Event | None = None
        self._stop = Event()
        self._worker = Thread(target=self._run, name="commands", daemon=True)
        self._safety = Thread(target=self._safety_loop, name="safety", daemon=True)

    def start(self) -> None:
        self._worker.start()
        self._safety.start()

    def submit_text(self, text: str) -> None:
        command = parse_command(text)
        if command is None:
            command = Command(CommandKind.UNKNOWN)
        with self._condition:
            self._queue.append(command)
            self._condition.notify()

    def cancel_all(self) -> None:
        with self._condition:
            self._queue.clear()
            if self._current_cancel is not None:
                self._current_cancel.set()
        self._drive.stop()
        self._speaker.stop()

    def _next(self) -> tuple[Command, Event] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._queue or self._stop.is_set())
            if self._stop.is_set():
                return None
            cancel = Event()
            self._current_cancel = cancel
            return self._queue.popleft(), cancel

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._next()
            if item is None:
                return
            command, cancel = item
            try:
                self._execute(command, cancel)
            finally:
                self._drive.stop()
                with self._condition:
                    self._current_cancel = None

    def _execute(self, command: Command, cancel: Event) -> None:
        if command.kind is CommandKind.GO:
            reached = self._go(command.amount, cancel)
            if not cancel.is_set() and not reached:
                self._speaker.speak("I stopped because something is in the way.", cancel)
        elif command.kind is CommandKind.TURN:
            self._turn(command.direction, command.amount, cancel)
        elif command.kind is CommandKind.DESCRIBE:
            self._speaker.speak(self._vision.describe(), cancel)
        elif command.kind is CommandKind.FIND:
            found = self._find(command.target, cancel)
            if not cancel.is_set():
                message = f"I found the {command.target}." if found else f"I couldn't find the {command.target}."
                self._speaker.speak(message, cancel)
        elif command.kind is CommandKind.UNKNOWN:
            self._speaker.speak("I didn't understand that command.", cancel)

    def _go(self, distance_m: float, cancel: Event) -> bool:
        self._drive.encoders.reset()
        while not cancel.is_set() and self._drive.encoders.distance_m < distance_m:
            if self._ultrasonic.blocked:
                return False
            self._drive.set(config.DRIVE_SPEED, config.DRIVE_SPEED)
            time.sleep(config.CONTROL_PERIOD_S)
        return not cancel.is_set()

    def _turn(self, direction: int, degrees: float, cancel: Event) -> bool:
        angle = 0.0
        previous = time.monotonic()
        while not cancel.is_set() and abs(angle) < degrees:
            speed = config.TURN_SPEED * direction
            self._drive.set(-speed, speed)
            now = time.monotonic()
            angle += self._imu.yaw_dps * (now - previous)
            previous = now
            time.sleep(config.CONTROL_PERIOD_S)
        return not cancel.is_set()

    def _find(self, target: str, cancel: Event) -> bool:
        angle = 0.0
        previous = time.monotonic()
        started = previous
        detection = None
        while (
            not cancel.is_set()
            and abs(angle) < 360
            and time.monotonic() - started < config.SEARCH_TIMEOUT_S
        ):
            detection = self._vision.find(target)
            if detection is not None:
                break
            self._drive.set(config.TURN_SPEED, -config.TURN_SPEED)
            now = time.monotonic()
            angle += self._imu.yaw_dps * (now - previous)
            previous = now
            time.sleep(config.VISION_FRAME_INTERVAL_S)
        self._drive.stop()
        if detection is None or cancel.is_set():
            return False

        while not cancel.is_set() and not self._ultrasonic.blocked:
            detection = self._vision.find(target)
            if detection is None:
                return False
            error = detection.center_x - 0.5
            correction = max(-0.35, min(0.35, error))
            self._drive.set(
                config.DRIVE_SPEED * (1 + correction),
                config.DRIVE_SPEED * (1 - correction),
            )
            time.sleep(config.VISION_FRAME_INTERVAL_S)
        return not cancel.is_set()

    def _safety_loop(self) -> None:
        while not self._stop.is_set():
            self._drive.inhibit(self._ultrasonic.blocked)
            time.sleep(config.CONTROL_PERIOD_S)

    def close(self) -> None:
        self.cancel_all()
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._worker.join(timeout=2)
        self._safety.join(timeout=2)
