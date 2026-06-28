
from math import pi
from threading import Event, Lock

from gpiozero import DigitalInputDevice, DigitalOutputDevice, Motor

from robot import config


class EncoderBank:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counts = [0] * len(config.ENCODER_PINS)
        self._devices = []
        for index, pin in enumerate(config.ENCODER_PINS):
            device = DigitalInputDevice(pin, pull_up=True)
            device.when_activated = lambda i=index: self._pulse(i)
            self._devices.append(device)

    def _pulse(self, index: int) -> None:
        with self._lock:
            self._counts[index] += 1

    def reset(self) -> None:
        with self._lock:
            self._counts = [0] * len(self._counts)

    @property
    def distance_m(self) -> float:
        with self._lock:
            mean_pulses = sum(self._counts) / len(self._counts)
        revolutions = mean_pulses / config.ENCODER_PULSES_PER_REVOLUTION
        return revolutions * pi * config.WHEEL_DIAMETER_M

    def close(self) -> None:
        for device in self._devices:
            device.close()


class Drive:
    def __init__(self) -> None:
        self.left = Motor(
            forward=config.MOTOR_LEFT_FORWARD_PIN,
            backward=config.MOTOR_LEFT_BACKWARD_PIN,
            enable=config.MOTOR_LEFT_PWM_PIN,
            pwm=True,
        )
        self.right = Motor(
            forward=config.MOTOR_RIGHT_FORWARD_PIN,
            backward=config.MOTOR_RIGHT_BACKWARD_PIN,
            enable=config.MOTOR_RIGHT_PWM_PIN,
            pwm=True,
        )
        self.standby = DigitalOutputDevice(config.MOTOR_STANDBY_PIN)
        self.encoders = EncoderBank()
        self._inhibited = Event()
        self.standby.on()

    def set(self, left: float, right: float) -> None:
        if self._inhibited.is_set() and left > 0 and right > 0:
            self.stop()
            return
        left = max(-1.0, min(1.0, left * config.LEFT_SPEED_SCALE))
        right = max(-1.0, min(1.0, right * config.RIGHT_SPEED_SCALE))
        self._set_motor(self.left, left)
        self._set_motor(self.right, right)

    @staticmethod
    def _set_motor(motor: Motor, speed: float) -> None:
        if speed >= 0:
            motor.forward(speed)
        else:
            motor.backward(-speed)

    def stop(self) -> None:
        self.left.stop()
        self.right.stop()

    def inhibit(self, blocked: bool) -> None:
        if blocked:
            self._inhibited.set()
            self.stop()
        else:
            self._inhibited.clear()

    def close(self) -> None:
        self.stop()
        self.encoders.close()
        self.left.close()
        self.right.close()
        self.standby.close()
