from gpiozero import DistanceSensor

from robot import config


class Ultrasonic:
    def __init__(self) -> None:
        self._sensor = DistanceSensor(
            echo=config.ULTRASONIC_ECHO_PIN,
            trigger=config.ULTRASONIC_TRIGGER_PIN,
            max_distance=4,
        )

    @property
    def distance_m(self) -> float:
        return self._sensor.distance

    @property
    def blocked(self) -> bool:
        return self.distance_m <= config.OBSTACLE_DISTANCE_M

    def close(self) -> None:
        self._sensor.close()
