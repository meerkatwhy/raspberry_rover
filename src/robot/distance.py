import time

from robot.hardware.ultrasonic import Ultrasonic


def main() -> None:
    sensor = Ultrasonic()
    try:
        while True:
            print(f"{sensor.distance_m:.3f} m", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
