import time

from gpiozero import Motor

from robot import config


SPEED = 0.12
SECONDS_PER_STEP = 3


def main() -> None:
    left = Motor(
        forward=config.MOTOR_LEFT_FORWARD_PIN,
        backward=config.MOTOR_LEFT_BACKWARD_PIN,
        enable=config.MOTOR_LEFT_PWM_PIN,
        pwm=True,
    )
    right = Motor(
        forward=config.MOTOR_RIGHT_FORWARD_PIN,
        backward=config.MOTOR_RIGHT_BACKWARD_PIN,
        enable=config.MOTOR_RIGHT_PWM_PIN,
        pwm=True,
    )

    steps = (
        ("left forward", lambda: left.forward(SPEED)),
        ("left backward", lambda: left.backward(SPEED)),
        ("right forward", lambda: right.forward(SPEED)),
        ("right backward", lambda: right.backward(SPEED)),
        ("both forward", lambda: (left.forward(SPEED), right.forward(SPEED))),
        ("both backward", lambda: (left.backward(SPEED), right.backward(SPEED))),
    )

    try:
        for name, action in steps:
            print(f"{name} at {SPEED:.0%} for {SECONDS_PER_STEP} seconds")
            action()
            time.sleep(SECONDS_PER_STEP)
            left.stop()
            right.stop()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMotor test interrupted")
    finally:
        left.stop()
        right.stop()
        left.close()
        right.close()


if __name__ == "__main__":
    main()
