from robot import config
from robot.hardware.camera import Camera
from robot.hardware.imu import MPU6050
from robot.hardware.microphone import Microphone
from robot.hardware.motors import Drive
from robot.hardware.amp import Amp
from robot.hardware.ultrasonic import Ultrasonic

from robot.services.commands import CommandService
from robot.services.listener import Listener
from robot.services.speaker import Speaker
from robot.services.vision import Vision


def main() -> None:
    microphone = Microphone()
    camera = Camera()
    drive = Drive()
    imu = MPU6050()
    ultrasonic = Ultrasonic()
    speaker = Speaker(Amp(config.APLAY_DEVICE))

    microphone.start()
    camera.start()
    imu.start()
    vision = Vision(camera)
    commands = CommandService(drive, imu, ultrasonic, vision, speaker)
    listener = Listener(microphone, commands.cancel_all, commands.submit_text)
    commands.start()
    listener.start()

    try:
        listener.wait()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        commands.close()
        speaker.stop()
        microphone.close()
        camera.close()
        ultrasonic.close()
        imu.close()
        drive.close()


if __name__ == "__main__":
    main()
