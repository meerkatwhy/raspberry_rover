from picamera2 import Picamera2

from robot import config


class Camera:
    def __init__(self) -> None:
        self._camera = Picamera2()
        setup = self._camera.create_video_configuration(
            main={"size": config.CAMERA_SIZE, "format": "RGB888"}
        )
        self._camera.configure(setup)

    def start(self) -> None:
        self._camera.start()

    def capture(self):
        return self._camera.capture_array()

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()
