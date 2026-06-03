from picamera2 import Picamera2
import cv2


class Camera:
    def __init__(self, size: tuple[int, int]) -> None:
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_still_configuration(main={"size": size}))
        self.camera.start()

    def capture_frame(self) -> list[dict[str, str | float]]:
        """
        Because the camera is mounted upside down, we need to rotate the frame 180 degrees.
        """
        frame = self.camera.capture_array("main")
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        return frame

    def close(self) -> None:
        self.camera.stop()
        self.camera.close()
