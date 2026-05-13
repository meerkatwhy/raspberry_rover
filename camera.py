from picamera2 import Picamera2
import cv2
from ultralytics import YOLO

class Camera:
    def __init__(self, model_path: str, size: tuple[int, int], conf: float = 0.25) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path, task="detect")
        self.conf = conf
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_still_configuration(main={"size": size}))
        self.camera.start()

    def detect_objects(self) -> list[dict[str, str | float]]:
        """
        Captures one frame and runs YOLO on it.
        log=True will save the frame, with detection boxes drawn on it, to LOCATION
        """
        frame = self.camera.capture_array("main")
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        result = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    verbose=False,
                )[0]
        class_ids = result.boxes.cls
        xyxyn = result.boxes.xyxyn

        detections = []
        for class_id, coordinates in zip(class_ids, xyxyn):
            x1, y1, x2, y2 = map(float, coordinates)
            detection = {
                "name": self.model.names[int(class_id)],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            detections.append(detection)
        return detections

    def close(self) -> None:
        self.camera.stop()
        self.camera.close()
