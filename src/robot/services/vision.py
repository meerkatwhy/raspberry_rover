import asyncio
from dataclasses import dataclass
from ultralytics import YOLO
from robot.hardware.camera import Camera

@dataclass(frozen=True, slots=True)
class Detection:
    """
    Represents a detection result of a single object.
    x1, y1 are top-left coordinates.
    x2, y2 are bottom-right coordinates.
    Coordinates are normalized to [0, 1].
    """
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

class Vision:
    def __init__(self, 
                 model_path: str, 
                 size:       tuple[int, int], 
                 confidence: float) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path, task="detect")
        self.confidence = confidence
        self.camera = Camera(size=size)

    async def detect_objects(self) -> list[Detection]:
        """
        Captures a frame from the camera and runs YOLO to detect objects.
        Returns a list of Detection objects.
        """
        return await asyncio.to_thread(self._detect_objects_sync)

    def _detect_objects_sync(self) -> list[Detection]:
        frame = self.camera.capture_frame()
        result = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    verbose=False,
                )[0]
        class_ids = result.boxes.cls
        xyxyn = result.boxes.xyxyn

        detections = []
        for class_id, coordinates in zip(class_ids, xyxyn):
            x1, y1, x2, y2 = map(float, coordinates)
            detection = Detection(
                name=self.model.names[int(class_id)],
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            )
            detections.append(detection)
        return detections

    def close(self) -> None:
        self.camera.close()