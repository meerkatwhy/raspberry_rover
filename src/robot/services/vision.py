
from collections import Counter
from dataclasses import dataclass

from ultralytics import YOLO

from robot import config
from robot.hardware.camera import Camera


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    center_x: float


class Vision:
    def __init__(self, camera: Camera) -> None:
        self._camera = camera
        self._model = YOLO(str(config.YOLO_MODEL), task="detect")

    def detect(self) -> list[Detection]:
        image = self._camera.capture()
        height, width = image.shape[:2]
        result = self._model.predict(
            image,
            conf=config.VISION_CONFIDENCE,
            verbose=False,
        )[0]
        detections = []
        for class_id, confidence, box in zip(
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.xyxy.cpu().numpy(),
        ):
            x1, _, x2, _ = box
            detections.append(
                Detection(
                    label=result.names[int(class_id)].lower(),
                    confidence=float(confidence),
                    center_x=float((x1 + x2) / (2 * width)),
                )
            )
        return detections

    def find(self, label: str) -> Detection | None:
        matches = [item for item in self.detect() if item.label == label.lower()]
        return max(matches, key=lambda item: item.confidence, default=None)

    def describe(self) -> str:
        counts = Counter(item.label for item in self.detect())
        if not counts:
            return "I don't see any recognized objects."
        objects = [
            f"{count} {label if count == 1 else label + 's'}"
            for label, count in sorted(counts.items())
        ]
        return "I see " + ", ".join(objects) + "."
