from camera import Camera
from config import MODELS_PATH

camera = Camera(model_path=MODELS_PATH / "yolo26n.onnx", size=(2560, 2560), conf=0.3)
print(camera.detect_objects())

