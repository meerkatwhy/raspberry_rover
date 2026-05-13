from timeit import timeit
from camera import Camera


camera = Camera(model_path="models/yolo26n.onnx", size=(1280, 1280), conf=0.25)

def test():
    camera.detect_objects()

print(timeit(test, number = 1))
