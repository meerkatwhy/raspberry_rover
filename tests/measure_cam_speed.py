from timeit import timeit
from robot.services.vision import Vision


camera = Vision(model_path="models/yolo26n.onnx", size=(1280, 1280), conf=0.25)
results = None

def test():
    global results
    results = camera.detect_objects()



print(results, timeit(test, number = 1))
