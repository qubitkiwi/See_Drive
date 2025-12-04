from ultralytics import YOLO
model_path = r"/home/elicer/sechan/0_ws/for_YOLO/runs/train/yolo11n_cls10_200epochs/weights/best.pt"
model = YOLO(model_path)
model.export(format="tflite", half=True)
