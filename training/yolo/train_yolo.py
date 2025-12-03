from ultralytics import YOLO

MODEL_PATH = "/home/elicer/sechan/0_ws/for_YOLO/runs/train/yolo11n_cls10_200epochs/weights/last.pt"
model = YOLO(MODEL_PATH)

# 학습 실행
model.train(
    data="/home/elicer/sechan/aihub_datasets/road_obstacle.yaml",
    epochs=200,
    imgsz=640,
    batch=-1,
    name="yolo11n_cls10_201_400ep",
    project="/home/elicer/sechan/0_ws/for_YOLO/runs/train",
    device=0,
    patience=20,
    save_period=20,
    # workers=8,                 
)

