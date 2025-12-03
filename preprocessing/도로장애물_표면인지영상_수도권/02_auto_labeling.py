from ultralytics import YOLO
import os
from tqdm import tqdm

DATA_ROOT = "/home/elicer/sechan/aihub_datasets/road_obstacle_split_dataset"
SPLITS = ["train", "val", "test"]
YOLO_LABEL_DIRNAME = "labels"

IMG_W = 1280
IMG_H = 720

CAR_ID   = 7
TRUCK_ID = 8
BUS_ID   = 9

model = YOLO("yolo11x.pt")  # COCO pretrained


def add_vehicle_labels(split):
    print(f"\n==============================")
    print(f"🚗 Auto-label START for [{split}]")
    print(f"==============================")

    img_dir = os.path.join(DATA_ROOT, "images", split)
    lbl_dir = os.path.join(DATA_ROOT, YOLO_LABEL_DIRNAME, split)

    img_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"📸 이미지 개수: {len(img_files)}")
    print(f"🏷 라벨 디렉토리: {lbl_dir}")

    processed = 0
    added_vehicle = 0

    for img_name in tqdm(img_files, desc=f"[{split}] Auto-labeling", ncols=90):
        processed += 1
        
        img_path = os.path.join(img_dir, img_name)
        stem, _ = os.path.splitext(img_name)
        txt_path = os.path.join(lbl_dir, stem + ".txt")

        # --------------------------
        # 1) 기존 라벨 읽기
        # --------------------------
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                existing = [ln.strip() for ln in f if ln.strip()]
        else:
            existing = []

        # 기존 vehicle 제거
        cleaned = []
        for ln in existing:
            try:
                cid = int(ln.split()[0])
            except:
                continue

            if cid not in [CAR_ID, TRUCK_ID, BUS_ID]:
                cleaned.append(ln)

        # --------------------------
        # 2) YOLO 차량 예측
        # --------------------------
        results = model(img_path, verbose=False)[0]

        new_lines = []
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            name = model.names[cls_idx]

            if name not in ["car", "truck", "bus"]:
                continue

            # 픽셀 좌표
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 정규화
            x_c = (x1 + x2) / 2 / IMG_W
            y_c = (y1 + y2) / 2 / IMG_H
            w   = (x2 - x1) / IMG_W
            h   = (y2 - y1) / IMG_H

            # ID 매핑
            if name == "car":
                cid = CAR_ID
            elif name == "truck":
                cid = TRUCK_ID
            else:
                cid = BUS_ID

            new_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # 차량 라벨 추가되면 카운트
        if len(new_lines) > 0:
            added_vehicle += 1

        # --------------------------
        # 3) 통합 후 TXT 저장
        # --------------------------
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned + new_lines))

    # --------------------------
    # 4) Summary
    # --------------------------
    print(f"\n🎉 [{split}] Auto-label 완료!")
    print(f"📄 총 이미지 처리: {processed}")
    print(f"🚙 차량 라벨이 추가된 이미지 수: {added_vehicle}")
    print(f"🗂 저장 위치: {os.path.join(DATA_ROOT, YOLO_LABEL_DIRNAME, split)}")
    print("------------------------------------------------------------\n")


# 전체 split 처리
for s in SPLITS:
    add_vehicle_labels(s)

print("\n🔥 전체 Auto-labeling 프로세스 완료!\n")
