import os
import json
import glob
from tqdm import tqdm

# ================== 사용자 설정 ==================

DATA_ROOT = "/home/elicer/sechan/aihub_datasets/road_obstacle_split_dataset"
SPLITS = ["train", "val", "test"]

# 이미지 크기 고정
IMG_W = 1280
IMG_H = 720

# 특정 클래스 제외하고 싶을 때:
# category_id → yolo_class_id 매핑
# ※ 9, 10 제거
CLASS_MAPPING = {
    1: 0,  # Animals(Dolls)
    2: 1,  # Person
    3: 2,  # Garbage bag & sacks
    4: 3,  # Construction signs & Parking prohibited board
    # 5: Traffic cone 
    6: 4,  # Box
    7: 5,  # Stones on road
    8: 6,  # Pothole on road
    # 9: 7,  # Filled pothole
    # 10: 8, # Manhole
}

# YOLO 라벨 폴더명
YOLO_LABEL_DIRNAME = "labels_yolo"

# =================================================


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, w, h = bbox
    x_c = x_min + w / 2.0
    y_c = y_min + h / 2.0
    return (x_c / img_w, y_c / img_h, w / img_w, h / img_h)


def convert_split(split_name):
    img_dir = os.path.join(DATA_ROOT, "images", split_name)
    json_dir = os.path.join(DATA_ROOT, "labels", split_name)
    yolo_dir = os.path.join(DATA_ROOT, YOLO_LABEL_DIRNAME, split_name)

    os.makedirs(yolo_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])

    print(f"\n===== Split: {split_name} =====")
    print(f"이미지 개수: {len(img_files)}")

    for img_name in tqdm(img_files, desc=f"Convert {split_name}"):
        stem, _ = os.path.splitext(img_name)

        img_path = os.path.join(img_dir, img_name)

        # JSON 찾기 (*_BBOX.json 또는 .json)
        json_glob = glob.glob(os.path.join(json_dir, stem + "*_BBOX.json"))
        if len(json_glob) == 0:
            json_glob = glob.glob(os.path.join(json_dir, stem + ".json"))

        if len(json_glob) == 0:
            print(f"\n⚠️ JSON 없음 → 이미지 유지 + 빈 txt 생성: {img_path}")
            open(os.path.join(yolo_dir, stem + ".txt"), "w").close()
            continue

        json_path = json_glob[0]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        yolo_lines = []

        # ========= annotation 처리 =========
        for ann in anns:
            cat_id = ann.get("category_id")
            bbox = ann.get("bbox")

            # bbox나 cat_id가 없으면 무시
            if bbox is None or cat_id is None:
                continue

            # 클래스 매핑 없으면 무시 (9,10 자동 제거)
            if cat_id not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[cat_id]

            # 정규화
            x, y, w, h = coco_bbox_to_yolo(bbox, IMG_W, IMG_H)

            # 보정
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            yolo_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # ========= txt 파일 저장 =========
        yolo_txt_path = os.path.join(yolo_dir, stem + ".txt")

        # bbox가 하나도 없는 경우 → 빈 파일 생성 (Negative sample)
        if len(yolo_lines) == 0:
            open(yolo_txt_path, "w").close()
        else:
            with open(yolo_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))


def main():
    for split in SPLITS:
        convert_split(split)

    print("\n🎯 YOLO txt 변환 완벽 완료!")
    print(f"➡ 저장 위치: {os.path.join(DATA_ROOT, YOLO_LABEL_DIRNAME)}")


if __name__ == "__main__":
    main()
