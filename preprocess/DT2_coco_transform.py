import os
import json
import numpy as np
from tqdm import tqdm
###############################################
# 0) class_name → class_id 고정 맵핑 (네가 준 거 그대로)
###############################################
ANNOTATION_LABEL = {
    "Undefined Stuff": 0, "Wall": 1, "Driving Area": 2, "Non Driving Area": 3,
    "Parking Area": 4, "No Parking Area": 5, "Big Notice": 6, "Pillar": 7,
    "Parking Area Number": 8, "Parking Line": 9, "Disabled Icon": 10,
    "Women Icon": 11, "Compact Car Icon": 12, "Speed Bump": 13,
    "Parking Block": 14, "Billboard": 15, "Toll Bar": 16, "Sign": 17,
    "No Parking Sign": 18, "Traffic Cone": 19, "Fire Extinguisher": 20,
    "Undefined Object": 21, "Two-wheeled Vehicle": 22, "Vehicle": 23,
    "Wheelchair": 24, "Stroller": 25, "Shopping Cart": 26, "Animal": 27, "Human": 28
}

# id -> name 으로 뒤집은 딕셔너리 (categories 생성용)
ID_TO_NAME = {v: k for k, v in ANNOTATION_LABEL.items()}


###############################################
# 1) 기존 NumPy 계산 함수 (그대로)
###############################################
def calculate_area(polygon):
    x = np.array(polygon[::2])
    y = np.array(polygon[1::2])
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calculate_bbox(polygon):
    x = polygon[::2]
    y = polygon[1::2]
    return [min(x), min(y), max(x) - min(x), max(y) - min(y)]


###############################################
# 2) segmentation 중첩 구조에서 polygon(dict list)만 추출하는 함수
###############################################
def extract_polygon_dicts(seg):
    """
    segmentation 안에서 [{x,y},{x,y}...] 형태의 polygon만 추출하여 리스트로 반환.
    new_seg(flat list) 변환은 기존 코드에서 처리한다.
    """
    polygons = []

    def traverse(item):
        # polygon 형태는 dict 리스트
        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
            polygons.append(item)
        # 리스트 안에 리스트가 더 있으면 계속 탐색
        elif isinstance(item, list):
            for elem in item:
                traverse(elem)

    traverse(seg)
    return polygons  # [{x,y},{x,y}...] 형태로 추출


###############################################
# 3) COCO 변환 메인 함수
#    -> 여기서 category_id를 ANNOTATION_LABEL 기준으로 고정
###############################################
def convert_to_coco(input_dir, output_file, directory):
    IMG_W = 4032
    IMG_H = 3040

    coco = {
        "info": [],
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": []
    }

    # 🔥 카테고리 리스트를 ANNOTATION_LABEL 기준으로 고정 생성
    # id 오름차순 정렬해서 넣기
    for cid in sorted(ID_TO_NAME.keys()):
        coco["categories"].append({
            "id": cid+1,
            "name": ID_TO_NAME[cid]
        })

    annotation_id = 0

    file_list = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for filename in tqdm(file_list, desc=f"[{directory}] COCO 변환 중", dynamic_ncols=True):

        with open(os.path.join(input_dir, filename), 'r') as f:
            data = json.load(f)

        img_filename = filename.replace('.json', '.png')

        image_info = {
            "id": len(coco["images"]),
            "file_name": img_filename,
            "width": IMG_W,
            "height": IMG_H
        }
        coco["images"].append(image_info)

        # objects 파싱
        for obj in data.get("objects", []):
            category_name = obj["class_name"]

            if category_name not in ANNOTATION_LABEL:
                print(f"[WARN] Unknown class_name '{category_name}' in {filename}, skip")
                continue

            category_id = ANNOTATION_LABEL[category_name]

            seg_raw = obj.get("annotation", [])
            polygons = extract_polygon_dicts(seg_raw)

            for poly_dict_list in polygons:

                new_seg = []
                for point in poly_dict_list:
                    new_seg.append(point["x"])
                    new_seg.append(point["y"])

                if len(new_seg) < 6:
                    continue

                area = calculate_area(new_seg)
                bbox = calculate_bbox(new_seg)

                ann = {
                    "id": annotation_id,
                    "image_id": image_info["id"],
                    "category_id": category_id + 1,
                    "segmentation": [new_seg],
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0
                }

                coco["annotations"].append(ann)
                annotation_id += 1

    # 저장
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=4)


###############################################
# 4) train / val / test 변환 실행
###############################################
for d in ('train', 'val', 'test'):
    print(f"\n===== {d} start =====")
    input_dir = f'new_data_set/{d}/labels'
    output_file = f'new_data_set/{d}.json'
    convert_to_coco(input_dir, output_file, d)

print("\n🎉 COCO 변환 완료!")