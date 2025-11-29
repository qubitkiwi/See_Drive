# seedrive_server.py
import io
import os
import math
import pathlib
from datetime import datetime
from typing import Literal, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float, select, desc, text
from sqlalchemy.dialects.postgresql import JSONB
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from ultralytics import YOLO

# =========================================================
# 앱 & 저장소 설정
# =========================================================
app = FastAPI(title="See: Drive API")

STORE_ROOT = os.getenv("IMG_STORE_DIR", "./img")
ORIG_DIR = os.path.join(STORE_ROOT, "orig")
OVERLAY_DIR = os.path.join(STORE_ROOT, "overlay")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# =========================================================
# DB 연결 및 스키마
# =========================================================

DEFAULT_DB_URL = "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
DATABASE_URL = os.getenv("DB_URL", DEFAULT_DB_URL)

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
metadata = MetaData()

lane_wear_results = Table(
    "lane_wear_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP")),
    Column("image_name", String(255)),
    Column("model", String(255)),
    Column("width", Integer), Column("height", Integer),
    Column("runtime_ms", Float),
    Column("overall", JSONB),
    Column("per_class", JSONB),
    Column("gps_lat", Float), Column("gps_lon", Float),
    Column("timestamp", DateTime),
    Column("device_id", String),
)

@app.on_event("startup")
def on_startup():
    metadata.create_all(engine)
    os.makedirs(ORIG_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

# =========================================================
# 모델 로딩
# =========================================================
MODEL_PATH = os.getenv("YOLO_MODEL", "./model/yolov11n-face.pt")                    # 얼굴
LP_MODEL_PATH = os.getenv("YOLO_LP_MODEL", "./model/license-plate-finetune-v1x.pt") # 차량 번호판

# LANE_MODEL_PATH = os.getenv("YOLO_LANE_MODEL", "best_model.pt") # 차선/정지선/횡단보도 세그


FACE_CONF       = float(os.getenv("FACE_CONF", "0.25"))
PLATE_CONF      = float(os.getenv("PLATE_CONF", "0.25"))
BLUR_IOU        = float(os.getenv("BLUR_IOU",  "0.50"))
BLUR_STRENGTH   = int(os.getenv("BLUR_STRENGTH", "31"))
PIXEL_SIZE      = int(os.getenv("PIXEL_SIZE", "16"))
BLUR_METHOD     = os.getenv("BLUR_METHOD", "gaussian")  # "gaussian" | "pixelate"

_model_face: Optional[YOLO] = None
# _model_lane: Optional[YOLO] = None
_model_lp: Optional[YOLO] = None

def _check_model_file(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} model not found: {path}")

def get_face_model() -> YOLO:
    global _model_face
    if _model_face is None:
        _check_model_file(MODEL_PATH, "Face")
        _model_face = YOLO(MODEL_PATH)
    return _model_face


def get_lp_model() -> YOLO:
    global _model_lp
    if _model_lp is None:
        _check_model_file(LP_MODEL_PATH, "License plate")
        _model_lp = YOLO(LP_MODEL_PATH)
    return _model_lp

# =========================================================
# 유틸 (IO/변환)
# =========================================================
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
ALLOWED_MIME = {"image/jpeg", "image/png"}

def read_image_from_upload(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported Content-Type: {file.content_type}")
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    img = Image.open(io.BytesIO(raw))
    try:
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        img = img.convert("RGB")
    return img

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()

def resize_long_edge(pil_img: Image.Image, max_edge: int) -> Image.Image:
    w, h = pil_img.size; m = max(w, h)
    if m <= max_edge: return pil_img
    s = max_edge / m
    return pil_img.resize((int(w * s), int(h * s)), Image.LANCZOS)

def _save_jpg(path: str, img_bgr: np.ndarray, quality: int = 92):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")

def _build_url(req: Optional[Request], path: str) -> str:
    if PUBLIC_BASE_URL: return f"{PUBLIC_BASE_URL}{path}"
    if req is not None:
        base = str(req.base_url).rstrip("/")
        return f"{base}{path}"
    return path


# =========================================================
# 블러 유틸
# =========================================================
def _detect_boxes(model: YOLO, frame_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> np.ndarray:
    r = model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    return r.boxes.xyxy.detach().cpu().numpy() if r and r.boxes is not None and len(r.boxes) > 0 else np.empty((0,4), float)

def apply_blur(img_bgr: np.ndarray, boxes_xyxy: np.ndarray,
               method: str = "gaussian", blur_strength: int = 31, pixel_size: int = 16):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(int(x1), 0); y1 = max(int(y1), 0)
        x2 = min(int(x2), w); y2 = min(int(y2), h)
        if x2 <= x1 or y2 <= y1: continue
        roi = out[y1:y2, x1:x2]
        if method == "pixelate":
            sh, sw = max(1, (y2-y1)//pixel_size), max(1, (x2-x1)//pixel_size)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            roi_blur = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        else:
            k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
        out[y1:y2, x1:x2] = roi_blur
    return out

# =========================================================
# 헬스체크
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "face_model": os.path.basename(MODEL_PATH),
        # "lane_model": os.path.basename(LANE_MODEL_PATH),
        "lp_model": os.path.basename(LP_MODEL_PATH),
        # "lane_classes": getattr(get_lane_model(), "names", {}),
    }

# =========================================================
# 오버레이 렌더
# =========================================================
COLOR_MAP = {0:(0,0,255),1:(255,0,0),2:(180,0,0),3:(255,255,255),4:(200,200,200),5:(0,255,255),6:(0,200,200)}
ALPHA_FILL = 0.35; CNT_THICK = 2


# =========================================================
# 테스트용 블러 API (얼굴/번호판)
# =========================================================
@app.post("/blur")
def blur(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou: float = Query(0.50, ge=0.05, le=0.95),
    method: Literal["gaussian", "pixelate"] = Query("gaussian"),
    blur_strength: int = Query(31, ge=3, le=199),
    pixel_size: int = Query(16, ge=2, le=128),
    max_size: int = Query(1280, ge=320, le=4096),
    jpeg_quality: int = Query(90, ge=60, le=100),
):
    pil_img = read_image_from_upload(file)
    pil_img_rs = resize_long_edge(pil_img, max_size)
    img_bgr = pil_to_cv2(pil_img_rs)

    boxes_face = _detect_boxes(get_face_model(), img_bgr, conf, iou, max_size)
    try:
        boxes_plate = _detect_boxes(get_lp_model(), img_bgr, conf, iou, max_size)
    except Exception:
        boxes_plate = np.empty((0,4), float)

    boxes = boxes_face if len(boxes_face) else np.empty((0,4), float)
    if len(boxes_plate): boxes = np.concatenate([boxes, boxes_plate], axis=0) if len(boxes) else boxes_plate

    if len(boxes):
        img_bgr = apply_blur(img_bgr, boxes, method=method, blur_strength=blur_strength, pixel_size=pixel_size)
    return Response(content=cv2_to_jpeg_bytes(img_bgr, quality=jpeg_quality), media_type="image/jpeg")

# =========================================================
# Lane wear 추론 + 저장 (패턴 지표 포함)
# =========================================================
@app.post("/lane_wear_infer")
def lane_wear_infer(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou: float  = Query(0.50, ge=0.05, le=0.95),
    max_size: int = Query(1280, ge=320, le=4096),
    gps_lat: float = Form(...),
    gps_lon: float = Form(...),
    timestamp: datetime = Form(...),
    device_id: str = Form(...),
):
    print("lane_wear_infer init")
    print("="*40)
    import time
    pil_img = read_image_from_upload(file)
    pil_img_rs = resize_long_edge(pil_img, max_size)
    frame = pil_to_cv2(pil_img_rs)
    H, W = frame.shape[:2]

    # (1) 얼굴/번호판 블러
    try: boxes_face = _detect_boxes(get_face_model(), frame, FACE_CONF, BLUR_IOU, max_size)
    except Exception: boxes_face = np.empty((0,4), float)
    try: boxes_plate = _detect_boxes(get_lp_model(), frame, PLATE_CONF, BLUR_IOU, max_size)
    except Exception: boxes_plate = np.empty((0,4), float)
    if len(boxes_face) or len(boxes_plate):
        all_boxes = boxes_face if len(boxes_face) else np.empty((0,4), float)
        if len(boxes_plate): all_boxes = np.concatenate([all_boxes, boxes_plate], axis=0) if len(all_boxes) else boxes_plate
        frame = apply_blur(frame, all_boxes, method=BLUR_METHOD, blur_strength=BLUR_STRENGTH, pixel_size=PIXEL_SIZE)

    # (2) DB 저장
    db_id = None; db_error = None
    try:
        with engine.begin() as conn:
            ins = conn.execute(
                lane_wear_results.insert().values(
                    image_name = getattr(file, "filename", None),
                    # model      = os.path.basename(LANE_MODEL_PATH),
                    model      = "None",
                    width      = W, height = H,
                    # runtime_ms = round(elapsed_ms, 2),
                    runtime_ms = 2,
                    # overall    = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics_all.items()},
                    overall    = {1 : 0.1},
                    # per_class  = per_class,
                    per_class  = 1,
                    gps_lat    = gps_lat, gps_lon = gps_lon,
                    timestamp  = timestamp, device_id = device_id,
                )
            )
            db_id = ins.inserted_primary_key[0]
    except Exception as e:
        db_error = str(e)

    # (3) 이미지 저장 (블러 반영 원본 + 오버레이)
    orig_url = overlay_url = None
    try:
        if db_id is not None:
            orig_path = os.path.join(ORIG_DIR, f"{db_id}.jpg")
            overlay_path = os.path.join(OVERLAY_DIR, f"{db_id}.jpg")
            _save_jpg(orig_path, frame, quality=92)
            # overlay_img = make_overlay_image(frame, class_masks)
            # _save_jpg(overlay_path, overlay_img, quality=92)
            orig_url    = _build_url(request, f"/lane_wear/image/{db_id}/orig")
            # overlay_url = _build_url(request, f"/lane_wear/image/{db_id}/overlay")
    except Exception as e:
        db_error = (db_error + " | " if db_error else "") + f"save_image: {e}"

    return {
        # "model": os.path.basename(LANE_MODEL_PATH),
        "image_size": {"width": W, "height": H},
        # "runtime_ms": round(elapsed_ms, 2),
        "runtime_ms": 0,
        # "overall": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics_all.items()},
        "overall" :  {1 : 0.1},
        # "per_class": per_class,
        "per_class": 1,
        "db_id": db_id,
        "db_error": db_error,
        "orig_url": orig_url,
        "overlay_url": overlay_url,
    }

# =========================================================
# 이미지 서빙
# =========================================================
@app.api_route("/lane_wear/image/{id:int}/{kind}", methods=["GET", "HEAD"])
def get_lane_wear_image_kind(id: int, kind: Literal["orig", "overlay"]):
    path = os.path.join(ORIG_DIR if kind == "orig" else OVERLAY_DIR, f"{id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(404, f"image not found: {kind} {id}")
    headers = {"Cache-Control": "public, max-age=86400"}
    return FileResponse(path, media_type="image/jpeg", headers=headers)

@app.api_route("/lane_wear/image/{id:int}", methods=["GET", "HEAD"])
def get_lane_wear_image_query(id: int, type: Literal["orig", "overlay"] = Query("orig")):
    return get_lane_wear_image_kind(id, type)

# =========================================================
# 조회/리스트
# =========================================================
def _shape_row(row: dict, req: Optional[Request] = None) -> dict:
    rid = row["id"]
    d = {
        "id": rid,
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "image_name": row.get("image_name"),
        "model": row.get("model"),
        "image_size": {"width": row.get("width"), "height": row.get("height")},
        "runtime_ms": row.get("runtime_ms"),
        "overall": row.get("overall") or {},
        "per_class": row.get("per_class") or {},
        "gps_lat": row.get("gps_lat"), "gps_lon": row.get("gps_lon"),
        "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
        "device_id": row.get("device_id"),
    }
    if os.path.exists(os.path.join(ORIG_DIR, f"{rid}.jpg")):
        d["orig_url"] = _build_url(req, f"/lane_wear/image/{rid}/orig")
    if os.path.exists(os.path.join(OVERLAY_DIR, f"{rid}.jpg")):
        d["overlay_url"] = _build_url(req, f"/lane_wear/image/{rid}/overlay")
    return d

@app.get("/lane_wear/latest")
def get_lane_wear_latest(request: Request, image_name: Optional[str] = None):
    stmt = select(lane_wear_results)
    if image_name:
        stmt = stmt.where(lane_wear_results.c.image_name == image_name)
    stmt = stmt.order_by(desc(lane_wear_results.c.created_at)).limit(1)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
        if not row: raise HTTPException(404, "no data")
        return _shape_row(row, request)

@app.get("/lane_wear/recent")
def get_lane_wear_recent(request: Request, limit: int = 20, offset: int = 0):
    limit = max(1, min(limit, 200))
    with engine.connect() as conn:
        rows = conn.execute(
            select(lane_wear_results)
            .order_by(desc(lane_wear_results.c.created_at))
            .limit(limit).offset(offset)
        ).mappings().all()
        return [_shape_row(r, request) for r in rows]

@app.get("/lane_wear/{id:int}")
def get_lane_wear(request: Request, id: int):
    with engine.connect() as conn:
        row = conn.execute(
            select(lane_wear_results).where(lane_wear_results.c.id == id)
        ).mappings().first()
        if not row: raise HTTPException(404, "not found")
        return _shape_row(row, request)

# =========================================================
# 요약 통계
# =========================================================
@app.get("/stats/summary")
def stats_summary(
    window_h: int = 24,
    warning: float = 40.0,
    critical: float = 70.0,
) -> Dict[str, Any]:
    with engine.begin() as conn:
        # 창 통계 (NULL 방지: COALESCE)
        q_window = text(f"""
            SELECT
              COUNT(*) AS detections,
              COUNT(DISTINCT CASE WHEN device_id IS NOT NULL THEN device_id END) AS active_devices,
              COALESCE(SUM(CASE WHEN (overall->>'wear_score')::float >= :critical THEN 1 ELSE 0 END), 0) AS alerts_critical,
              COALESCE(SUM(CASE WHEN (overall->>'wear_score')::float >= :warning
                                 AND (overall->>'wear_score')::float < :critical
                           THEN 1 ELSE 0 END), 0) AS alerts_warning
            FROM lane_wear_results
            WHERE created_at >= NOW() - INTERVAL '{int(window_h)} hour'
        """)
        row = conn.execute(q_window, {"warning": warning, "critical": critical}).mappings().one()

        # 디바이스 최신 상태 요약 (NULL 방지)
        q_latest = text("""
            WITH latest AS (
              SELECT DISTINCT ON (device_id)
                device_id, created_at, (overall->>'wear_score')::float AS wear
              FROM lane_wear_results
              WHERE device_id IS NOT NULL
              ORDER BY device_id, created_at DESC
            )
            SELECT
              COALESCE(SUM(CASE WHEN wear < :warning THEN 1 ELSE 0 END), 0)                       AS ok,
              COALESCE(SUM(CASE WHEN wear >= :warning AND wear < :critical THEN 1 ELSE 0 END), 0) AS warning_cnt,
              COALESCE(SUM(CASE WHEN wear >= :critical THEN 1 ELSE 0 END), 0)                     AS critical_cnt
            FROM latest;
        """)
        dev = conn.execute(q_latest, {"warning": warning, "critical": critical}).mappings().one()

        # 트렌드 (COUNT(*)는 0을 반환하므로 그대로 OK)
        q_trend = text(f"""
            WITH cur AS (
              SELECT COUNT(*) AS c FROM lane_wear_results
              WHERE created_at >= NOW() - INTERVAL '{int(window_h)} hour'
                AND (overall->>'wear_score')::float >= :critical
            ),
            prev AS (
              SELECT COUNT(*) AS p FROM lane_wear_results
              WHERE created_at >= NOW() - INTERVAL '{int(window_h*2)} hour'
                AND created_at <  NOW() - INTERVAL '{int(window_h)} hour'
                AND (overall->>'wear_score')::float >= :critical
            )
            SELECT c, p FROM cur, prev;
        """)
        t = conn.execute(q_trend, {"critical": critical}).mappings().one()
        c = float(t["c"] or 0)
        p = float(t["p"] or 0)
        delta = (c - p) / p if p > 0 else (1.0 if c > 0 else None)

        # 유지보수 후보 count (COUNT(*)는 0)
        q_maint = text("""
            WITH ranked AS (
              SELECT device_id, created_at, (overall->>'wear_score')::float AS wear,
                     ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY created_at DESC) AS rn
              FROM lane_wear_results
              WHERE device_id IS NOT NULL
            ),
            last3 AS (
              SELECT device_id,
                     SUM( (wear >= :critical)::int ) AS crit3,
                     COUNT(*) AS n
              FROM ranked
              WHERE rn <= 3
              GROUP BY device_id
            )
            SELECT COUNT(*) AS candidates
            FROM last3
            WHERE n = 3 AND crit3 = 3;
        """)
        maint = conn.execute(q_maint, {"critical": critical}).mappings().one()

    return {
        "window_h": window_h,
        "thresholds": {"warning": warning, "critical": critical},
        "detections_24h": int(row["detections"] or 0),
        "active_devices_24h": int(row["active_devices"] or 0),
        "alerts_24h": {
            "critical": int(row["alerts_critical"] or 0),
            "warning": int(row["alerts_warning"] or 0),
            "trend_vs_prev": delta
        },
        "latest_device_state": {
            "ok": int(dev["ok"] or 0),
            "warning": int(dev["warning_cnt"] or 0),
            "critical": int(dev["critical_cnt"] or 0),
        },
        "maintenance_candidates": int(maint["candidates"] or 0),
    }

# =========================================================
# 공간 집계 & 후보 랭크
# =========================================================
@app.get("/geo/cells")
def geo_cells(
    min_lat: float = Query(...), min_lon: float = Query(...),
    max_lat: float = Query(...), max_lon: float = Query(...),
    step_m: int = Query(50, ge=10, le=500),
    window_h: int = Query(24, ge=1, le=168),
    agg: str = Query("p90", pattern="^(avg|max|p90)$"),
    min_count: int = Query(1, ge=1, le=100),
) -> Dict[str, Any]:
    if min_lat > max_lat or min_lon > max_lon:
        raise HTTPException(status_code=400, detail="invalid bbox")
    lat0 = (min_lat + max_lat) / 2.0
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat0)) + 1e-9)
    lat_step = step_m * lat_deg_per_m; lon_step = step_m * lon_deg_per_m

    with engine.begin() as conn:
        q = text(f"""
            WITH raw AS (
              SELECT
                FLOOR( (gps_lat - :min_lat) / :lat_step )::int AS cy,
                FLOOR( (gps_lon - :min_lon) / :lon_step )::int AS cx,
                (overall->>'wear_score')::float AS wear,
                "timestamp" AS ts
              FROM lane_wear_results
              WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL
                AND created_at >= NOW() - INTERVAL '{int(window_h)} hour'
                AND gps_lat BETWEEN :min_lat AND :max_lat
                AND gps_lon BETWEEN :min_lon AND :max_lon
            ),
            ag AS (
              SELECT
                cy, cx, COUNT(*) AS n,
                AVG(wear) AS wear_avg,
                MAX(wear) AS wear_max,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY wear) AS wear_p90,
                MAX(ts) AS last_ts
              FROM raw
              GROUP BY cy, cx
            )
            SELECT * FROM ag WHERE n >= :min_count
        """)
        rows = conn.execute(q, {"min_lat": min_lat, "max_lat": max_lat,
                                "min_lon": min_lon, "max_lon": max_lon,
                                "lat_step": lat_step, "lon_step": lon_step,
                                "min_count": min_count}).mappings().all()

    cells = []
    for r in rows:
        cy, cx = r["cy"], r["cx"]
        lat = min_lat + (cy + 0.5) * lat_step
        lon = min_lon + (cx + 0.5) * lon_step
        rep = r["wear_p90"] if agg == "p90" else (r["wear_max"] if agg == "max" else r["wear_avg"])
        cells.append({
            "cy": int(cy), "cx": int(cx),
            "lat": float(lat), "lon": float(lon),
            "count": int(r["n"]),
            "wear_avg": float(r["wear_avg"]),
            "wear_max": float(r["wear_max"]),
            "wear_p90": float(r["wear_p90"]),
            "rep": float(rep),
            "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None,
        })
    return {"bbox": [min_lat, min_lon, max_lat, max_lon], "step_m": step_m, "window_h": window_h, "agg": agg, "cells": cells}

def _priority_formula_sql(window_h: int) -> str:
    return f"""
    WITH ranked AS (
      SELECT id, device_id, created_at, (overall->>'wear_score')::float AS wear,
             ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY created_at DESC) AS rn
      FROM lane_wear_results
      WHERE created_at >= NOW() - INTERVAL '{int(window_h)} hour' AND device_id IS NOT NULL
    ),
    agg AS (
      SELECT device_id,
             MAX(CASE WHEN rn=1 THEN wear END)               AS w_last,
             AVG(CASE WHEN rn<=3 THEN wear END)              AS w_last3,
             AVG(CASE WHEN rn BETWEEN 4 AND 6 THEN wear END) AS w_prev3,
             SUM(CASE WHEN rn<=3 AND wear >= :critical THEN 1 ELSE 0 END) AS crit3,
             COUNT(*) AS n,
             MAX(created_at) AS last_ts
      FROM ranked GROUP BY device_id
    ),
    score AS (
      SELECT device_id, w_last, w_last3, w_prev3,
             COALESCE(w_last3 - COALESCE(w_prev3, w_last3), 0) AS trend,
             crit3, n, last_ts,
             EXTRACT(EPOCH FROM (NOW() - last_ts))/3600.0 AS hours_since,
             (
               0.50 * LEAST(GREATEST(w_last,0),100)/100.0 +
               0.20 * (COALESCE(w_last3 - COALESCE(w_prev3, w_last3),0)/100.0) +
               0.20 * (crit3/3.0) +
               0.10 * LEAST(n/10.0, 1.0)
             ) * EXP(- LEAST(EXTRACT(EPOCH FROM (NOW() - last_ts))/3600.0, 168)/72.0)
             AS priority
      FROM agg
    )
    SELECT * FROM score
    """

@app.get("/candidates/rank")
def candidates_rank(window_h: int = 168, critical: float = 70.0, limit: int = 20, offset: int = 0):
    sql = _priority_formula_sql(window_h) + " ORDER BY priority DESC LIMIT :limit OFFSET :offset"
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"critical": critical, "limit": limit, "offset": offset}).mappings().all()
    return [
        {"device_id": r["device_id"], "priority": float(r["priority"] or 0.0),
         "w_last": float(r["w_last"] or 0.0), "trend": float(r["trend"] or 0.0),
         "crit3": int(r["crit3"] or 0), "n": int(r["n"] or 0),
         "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None,
         "hours_since": float(r["hours_since"] or 0.0)}
        for r in rows
    ]

@app.get("/candidates/rank_for_id/{id:int}")
def candidate_rank_for_id(id: int, window_h: int = 168, critical: float = 70.0):
    with engine.begin() as conn:
        row = conn.execute(select(lane_wear_results.c.device_id).where(lane_wear_results.c.id == id)).first()
    if not row or not row[0]: return {"rank": None, "total": 0, "row": None}
    device_id = row[0]
    sql_all = _priority_formula_sql(window_h)
    with engine.begin() as conn:
        all_rows = conn.execute(text(sql_all), {"critical": critical}).mappings().all()
    all_rows = sorted(all_rows, key=lambda r: (r["priority"] or 0.0), reverse=True)
    total = len(all_rows)
    rank = next((i+1 for i, r in enumerate(all_rows) if r["device_id"] == device_id), None)
    cur = next((r for r in all_rows if r["device_id"] == device_id), None)
    if cur is None: return {"rank": None, "total": total, "row": None}
    def _shape(r):
        return {"device_id": r["device_id"], "priority": float(r["priority"] or 0.0),
                "w_last": float(r["w_last"] or 0.0), "trend": float(r["trend"] or 0.0),
                "crit3": int(r["crit3"] or 0), "n": int(r["n"] or 0),
                "last_ts": r["last_ts"].isoformat() if r["last_ts"] else None}
    top10 = [_shape(r) for r in all_rows[:10]]
    return {"rank": rank, "total": total, "row": _shape(cur), "top10": top10}

HTML_FILE_PATH = "./templates/index.html"


@app.get("/")
async def serve_form():
    """
    루트 경로에 접속하면 HTML 폼 파일(index.html)을 반환합니다.
    FileResponse를 사용하여 파일을 직접 읽어 전송합니다.
    """
    if not os.path.exists(HTML_FILE_PATH):
        return {"error": f"HTML file not found at {HTML_FILE_PATH}"}
        
    # FileResponse를 사용하여 파일을 효율적으로 전송합니다.
    return FileResponse(HTML_FILE_PATH, media_type="text/html")