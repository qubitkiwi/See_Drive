import numpy as np
import cv2
from line_model import Line

# 새로 분리한 거리/색상 모듈

from distance_visualizer import render_bezier_safe_distance
# from test import render_bezier_safe_distance


# =====================
# 기본 하이퍼파라미터
# =====================
INNER = 0 # 중앙제거=0, 활성화=255

CANNY_LOW_THRESHOLD = 20
CANNY_HIGH_THRESHOLD = 80
GAUSSIAN_KERNEL_SIZE = 5

HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_LEN = 10
HOUGH_MAX_LINE_GAP = 50

MIN_ABS_SLOPE = 0.7
MAX_ABS_SLOPE = 1.8

# 실제 주차칸 폭(m)
SLOT_WIDTH_M = 2.5


# =====================
# 기본 유틸 함수
# =====================

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        ignore_mask_color = (255,) * img.shape[2]
    else:
        ignore_mask_color = 255

    vertices = np.array(vertices, dtype=np.int32)
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask



def hough_lines_detection(img):
    return cv2.HoughLinesP(
        img,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        np.array([]),
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )


def weighted_img(overlay, base, alpha=0.8, beta=1.0, gamma=0.0):
    overlay = np.uint8(overlay)
    if len(overlay.shape) == 2:
        overlay = np.dstack((overlay, overlay, overlay))

    if overlay.shape[:2] != base.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))

    return cv2.addWeighted(base, alpha, overlay, beta, gamma)


# =====================
# 차선 계산
# =====================

def compute_lane_from_candidates(line_candidates, img_shape):
    if not line_candidates:
        return None, None
    
    h, w = img_shape
    mid_x = w // 2
    
    pos_lines = [l for l in line_candidates if l.slope > 0]  # right candidates
    neg_lines = [l for l in line_candidates if l.slope < 0]

    original_left = None
    original_right = None

    # 왼쪽 차선
    if len(neg_lines) > 0:
        neg_bias = np.median([l.bias for l in neg_lines])
        neg_slope = np.median([l.slope for l in neg_lines])

        if neg_slope != 0:
            y1 = img_shape[0]  # 540 가장 하단
            y2 = int(img_shape[0] * 0.4)  # y 상단
            x1 = int((y1 - neg_bias) / neg_slope)
            x2 = int((y2 - neg_bias) / neg_slope)
            original_left = Line(x1, y1, x2, y2)

    # 오른쪽 차선
    if len(pos_lines) > 0:
        pos_bias = np.median([l.bias for l in pos_lines])
        pos_slope = np.median([l.slope for l in pos_lines])

        if pos_slope != 0:
            y1 = img_shape[0]
            y2 = int(img_shape[0] * 0.4)
            x1 = int((y1 - pos_bias) / pos_slope)
            x2 = int((y2 - pos_bias) / pos_slope)
            original_right = Line(x1, y1, x2, y2)
            
    # ====================================================
    # 🔥 추가한 핵심 안정화 (중앙 침범 방지)
    # ====================================================
    # 왼쪽 차선이 중앙보다 오른쪽이면 제거
    if original_left is not None:
        # cx_left = (original_left.x1 + original_left.x2) // 2
        xs = [original_left.x1, original_left.x2]
        if max(xs) > 360:
            original_left = None

    # 오른쪽 차선이 중앙보다 왼쪽이면 제거
    if original_right is not None:
        xs = [original_right.x1, original_right.x2]
        if min(xs) < 600:
            original_right = None

    return original_left, original_right

def inside_roi(pt, roi): 
    # pt = (x,y), roi = np.array of polygon
    return cv2.pointPolygonTest(roi, pt, False) >= 0


def get_lane_lines(color_image):
    color_image = cv2.resize(color_image, (960, 540))
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
    img_edge = cv2.Canny(img_blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    h, w = img_edge.shape
    
    # ROI 설정
    '''
    outer = np.array([
        (int(0.01*w), int(0.70*h)),
        (int(0.30*w), int(0.10*h)),
        (int(0.70*w), int(0.10*h)),
        (int(0.99*w), int(0.70*h)),
    ])
    
    inner = np.array([
    (int(0.15*w), int(0.70*h)),
    (int(0.35*w), int(0.20*h)), 
    (int(0.65*w), int(0.20*h)),
    (int(0.85*w), int(0.70*h))            
])
    '''
    ############################################################################################
    outer = np.array([
        (int(0.09*w), int(1.00*h)),  # left-bottom
        (int(0.35*w), int(0.40*h)),  # left-top
        (int(0.65*w), int(0.40*h)),  # right-top
        (int(0.93*w), int(1.00*h)),  # right-bottom
    ])
    
    inner = np.array([
        (int(0.15*w), int(1.00*h)),  # 기본값 : 0.70
        (int(0.35*w), int(0.5*h)),  # 기본값 : 0.20
        (int(0.65*w), int(0.5*h)),
        (int(0.88*w), int(1.00*h)),
    ])
    

    # ROI 마스크 생성
    mask = np.zeros_like(img_edge)
    cv2.fillPoly(mask, [outer], 255)
    cv2.fillPoly(mask, [inner], INNER)  # 중앙 제거

    img_edge_roi = cv2.bitwise_and(img_edge, mask)
#############################################################################################    
    #cv2.imshow("DEBUG-EDGE", img_edge)
    #cv2.imshow("ROI MAS_", mask)
    #cv2.imshow("ROI EDGE", img_edge_roi)

    # ============ 허프 탐지 =============
    detected = hough_lines_detection(img_edge_roi)
    if detected is None:
        return None, None

    detected_lines = [Line(x1, y1, x2, y2) for [[x1, y1, x2, y2]] in detected]

    # ---------------------------------------------
    # 🎯 (1) ROI 내부 중심점 필터링 추가
    # ---------------------------------------------
    filtered_lines = []
    for line in detected_lines:
        cx = (line.x1 + line.x2) // 2
        cy = (line.y1 + line.y2) // 2

        if inside_roi((cx, cy), outer):   # outer ROI 안에 중심점이 있어야만 허용
            filtered_lines.append(line)

    # ---------------------------------------------
    # 🎯 (2) 너무 짧은 가짜 선 제거
    # ---------------------------------------------
    min_len = 40  # 너 상황에 맞춰 조정
    filtered_lines = [
        l for l in filtered_lines
        if l.length() > min_len
    ]

    # 기울기 필터 적용    
    candidate_lines = [
        l for l in filtered_lines
        if MIN_ABS_SLOPE <= abs(l.slope) <= MAX_ABS_SLOPE
    ]

    if not candidate_lines:
        return (None, None)

    return compute_lane_from_candidates(candidate_lines, img_gray.shape)

# =====================
# Temporal Smoothing
# =====================

def smoothen_over_time(lane_lines_history, alpha=0.2):
    ema_left = None
    ema_right = None

    for origL, origR in lane_lines_history:
        # Left Line
        if origL is not None:
            if ema_left is None:
                ema_left = origL  # 첫 프레임은 그대로
            else:
                Lx1, Ly1, Lx2, Ly2 = origL.get_coords()
                Ex1, Ey1, Ex2, Ey2 = ema_left.get_coords()
                # 지수 이동 평균 적용
                new_x1 = alpha * Lx1 + (1 - alpha) * Ex1
                new_y1 = alpha * Ly1 + (1 - alpha) * Ey1
                new_x2 = alpha * Lx2 + (1 - alpha) * Ex2
                new_y2 = alpha * Ly2 + (1 - alpha) * Ey2
                ema_left = Line(new_x1, new_y1, new_x2, new_y2)

        # Right Line
        if origR is not None:
            if ema_right is None:
                ema_right = origR
            else:
                Rx1, Ry1, Rx2, Ry2 = origR.get_coords()
                Ex1, Ey1, Ex2, Ey2 = ema_right.get_coords()
                new_x1 = alpha * Rx1 + (1 - alpha) * Ex1
                new_y1 = alpha * Ry1 + (1 - alpha) * Ey1
                new_x2 = alpha * Rx2 + (1 - alpha) * Ex2
                new_y2 = alpha * Ry2 + (1 - alpha) * Ey2
                ema_right = Line(new_x1, new_y1, new_x2, new_y2)

    return ema_left, ema_right


# =====================
# 거리 계산용
# =====================

def get_x_on_line_at_y(line, y):
    x1, y1, x2, y2 = line.get_coords()
    if x1 == x2:
        return x1
    m = (y2 - y1) / float(x2 - x1)
    b = y1 - m*x1
    return (y - b) / m


# =====================
# 최종 렌더링
# =====================

def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    assert len(frames) >= 1

    # Lane smoothing
    lane_lines_history = []
    for f in frames:
        lanes = get_lane_lines(f)
        lane_lines_history.append(lanes)

    # 최근 결과 사용
    if temporal_smoothing:
        original_left, original_right = smoothen_over_time(lane_lines_history)
    else:
        original_left, original_right = lane_lines_history[-1]

    base_frame = cv2.resize(frames[-1], (960, 540))
    line_img = np.zeros_like(base_frame, dtype=np.uint8)

    h, w = line_img.shape[:2]

    # -------------------------
    # 가이드라인 y 좌표 (top/mid/bottom)
    # -------------------------
    y_bottom = int(h * 0.95)
    y_mid    = int(h * 0.75)
    y_top    = int(h * 0.55)

    # 좌우 x 좌표
    xL_bottom = int(w * 0.07)
    xR_bottom = int(w * 0.93)
    xL_mid    = int(w * 0.18)
    xR_mid    = int(w * 0.82)
    xL_top    = int(w * 0.32)
    xR_top    = int(w * 0.68)

    # 좌표 세트 구성
    top_left  = (xL_top, y_top)
    top_right = (xR_top, y_top)

    mid_left  = (xL_mid, y_mid)
    mid_right = (xR_mid, y_mid)

    bot_left  = (xL_bottom, y_bottom)
    bot_right = (xR_bottom, y_bottom)

    # -------------------------
    # 스케일 계산
    # -------------------------
    if original_left is not None and original_right is not None:
        xl_top = get_x_on_line_at_y(original_left,  y_top)
        xr_top = get_x_on_line_at_y(original_right, y_top)
        xl_mid = get_x_on_line_at_y(original_left,  y_mid)
        xr_mid = get_x_on_line_at_y(original_right, y_mid)
        xl_bot = get_x_on_line_at_y(original_left,  y_bottom)
        xr_bot = get_x_on_line_at_y(original_right, y_bottom)

        w_top = abs(xr_top - xl_top)
        w_mid = abs(xr_mid - xl_mid)
        w_bot = abs(xr_bot - xl_bot)

        scale_top = SLOT_WIDTH_M / w_top if w_top > 1 else 0
        scale_mid = SLOT_WIDTH_M / w_mid if w_mid > 1 else 0
        scale_bot = SLOT_WIDTH_M / w_bot if w_bot > 1 else 0
    else:
        scale_top = scale_mid = scale_bot = 0

    # =====================================================
    # 🔥 핵심: 선 그리기 + 거리 기반 색상 변경은 모두 여기로 위임
    # =====================================================
    '''
    render_safe_distance_overlay(
        line_img,
        original_left, original_right,
        top_left, top_right,
        mid_left, mid_right,
        bot_left, bot_right,
        scale_top, scale_mid, scale_bot,
    )
    '''
    render_bezier_safe_distance(
    line_img,
    original_left,
    original_right,
    scale_top,
    scale_mid,
    scale_bot
    )
    
    blended = weighted_img(line_img, base_frame)
    return blended

