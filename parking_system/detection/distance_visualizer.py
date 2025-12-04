# =========================================
# color_distance.py  (베지어 + 위험/안전 색상 시스템 포함)
# =========================================

import cv2
import numpy as np
from line_model import Line

Color = (255, 255, 255)
# =========================================
# 0) 유틸: y에서 직선(Line)과 만나는 x 좌표 찾기
# =========================================
def get_x_on_line_at_y(line: Line, y):
    x1, y1, x2, y2 = line.get_coords()
    if x1 == x2:
        return x1
    m = (y2 - y1) / float(x2 - x1)
    b = y1 - m * x1
    return int((y - b) / m)


# =========================================
# 1) 안전/위험 색상 단일 구간
# =========================================

def choose_color_top(dist_cm):
    if dist_cm <= 21:
        return (255, 0,0)
    elif dist_cm <= 25:
        return (255, 255, 0)
    else:
        return (255, 255, 255)

def choose_color_bot(dist_cm):
    return (255, 0, 0) if dist_cm <= 10 else (255, 255, 255)


# =========================================
# 2) 거리 계산
# =========================================
def compute_distance_cm(point, lane_line: Line, scale):
    if lane_line is None:
        return 999
    px, py = point
    fx = get_x_on_line_at_y(lane_line, py)
    px_dist = abs((fx - px)*0.8)
    return px_dist * scale * 100


# =========================================
# 3) 베지어 곡선 생성
# =========================================
def bezier_curve(p0, p1, p2, steps=40):
    curve = []
    for t in np.linspace(0, 1, steps):
        x = (1-t)**2 * p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
        curve.append([int(x), int(y)])
    return np.array(curve)


# =========================================
# 4) top / mid / bottom 좌표 자동 생성 ########################################################################################################
# =========================================
def generate_bezier_guide_points():
    # bottom
    x1L, y1L = 150, 550
    x1R, y1R = 800, 550

    # top
    x2L, y2L = 295, 350
    x2R, y2R = 665, 350

    # control point (곡률 조정)
    cx, cy = 450, 310

    # 베지어 top curve
    curve = bezier_curve(
        (x2L, y2L),
        (cx, cy),
        (x2R, y2R),
        steps=40
    )

    top_left = (curve[0][0], curve[0][1])
    top_right = (curve[-1][0], curve[-1][1])

    # mid = top과 bottom의 중간
    mid_left = (
        int((x1L + x2L) / 2),
        int((y1L + y2L) / 2)
    )
    mid_right = (
        int((x1R + x2R) / 2),
        int((y1R + y2R) / 2)
    )

    bot_left = (x1L, y1L)
    bot_right = (x1R, y1R)

    return (
        top_left, top_right,
        mid_left, mid_right,
        bot_left, bot_right,
        curve
    )
    
# 텍스트에 검은 테두리(Outline)를 줘서 잘 보이게 하는 함수
def draw_text(img, text, pos, font_scale=0.6, color=(255, 255, 255), thickness=1):
    x, y = pos
    # 1. 검은색으로 두껍게 먼저 그리기 (테두리 역할)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # 2. 원하는 색으로 그 위에 덮어쓰기
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def choose_text_color(dist_cm):
    if dist_cm <= 21:       # 매우 가까움
        return (255, 0, 0)  # 빨강
    elif dist_cm <= 25:     # 주의
        return (255, 255, 0)  # 노랑
    else:
        return (255, 255, 255)  # 흰색


# =========================================
# 5) 전체 가이드라인 + 색상 위험 처리
# =========================================
def render_bezier_safe_distance(
    line_img,
    lane_left: Line,
    lane_right: Line,
    scale_top, scale_mid, scale_bot
):
    """
    lane_detection_16.py에서 scale만 받아서
    베지어 기반 top/mid/bottom 가이드를 그리면서
    위험/안전 색상 적용.
    """

    # -----------------------------
    # top / mid / bottom 좌표 생성
    # -----------------------------
    (
        top_left, top_right,
        mid_left, mid_right,
        bot_left, bot_right,
        curve
    ) = generate_bezier_guide_points()

    # -----------------------------
    # 각 레벨별 거리 계산
    # -----------------------------
    dist_top_L = compute_distance_cm(top_left, lane_left, scale_top)
    dist_mid_L = compute_distance_cm(mid_left, lane_left, scale_mid)
    dist_bot_L = compute_distance_cm(bot_left, lane_left, scale_bot)

    dist_top_R = compute_distance_cm(top_right, lane_right, scale_top)
    dist_mid_R = compute_distance_cm(mid_right, lane_right, scale_mid)
    dist_bot_R = compute_distance_cm(bot_right, lane_right, scale_bot)

    # -----------------------------
    # 색상 결정
    # -----------------------------
    c_tl = choose_color_top(dist_top_L)
    # c_ml = choose_color(dist_mid_L)
    c_bl = choose_color_bot(dist_bot_L)

    c_tr = choose_color_top(dist_top_R)
    # c_mr = choose_color(dist_mid_R)
    c_br = choose_color_bot(dist_bot_R)

    th = 5

    # 중앙점
    top_center = ((top_left[0] + top_right[0]) // 2, top_left[1])
    mid_center = ((mid_left[0] + mid_right[0]) // 2, mid_left[1])
    bot_center = ((bot_left[0] + bot_right[0]) // 2, bot_left[1])


    # ==============================================================================================================

    # 가로선 mid, bottom
    cv2.line(line_img, mid_left, mid_right, Color, 5)
    cv2.line(line_img, bot_left, bot_right, Color, 5)
    
    # 세로선 bot-mid
    cv2.line(line_img, bot_left,  mid_left,  c_tl, th, cv2.LINE_AA)
    cv2.line(line_img, bot_right, mid_right, c_tr, th, cv2.LINE_AA)
    
    # 세로선 mid-top
    cv2.line(line_img, mid_left,  top_left,  c_tl, th, cv2.LINE_AA)
    cv2.line(line_img, mid_right, top_right, c_tr, th, cv2.LINE_AA)


    # ==============================================================================================================
    # 선 중앙 채우기(선택)
    # cv2.line(line_img, bot_left, top_left, Color, 2)
    # cv2.line(line_img, bot_right, top_right, Color, 2)
    # ==============================================================================================================
    
    # -----------------------------
    # 윗쪽 베지어 곡선도 그리기 (흰색)
    # -----------------------------
    cv2.polylines(
        line_img,
        [curve.astype(np.int32)],
        isClosed=False,
        color=Color,
        thickness=3,
        lineType=cv2.LINE_AA
    )
    # ==================================================================================================================
    # 차선 빨간색 표시
    # ==================================================================================================================
    
    if lane_left is not None:
        lane_left.draw(line_img, (255, 255, 255), 10)
    if lane_right is not None:
        lane_right.draw(line_img, (255, 255, 255), 10)
    
# -----------------------------------------------------
    # [추가됨] 거리 수치 텍스트 표시 (HUD)
    # -----------------------------------------------------
    
    # 왼쪽 차선 거리 표시
    if lane_left is not None:
        # f-string을 써서 소수점 1자리까지 표기 (예: "12.5cm")
        text_L = f"{dist_top_L:.1f}cm" 
        
        # 텍스트 위치: 가이드라인 왼쪽(mid_left)에서 x좌표를 -120만큼 이동
        pos_L = (mid_left[0] - 50, mid_left[1]-50) 
        
        text_color_L = choose_text_color(dist_top_L)
        
        # 화면에 그리기 (color_L은 현재 상태 색상: 초록/노랑/빨강)
        draw_text(line_img, text_L, pos_L, font_scale=0.5, color=text_color_L)

    # 오른쪽 차선 거리 표시
    if lane_right is not None:
        text_R = f"{dist_top_R:.1f}cm"
        
        # 텍스트 위치: 가이드라인 오른쪽(mid_right)에서 x좌표를 +20만큼 이동
        pos_R = (mid_right[0] - 30, mid_right[1]-50)
        
        text_color_R = choose_text_color(dist_top_R)
        draw_text(line_img, text_R, pos_R, font_scale=0.5, color=text_color_R)
    
    return line_img
