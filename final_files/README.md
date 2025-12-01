## 🚗 Parking Guidance System Demo

주차 가이드 시스템 시연 코드

이 프로젝트는 OpenCV 기반의 주차 보조(가이드라인) 시스템을 구현한 데모 코드입니다.
카메라 영상을 입력받아 주차 가능 영역 탐지, 사용자 선택 기반 주차 경로 생성,
후진 가이드라인 시각화 등을 수행합니다.

## ✨ Features

🅿️ Parking Slot Detection
Detectron2 또는 영상처리 기반 슬롯 탐지

🎯 User-Selected Target Parking Area
마우스 클릭으로 원하는 주차 공간 지정

🌀 Bezier Curve / Straight-line Parking Path
테슬라 스타일 자연스러운 후진 경로 생성

🎥 Real-time Overlay Rendering
카메라 영상 위에 곡선/직선 가이드라인 시각화

🔄 Multiple Camera Support
전방 → 후방 영상 전환, 단계별 시뮬레이션 진행

▶️ Demo Execution
python parking_guidance.py
