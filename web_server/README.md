# Accident Data Server

이 프로젝트는 Flutter 기반의 클라이언트 애플리케이션에서 사고 발생 시 전송되는 이미지와 센서 데이터를 수신하기 위해 설계된 FastAPI 서버입니다.

클라이언트(Flutter 앱)는 사고가 감지되면 관련 이미지 파일과 센서 데이터(JSON 형식)를 이 서버로 전송하며, 서버는 이를 안전하게 저장하고 관리합니다.



전송된 사진에서 차량번호판과 사람 얼굴은 블러처리 됩니다.

# 모델 출처

| 모델 이름 | 출처 |
| -------| ---|
| yolov11n-face.pt | [링크1] |
| license-plate-finetune-v1x.pt | [링크2] |

# 환경 설정
``` shell
# 가상 환경 생성 (선택 사항)
python -m venv web_env
source web_env/bin/activate  # Linux/macOS
# .\web_env\Scripts\activate # Windows

pip install -r requirements.txt
```



# 서버 실행
``` shell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```


[링크1]: https://huggingface.co/deepghs/yolo-face/blob/b6b06ab1a58eba921209ee6431a79ffedc498eb1/yolov11n-face/model.pt

[링크2]: https://huggingface.co/morsetechlab/yolov11-license-plate-detection/blob/main/license-plate-finetune-v1x.pt