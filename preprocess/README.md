# 데이터 전처리

이 프로그램은 리눅스 환경에서 aihub 데이터를 받고 cocodata set을 변환합니다.


|사용한 데이터| 출처 |
|-----| ----|
|실내 자율주차용 데이터| [링크1]|
|주차 공간 탐색을 위한 차량 관점 복합 데이터| [링크2]|




## 설치 및 환경 구성
``` shell
#aihub shell down
curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do
chmod +x aihubshell

# 가상 환경 생성 (선택 사항)
python -m venv down_env
source down_env/bin/activate  # Linux/macOS
# .\down_env\Scripts\activate # Windows

pip install -r requirements.txt

# 환경 나가기
deactivate
```

## 실내 자율주차용 데이터 실행 순서
``` shell
python aihub_down.py
python DT2_split_pre.py
python DT2_coco_transform.py
```

## 주차 공간 탐색을 위한 차량 관점 복합 데이터
``` shell
python parking/parking_down.py
python parking/unzip.py
python parking/preprocessing.py
python parking/coco_transform.py
```

## 결과물
![preprocess result](../img/preprocess_result.png)



[링크1]: https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%8B%A4%EB%82%B4&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM003&aihubDataSe=data&dataSetSn=71576

[링크2]: https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%A3%BC%EC%B0%A8+%EA%B3%B5%EA%B0%84+%ED%83%90%EC%83%89&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=598