# 2카메라 시선 오버레이 프로토타입 (MediaPipe + OpenCV, 한국어 주석)

**목적**:

- **월드 카메라**: 외부 장면을 촬영하여 화면에 표시
- **눈 카메라**: MediaPipe Face Mesh(홍채 랜드마크 포함)를 이용해 **홍채 중심 근사**를 구하고, 월드 영상 위에 **시선 마커**를 표시

> ⚠️ 본 코드는 **정확한 캘리브레이션/머리자세 보정 없이**, MediaPipe가 주는 [0,1] **정규화 좌표를 월드 프레임에 직접 매핑**하는 **시범용 프로토타입**입니다.  
> 실제 연구/제품 수준 정확도를 위해서는 **캘리브레이션(월드↔눈 특징 매핑)**, **Head Pose 보정**, **칼만/저역필터 등 시간 필터링**이 필요합니다.

---

## 1) 환경 준비 (Windows, PowerShell 기준)

```powershell
python -m venv venv
..\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install mediapipe opencv-python numpy
```

- Python 3.9 ~ 3.12 권장(Windows + Python 3.11 테스트됨)
- 웹캠 2대 필요(장치 인덱스는 PC마다 상이)

---

## 2) 실행

동일 폴더에 `dual_camera_iris_gaze.py`를 두고 아래처럼 실행합니다.

```powershell
python dual_camera_iris_gaze.py --world_cam 0 --eye_cam 1 --world_w 1280 --world_h 720 --eye_w 640 --eye_h 480 --flip_eye
```

### 주요 인자

- `--world_cam`, `--eye_cam`: OpenCV 카메라 인덱스(0/1/2…). 서로 바뀌었으면 값을 교체.
- `--world_w`, `--world_h`: 월드 카메라 캡처 해상도(요청값, 장치가 무시할 수 있음)
- `--eye_w`, `--eye_h`: 눈 카메라 캡처 해상도(요청값)
- `--flip_eye`: 눈 카메라 영상을 좌우 반전(셀피 카메라용)
- `--min_det_conf`, `--min_trk_conf`: MediaPipe Face Mesh confidence
- `--ema_alpha`: 0 < alpha ≤ 1, 클수록 반응 빠르나 노이즈 증가
- `--use_left_eye_only` / `--use_right_eye_only`: 좌/우 한쪽만 사용

### 실행 중 키

- `q`: 종료
- `h`: 도움말 표시 토글
- `d`: 눈 프레임 디버그(홍채 중심 원) 토글
- `r`: EMA 스무딩 상태 초기화

---

## 3) 동작 개요

1. OpenCV로 **월드/눈** 카메라에서 프레임 캡처
2. 눈 프레임을 **BGR→RGB**로 변환 후 **MediaPipe Face Mesh(refine_landmarks=True)** 수행
   - 커뮤니티에서 자주 쓰는 홍채 랜드마크 인덱스:
     - `LEFT_IRIS  = [474, 475, 476, 477]`
     - `RIGHT_IRIS = [469, 470, 471, 472]`
3. 각 눈에 대해 **홍채 경계 4점의 평균**을 **중심 근사**로 사용(정규화 좌표 [0,1])
4. 좌/우 중 선택(혹은 평균) 후, **EMA(지수이동평균)**로 스무딩
5. 정규화 좌표를 월드 프레임 픽셀로 매핑하여 **마커(원 + 십자)**를 오버레이

---

## 4) 트러블슈팅

- **카메라가 뒤바뀜**: `--world_cam`/`--eye_cam` 값을 서로 교체
- **프레임 드랍/지연**: 해상도 줄이기(`--world_w 640 --world_h 360`, `--eye_w 320 --eye_h 240`)
- **홍채가 검출되지 않음**: 조명 확보, `--min_det_conf`/`--min_trk_conf` 낮춰보기, 얼굴이 충분히 크게 나오게 위치 조정
- **마커가 좌우 반대로 움직임**: `--flip_eye` 옵션 시도

---

## 5) 확장 아이디어(향후 연구/개발)

- **캘리브레이션**: 화면/월드의 여러 점을 바라보게 하여 (눈 특징 → 월드 좌표) 회귀/호모그래피/다항식 매핑 학습
- **머리자세 보정**: Face Mesh 3D 랜드마크로 PnP 추정 → 안구 좌표계/카메라 좌표계 정렬
- **고급 필터**: 칼만 필터, 저역 필터, 저지터 스무딩
- **커스텀 모델 교체**: MediaPipe 대신 자체 학습한 홍채/동공 추정 모델로 모듈 바꾸기

---

## 6) 파일

- `dual_camera_iris_gaze_ko.py` — 한국어 주석 전체 코드
- `README_ko.md` — 이 문서

---

## 7) 라이선스

MIT
