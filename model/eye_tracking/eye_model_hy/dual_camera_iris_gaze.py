#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dual_camera_iris_gaze_ko.py

프로토타입: 2대의 카메라와 MediaPipe Face Mesh(iris 랜드마크) + OpenCV를 활용한 시선 마커 오버레이
- 카메라 A("world"): 바깥 세상을 촬영 → 배경 영상으로 사용
- 카메라 B("eye"): 얼굴/눈을 촬영 → MediaPipe로 홍채(iris) 좌표를 구해서 월드 영상 위에 시선 마커 표시

※ 이 코드는 정확한 지오메트리 보정 없이, MediaPipe가 제공하는 [0,1] 정규화 좌표를
  월드 프레임에 단순 매핑하는 '시범용 프로토타입'입니다. 연구/제품 수준의 정확도를 원하면
  반드시 캘리브레이션, 머리자세(Head Pose) 보정, 고급 필터링(칼만 등)을 추가하세요.

필요 환경:
- Python 3.9 이상(3.11 테스트됨)
- mediapipe >= 0.10.x, opencv-python, numpy

실행 예시(장치 인덱스는 환경에 따라 다름):
    python dual_camera_iris_gaze_ko.py --world_cam 0 --eye_cam 1 --world_w 1280 --world_h 720 --eye_w 640 --eye_h 480 --flip_eye

키 조작:
    q : 종료
    h : 도움말 토글
    d : 눈 프레임에 홍채 디버그 원 표시 토글
    r : EMA(지수이동평균) 스무딩 상태 리셋

작성: ChatGPT (사용자 요청에 따른 프로토타입)
라이선스: MIT
"""
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# -------- MediaPipe 임포트 --------
try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit(
        "mediapipe 임포트 실패. 다음으로 설치하세요: pip install mediapipe\n"
        f"원본 오류: {e}"
    )

# ----------------------------
# 상수 / 랜드마크 인덱스
# ----------------------------
# MediaPipe Face Mesh에서 refine_landmarks=True로 설정하면 홍채(iris) 랜드마크가 포함됩니다.
# 커뮤니티에서 널리 쓰이는 홍채(iris) 4점 인덱스(좌/우):
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# 일부 자료는 468(우), 473(좌)을 iris 중심으로 쓰기도 하나(참고용),
# 본 코드는 각 눈의 4개 경계점 평균을 중심 근사로 사용합니다.


@dataclass
class EMA2D:
    """2차원 점([0,1] 정규화 좌표)에 대한 간단한 지수이동평균(EMA) 필터."""
    alpha: float = 0.25                 # EMA 계수, 알파가 클수록 최신값 반영이 커짐(반응 빠름)
    value: Optional[np.ndarray] = None  # 현재 EMA 값 (shape: (2,))

    def update(self, pt01: np.ndarray) -> np.ndarray:
        """새 좌표(pt01)를 반영하여 EMA 값을 갱신하고 반환."""
        if self.value is None or not np.isfinite(self.value).all():
            self.value = pt01.astype(np.float32)
        else:
            self.value = self.alpha * pt01.astype(np.float32) + (1.0 - self.alpha) * self.value
        # 수치적 안전을 위해 [0,1]로 클램프
        self.value = np.clip(self.value, 0.0, 1.0)
        return self.value

    def reset(self):
        """EMA 내부 상태를 리셋."""
        self.value = None


def parse_args():
    """명령행 인자 파서 구성."""
    ap = argparse.ArgumentParser(description="MediaPipe + OpenCV 2카메라 시선 오버레이 프로토타입(한국어 주석)")
    ap.add_argument("--world_cam", type=int, default=0, help="월드(배경) 카메라 OpenCV 인덱스")
    ap.add_argument("--eye_cam", type=int, default=1, help="눈/얼굴 카메라 OpenCV 인덱스")
    ap.add_argument("--world_w", type=int, default=1280, help="월드 카메라 캡처 가로 해상도(요청값)")
    ap.add_argument("--world_h", type=int, default=720, help="월드 카메라 캡처 세로 해상도(요청값)")
    ap.add_argument("--eye_w", type=int, default=640, help="눈 카메라 캡처 가로 해상도(요청값)")
    ap.add_argument("--eye_h", type=int, default=480, help="눈 카메라 캡처 세로 해상도(요청값)")
    ap.add_argument("--flip_eye", action="store_true", help="눈 카메라 영상을 좌우 반전(셀피 느낌)")
    ap.add_argument("--draw_scale", type=float, default=1.0, help="도형/텍스트 스케일 팩터")
    ap.add_argument("--min_det_conf", type=float, default=0.5, help="FaceMesh min_detection_confidence")
    ap.add_argument("--min_trk_conf", type=float, default=0.5, help="FaceMesh min_tracking_confidence")
    ap.add_argument("--ema_alpha", type=float, default=0.25, help="EMA 알파(0<alpha<=1)")
    ap.add_argument("--use_left_eye_only", action="store_true", help="좌안(Left)만 사용")
    ap.add_argument("--use_right_eye_only", action="store_true", help="우안(Right)만 사용")
    return ap.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    OpenCV 카메라 열기 유틸.
    - 일부 Windows 환경에서 CAP_DSHOW 백엔드가 더 안정적이라 먼저 시도
    - 해상도 설정은 장치에 따라 무시될 수 있음
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # 백엔드 힌트 없이 재시도
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패 (index={index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """BGR(OpenCV 기본) → RGB(MediaPipe가 선호) 변환."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def iris_center_from_landmarks(landmarks, frame_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    주어진 얼굴 랜드마크들(정규화 좌표)을 사용해 좌/우 홍채 중심 근사값을 구함.
    - landmarks: MediaPipe가 반환한 얼굴 랜드마크 리스트(정규화 [0,1] 좌표)
    - frame_shape: (H, W) (참고용; 본 함수에서는 정규화 좌표만 사용)
    반환: (left_center01, right_center01) — 둘 중 없을 수도 있으므로 Optional
    """
    # 내부 헬퍼: 특정 인덱스 집합의 (x,y) 정규화 좌표 배열 반환
    def gather_xy(idxs):
        pts = []
        for i in idxs:
            if i < len(landmarks):
                lm = landmarks[i]
                pts.append((lm.x, lm.y))
        if len(pts) != len(idxs):
            return None  # 일부 누락 시 None
        return np.array(pts, dtype=np.float32)

    left_pts01 = gather_xy(LEFT_IRIS)
    right_pts01 = gather_xy(RIGHT_IRIS)

    # 4개 경계점 평균을 중심 근사로 사용
    def center01(pts01):
        if pts01 is None:
            return None
        return pts01.mean(axis=0)  # (2,)

    return center01(left_pts01), center01(right_pts01)


def pick_gaze_point(left01: Optional[np.ndarray],
                    right01: Optional[np.ndarray],
                    use_left_only: bool,
                    use_right_only: bool) -> Optional[np.ndarray]:
    """
    좌/우 중심 중 하나의 최종 시선 포인트(정규화 [0,1])를 선택.
    정책:
      - use_left_only / use_right_only 가 지정되면 해당 눈만 사용
      - 둘 다 가능하면 단순 평균(헤uristic)
      - 아니면 존재하는 쪽만 사용
    """
    if use_left_only and left01 is not None:
        return left01
    if use_right_only and right01 is not None:
        return right01
    if left01 is not None and right01 is not None:
        return (left01 + right01) / 2.0
    return left01 if left01 is not None else right01


def draw_hud(frame_bgr: np.ndarray, text_lines, scale: float = 1.0):
    """세계(월드) 프레임 위 왼쪽 상단에 간단한 도움말/상태 텍스트 표시."""
    x, y = 10, 20
    for t in text_lines:
        # 가독성을 위한 테두리(검정) + 본문(흰색) 두 번 그리기
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(20 * scale)


def main():
    args = parse_args()

    # ---------- 카메라 열기 ----------
    world_cap = open_camera(args.world_cam, args.world_w, args.world_h)  # 월드(배경)
    eye_cap = open_camera(args.eye_cam, args.eye_w, args.eye_h)          # 눈(얼굴)

    # ---------- MediaPipe Face Mesh 초기화 ----------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,          # 동영상 스트림 모드
        max_num_faces=1,                  # 최대 1명
        refine_landmarks=True,            # 홍채 랜드마크 포함(필수)
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf,
    )

    # ---------- 시선 좌표 스무딩(EMA) ----------
    ema = EMA2D(alpha=args.ema_alpha)

    # ---------- UI 토글 상태 ----------
    show_help = True     # 도움말 표시
    show_debug = False   # 눈 프레임에 디버그 원 표시

    while True:
        # 1) 월드/눈 프레임 읽기
        ok_world, world_bgr = world_cap.read()
        ok_eye, eye_bgr = eye_cap.read()
        if not ok_world or not ok_eye:
            print("경고: 카메라 프레임 읽기 실패(월드/눈 중 하나).")
            break

        # 2) 눈 프레임 전처리(좌우 반전 옵션)
        if args.flip_eye:
            eye_bgr = cv2.flip(eye_bgr, 1)

        # 3) MediaPipe 처리용 RGB로 변환(BGR→RGB)
        eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
        eye_rgb.flags.writeable = False  # 성능 최적화: 참조 전달
        res = face_mesh.process(eye_rgb) # 얼굴 랜드마크 좌표를 제공합니다. 여기서 반환되는 results.multi_face_landmarks 안에 각 점의 (x, y, z) 정규화 좌표가 포함됩니다.
        eye_rgb.flags.writeable = True

        # 4) 홍채 중심(정규화 좌표) 추정
        gaze_pt01 = None  # 최종 시선 좌표를 담을 변수 (x_norm, y_norm), [0,1] 값
        if res.multi_face_landmarks: # 얼굴 랜드마크 리스트가 있으면..
            # 첫 번째 얼굴
            face_landmarks = res.multi_face_landmarks[0].landmark # NormalizedLandmark(x, y, z) 구조. x,y는 입력 프레임 기준 정규화 좌표(가로/세로를 1로 본 상대좌표)입니다. z는 “깊이(depth)”. 값이 작을수록(대개 더 음수일수록) 카메라에 더 가까움을 뜻합니다

            # 좌/우 홍채 중심 근사값 계산
            left01, right01 = iris_center_from_landmarks(face_landmarks, eye_bgr.shape[:2]) # eye_bgr.shape[:2]: (H, W)

            # 좌/우 중 사용할 최종 포인트 선택
            gaze01 = pick_gaze_point(left01, right01, args.use_left_eye_only, args.use_right_eye_only)

            # 스무딩(EMA) 적용
            if gaze01 is not None and np.isfinite(gaze01).all():
                gaze_pt01 = ema.update(gaze01)

            # (옵션) 디버그: 눈 프레임에 좌/우 중심 근사 그리기
            if show_debug and (left01 is not None or right01 is not None):
                h_eye, w_eye = eye_bgr.shape[:2]
                if left01 is not None:
                    lx, ly = int(left01[0] * w_eye), int(left01[1] * h_eye)
                    cv2.circle(eye_bgr, (lx, ly), int(6 * args.draw_scale), (0, 255, 0), 2, cv2.LINE_AA)
                if right01 is not None:
                    rx, ry = int(right01[0] * w_eye), int(right01[1] * h_eye)
                    cv2.circle(eye_bgr, (rx, ry), int(6 * args.draw_scale), (0, 255, 255), 2, cv2.LINE_AA)

        # 5) 월드 프레임에 시선 마커 오버레이
        if gaze_pt01 is not None:
            h_w, w_w = world_bgr.shape[:2]
            gx, gy = int(gaze_pt01[0] * w_w), int(gaze_pt01[1] * h_w)  # 정규화→픽셀
            r = int(10 * args.draw_scale)
            # 원 + 십자 마커
            cv2.circle(world_bgr, (gx, gy), r, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(world_bgr, (gx, gy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # 6) HUD(도움말/상태) 그리기
        if show_help:
            draw_hud(
                world_bgr,
                [
                    "Dual-Camera Iris Gaze (Prototype)",
                    "[q] 종료  [h] 도움말  [d] 디버그  [r] EMA리셋",
                    f"옵션: left-only={args.use_left_eye_only}, right-only={args.use_right_eye_only}, flip_eye={args.flip_eye}",
                ],
                scale=args.draw_scale,
            )

        # 7) 결과 표시(월드 / 눈)
        cv2.imshow("World (Gaze Overlay)", world_bgr)
        cv2.imshow("Eye (Debug)", eye_bgr if show_debug else cv2.resize(
            eye_bgr,
            (min(480, eye_bgr.shape[1]), int(min(480, eye_bgr.shape[1]) * eye_bgr.shape[0] / max(1, eye_bgr.shape[1])))
        ))

        # 8) 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('r'):
            ema.reset()

    # 자원 해제
    world_cap.release()
    eye_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
