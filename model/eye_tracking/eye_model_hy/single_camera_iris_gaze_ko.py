#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single_camera_iris_gaze_ko.py

차선책 프로토타입: 웹캠 1대로 MediaPipe Face Mesh(iris 랜드마크) + OpenCV로 시선 마커를 같은 프레임에 오버레이
- 카메라 하나로 얼굴/눈을 촬영하고, 같은 프레임을 "월드"처럼 사용하여 시선 마커를 그립니다.
- 정확한 지오메트리 보정 없이 [0,1] 정규화 좌표를 동일 프레임에 매핑하는 시범용 코드입니다.
  연구/제품 용도로는 반드시 캘리브레이션, 머리자세 보정(Head Pose), 필터링(칼만/1€) 등을 추가하세요.

필요 환경:
- Python 3.9 이상(3.11 테스트됨)
- mediapipe >= 0.10.x, opencv-python, numpy

실행 예시(장치 인덱스는 환경에 따라 다름):
    python single_camera_iris_gaze_ko.py --cam 0 --w 1280 --h 720 --flip

키 조작:
    q : 종료
    h : 도움말 토글
    d : 디버그 토글(홍채 원, PIP 미니뷰)
    t : 시선 궤적(트레일) 토글
    r : EMA(지수이동평균) 및 트레일 리셋

작성: ChatGPT (사용자 요청에 따른 프로토타입)
라이선스: MIT
"""
import argparse
from collections import deque
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
# 널리 쓰이는 홍채(iris) 4점 인덱스(좌/우):
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

@dataclass
class EMA2D:
    """2차원 점([0,1] 정규화 좌표)에 대한 간단한 지수이동평균(EMA) 필터."""
    alpha: float = 0.25                 # 알파↑ → 최신값 반영↑(반응 빠름)
    value: Optional[np.ndarray] = None  # (2,)

    def update(self, pt01: np.ndarray) -> np.ndarray:
        if self.value is None or not np.isfinite(self.value).all():
            self.value = pt01.astype(np.float32)
        else:
            self.value = self.alpha * pt01.astype(np.float32) + (1.0 - self.alpha) * self.value
        self.value = np.clip(self.value, 0.0, 1.0)
        return self.value

    def reset(self):
        self.value = None


def parse_args():
    ap = argparse.ArgumentParser(description="웹캠 1대용 시선 오버레이 프로토타입(한국어 주석)")
    ap.add_argument("--cam", type=int, default=0, help="카메라 OpenCV 인덱스")
    ap.add_argument("--w", type=int, default=1280, help="캡처 가로 해상도(요청값)")
    ap.add_argument("--h", type=int, default=720, help="캡처 세로 해상도(요청값)")
    ap.add_argument("--flip", action="store_true", help="영상 좌우 반전(셀피 느낌)")
    ap.add_argument("--draw_scale", type=float, default=1.0, help="도형/텍스트 스케일")
    ap.add_argument("--min_det_conf", type=float, default=0.5, help="FaceMesh min_detection_confidence")
    ap.add_argument("--min_trk_conf", type=float, default=0.5, help="FaceMesh min_tracking_confidence")
    ap.add_argument("--ema_alpha", type=float, default=0.25, help="EMA 알파(0<alpha<=1)")
    ap.add_argument("--use_left_eye_only", action="store_true", help="좌안만 사용")
    ap.add_argument("--use_right_eye_only", action="store_true", help="우안만 사용")
    ap.add_argument("--trail_len", type=int, default=50, help="시선 트레일 길이(프레임) 0이면 끔")
    return ap.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """OpenCV 카메라 열기 유틸(Windows: CAP_DSHOW 우선)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패 (index={index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def iris_center_from_landmarks(landmarks, frame_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    좌/우 홍채 중심 근사([0,1])와 대략적인 반경을 추정.
    반환: (left_center01, right_center01, left_radius01, right_radius01)
    radius01은 정규화 좌표계에서의 대략적 반경(평균 거리).
    """
    def gather_xy(idxs):
        pts = []
        for i in idxs:
            if i < len(landmarks):
                lm = landmarks[i]
                pts.append((lm.x, lm.y))
        if len(pts) != len(idxs):
            return None
        return np.array(pts, dtype=np.float32)

    def center_radius01(pts01):
        if pts01 is None:
            return None, None
        c = pts01.mean(axis=0)
        r = np.mean(np.linalg.norm(pts01 - c[None, :], axis=1))
        return c, float(r)

    left_pts01 = gather_xy(LEFT_IRIS)
    right_pts01 = gather_xy(RIGHT_IRIS)

    lc, lr = center_radius01(left_pts01)
    rc, rr = center_radius01(right_pts01)
    return lc, rc, lr, rr


def pick_gaze_point(left01: Optional[np.ndarray], right01: Optional[np.ndarray], use_left_only: bool, use_right_only: bool) -> Optional[np.ndarray]:
    if use_left_only and left01 is not None:
        return left01
    if use_right_only and right01 is not None:
        return right01
    if left01 is not None and right01 is not None:
        return (left01 + right01) / 2.0
    return left01 if left01 is not None else right01


def draw_hud(frame_bgr: np.ndarray, text_lines, scale: float = 1.0):
    x, y = 10, 22
    for t in text_lines:
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(22 * scale)


def draw_pip_eye(frame_bgr: np.ndarray, eye_bgr: np.ndarray, topleft=(10, 10), max_w=300):
    """메인 프레임 위에 미니뷰(PIP)를 그립니다."""
    h, w = eye_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        eye_bgr = cv2.resize(eye_bgr, (int(w * scale), int(h * scale)))
    x, y = topleft
    h2, w2 = eye_bgr.shape[:2]
    # 배경 윤곽(반투명 배경을 흉내):
    roi = frame_bgr[y : y + h2 + 8, x : x + w2 + 8]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        overlay = roi.copy()
        overlay[:] = (32, 32, 32)
        cv2.addWeighted(overlay, 0.35, roi, 0.65, 0, roi)
        frame_bgr[y + 4 : y + 4 + h2, x + 4 : x + 4 + w2] = eye_bgr
        cv2.rectangle(frame_bgr, (x + 4, y + 4), (x + 4 + w2, y + 4 + h2), (200, 200, 200), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    cap = open_camera(args.cam, args.w, args.h)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # 홍채 랜드마크 포함(필수)
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf,
    )

    ema = EMA2D(alpha=args.ema_alpha)

    show_help = True
    show_debug = False
    show_trail = False
    trail = deque(maxlen=max(0, args.trail_len))  # 픽셀 좌표 저장

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("경고: 카메라 프레임 읽기 실패")
            break

        if args.flip:
            frame_bgr = cv2.flip(frame_bgr, 1)

        # MediaPipe 입력 준비
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        h, w = frame_bgr.shape[:2]
        gaze_pt01 = None
        left01 = right01 = None
        left_r01 = right_r01 = None

        if res.multi_face_landmarks:
            face_landmarks = res.multi_face_landmarks[0].landmark
            left01, right01, left_r01, right_r01 = iris_center_from_landmarks(face_landmarks, (h, w))
            gaze01 = pick_gaze_point(left01, right01, args.use_left_eye_only, args.use_right_eye_only)
            if gaze01 is not None and np.isfinite(gaze01).all():
                gaze_pt01 = ema.update(gaze01)

        # 메인 프레임(=월드 프레임) 위에 시선 마커 오버레이
        if gaze_pt01 is not None:
            gx, gy = int(gaze_pt01[0] * w), int(gaze_pt01[1] * h)
            r = int(10 * args.draw_scale)
            cv2.circle(frame_bgr, (gx, gy), r, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(frame_bgr, (gx, gy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            if show_trail:
                trail.append((gx, gy))
        else:
            # 검출 끊기면 트레일만 이어질 수 있으나, 여기서는 유지
            pass

        # 트레일 그리기
        if show_trail and len(trail) >= 2:
            for i in range(1, len(trail)):
                cv2.line(frame_bgr, trail[i - 1], trail[i], (0, 0, 255), 2, cv2.LINE_AA)

        # 디버그: 홍채 근사 원 및 PIP(미니뷰)
        if show_debug and (left01 is not None or right01 is not None):
            # 홍채 근사 원: 정규화 반경 → 픽셀 반경으로 변환(대략 가로 기준)
            if left01 is not None and left_r01 is not None:
                cx, cy = int(left01[0] * w), int(left01[1] * h)
                rr = int(max(2, left_r01 * w))
                cv2.circle(frame_bgr, (cx, cy), rr, (0, 255, 0), 2, cv2.LINE_AA)
            if right01 is not None and right_r01 is not None:
                cx, cy = int(right01[0] * w), int(right01[1] * h)
                rr = int(max(2, right_r01 * w))
                cv2.circle(frame_bgr, (cx, cy), rr, (0, 255, 255), 2, cv2.LINE_AA)

            # 간단한 PIP: 두 눈의 중심 평균 주변을 확대해서 표시
            if left01 is not None or right01 is not None:
                if left01 is not None and right01 is not None:
                    c01 = (left01 + right01) / 2.0
                    r01 = (left_r01 if left_r01 is not None else 0.0 + right_r01 if right_r01 is not None else 0.0) / 2.0
                elif left01 is not None:
                    c01, r01 = left01, left_r01 if left_r01 is not None else 0.03
                else:
                    c01, r01 = right01, right_r01 if right_r01 is not None else 0.03
                cx, cy = int(c01[0] * w), int(c01[1] * h)
                rad = int(max(16, (r01 or 0.02) * w * 5))  # 대략 눈 주변 크롭 폭
                x0, y0 = max(0, cx - rad), max(0, cy - rad)
                x1, y1 = min(w, cx + rad), min(h, cy + rad)
                eye_crop = frame_bgr[y0:y1, x0:x1].copy()
                if eye_crop.size > 0:
                    # 시선/홍채 표시를 PIP에도 겹쳐주기(옵션)
                    if gaze_pt01 is not None:
                        gx, gy = int(gaze_pt01[0] * w), int(gaze_pt01[1] * h)
                        pgx, pgy = gx - x0, gy - y0
                        cv2.drawMarker(eye_crop, (pgx, pgy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
                    draw_pip_eye(frame_bgr, eye_crop, topleft=(10, 10), max_w=320)

        # HUD
        if show_help:
            draw_hud(
                frame_bgr,
                [
                    "Single-Camera Iris Gaze (Prototype)",
                    "[q] 종료  [h] 도움말  [d] 디버그  [t] 트레일  [r] EMA리셋",
                    f"옵션: left-only={args.use_left_eye_only}, right-only={args.use_right_eye_only}, flip={args.flip}",
                ],
                scale=args.draw_scale,
            )

        cv2.imshow("World (Same Frame with Gaze)", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('t'):
            show_trail = not show_trail
            if not show_trail:
                trail.clear()
        elif key == ord('r'):
            ema.reset()
            trail.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
