import cv2, time, numpy as np
from collections import deque
import mediapipe as mp
from enum import Enum

# =====================[ 설정값 ]=====================
CAM_INDEX = 0

# 9점(3x3) 캘리브 포인트
CALIB_POINTS = [
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
]

BASE_DWELL = 0.9  # 캘리브 1점 응시 시간(초)

# ===== FSM / Fixation (완화된 파라미터) =====
FSM_FREQ = 30.0
VEL_THRESH_ENTER   = 0.007   # 진입 속도 문턱 (완화)
VEL_THRESH_EXIT    = 0.018   # 이탈 속도 문턱 (더 느슨)
DISP_THRESH_ENTER  = 0.020   # 진입 분산 문턱 (완화)
DISP_THRESH_EXIT   = 0.050   # 이탈 분산 문턱 (더 느슨)
MIN_FIXATING_SEC   = 0.30    # CANDIDATE -> FIXATING 최소 시간
MIN_FIXED_SEC      = 0.25    # FIXATING -> FIXED 최소 시간
EVENT_HOLD_SEC     = 1.5     # FIXED 유지 1.5s면 이벤트
EVENT_COOLDOWN     = 2.0     # 이벤트 후 재발사 최소 대기
SMOOTH_WIN         = 11      # 중앙값 창 확대 → 더 차분

# ===== Zoom PiP (유지 길게) =====
ZOOM_BOX_FRAC = 0.28
ZOOM_SCALE    = 2.0
ZOOM_PIP_FRAC = 0.35
ZOOM_DURATION = 3.0          # 3초 유지

# ===== 표시용 EMA + 데드존(강화) =====
DISPLAY_EMA_ALPHA = 0.08     # 작을수록 더 느릿/차분
DISPLAY_DEAD_FRAC = 0.012    # 화면 짧은 변의 1.2% 이내 이동 무시

# MediaPipe FaceMesh iris 인덱스
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
mp_face = mp.solutions.face_mesh
# ====================================================


# ---------------------[ 유틸 ]---------------------
def iris_center(landmarks, w, h):
    def center(ids):
        xs, ys = [], []
        for i in ids:
            xs.append(landmarks[i].x * w)
            ys.append(landmarks[i].y * h)
        return np.array([np.mean(xs), np.mean(ys)], dtype=np.float32)
    lc = center(LEFT_IRIS)
    rc = center(RIGHT_IRIS)
    both = (lc + rc) / 2.0
    return lc, rc, both

def target_radius(w, h):
    s = min(w, h)
    return max(14, int(s * 0.03))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_around_center(frame, cx, cy, box_frac):
    h, w = frame.shape[:2]
    box = int(min(w, h) * box_frac)
    half = box // 2
    x1 = clamp(int(cx) - half, 0, w-1)
    y1 = clamp(int(cy) - half, 0, h-1)
    x2 = clamp(x1 + box, 0, w)
    y2 = clamp(y1 + box, 0, h)
    if x2 - x1 < box:
        x1 = clamp(x2 - box, 0, w - box)
    if y2 - y1 < box:
        y1 = clamp(y2 - box, 0, h - box)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def draw_zoom_pip(frame, center_xy, box_frac=0.25, scale=2.0, pip_frac=0.35, pos='tr'):
    h, w = frame.shape[:2]
    cx, cy = center_xy
    crop, box_xyxy = crop_around_center(frame, cx, cy, box_frac)
    zoomed = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pip_size = int(min(w, h) * pip_frac)
    zoomed = cv2.resize(zoomed, (pip_size, pip_size))

    margin = max(10, pip_size // 20)
    if pos == 'tr':
        px1, py1 = w - pip_size - margin, margin
    elif pos == 'tl':
        px1, py1 = margin, margin
    elif pos == 'br':
        px1, py1 = w - pip_size - margin, h - pip_size - margin
    else:
        px1, py1 = margin, h - pip_size - margin
    px2, py2 = px1 + pip_size, py1 + pip_size

    shadow = frame.copy()
    cv2.rectangle(shadow, (px1+5, py1+5), (px2+5, py2+5), (0,0,0), -1)
    alpha = 0.35
    frame[:] = cv2.addWeighted(shadow, alpha, frame, 1-alpha, 0)
    cv2.rectangle(frame, (px1-2, py1-2), (px2+2, py2+2), (0,0,0), -1)
    frame[py1:py2, px1:px2] = zoomed
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,0), 2)

    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, "ZOOM", (px1, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    return frame
# -------------------------------------------------


# ----------------[ 시계열 필터 ]------------------
class Kalman2D:
    def __init__(self, freq=30.0, proc=2e-4, meas=8e-3):  # ← 더 느긋: proc↓ meas↑
        dt = 1.0/freq
        self.A = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32)*proc
        self.R = np.eye(2, dtype=np.float32)*meas
        self.x = np.zeros((4,1), np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.inited = False
    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
    def update(self, z):
        z = np.asarray(z, np.float32).reshape(2,1)
        if not self.inited:
            self.x[:2] = z
            self.inited = True
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
    def step(self, meas_uv_norm):
        self.predict()
        self.update(meas_uv_norm)
        return self.x[:2].ravel(), self.x[2:].ravel()

class TemporalMedian:
    def __init__(self, win=11):
        self.buf_u = deque(maxlen=win)
        self.buf_v = deque(maxlen=win)
    def push(self, u, v):
        self.buf_u.append(float(u)); self.buf_v.append(float(v))
    def value(self):
        if not self.buf_u: return None
        return np.median(self.buf_u), np.median(self.buf_v)
    def dispersion(self):
        if len(self.buf_u) < 3: return 1e9
        du = max(self.buf_u)-min(self.buf_u)
        dv = max(self.buf_v)-min(self.buf_v)
        return du + dv
# -------------------------------------------------


# -----------------[ FSM 정의 ]-------------------
class State(Enum):
    SACCADE   = 0
    CANDIDATE = 1
    FIXATING  = 2
    FIXED     = 3

class GazeFSM:
    def __init__(self):
        self.state = State.SACCADE
        self.t_candidate = None
        self.t_fixating = None
        self.t_fixed = None
    def step(self, uv_norm, vel_norm, disp_norm):
        now = time.time()
        s = self.state
        if s == State.FIXED:
            if (vel_norm > VEL_THRESH_EXIT) or (disp_norm > DISP_THRESH_EXIT):
                self.__init__(); return State.SACCADE
        if s == State.SACCADE:
            if (vel_norm < VEL_THRESH_ENTER) and (disp_norm < DISP_THRESH_ENTER):
                self.state = State.CANDIDATE
                self.t_candidate = now
        elif s == State.CANDIDATE:
            if (vel_norm < VEL_THRESH_ENTER) and (disp_norm < DISP_THRESH_ENTER):
                if now - self.t_candidate >= MIN_FIXATING_SEC:
                    self.state = State.FIXATING
                    self.t_fixating = now
            else:
                self.__init__()
        elif s == State.FIXATING:
            if (vel_norm < VEL_THRESH_ENTER) and (disp_norm < DISP_THRESH_ENTER):
                if now - self.t_fixating >= MIN_FIXED_SEC:
                    self.state = State.FIXED
                    self.t_fixed = now
            else:
                self.__init__()
        return self.state
# -------------------------------------------------


# ----------------[ 캘리브레이션 ]----------------
def collect_calibration(cap, face_mesh, screen_w, screen_h):
    X_feat, Y_screen = [], []
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for idx, (nx, ny) in enumerate(CALIB_POINTS):
        target = (int(nx*screen_w), int(ny*screen_h))
        start_t = None; samples = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1); h, w = frame.shape[:2]
            disp = frame.copy()
            R = target_radius(w, h)
            cv2.circle(disp, target, R, (255,255,255), -1)
            cv2.circle(disp, target, R+6, (0,0,0), 2)
            font_scale = max(1.2, min(w, h)/500)
            cv2.putText(disp, f"Look at dot {idx+1}/{len(CALIB_POINTS)}",
                        (50, int(80*font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255,255,255), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                try:
                    lc, rc, both = iris_center(lm, w, h)
                    inter_eye = np.linalg.norm(lc-rc)+1e-6
                    feat = np.array([both[0]/w, both[1]/h, inter_eye/w, 1.0], np.float32)
                    cv2.circle(disp, (int(both[0]), int(both[1])), 7, (0,255,255), -1)
                    if start_t is None: start_t = time.time()
                    samples.append(feat)
                    elapsed = time.time()-start_t
                    cv2.putText(disp, f"Hold:{elapsed:.1f}/{BASE_DWELL:.1f}s",
                                (50, int(140*font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale*0.9, (0,255,255), 2, cv2.LINE_AA)
                    if elapsed>=BASE_DWELL:
                        feat_mean = np.mean(np.stack(samples,0),0)
                        X_feat.append(feat_mean)
                        Y_screen.append(np.array([target[0], target[1]], np.float32))
                        break
                except: pass
            cv2.imshow("Calibration", disp)
            if cv2.waitKey(1)==27:
                cv2.destroyWindow("Calibration")
                return np.empty((0,4),np.float32), np.empty((0,2),np.float32)
    cv2.destroyWindow("Calibration")
    return np.stack(X_feat,0), np.stack(Y_screen,0)

def fit_linear_map(X, Y):
    W,*_ = np.linalg.lstsq(X, Y, rcond=None)
    return W
# -------------------------------------------------


# =====================[ 메인 ]=====================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): return
    ok, frame = cap.read()
    if not ok: return
    frame = cv2.flip(frame,1); H,W = frame.shape[:2]

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        print("[1] Calibration start (9 dots)")
        X,Y = collect_calibration(cap, fm, W, H)
        if len(X)==0: return
        Wmap = fit_linear_map(X, Y)

        cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 시계열 필터/상태
        kf   = Kalman2D(freq=FSM_FREQ, proc=2e-4, meas=8e-3)
        tmed = TemporalMedian(win=SMOOTH_WIN)
        fsm  = GazeFSM()

        # 이벤트 상태
        zoom_until_ts = 0.0
        event_armed_at = None
        last_event_ts = 0.0

        # 표시용 EMA + 데드존 상태
        disp_lp = None

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame,1); h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                try:
                    lc, rc, both = iris_center(lm, w, h)
                    inter_eye = np.linalg.norm(lc-rc)+1e-6
                    xfeat = np.array([both[0]/w, both[1]/h, inter_eye/w, 1.0], np.float32)
                    g = xfeat @ Wmap
                    g[0] = np.clip(g[0],0,w-1); g[1] = np.clip(g[1],0,h-1)

                    # 정규화 좌표
                    un, vn = g[0]/w, g[1]/h

                    # 칼만 + 중앙값
                    (u_hat, v_hat), (du, dv) = kf.step((un, vn))
                    tmed.push(u_hat, v_hat)
                    disp_norm = tmed.dispersion()
                    vel_norm  = float(np.hypot(du, dv))
                    state = fsm.step((u_hat, v_hat), vel_norm, disp_norm)

                    # 표시용 좌표(정규화 → 픽셀)
                    u_disp, v_disp = tmed.value() or (u_hat, v_hat)
                    u_px = int(np.clip(u_disp * w, 0, w-1))
                    v_px = int(np.clip(v_disp * h, 0, h-1))

                    # ===== 표시용 EMA + 데드존 (강화) =====
                    dead_px = max(8, int(min(w, h) * DISPLAY_DEAD_FRAC))
                    if disp_lp is None:
                        disp_lp = np.array([u_px, v_px], dtype=np.float32)
                    else:
                        delta = np.array([u_px, v_px], np.float32) - disp_lp
                        if np.hypot(delta[0], delta[1]) >= dead_px:
                            disp_lp = (1 - DISPLAY_EMA_ALPHA) * disp_lp + DISPLAY_EMA_ALPHA * np.array([u_px, v_px], np.float32)
                    draw_x, draw_y = int(disp_lp[0]), int(disp_lp[1])
                    # ====================================

                    # 상태별 색상
                    color = {
                        State.SACCADE:(0,165,255),
                        State.CANDIDATE:(0,255,255),
                        State.FIXATING:(0,255,128),
                        State.FIXED:(0,255,0),
                    }[state]

                    # 이벤트(줌) 완화 로직
                    now = time.time()
                    if state == State.FIXED:
                        if event_armed_at is None:
                            event_armed_at = fsm.t_fixed
                        held = now - event_armed_at
                        if (held >= EVENT_HOLD_SEC) and (now - last_event_ts >= EVENT_COOLDOWN):
                            zoom_until_ts = now + ZOOM_DURATION
                            last_event_ts = now
                    else:
                        event_armed_at = None

                    # HUD
                    held = 0.0 if event_armed_at is None else (now - event_armed_at)
                    cv2.putText(frame, f"STATE:{state.name}", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                    cv2.putText(frame, f"Fixed hold:{held:4.1f}/{EVENT_HOLD_SEC:.1f}s",
                                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)

                    # 시선 원 (부드러운 좌표로 그림)
                    r = max(10, int(min(w, h)*0.015))
                    cv2.circle(frame, (draw_x, draw_y), r, color, -1)

                    # PiP 유지
                    if now < zoom_until_ts:
                        frame = draw_zoom_pip(frame, (draw_x, draw_y),
                                              box_frac=ZOOM_BOX_FRAC,
                                              scale=ZOOM_SCALE,
                                              pip_frac=ZOOM_PIP_FRAC,
                                              pos='tr')
                except:
                    pass

            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) == 27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

