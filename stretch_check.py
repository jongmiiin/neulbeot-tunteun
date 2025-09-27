import cv2
import time
import math
import numpy as np
import mediapipe as mp

# ===== 설정값 =====
MIRROR_INPUT = True          # 웹캠 프리뷰를 좌우 반전(거울화)해서 보여줄지
TILT_DEG_TARGET = 20         # 좌우 기울이기 목표 각도(도)
HOLD_MS_UP = 1000            # '손깍지 위로' 유지 시간(ms)
HOLD_MS_TILT = 800           # 좌/우 기울이기 유지 시간(ms)
MIN_VIS = 0.5                # 랜드마크 가시성 threshold

# 손깍지(가까움) 판정 난이도(0.0~1.0). 값이 작을수록 '더 가깝게' 요구
WRIST_CLOSE_RATIO_MAX = 0.65   # (양손목 거리 / 어깨폭) 최대 허용 비율
WRIST_LEVEL_EPS = 0.12         # 양손목 높이 차 허용(정상화 좌표)

# ===== 추가: 팔 펴짐/머리기준선/베이스라인 =====
ELBOW_MIN_ANGLE_DEG = 160    # 팔꿈치 펴짐 최소 각도(권장: 160~170)
HEAD_REF_MARGIN = 0.01       # 머리 기준선 여유
BASELINE_WINDOW = 40         # 업 자세 동안 기울기 평균에 쓸 샘플 수

# 프레임 종횡비(가로/세로). main 루프에서 갱신
ASPECT = 1.0

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def euc(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def safe_get(lms, idx):
    lm = lms[idx]
    return (lm.x, lm.y, lm.visibility)

def center(p1, p2):
    return ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)

# ===== 추가 유틸 =====
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def joint_angle_deg(a, b, c, aspect):
    """
    b(관절)에서의 내각(어깨-팔꿈치-손목).
    MediaPipe는 x,y를 각각 [0,1]로 정규화하므로, x에 aspect(=width/height)를 곱해 각도 왜곡을 보정.
    """
    ax, ay = a[0], a[1]; bx, by = b[0], b[1]; cx, cy = c[0], c[1]
    v1x, v1y = (ax - bx) * aspect, (ay - by)
    v2x, v2y = (cx - bx) * aspect, (cy - by)
    n1 = math.hypot(v1x, v1y); n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = clamp((v1x*v2x + v1y*v2y) / (n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def best_head_point(lms):
    """코/양 눈/양 귀 중 가시성 높은 것들 중에서 가장 위(작은 y)에 있는 점을 선택."""
    ids = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE.value,
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
    ]
    cand = []
    for i in ids:
        x, y, v = safe_get(lms, i)
        if v >= MIN_VIS:
            cand.append((x, y, v))
    if not cand:
        return None
    cand.sort(key=lambda t: (t[1], -t[2]))  # 더 위(+가시성) 선호
    return cand[0]  # (x,y,v)

def torso_tilt_deg_user(lms, mirrored=False):
    """
    수직 대비 기울기(도). 기본은 엉덩이→어깨 축,
    엉덩이가 안 보이면 어깨중심→머리(코/눈/귀) 축으로 대체.
    x축은 프레임 종횡비(ASPECT)로 보정.
    """
    global ASPECT
    # 어깨/엉덩이 중앙 좌표
    ls = safe_get(lms, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs = safe_get(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    lh = safe_get(lms, mp_pose.PoseLandmark.LEFT_HIP.value)
    rh = safe_get(lms, mp_pose.PoseLandmark.RIGHT_HIP.value)

    sc = center(ls, rs)
    hc = center(lh, rh)

    if ls[2] >= MIN_VIS and rs[2] >= MIN_VIS and lh[2] >= MIN_VIS and rh[2] >= MIN_VIS:
        dx = (sc[0] - hc[0]) * ASPECT
        dy = (hc[1] - sc[1])   # 위가 작음 → 보통 양수
        deg = 0.0 if abs(dy) < 1e-6 else math.degrees(math.atan2(dx, dy))
    else:
        # 대체 축: 어깨중심→머리
        head = best_head_point(lms)
        if head is None or ls[2] < MIN_VIS or rs[2] < MIN_VIS:
            return 0.0  # 신뢰 불가
        hx, hy, _ = head
        dx = (hx - sc[0]) * ASPECT
        dy = (sc[1] - hy)
        deg = 0.0 if abs(dy) < 1e-6 else math.degrees(math.atan2(dx, dy))

    if mirrored:
        deg = -deg
    return deg

def arms_up_and_clasped(lms):
    """
    '손깍지 위로'의 엄격 판정:
    1) 두 손목이 머리선(head_ref)보다 위
    2) 손목이 서로 가깝고 높이 차가 작음(손깍지 근사)
    3) 두 팔꿈치도 머리선보다 위
    4) 양 팔꿈치 각도(어깨-팔꿈치-손목) >= ELBOW_MIN_ANGLE_DEG (팔을 쭉 편 상태)
    """
    global ASPECT

    # 필요한 랜드마크
    nose = safe_get(lms, mp_pose.PoseLandmark.NOSE.value)
    leye = safe_get(lms, mp_pose.PoseLandmark.LEFT_EYE.value)
    reye = safe_get(lms, mp_pose.PoseLandmark.RIGHT_EYE.value)
    ls = safe_get(lms, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs = safe_get(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    le = safe_get(lms, mp_pose.PoseLandmark.LEFT_ELBOW.value)
    re = safe_get(lms, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    lw = safe_get(lms, mp_pose.PoseLandmark.LEFT_WRIST.value)
    rw = safe_get(lms, mp_pose.PoseLandmark.RIGHT_WRIST.value)

    # 가시성 체크(팔 관련은 필수)
    need = [ls, rs, le, re, lw, rw]
    if any(v < MIN_VIS for (_, _, v) in need):
        return False, {"reason": "low_visibility"}

    # 머리 기준선(코/양 눈 중 보이는 것의 최소 y)
    head_y_candidates = [p[1] for p in [nose, leye, reye] if p[2] >= MIN_VIS]
    head_y = min(head_y_candidates) if head_y_candidates else nose[1]
    head_ref = head_y - HEAD_REF_MARGIN

    # 1) 손이 머리선보다 위?
    if not (lw[1] < head_ref and rw[1] < head_ref):
        return False, {"reason": "wrists_not_high_enough"}

    # 2) 손목이 서로 가깝고 높이 차가 작음? (손깍지 근사)
    shoulder_w = euc((ls[0], ls[1]), (rs[0], rs[1]))
    wrist_dist = euc((lw[0], lw[1]), (rw[0], rw[1]))
    if shoulder_w <= 1e-6:
        return False, {"reason": "invalid_shoulder_width"}
    ratio = wrist_dist / shoulder_w
    if not (ratio < WRIST_CLOSE_RATIO_MAX and abs(lw[1] - rw[1]) < WRIST_LEVEL_EPS):
        return False, {"reason": "wrists_not_close_enough", "ratio": ratio}

    # 3) 팔꿈치도 머리선 이상(=위)
    if not (le[1] < head_ref and re[1] < head_ref):
        return False, {"reason": "elbows_not_high_enough"}

    # 4) 팔꿈치가 충분히 펴짐(각도 >= 기준)
    ang_l = joint_angle_deg(ls, le, lw, ASPECT)
    ang_r = joint_angle_deg(rs, re, rw, ASPECT)
    if ang_l is None or ang_r is None:
        return False, {"reason": "angle_nan"}
    if not (ang_l >= ELBOW_MIN_ANGLE_DEG and ang_r >= ELBOW_MIN_ANGLE_DEG):
        return False, {"reason": "elbows_not_straight", "ang_l": ang_l, "ang_r": ang_r}

    # (선택) 기존 스택 순서도 유지하면 더 엄격해짐: 손목 < 팔꿈치 < 어깨
    if not (lw[1] < le[1] < ls[1] and rw[1] < re[1] < rs[1]):
        return False, {"reason": "arm_stack_order_failed"}

    return True, {"shoulder_w": shoulder_w, "ang_l": ang_l, "ang_r": ang_r}

class HoldTimer:
    def __init__(self, need_ms):
        self.need_ms = need_ms
        self.start_ts = None

    def update(self, cond: bool):
        now = time.perf_counter()
        if cond:
            if self.start_ts is None:
                self.start_ts = now
        else:
            self.start_ts = None

    def done(self):
        if self.start_ts is None:
            return False
        return (time.perf_counter() - self.start_ts) * 1000.0 >= self.need_ms

    def progress01(self):
        if self.start_ts is None:
            return 0.0
        p = ((time.perf_counter() - self.start_ts) * 1000.0) / self.need_ms
        return float(max(0.0, min(1.0, p)))

class StretchFSM:
    def __init__(self, mirror_input=False):
        self.state = "CENTER"  # CENTER → TILT_RIGHT → TILT_LEFT → DONE
        self.mirror_input = mirror_input
        self.hold_up = HoldTimer(HOLD_MS_UP)
        self.hold_tilt = HoldTimer(HOLD_MS_TILT)
        # 추가: 기울기 베이스라인(업 자세 유지 중 샘플링 후 고정)
        self.baseline_buf = []
        self.baseline_tilt = 0.0

    def step(self, lms):
        instruction = "Do some stretching up!"
        progress = 0.0
        tilt_deg = 0.0
        debug = {}

        if lms is None:
            self.hold_up.update(False)
            self.hold_tilt.update(False)
            return self.state, "I’m looking for a person...", 0.0, 0.0, debug

        # 절대 기울기(수직 대비, 화면 기준)
        tilt_abs = torso_tilt_deg_user(lms, mirrored=self.mirror_input)
        debug["tilt_abs"] = tilt_abs

        # 팔 위로 + 펴짐 판정
        arms_ok, info = arms_up_and_clasped(lms)
        debug.update(info)

        if self.state == "CENTER":
            # 업 유지 체크 + 베이스라인 수집
            self.hold_up.update(arms_ok)
            instruction = "Do some stretching up!"
            progress = self.hold_up.progress01()

            if arms_ok:
                self.baseline_buf.append(tilt_abs)
                if len(self.baseline_buf) > BASELINE_WINDOW:
                    self.baseline_buf = self.baseline_buf[-BASELINE_WINDOW:]

            if self.hold_up.done():
                self.baseline_tilt = float(np.median(self.baseline_buf)) if self.baseline_buf else tilt_abs
                self.state = "TILT_RIGHT"
                self.hold_tilt = HoldTimer(HOLD_MS_TILT)

            # 표시용 상대각(현재-최근값)
            tilt_deg = tilt_abs - (self.baseline_buf[-1] if self.baseline_buf else tilt_abs)

        elif self.state == "TILT_RIGHT":
            tilt_rel = tilt_abs - self.baseline_tilt
            cond = arms_ok and (tilt_rel <= -TILT_DEG_TARGET)
            self.hold_tilt.update(cond)
            instruction = "Tilt to the right!"
            progress = self.hold_tilt.progress01()
            tilt_deg = tilt_rel
            if self.hold_tilt.done():
                self.state = "TILT_LEFT"
                self.hold_tilt = HoldTimer(HOLD_MS_TILT)

        elif self.state == "TILT_LEFT":
            tilt_rel = tilt_abs - self.baseline_tilt
            cond = arms_ok and (tilt_rel >= TILT_DEG_TARGET)
            self.hold_tilt.update(cond)
            instruction = "Tilt to the left!"
            progress = self.hold_tilt.progress01()
            tilt_deg = tilt_rel
            if self.hold_tilt.done():
                self.state = "DONE"

        elif self.state == "DONE":
            instruction = "Stretching complete :)"
            progress = 1.0
            tilt_deg = tilt_abs - self.baseline_tilt

        # 팔이 내려오면 안내 보강(상태는 유지)
        if self.state in ("TILT_LEFT", "TILT_RIGHT") and not arms_ok:
            instruction += "  (Keep your arms up!)"

        debug["baseline_tilt"] = self.baseline_tilt
        return self.state, instruction, progress, tilt_deg, debug

def draw_overlay(frame, instruction, progress, tilt_deg, state):
    h, w = frame.shape[:2]

    # 상단 안내 텍스트
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, instruction, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # 진행 바
    bar_w, bar_h = int(w * 0.5), 18
    x0 = (w - bar_w) // 2
    y0 = 80
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 2)
    filled = int(bar_w * max(0.0, min(1.0, progress)))
    cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + bar_h), (255, 255, 255), -1)

    # 우측 상단에 기울기 표시(베이스라인 대비 상대각)
    deg_txt = f"tilt: {tilt_deg:+.1f} deg"
    cv2.putText(frame, deg_txt, (w - 250, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2, cv2.LINE_AA)

    # 상태 표시
    cv2.putText(frame, f"[{state}]", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("I can’t open the webcam.")
        return

    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    fsm = StretchFSM(mirror_input=MIRROR_INPUT)
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ===== 추가: 프레임 종횡비 반영(각도 보정용) =====
            global ASPECT
            h, w = frame.shape[:2]
            ASPECT = w / float(h)

            if MIRROR_INPUT:
                frame = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            lms = res.pose_landmarks.landmark if res.pose_landmarks else None

            # 상태 업데이트
            state, instruction, progress, tilt_deg, debug = fsm.step(lms)

            # 랜드마크 그리기(디버그용)
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            draw_overlay(frame, instruction, progress, tilt_deg, state)

            # FPS 표시
            now = time.perf_counter()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"{fps:4.1f} FPS", (frame.shape[1]-120, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow("Stretch Demo (MediaPipe Pose)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()
