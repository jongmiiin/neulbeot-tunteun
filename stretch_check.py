import cv2
import time
import math
import numpy as np
import mediapipe as mp

# ===== 설정값 =====
MIRROR_INPUT = True          # 웹캠 프리뷰를 좌우 반전(거울화)해서 보여줄지
HOLD_MS_UP = 1000            # '손깍지 위로' 유지 시간(ms)
MIN_VIS = 0.5                # 랜드마크 가시성 threshold

# 손깍지(가까움) 판정 난이도(0.0~1.0). 값이 작을수록 '더 가깝게' 요구
WRIST_CLOSE_RATIO_MAX = 0.65   # (양손목 거리 / 어깨폭) 최대 허용 비율
WRIST_LEVEL_EPS = 0.12         # 양손목 높이 차 허용(정상화 좌표)

# 팔 펴짐/머리 기준선
ELBOW_MIN_ANGLE_DEG = 160    # 팔꿈치 펴짐 최소 각도(권장: 160~170)
HEAD_REF_MARGIN = 0.01       # 머리 기준선 여유

# ===== 타깃 원(좌/우) 설정 =====
TARGET_OFFSET_X_SW = 1.20    # 어깨폭 대비 좌우 오프셋 (기본 1.2 * 어깨폭)
TARGET_RADIUS_PX   = 50      # 원 반지름(px)
TARGET_HIT_HYSTERESIS_MS = 120  # 원에 들어온 뒤 이 시간 이상 유지 시 히트 처리(오검출 방지)

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

def clamp01(x):
    return max(0.0, min(1.0, x))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def joint_angle_deg(a, b, c, aspect):
    """ b(관절)에서의 내각(어깨-팔꿈치-손목). x축은 aspect(=width/height)로 보정 """
    ax, ay = a[0], a[1]; bx, by = b[0], b[1]; cx, cy = c[0], c[1]
    v1x, v1y = (ax - bx) * aspect, (ay - by)
    v2x, v2y = (cx - bx) * aspect, (cy - by)
    n1 = math.hypot(v1x, v1y); n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = clamp((v1x*v2x + v1y*v2y) / (n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def arms_up_and_clasped(lms):
    """
    '손깍지 위로'의 엄격 판정:
    1) 두 손목이 머리선(head_ref)보다 위
    2) 손목이 서로 가깝고 높이 차가 작음(손깍지 근사)
    3) 두 팔꿈치도 머리선보다 위
    4) 양 팔꿈치 각도(어깨-팔꿈치-손목) >= ELBOW_MIN_ANGLE_DEG (팔을 쭉 편 상태)
    5) (권장) 손목 < 팔꿈치 < 어깨 스택 순서
    """
    global ASPECT

    nose = safe_get(lms, mp_pose.PoseLandmark.NOSE.value)
    leye = safe_get(lms, mp_pose.PoseLandmark.LEFT_EYE.value)
    reye = safe_get(lms, mp_pose.PoseLandmark.RIGHT_EYE.value)
    ls = safe_get(lms, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs = safe_get(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    le = safe_get(lms, mp_pose.PoseLandmark.LEFT_ELBOW.value)
    re = safe_get(lms, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    lw = safe_get(lms, mp_pose.PoseLandmark.LEFT_WRIST.value)
    rw = safe_get(lms, mp_pose.PoseLandmark.RIGHT_WRIST.value)

    need = [ls, rs, le, re, lw, rw]
    if any(v < MIN_VIS for (_, _, v) in need):
        return False, {"reason": "low_visibility"}

    # 머리선
    head_y_candidates = [p[1] for p in [nose, leye, reye] if p[2] >= MIN_VIS]
    head_y = min(head_y_candidates) if head_y_candidates else nose[1]
    head_ref = head_y - HEAD_REF_MARGIN

    # 손목 머리선 위
    if not (lw[1] < head_ref and rw[1] < head_ref):
        return False, {"reason": "wrists_not_high_enough"}

    # 손깍지 근사: 손목 근접 + 높이 차 작음
    shoulder_w = euc((ls[0], ls[1]), (rs[0], rs[1]))
    if shoulder_w <= 1e-6:
        return False, {"reason": "invalid_shoulder_width"}

    wrist_dist = euc((lw[0], lw[1]), (rw[0], rw[1]))
    ratio = wrist_dist / shoulder_w
    if not (ratio < WRIST_CLOSE_RATIO_MAX and abs(lw[1] - rw[1]) < WRIST_LEVEL_EPS):
        return False, {"reason": "wrists_not_close_enough", "ratio": ratio}

    # 팔꿈치도 머리선 위
    if not (le[1] < head_ref and re[1] < head_ref):
        return False, {"reason": "elbows_not_high_enough"}

    # 팔꿈치 펴짐
    ang_l = joint_angle_deg(ls, le, lw, ASPECT)
    ang_r = joint_angle_deg(rs, re, rw, ASPECT)
    if ang_l is None or ang_r is None:
        return False, {"reason": "angle_nan"}
    if not (ang_l >= ELBOW_MIN_ANGLE_DEG and ang_r >= ELBOW_MIN_ANGLE_DEG):
        return False, {"reason": "elbows_not_straight", "ang_l": ang_l, "ang_r": ang_r}

    # 스택 순서
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
    """
    상태: CENTER(업 자세 인식/베이스라인 확보) -> TARGETS(좌/우 원 클릭) -> DONE
    """
    def __init__(self, mirror_input=False):
        self.state = "CENTER"
        self.mirror_input = mirror_input
        self.hold_up = HoldTimer(HOLD_MS_UP)

        # 베이스라인(업 순간)
        self.baseline = None  # dict: { 'sc':(x,y), 'sw':float, 'hands_center':(x,y) }

        # 타깃 (정규화 좌표; 0~1)
        self.target_L = None
        self.target_R = None
        self.hit_L = False
        self.hit_R = False

        # 히트 방지용 홀드 타이머
        self.hit_hold_L = HoldTimer(TARGET_HIT_HYSTERESIS_MS)
        self.hit_hold_R = HoldTimer(TARGET_HIT_HYSTERESIS_MS)

    def _make_targets(self, lms):
        """업 자세 완료 시 타깃 좌표 생성 (y는 손중심 y 그대로, x는 좌우로 어깨폭*계수)"""
        ls = safe_get(lms, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        rs = safe_get(lms, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        lw = safe_get(lms, mp_pose.PoseLandmark.LEFT_WRIST.value)
        rw = safe_get(lms, mp_pose.PoseLandmark.RIGHT_WRIST.value)

        sc = center(ls, rs)                 # 어깨 중심
        sw = euc((ls[0], ls[1]), (rs[0], rs[1]))  # 어깨 폭
        hc = center(lw, rw)                 # 두 손목의 중심(손 중심으로 사용)

        # 좌/우 타깃 (정규화 좌표)
        offset = TARGET_OFFSET_X_SW * sw
        txL = clamp01(hc[0] - offset)
        txR = clamp01(hc[0] + offset)
        ty  = clamp01(hc[1])  # 요청사항대로 y는 동일 유지

        self.baseline = {"sc": sc, "sw": sw, "hands_center": hc}
        self.target_L = (txL, ty)
        self.target_R = (txR, ty)
        self.hit_L = self.hit_R = False
        # 히스테리시스 타이머 초기화
        self.hit_hold_L = HoldTimer(TARGET_HIT_HYSTERESIS_MS)
        self.hit_hold_R = HoldTimer(TARGET_HIT_HYSTERESIS_MS)

    def step(self, lms, frame_size):
        instruction = "Interlace your fingers and stretch upward!"
        progress = 0.0
        debug = {}
        h, w = frame_size

        if lms is None:
            self.hold_up.update(False)
            return self.state, "I’m looking for a person...", 0.0, debug

        if self.state == "CENTER":
            arms_ok, info = arms_up_and_clasped(lms)
            debug.update(info)
            self.hold_up.update(arms_ok)
            instruction = "Interlace your fingers and stretch upward!"
            progress = self.hold_up.progress01()

            if self.hold_up.done():
                self._make_targets(lms)
                self.state = "TARGETS"
                instruction = "Reach to the circles!"

        elif self.state == "TARGETS":
            instruction = "Reach to the circles!"
            progress = 0.0

            # 현재 손 중심(두 손목 중점)
            lw = safe_get(lms, mp_pose.PoseLandmark.LEFT_WRIST.value)
            rw = safe_get(lms, mp_pose.PoseLandmark.RIGHT_WRIST.value)
            if lw[2] >= MIN_VIS and rw[2] >= MIN_VIS:
                hc = center(lw, rw)  # 정규화 좌표

                # 정규화 → 픽셀
                def norm2px(p):
                    x = int(p[0] * w)
                    y = int(p[1] * h)
                    return x, y

                if (self.target_L is not None) and (not self.hit_L):
                    cx, cy = norm2px(self.target_L)
                    hx, hy = norm2px(hc)
                    d = math.hypot(hx - cx, hy - cy)
                    self.hit_hold_L.update(d <= TARGET_RADIUS_PX)
                    if self.hit_hold_L.done():
                        self.hit_L = True

                if (self.target_R is not None) and (not self.hit_R):
                    cx, cy = norm2px(self.target_R)
                    hx, hy = norm2px(hc)
                    d = math.hypot(hx - cx, hy - cy)
                    self.hit_hold_R.update(d <= TARGET_RADIUS_PX)
                    if self.hit_hold_R.done():
                        self.hit_R = True

                if self.hit_L and self.hit_R:
                    self.state = "DONE"
                    instruction = "Stretching complete! Great job :)"
                    progress = 1.0

        elif self.state == "DONE":
            instruction = "Stretching complete! Great job :)"
            progress = 1.0

        debug["state"] = self.state
        debug["targets"] = {"L": self.target_L, "R": self.target_R, "hit_L": self.hit_L, "hit_R": self.hit_R}
        return self.state, instruction, progress, debug

def draw_overlay(frame, instruction, progress, state, targets_info):
    h, w = frame.shape[:2]

    # 상단 안내 텍스트
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, instruction, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # 진행 바 (CENTER/DONE 중심으로 표시)
    if state in ("CENTER", "DONE"):
        bar_w, bar_h = int(w * 0.5), 18
        x0 = (w - bar_w) // 2
        y0 = 80
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 2)
        filled = int(bar_w * max(0.0, min(1.0, progress)))
        cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + bar_h), (255, 255, 255), -1)

    # 타깃 원 표시
    if state in ("TARGETS", "DONE") and targets_info:
        L = targets_info.get("L")
        R = targets_info.get("R")
        hit_L = targets_info.get("hit_L", False)
        hit_R = targets_info.get("hit_R", False)

        def norm2px(p):
            x = int(p[0] * w)
            y = int(p[1] * h)
            return x, y

        if L and not hit_L:
            cx, cy = norm2px(L)
            cv2.circle(frame, (cx, cy), TARGET_RADIUS_PX, (0, 255, 255), 3)  # 노란색 테두리
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

        if R and not hit_R:
            cx, cy = norm2px(R)
            cv2.circle(frame, (cx, cy), TARGET_RADIUS_PX, (0, 255, 255), 3)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

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

            # 종횡비 갱신
            global ASPECT
            h, w = frame.shape[:2]
            ASPECT = w / float(h)

            if MIRROR_INPUT:
                frame = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            lms = res.pose_landmarks.landmark if res.pose_landmarks else None

            # 상태 업데이트
            state, instruction, progress, debug = fsm.step(lms, (h, w))

            # 랜드마크(디버그용)
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            # 오버레이(타깃 포함)
            draw_overlay(
                frame,
                instruction,
                progress,
                state,
                targets_info=debug.get("targets")
            )

            # FPS 표시
            now = time.perf_counter()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"{fps:4.1f} FPS", (frame.shape[1]-120, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow("Stretch Target Demo (MediaPipe Pose)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()
