import cv2
import numpy as np
import mediapipe as mp

# 1) 설정값
UP_THRESHOLD = 170           # 상단: 무릎 각도 ≥ 이 값
DOWN_THRESHOLD = 100         # 하단: 무릎 각도 ≤ 이 값
ANKLE_MOVEMENT_LIMIT = 5     # 발 y좌표 프레임 간 최대 이동 허용치(px)

# 2) 각도 계산 함수
def calculate_angle(a, b, c):
    """
    세 점 a, b, c (각 b를 꼭짓점으로) 사이의 각도를 계산
    a, b, c: (x, y) 튜플
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

# 3) MediaPipe Pose 초기화
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 4) 비디오 캡처 준비
cap = cv2.VideoCapture(0)  # 카메라 번호 조정
count = 0
stage = None               # None / "up" / "down"
prev_left_ankle_y  = None
prev_right_ankle_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전(거울 모드) — 원치 않으면 제거
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # BGR → RGB 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # 왼쪽 관절
        hip_l   = (lm[mp_pose.PoseLandmark.LEFT_HIP].x  * w,
                   lm[mp_pose.PoseLandmark.LEFT_HIP].y  * h)
        knee_l  = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w,
                   lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h)
        ankle_l = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w,
                   lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)

        # 오른쪽 관절
        hip_r   = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x  * w,
                   lm[mp_pose.PoseLandmark.RIGHT_HIP].y  * h)
        knee_r  = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                   lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)
        ankle_r = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
                   lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h)

        # 무릎 각도 계산
        angle_l = calculate_angle(hip_l, knee_l, ankle_l)
        angle_r = calculate_angle(hip_r, knee_r, ankle_r)

        # 발 움직임 안정성 체크
        stable = True
        if prev_left_ankle_y is not None:
            if abs(ankle_l[1] - prev_left_ankle_y) > ANKLE_MOVEMENT_LIMIT:
                stable = False
        if prev_right_ankle_y is not None:
            if abs(ankle_r[1] - prev_right_ankle_y) > ANKLE_MOVEMENT_LIMIT:
                stable = False

        prev_left_ankle_y  = ankle_l[1]
        prev_right_ankle_y = ankle_r[1]

        # 상태 머신: 양쪽 무릎이 모두 조건 만족하고 발도 안정적일 때만 count/상태 변경
        if stable and angle_l > UP_THRESHOLD and angle_r > UP_THRESHOLD:
            if stage == "down":
                count += 1
            stage = "up"

        if stable and angle_l < DOWN_THRESHOLD and angle_r < DOWN_THRESHOLD:
            if stage == "up" or stage is None:
                stage = "down"

        # 화면에 정보 표시
        cv2.putText(frame, f'L:{int(angle_l)} R:{int(angle_r)}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f'Stage: {stage}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f'Count: {count}', (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    # 결과 화면 출력 및 종료 조건
    cv2.imshow('Squat Counter', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC키로 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
