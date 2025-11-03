import cv2
import numpy as np
import mediapipe as mp

#################################### SETUP ####################################

cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")

board_img = cv2.imread("board.png")
board_h, board_w = board_img.shape[:2]

#################################### FUNCTIONS #################################

def detect_pupil_center_2(img):
    """Detects pupil by thresholding and morphology."""
    inverted = cv2.bitwise_not(img)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    _, thresh = cv2.threshold(eroded, 210, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def extend_gaze_vector(eye_center, pupil_center_abs, frame_size, board_size, scale=5.0):
    frame_w, frame_h = frame_size
    board_w, board_h = board_size

    dx = pupil_center_abs[0] - eye_center[0]
    dy = pupil_center_abs[1] - eye_center[1]

    gx = int(board_w/2 - dx * (board_w / frame_w) * scale)
    gy = int(board_h/2 + dy * (board_h / frame_h) * scale)

    return gx, gy

#################################### MEDIAPIPE #################################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#################################### MAIN LOOP #################################

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    gaze_visual = board_img.copy()

    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Iris landmark indices
        left_iris_idx = [468, 469, 470, 471]
        right_iris_idx = [473, 474, 475, 476]

        # Approximate eyelid corners for bounding box
        left_eye_bounds = [33, 133]   # left/right corners
        right_eye_bounds = [362, 263]

        eyes = []

        for iris_idx, bound_idx in [(left_iris_idx, left_eye_bounds),
                                    (right_iris_idx, right_eye_bounds)]:
            iris_pts = np.array([[face_landmarks.landmark[i].x * frame_w,
                                  face_landmarks.landmark[i].y * frame_h] for i in iris_idx])
            eye_corners = np.array([[face_landmarks.landmark[i].x * frame_w,
                                     face_landmarks.landmark[i].y * frame_h] for i in bound_idx])

            # Compute rough eye bounding box
            x_min = int(min(eye_corners[:,0]) - 5)
            x_max = int(max(eye_corners[:,0]) + 5)
            y_min = int(min(iris_pts[:,1].min(), eye_corners[:,1].min()) - 10)
            y_max = int(max(iris_pts[:,1].max(), eye_corners[:,1].max()) + 10)

            # Clamp to frame
            x_min = max(0, x_min); y_min = max(0, y_min)
            x_max = min(frame_w-1, x_max); y_max = min(frame_h-1, y_max)

            eyes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        # Process each detected eye
        for (x, y, w, h) in eyes:
            eye_crop = frame[y:y+h, x:x+w].copy()
            pupil_center_rel = detect_pupil_center_2(eye_crop)
            if pupil_center_rel is None:
                continue

            px, py = pupil_center_rel
            pupil_center_abs = (x + px, y + py)
            eye_center = (x + w//2, y + h//2)

            # Draw detections
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, pupil_center_abs, 4, (0, 0, 255), -1)
            cv2.circle(frame, eye_center, 4, (255, 255, 0), -1)

            # Compute gaze projection
            gx, gy = extend_gaze_vector(
                eye_center, pupil_center_abs,
                frame_size=(frame_w, frame_h),
                board_size=(board_w, board_h),
                scale=32.0
            )

            # Draw on board
            cv2.circle(gaze_visual, (gx, gy), 8, (0, 0, 255), -1)
            cv2.putText(gaze_visual, f"({gx},{gy})", (gx+10, gy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Combine video and board
    vis = np.hstack((frame, cv2.resize(gaze_visual, (frame_w, frame_h))))
    cv2.imshow("Gaze on Board", vis)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
