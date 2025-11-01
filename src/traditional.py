import cv2
import numpy as np
import mediapipe as mp


# cap = cv2.VideoCapture("test_AMELINE_30cm_eye_HD_rgb_1761970095.mp4")
# cap = cv2.VideoCapture("test_JESTIN_30cm_eye_HD_rgb_1761970129.mp4")
# cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")
cap = cv2.VideoCapture("ameline_with_light_rgb.mp4")

#################################### FUNCTIONS #################################

def detect_pupil_center_2(img):
    """
    Simple pupil detection using threshold + contour.
    Returns center (cx, cy) relative to the eye crop, or None.
    """
    inverted = cv2.bitwise_not(img)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(gray, kernel, iterations=1)
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
    """
    Extend vector from eye center through pupil center.
    Maps camera frame pixels -> board image pixels, centered on board.
    """
    frame_w, frame_h = frame_size
    board_w, board_h = board_size

    dx = pupil_center_abs[0] - eye_center[0]
    dy = pupil_center_abs[1] - eye_center[1]


    # Scale vector from camera frame to board size
    gx = int(board_w/2 + dx * (board_w / frame_w) * scale)
    gy = int(board_h/2 + dy * (board_h / frame_h) * scale)

    # Clamp to board boundaries
    # gx = np.clip(gx, 0, board_w-1)
    # gy = np.clip(gy, 0, board_h-1)

    return gx, gy

################################### TRADITIONAL SETUP ################################

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

board_img = cv2.imread("board.png")
board_h, board_w = board_img.shape[:2]


################################### MEDIAPIPE SETUP ################################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))

    # Keep lower 2 eyes if >2 detected
    if len(eyes) > 2:
        eyes = sorted(eyes, key=lambda e: e[1])[1:]

    gaze_visual = board_img.copy()

    for (x, y, w, h) in eyes:
        eye_crop = frame[y:y+h, x:x+w].copy()
        pupil_center_rel = detect_pupil_center_2(eye_crop)

        if pupil_center_rel is None:
            continue

        px, py = pupil_center_rel
        pupil_center_abs = (x + px, y + py)

        # Eye center in camera frame
        eye_center = (x + w//2, y + h//2)

        # Draw rectangles and points
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, pupil_center_abs, 4, (0, 0, 255), -1)
        cv2.circle(frame, eye_center, 4, (255, 255, 0), -1)  # eye center

        # Compute gaze projection on board
        gx, gy = extend_gaze_vector(eye_center, pupil_center_abs,
                                    frame_size=(frame_w, frame_h),
                                    board_size=(board_w, board_h),
                                    scale=35.0)

        # Draw gaze on board
        cv2.circle(gaze_visual, (gx, gy), 8, (0, 0, 255), -1)
        cv2.putText(gaze_visual, f"({gx},{gy})", (gx+10, gy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Stack camera and board views
    vis = np.hstack((frame, cv2.resize(gaze_visual, (frame_w, frame_h))))
    cv2.imshow("Left: Detection | Right: Gaze on Board", vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
