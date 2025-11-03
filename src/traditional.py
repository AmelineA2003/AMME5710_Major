import cv2
import numpy as np
import mediapipe as mp

##################################### VIDEOS ##################################
# cap = cv2.VideoCapture("test_AMELINE_30cm_eye_HD_rgb_1761970095.mp4")
# cap = cv2.VideoCapture("test_JESTIN_30cm_eye_HD_rgb_1761970129.mp4")
cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")
# cap = cv2.VideoCapture("ameline_with_light_rgb.mp4")

################################# KALMAN FILTER ###############################

def create_kalman():
    kf = cv2.KalmanFilter(4, 2)  # [x, y, dx, dy] -> [x, y]
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    return kf

# We'll create two separate filters for two eyes
kf_left = create_kalman()
kf_right = create_kalman()

#################################### FUNCTIONS #################################

def detect_pupil_center_2(img):
    inverted = cv2.bitwise_not(img)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3,3), np.uint8)
    # eroded = cv2.erode(gray, kernel, iterations=1)
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

def detect_eye_center(img): 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = gray.copy()
    mean = img_copy.mean()
    stddev = img_copy.std()

    mask = img_copy < (mean - 1 * stddev)
    img_copy[mask] = 0

    ys, xs = np.where(img_copy == 0)

    if len(xs) > 0:
        leftmost_zero = xs.min()
        rightmost_zero = xs.max()

        y_left = ys[xs == leftmost_zero]
        y_right = ys[xs == rightmost_zero]

        top_y = min(y_left.min(), y_right.min())
        bottom_y = max(y_left.max(), y_right.max())
        vertical_mid = (top_y + bottom_y) // 2

        eye_center = ((leftmost_zero + rightmost_zero) // 2, bottom_y)
        return eye_center
    else: 
        return None
    

def extend_gaze_vector(eye_center, pupil_center_abs, frame_size, board_size, scale=5.0):
    frame_w, frame_h = frame_size
    board_w, board_h = board_size

    dx = pupil_center_abs[0] - eye_center[0]
    dy = pupil_center_abs[1] - eye_center[1]


    # Scale vector from camera frame to board size
    gx = int(board_w/2 - dx * (board_w / frame_w) * scale)
    gy = int(board_h/2 + dy * (board_h / frame_h) * scale)

    # Clamp to board boundaries
    # gx = np.clip(gx, 0, board_w-1)
    # gy = np.clip(gy, 0, board_h-1)

    return gx, gy

################################### TRADITIONAL SETUP ################################

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

board_img = cv2.imread("board.png")
board_h, board_w = board_img.shape[:2]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))

    # Keep lower 2 eyes 
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
        # eye_center = detect_eye_center(eye_crop)

        # eye_center_rel = detect_eye_center(eye_crop)
        # if eye_center_rel is None:
        #     continue

        # # Convert to frame coordinates
        # eye_center = (x + eye_center_rel[0], y + eye_center_rel[1])

        # Draw rectangles and points
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, pupil_center_abs, 4, (0, 0, 255), -1)
        cv2.circle(frame, eye_center, 4, (255, 255, 0), -1)  # eye center

        # Compute gaze projection on board
        gx, gy = extend_gaze_vector(eye_center, pupil_center_abs,
                                    frame_size=(frame_w, frame_h),
                                    board_size=(board_w, board_h),
                                    scale=32.0)

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
