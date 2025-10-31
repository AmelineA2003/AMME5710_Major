import cv2
import numpy as np

################################### FUNCTIONS ################################

def detect_pupil_center(img):
    inverted = cv2.bitwise_not(img)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(gray, kernel, iterations=1)
    _, thresh = cv2.threshold(eroded, 200, 255, cv2.THRESH_BINARY)
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


# Get rid of intrinsics - perspective projection 

def project_gaze(K, pupil_center, depth):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u, v = pupil_center
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    ray = ray / np.linalg.norm(ray)
    scale = depth / ray[2]
    point_3d = ray * scale
    uv_proj = K @ (point_3d / point_3d[2])
    return (uv_proj[0], uv_proj[1])



################################### MAIN LOOP ################################
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")

K = np.array([[613.94, 0, 320],
              [0, 613.616, 240],
              [0,  0,  1]])   

depth = 0.02  # meters to A4 board

# Load the board image
board_img = cv2.imread("board.png")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))

    # TODO: BETTER OUTLIER REMOVAL METHOD
    if len(eyes) > 2:
        # Sort by y (top to bottom)
        eyes = sorted(eyes, key=lambda e: e[1])
        # Remove the highest one
        eyes = eyes[1:]

    gaze_visual = board_img.copy()

    for (x, y, w, h) in eyes:

        eye_crop = frame[y:y+h, x:x+w].copy()
        pupil_center_rel = detect_pupil_center(eye_crop)

        if pupil_center_rel is not None:
            px, py = pupil_center_rel
            pupil_center_abs = (x + px, y + py)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, pupil_center_abs, 4, (0, 0, 255), -1)

            gaze_point = project_gaze(K, pupil_center_abs, depth)

            board_h, board_w = board_img.shape[:2]
            frame_h, frame_w = frame.shape[:2]

            # Scale gaze point from camera frame to board image
            gx = int(np.clip(gaze_point[0] * board_w / frame_w, 0, board_w-1))
            gy = int(np.clip(gaze_point[1] * board_h / frame_h, 0, board_h-1))


            cv2.circle(frame, (int(gaze_point[0]), int(gaze_point[1])), 5, (255, 0, 0), -1)
            cv2.circle(gaze_visual, (gx, gy), 8, (0, 0, 255), -1)
            cv2.putText(gaze_visual, f"({gx}, {gy})", (gx+10, gy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Stack both views 
    vis = np.hstack((frame, cv2.resize(gaze_visual, (frame.shape[1], frame.shape[0]))))
    cv2.imshow("Left: Detection | Right: Gaze on Board", vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
