""" 
This file stores the functions for our reprojection testing.
"""

import cv2
import numpy as np
import mediapipe as mp

from traditional_gaze_functions import detect_eye_center, detect_pupil_center_2

############################# VIDEO SETUP #############################

cap = cv2.VideoCapture("videos/ameline_test_d-_rgb_1761129562.mp4")
board_img = cv2.imread("board.png")
board_h, board_w = board_img.shape[:2]

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video: stack camera frame + gaze board side by side for each method
out = cv2.VideoWriter(
    "outputs/gaze_comparison_with_board.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_w*2, frame_h*2)  # width: two columns, height: two rows
)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

############################# FUNCTIONS #############################
""" 
This function defines the geometry for gaze reprojection onto the advertisement board. 
It multiplies the dx and dy distances by the depth of the subject (scale)
"""
def extend_gaze_vector(eye_center, pupil_center_abs, frame_size, board_size, scale=32.0):
    frame_w, frame_h = frame_size
    board_w, board_h = board_size

    # Find 2D gaze vector 
    dx = pupil_center_abs[0] - eye_center[0]
    dy = pupil_center_abs[1] - eye_center[1]
    gx = int(board_w/2 - dx * (board_w / frame_w) * scale)
    gy = int(board_h/2 + dy * (board_h / frame_h) * scale)

    # Clamp to board
    gx = np.clip(gx, 0, board_w-1)
    gy = np.clip(gy, 0, board_h-1)
    return gx, gy


############################# MEDIAPIPE SETUP #############################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

left_iris_idx = [468, 469, 470, 471]
right_iris_idx = [473, 474, 475, 476]
left_eye_bounds = [33, 133]
right_eye_bounds = [362, 263]


# Smoothing alpha to smooth gaze reprojection
alpha = 0.4

# Previous gaze positions 
prev_gaze_mp = None
prev_gaze_haar = None

############################# MAIN LOOP #############################
""" 
This loop reads in an input video. 

For each frame, it calculates the eye center and pupil center using our methods, comparing it to MediaPipe. 
The gaze from each method is then reprojected onto a board. 
"""
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Copies for each method
    frame_mp = frame.copy()
    frame_haar = frame.copy()
    gaze_board_mp = board_img.copy()
    gaze_board_haar = board_img.copy()

    ####### MediaPipe #######
    frame_rgb = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        for iris_idx, bound_idx in [(left_iris_idx, left_eye_bounds), (right_iris_idx, right_eye_bounds)]:
            iris_pts = np.array([[face_landmarks.landmark[i].x * frame_w,
                                  face_landmarks.landmark[i].y * frame_h] for i in iris_idx])
            eye_corners = np.array([[face_landmarks.landmark[i].x * frame_w,
                                     face_landmarks.landmark[i].y * frame_h] for i in bound_idx])
            x_min = int(min(eye_corners[:,0]) - 5)
            x_max = int(max(eye_corners[:,0]) + 5)
            y_min = int(min(iris_pts[:,1].min(), eye_corners[:,1].min()) - 10)
            y_max = int(max(iris_pts[:,1].max(), eye_corners[:,1].max()) + 10)
            x_min = max(0, x_min); y_min = max(0, y_min)
            x_max = min(frame_w-1, x_max); y_max = min(frame_h-1, y_max)

            eye_crop = frame_mp[y_min:y_max, x_min:x_max].copy()
            pupil_center_rel = detect_pupil_center_2(eye_crop)
            if pupil_center_rel is None:
                continue
            px, py = pupil_center_rel
            pupil_center_abs = (x_min + px, y_min + py)
            eye_center = (x_min + (x_max - x_min)//2, y_min + (y_max - y_min)//2)

            # Draw on frame
            cv2.rectangle(frame_mp, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            cv2.circle(frame_mp, pupil_center_abs, 5, (0,0,255), -1)
            cv2.circle(frame_mp, eye_center, 4, (255,255,0), -1)

            # Draw gaze on board
            gx, gy = extend_gaze_vector(eye_center, pupil_center_abs,
                            frame_size=(frame_w, frame_h),
                            board_size=(board_w, board_h))

            # Apply EMA smoothing
            if prev_gaze_mp is None:
                smoothed_gaze_mp = (gx, gy)
            else:
                smoothed_gaze_mp = (int(alpha*gx + (1-alpha)*prev_gaze_mp[0]),
                                    int(alpha*gy + (1-alpha)*prev_gaze_mp[1]))
            prev_gaze_mp = smoothed_gaze_mp

            cv2.circle(gaze_board_mp, smoothed_gaze_mp, 18, (0,0,255), -1)

    ####### Haar Cascade #######
    gray = cv2.cvtColor(frame_haar, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20,20))
    if len(eyes) > 2:
        eyes = sorted(eyes, key=lambda e: e[1])[1:]

    for (x, y, w, h) in eyes:
        eye_crop = frame_haar[y:y+h, x:x+w].copy()
        pupil_center_rel = detect_pupil_center_2(eye_crop)
        if pupil_center_rel is None:
            continue
        px, py = pupil_center_rel
        pupil_center_abs = (x + px, y + py)


        eye_center_rel = detect_eye_center(eye_crop)
        if eye_center_rel is None:
            continue
        eye_center = (x + eye_center_rel[0], y + eye_center_rel[1])

        # Draw on frame
        cv2.rectangle(frame_haar, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame_haar, pupil_center_abs, 4, (0,0,255), -1)
        cv2.circle(frame_haar, eye_center, 4, (255,255,0), -1)

        # Draw gaze on board
        gx, gy = extend_gaze_vector(eye_center, pupil_center_abs,
                            frame_size=(frame_w, frame_h),
                            board_size=(board_w, board_h))

        # Apply smoothing
        if prev_gaze_haar is None:
            smoothed_gaze_haar = (gx, gy)
        else:
            smoothed_gaze_haar = (int(alpha*gx + (1-alpha)*prev_gaze_haar[0]),
                                int(alpha*gy + (1-alpha)*prev_gaze_haar[1]))
        prev_gaze_haar = smoothed_gaze_haar

        cv2.circle(gaze_board_haar, smoothed_gaze_haar, 18, (0,0,255), -1)

    # Combine all videos into one
    mp_combined = np.vstack((frame_mp, cv2.resize(gaze_board_mp, (frame_w, frame_h))))
    haar_combined = np.vstack((frame_haar, cv2.resize(gaze_board_haar, (frame_w, frame_h))))
    combined_vis = np.hstack((mp_combined, haar_combined))

    out.write(combined_vis)
    cv2.imshow("MediaPipe | Haar", combined_vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
