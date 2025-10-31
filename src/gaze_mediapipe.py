import cv2
import numpy as np
import mediapipe as mp

################################### SETUP ################################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")
board_img = cv2.imread("board.png")
board_h, board_w = board_img.shape[:2]

def extend_gaze_vector(eye_center, pupil_center_abs, frame_size, board_size, scale=5.0):
    frame_w, frame_h = frame_size
    board_w, board_h = board_size

    dx = pupil_center_abs[0] - eye_center[0]
    dy = pupil_center_abs[1] - eye_center[1]

    gx = int(board_w/2 + dx * (board_w / frame_w) * scale)
    gy = int(board_h/2 + dy * (board_h / frame_h) * scale)

    gx = np.clip(gx, 0, board_w-1)
    gy = np.clip(gy, 0, board_h-1)
    return gx, gy

################################### MAIN LOOP ################################

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    frame_h, frame_w = frame.shape[:2]
    gaze_visual = board_img.copy()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # MediaPipe eye indices (refined landmarks)
        left_eye_indices = [33, 133]  # approximate left eye corners
        right_eye_indices = [362, 263]  # approximate right eye corners
        left_pupil_index = 468  # iris center
        right_pupil_index = 473

        eyes = []

        # Left eye
        leye_pts = [face_landmarks.landmark[i] for i in left_eye_indices]
        lpupil = face_landmarks.landmark[left_pupil_index]
        left_eye_center = ((leye_pts[0].x + leye_pts[1].x)/2 * frame_w,
                           (leye_pts[0].y + leye_pts[1].y)/2 * frame_h)
        left_pupil_abs = (lpupil.x * frame_w, lpupil.y * frame_h)
        eyes.append((left_eye_center, left_pupil_abs))

        # Right eye
        reye_pts = [face_landmarks.landmark[i] for i in right_eye_indices]
        rpupil = face_landmarks.landmark[right_pupil_index]
        right_eye_center = ((reye_pts[0].x + reye_pts[1].x)/2 * frame_w,
                            (reye_pts[0].y + reye_pts[1].y)/2 * frame_h)
        right_pupil_abs = (rpupil.x * frame_w, rpupil.y * frame_h)
        eyes.append((right_eye_center, right_pupil_abs))

        for eye_center, pupil_center_abs in eyes:
            # Compute gaze projection
            gx, gy = extend_gaze_vector(
                eye_center, pupil_center_abs,
                frame_size=(frame_w, frame_h),
                board_size=(board_w, board_h),
                scale=30.0
            )

            # Draw on frame
            cv2.circle(frame, (int(eye_center[0]), int(eye_center[1])), 4, (255,255,0), -1)
            cv2.circle(frame, (int(pupil_center_abs[0]), int(pupil_center_abs[1])), 4, (0,0,255), -1)

            # Draw gaze on board
            cv2.circle(gaze_visual, (gx, gy), 8, (0,0,255), -1)
            cv2.putText(gaze_visual, f"({gx},{gy})", (gx+10, gy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    vis = np.hstack((frame, cv2.resize(gaze_visual, (frame_w, frame_h))))
    cv2.imshow("Left: Detection | Right: Gaze on Board", vis)

    if cv2.waitKey(15) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
