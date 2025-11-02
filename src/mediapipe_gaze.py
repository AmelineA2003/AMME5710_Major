import cv2
import numpy as np
import mediapipe as mp

#################################### FUNCTIONS #################################

def project_pupil_to_board(eye_bbox, pupil_pos, board_size):
    """
    Map the pupil's position relative to the eye bounding box to board coordinates.
    eye_bbox: (x, y, w, h)
    pupil_pos: (px, py) absolute pixel position in frame
    board_size: (board_w, board_h)
    """
    x, y, w, h = eye_bbox
    px, py = pupil_pos
    board_w, board_h = board_size

    # Normalize pupil position inside eye bbox
    nx = (px - x) / w
    ny = (py - y) / h

    # Map to board coordinates
    gx = int(nx * board_w)
    gy = int(ny * board_h)

    # Clamp
    gx = np.clip(gx, 0, board_w-1)
    gy = np.clip(gy, 0, board_h-1)
    return gx, gy

################################### SETUP ################################

cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")
board_img = cv2.imread("board.png")
if board_img is None:
    raise FileNotFoundError("❌ board.png not found in current directory!")

board_h, board_w = board_img.shape[:2]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

################################### MAIN LOOP ################################

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    gaze_visual = board_img.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x * frame_w, lm.y * frame_h) for lm in face_landmarks.landmark])

            # Mediapipe indices
            LEFT_EYE_IDX = [33, 133, 160, 159, 158, 153, 144, 145]
            RIGHT_EYE_IDX = [263, 362, 385, 386, 387, 373, 380, 374]
            LEFT_IRIS_IDX = [468, 469, 470, 471]
            RIGHT_IRIS_IDX = [473, 474, 475, 476]

            # Compute eye bbox
            left_eye_pts = landmarks[LEFT_EYE_IDX]
            right_eye_pts = landmarks[RIGHT_EYE_IDX]
            left_bbox_x, left_bbox_y = left_eye_pts[:,0].min(), left_eye_pts[:,1].min()
            left_bbox_w, left_bbox_h = left_eye_pts[:,0].max() - left_bbox_x, left_eye_pts[:,1].max() - left_bbox_y
            right_bbox_x, right_bbox_y = right_eye_pts[:,0].min(), right_eye_pts[:,1].min()
            right_bbox_w, right_bbox_h = right_eye_pts[:,0].max() - right_bbox_x, right_eye_pts[:,1].max() - right_bbox_y

            # Pupil centers
            left_pupil = np.mean(landmarks[LEFT_IRIS_IDX], axis=0)
            right_pupil = np.mean(landmarks[RIGHT_IRIS_IDX], axis=0)

            # Draw eye bounding boxes and pupils
            cv2.rectangle(frame, (int(left_bbox_x), int(left_bbox_y)),
                          (int(left_bbox_x+left_bbox_w), int(left_bbox_y+left_bbox_h)), (0,255,0), 1)
            cv2.rectangle(frame, (int(right_bbox_x), int(right_bbox_y)),
                          (int(right_bbox_x+right_bbox_w), int(right_bbox_y+right_bbox_h)), (0,255,0), 1)
            cv2.circle(frame, tuple(left_pupil.astype(int)), 3, (0,0,255), -1)
            cv2.circle(frame, tuple(right_pupil.astype(int)), 3, (0,0,255), -1)

            # Project pupils to board
            gx_left, gy_left = project_pupil_to_board((left_bbox_x, left_bbox_y, left_bbox_w, left_bbox_h),
                                                      left_pupil, (board_w, board_h))
            gx_right, gy_right = project_pupil_to_board((right_bbox_x, right_bbox_y, right_bbox_w, right_bbox_h),
                                                        right_pupil, (board_w, board_h))

            # Draw on board
            cv2.circle(gaze_visual, (gx_left, gy_left), 8, (0,0,255), -1)
            cv2.circle(gaze_visual, (gx_right, gy_right), 8, (0,0,255), -1)

    vis = np.hstack((frame, cv2.resize(gaze_visual, (frame_w, frame_h))))
    cv2.imshow("Left: Frame | Right: Board", vis)

    if cv2.waitKey(50) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
