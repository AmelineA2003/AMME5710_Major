import sys
import os
import math
import pickle
import warnings
import cv2
import numpy as np
import mediapipe as mp

"""
GetHeadPoseVideo.py

This file defines two functions:

load_cascade: which returns a haar cascade file which it finds in the current working directory as a .xml file.

GetHeadPose: takes in the face, eye, mouth and nose cascades
"""



# Suppress specific deprecation warning
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*Arrays of 2-dimensional vectors.*"
)

# Add script directory to path
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)

# Set Qt platform (Linux)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')


def load_cascade(name: str) -> cv2.CascadeClassifier:
    """
    Load a Haar cascade classifier from the current working directory.
    """
    path = os.path.join(os.getcwd(), name)
    if not os.path.isfile(path):
        print(f"ERROR: Cascade file not found → {path}")
        return cv2.CascadeClassifier()
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        print(f"ERROR: Failed to load cascade → {path}")
    else:
        print(f"Loaded: {name}")
    return clf


def GetHeadPose(
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
    mouth_cascade: cv2.CascadeClassifier,
    nose_cascade: cv2.CascadeClassifier,
    img_rgb: np.ndarray,
    height_map: np.ndarray
) -> dict:
    """
    Perform head pose estimation using MediaPipe Face Mesh and custom Haar cascade method.
    """
    img1 = img_rgb.copy()
    img2 = img_rgb.copy()
    height_m = height_map.copy()

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    original = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    default_return = {
        'angles': None,
        'camera_to_head_rotation': None,
        'nose_2d': None,
        'axis_offsets_px': None,
        'Mine': cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
        'MediaPipe': original
    }

    # ------------------------------------------------------------------ #
    # MediaPipe Face Mesh
    # ------------------------------------------------------------------ #
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            print("No face detected by MediaPipe.")
            return default_return
        landmarks = results.multi_face_landmarks[0]

    # 3D model points
    model_pts = np.array([
        [0.0, 0.0, 0.0],           # Nose tip
        [0.0, -63.6, -12.5],       # Chin
        [43.3, 32.7, -26.0],       # Left eye outer
        [-43.3, 32.7, -26.0],      # Right eye outer
        [28.9, -28.9, -24.1],      # Left mouth outer
        [-28.9, -28.9, -24.1],     # Right mouth outer
    ], dtype=np.float64)

    # 2D image points
    img_pts = np.array([
        [landmarks.landmark[4].x * w, landmarks.landmark[4].y * h],
        [landmarks.landmark[152].x * w, landmarks.landmark[152].y * h],
        [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],
        [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],
        [landmarks.landmark[61].x * w, landmarks.landmark[61].y * h],
        [landmarks.landmark[291].x * w, landmarks.landmark[291].y * h],
    ], dtype=np.float64)

    # Camera intrinsics
    focal = 1.5 * w
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam_mat, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not success:
        raise RuntimeError("solvePnP failed.")

    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_to_head = R_world_to_cam.T

    # Euler angles (Z-Y-X)
    sy = math.sqrt(R_cam_to_head[0, 0]**2 + R_cam_to_head[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-R_cam_to_head[2, 1], R_cam_to_head[2, 2])
        yaw = math.atan2(R_cam_to_head[2, 0], sy)
        roll = math.atan2(-R_cam_to_head[1, 0], R_cam_to_head[0, 0])
    else:
        pitch = math.atan2(R_cam_to_head[1, 2], R_cam_to_head[1, 1])
        yaw = math.atan2(R_cam_to_head[2, 0], sy)
        roll = 0.0

    pitch_deg_mp = math.degrees(pitch)
    yaw_deg_mp = math.degrees(yaw)
    roll_deg_mp = math.degrees(roll)

    # Draw axes
    axes_local = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    arrow_len = 50.0
    dirs_cam = (R_cam_to_head @ axes_local.T).T
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1)[:, None]
    dirs_cam *= arrow_len
    offsets_2d = np.round(dirs_cam[:, :2]).astype(int)
    nose_2d = tuple(img_pts[0].astype(int))
    colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, (dx, dy) in enumerate(offsets_2d):
        end = (nose_2d[0] + dx, nose_2d[1] + dy)
        cv2.arrowedLine(original, nose_2d, end, colors_bgr[i], thickness=4, tipLength=0.25)

    # Overlay angles
    cv2.putText(original, f'Pitch: {pitch_deg_mp:+.2f} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Yaw:   {yaw_deg_mp:+.2f} deg', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Roll:  {roll_deg_mp:+.2f} deg', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ------------------------------------------------------------------ #
    # Custom Haar Cascade + Depth Map
    # ------------------------------------------------------------------ #
    try:
        gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            print("No faces detected by Haar cascade.")
            return {**default_return, 'MediaPipe': original}

        # Interpolate NaN in height map
        arr = height_m.copy().astype(float)
        rows, cols = arr.shape
        for i in range(rows):
            for j in range(cols):
                if np.isnan(arr[i, j]):
                    neighbors = [
                        arr[ni, nj] for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if (ni := i + di) >= 0 and ni < rows and (nj := j + dj) >= 0 and nj < cols and not np.isnan(arr[ni, nj])
                    ]
                    if neighbors:
                        arr[i, j] = np.mean(neighbors)
        height_m = arr

        x, y, w, h = faces[0]
        cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Annotate corners
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        labels = [f"({cx},{cy})" for cx, cy in corners]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        positions = [(x - 5, y - 5), (x + w + 5, y - 5), (x - 5, y + h + 15), (x + w + 5, y + h + 15)]
        for label, pos in zip(labels, positions):
            cv2.putText(img2, label, pos, font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(img2, label, pos, font, font_scale, (255, 255, 255), thickness)

        # Reference lines
        cv2.line(img2, (x + w // 2, y), (x + w // 2, y + h), (255, 255, 255), 1)
        for y_line in [y + h // 3, y + h // 2, y + 3 * h // 5]:
            cv2.line(img2, (x, y_line), (x + w, y_line), (255, 255, 255), 1)

        print("found a face")
        print(h, w, "head box dimensions\n")

        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        nose = nose_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        features = {'face': (x, y, w, h)}

        # Eye Detection
        exp = 10
        for (ex, ey, ew, eh) in eyes:
            center = (x + ex + ew // 2, y + ey + eh // 2)
            if center[0] < x + w // 2 and center[1] < y + h // 2:
                if 'eye_left' not in features:
                    depth = height_m[y + ey + eh // 2, x + ex + ew // 2]
                    features['eye_left'] = [center[0], center[1], depth]
                    cv2.circle(img2, center, 5, (0, 0, 255), 2)
                    print("Left eye detected at:", center)
            if center[0] > x + w // 2 and center[1] < y + h // 2:
                if 'eye_right' not in features:
                    depth = height_m[y + ey + eh // 2, x + ex + ew // 2]
                    features['eye_right'] = [center[0], center[1], depth]
                    cv2.circle(img2, center, 5, (0, 0, 255), 2)
                    print("Right eye detected at:", center)

        if 'eye_left' in features and 'eye_right' in features:
            if abs(features['eye_left'][1] - features['eye_right'][1]) < (h // 2 - h // 3):
                print("EYES valid\n")
                left_vec = np.array(features["eye_left"])[:2]
                right_vec = np.array(features["eye_right"])[:2]
                mid_x = (left_vec[0] + right_vec[0]) // 2
                mid_y = (left_vec[1] + right_vec[1]) // 2
                mid_depth = height_m[mid_y, mid_x]
                features["eye_midpoint"] = [mid_x, mid_y, mid_depth]
            else:
                print("EYES not valid\n")

        # Nose Detection
        for (nx, ny, nw, nh) in nose:
            if (nw * nh) / (h * w) < 0.02:
                continue
            nose_center = (x + nx + nw // 2, y + ny + nh // 2)
            if ('eye_left' in features and 'eye_right' in features and
                nose_center[1] > features['eye_left'][1] and
                nose_center[1] > features['eye_right'][1] and
                features['eye_left'][0] < nose_center[0] < features['eye_right'][0]):
                
                nose_map = height_m[y + ny:y + ny + nh, x + nx:x + nx + nw]
                row, col = np.unravel_index(np.argmin(nose_map), nose_map.shape)
                nose_depth = np.min(nose_map)
                nose_center = (x + nx + col, y + ny + row)
                cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)
                print("Nose detected at:", nose_center)
                features['nose'] = [nose_center[0], nose_center[1], nose_depth]
                break
        else:
            print("No valid nose detected.\n")

        try:
            print(features["nose"], "\n")
        except KeyError:
            print("No nose found\n")

        # Mouth Detection
        exp_y = 0
        for (mx, my, mw, mh) in mouth:
            mouth_center = (x + mx + mw // 2, y + my + mh // 2)
            if 'nose' not in features or mouth_center[1] < features['nose'][1] + 0.2 * h:
                continue
            exp_y = int(mh * 0.15)
            mouth_region = img1[y + my + exp_y:y + my + mh - exp_y, x + mx:x + mx + mw]
            cv2.circle(img2, mouth_center, 5, (0, 0, 255), 2)
            features['mouth'] = mouth_center
            print("Mouth detected at:", mouth_center)

            gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2GRAY)
            gray_mouth = cv2.equalizeHist(gray_mouth)
            hist = cv2.calcHist([gray_mouth], [0], None, [256], [0, 256])
            mean, std = np.mean(hist), np.std(hist)
            A, B, C, D = hist.min(), mean + std, hist.min(), hist.max()
            adjusted = ((D - C) / (B - A)) * (gray_mouth - A) + C
            adjusted = np.clip(adjusted, 0, 255).astype('uint8')
            _, thresh = cv2.threshold(adjusted, 240, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=1)
            black_mask = dilated == 0
            rows, cols = np.where(black_mask)
            if len(cols) == 0:
                continue
            left_mouth = np.array([x + mx + cols[np.argmin(cols)], y + my + rows[np.argmin(cols)] + exp_y])
            right_mouth = np.array([x + mx + cols[np.argmax(cols)], y + my + rows[np.argmax(cols)] + exp_y])
            features["left_mouth"] = left_mouth
            features["right_mouth"] = right_mouth
            mid_x = (left_mouth[0] + right_mouth[0]) // 2
            mid_y = (left_mouth[1] + right_mouth[1]) // 2
            mid_depth = height_m[mid_y, mid_x]
            features["mid_mouth"] = np.array([mid_x, mid_y, mid_depth])
            cv2.circle(img2, (left_mouth[0], left_mouth[1]), 2, (0, 255, 255), 1)
            cv2.circle(img2, (right_mouth[0], right_mouth[1]), 2, (0, 255, 255), 1)
            break

        try:
            print(features["right_mouth"], "right mouth")
            print(features["left_mouth"], "left mouth\n")
        except KeyError:
            print("no valid mouth\n")

        # Facial connections
        connections = [
            (features["eye_left"][:2], features["right_mouth"][:2]),
            (features["eye_right"][:2], features["left_mouth"][:2]),
            (features["right_mouth"][:2], features["left_mouth"][:2]),
            (features["eye_right"][:2], features["eye_left"][:2]),
            (features["eye_left"][:2], features["left_mouth"][:2]),
            (features["eye_right"][:2], features["right_mouth"][:2]),
        ]
        for p1, p2 in connections:
            cv2.line(img2, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 1)

        # Pose Estimation (Custom)
        yaw_deg = roll_deg = pitch_deg = 0.0

        if all(k in features for k in ["eye_left", "eye_right", "nose"]):
            left_eye = np.array(features["eye_left"])[:2]
            right_eye = np.array(features["eye_right"])[:2]
            nose = np.array(features["nose"])[:2]
            eye_axis = right_eye - left_eye
            inter_eye_dist = np.linalg.norm(eye_axis)
            if inter_eye_dist > 0:
                eye_unit = eye_axis / inter_eye_dist
                left_to_nose = nose - left_eye
                proj = np.dot(left_to_nose, eye_unit)
                norm_pos = proj / inter_eye_dist
                denom_l = 0.5 - norm_pos
                denom_r = 0.5 + norm_pos
                yaw_deg = (denom_r / denom_l if denom_l != 0 else float('inf')) * 180

        if 'eye_left' in features and 'eye_right' in features:
            eye_axis = np.array(features["eye_right"])[:2] - np.array(features["eye_left"])[:2]
            if np.linalg.norm(eye_axis) > 0:
                eye_unit = eye_axis / np.linalg.norm(eye_axis)
                orth = np.array([-eye_unit[1], eye_unit[0]])
                planar = np.array([0, 1])
                roll_rad = np.arctan2(np.cross(planar, orth), np.dot(planar, orth))
                roll_deg = np.degrees(roll_rad)

        if all(k in features for k in ["eye_midpoint", "mid_mouth", "nose"]):
            eye_mid = np.array(features["eye_midpoint"])[:2]
            mouth_mid = np.array(features["mid_mouth"])[:2]
            nose_pt = np.array(features["nose"])[:2]
            mid_vec = mouth_mid - eye_mid
            nose_vec = nose_pt - eye_mid
            if np.linalg.norm(mid_vec) > 0:
                proj = np.dot(nose_vec, mid_vec) / np.linalg.norm(mid_vec)
                norm_pos = proj / np.linalg.norm(mid_vec)
                denom_d = 0.5 - norm_pos
                denom_u = 0.5 + norm_pos
                pitch_deg = (denom_d / denom_u if denom_u != 0 else float('inf')) * 180

        print(f"pitch: {pitch_deg:.2f}")
        print(f"roll: {roll_deg:.2f}")
        print(f"yaw: {yaw_deg:.2f}")

        # Draw custom axes
        rvec_custom = np.array([np.radians(pitch_deg), np.radians(roll_deg), np.radians(yaw_deg)])
        R_custom, _ = cv2.Rodrigues(rvec_custom)
        axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        directions = R_custom @ axes
        directions /= np.linalg.norm(directions, axis=0)
        directions *= 50
        nose_pos = (int(features["nose"][0]), int(features["nose"][1]))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ['X', 'Y', 'Z']
        offset = 5
        for i in range(3):
            dx, dy, _ = directions[:, i]
            end = (int(nose_pos[0] + dx), int(nose_pos[1] + dy))
            cv2.arrowedLine(img2, nose_pos, end, colors[i], 2, tipLength=0.3)
            label_x = end[0] + offset
            label_y = end[1] - offset
            cv2.putText(img2, labels[i], (label_x, label_y), font, 0.5, colors[i], 2, cv2.LINE_AA)

        # Overlay custom angles
        cv2.putText(img2, f"Pitch: {pitch_deg:.2f}", (5, 30), font, 1, (255, 255, 255), 3)
        cv2.putText(img2, f"Yaw: {yaw_deg:.2f}", (5, 60), font, 1, (255, 255, 255), 3)
        cv2.putText(img2, f"Roll: {roll_deg:.2f}", (5, 90), font, 1, (255, 255, 255), 3)

    except (KeyError, ValueError, IndexError) as e:
        print(f"Feature detection failed: {e}")
        return {**default_return, 'MediaPipe': original}

    return {
        'angles': {'pitch': pitch_deg, 'yaw': yaw_deg, 'roll': roll_deg},
        'camera_to_head_rotation': R_cam_to_head,
        'nose_2d': nose_2d,
        'axis_offsets_px': offsets_2d,
        'Mine': cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
        'MediaPipe': original
    }


if __name__ == "__main__":
    pickle_name = "jestin_video_3.pkl"
    try:
        with open(os.path.join(os.getcwd(), pickle_name), 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: The file {pickle_name} was not found.")
        sys.exit(1)
    except pickle.PicklingError:
        print("Error: Failed to load the pickle file.")
        sys.exit(1)

    rgb_frames = data['rgb_frames']
    height_frames = data['height_frames']
    if len(rgb_frames) != len(height_frames):
        raise ValueError("Mismatch between RGB and height frame counts.")
    print(f"Loaded {len(rgb_frames)} frames from '{pickle_name}'.")

    face_cascade = load_cascade('haarcascade_frontalface_default.xml')
    eye_cascade = load_cascade('haarcascade_eye.xml')
    mouth_cascade = load_cascade('haarcascade_mcs_mouth.xml')
    nose_cascade = load_cascade('haarcascade_mcs_nose.xml')

    fps = 30
    delay_ms = int(1000 / fps)
    print("Starting playback. Press 'q' to quit, 'p' to pause/resume, 'r' to rewind, 'n'/'b' for next/prev (when paused).")

    paused = False
    frame_idx = 0
    window_mine = "Mine"
    window_mp = "MediaPipe"

    while frame_idx < len(rgb_frames):
        if not paused:
            rgb_bgr = rgb_frames[frame_idx]
            height = height_frames[frame_idx]
            rgb_input = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            output = GetHeadPose(face_cascade, eye_cascade, mouth_cascade, nose_cascade, rgb_input, height)

            desired_width, desired_height = 800, 600
            cv2.namedWindow(window_mine, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_mine, desired_width, desired_height)
            cv2.imshow(window_mine, output["Mine"])

            cv2.namedWindow(window_mp, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_mp, desired_width, desired_height)
            cv2.imshow(window_mp, output["MediaPipe"])

            frame_idx += 1
            if frame_idx >= len(rgb_frames):
                print("Playback complete.")
                break

        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF
        if key == ord('q'):
            print("Playback quit by user.")
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            frame_idx = 0
            print("Rewound to start.")
        if paused:
            if key == ord('n'):
                frame_idx = min(frame_idx + 1, len(rgb_frames) - 1)
            elif key == ord('b'):
                frame_idx = max(frame_idx - 1, 0)

    cv2.destroyAllWindows()
    print("Playback window closed.")