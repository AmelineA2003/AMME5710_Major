import sys, os
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import math
# import mediapipe as mp
import joblib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Arrays of 2-dimensional vectors.*")
import mediapipe as mp


def load_cascade(name):
    cascade_dir = os.getcwd()   
    path = os.path.join(cascade_dir, name)
    if not os.path.isfile(path):
        print(f"ERROR: Cascade file not found → {path}")
        return cv2.CascadeClassifier()      # returns an empty classifier
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        print(f"ERROR: Failed to load cascade → {path}")
    else:
        print(f"Loaded: {name}")
    return clf


def GetHeadPose(face_cascade, eye_cascade, mouth_cascade, nose_cascade, img, height_map):
    img1 = img.copy()
    img2 = img.copy()
    height_m = height_map.copy()
    
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No faces detected")
        return img2  # Early return with original image if no face

    arr = height_m.copy().astype(float)
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if np.isnan(arr[i, j]):
                neighbors = []
                # 4-connectivity: up, down, left, right
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(arr[ni, nj]):
                        neighbors.append(arr[ni, nj])
                if neighbors:
                    arr[i, j] = np.mean(neighbors)
    height_m = arr

    # Finding Face
    x, y, w, h = faces[0]

    # Draw rectangle around face
    cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Annotate corners with coordinates
    tl = f"({x},{y})"
    tr = f"({x+w},{y})"
    bl = f"({x},{y+h})"
    br = f"({x+w},{y+h})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    cv2.putText(img2, tl, (x-5, y-5), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(img2, tr, (x+w+5, y-5), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(img2, bl, (x-5, y+h+15), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(img2, br, (x+w+5, y+h+15), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(img2, tl, (x-5, y-5), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(img2, tr, (x+w+5, y-5), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(img2, bl, (x-5, y+h+15), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(img2, br, (x+w+5, y+h+15), font, font_scale, (255, 255, 255), thickness)

    # Extract ROI for facial features
    roi_gray = gray[y:y+h, x:x+w]
    roi_height = height_m[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    nose = nose_cascade.detectMultiScale(roi_gray)
    mouth = mouth_cascade.detectMultiScale(roi_gray)

    features = {'face': (x, y, w, h)}

    # Draw reference lines
    cv2.line(img2, (x + int(w/2), y), (x + int(w/2), y + h), (255, 255, 255), 1)
    cv2.line(img2, (x, y + int(h/3)), (x + w, y + int(h/3)), (255, 255, 255), 1)
    cv2.line(img2, (x, y + int(h/2)), (x + w, y + int(h/2)), (255, 255, 255), 1)
    cv2.line(img2, (x, y + int(3*h/5)), (x + w, y + int(3*h/5)), (255, 255, 255), 1)

    print("found a face")
    print(h, w, "head box dimensions\n")

    # Finding eyes
    exp = 10

    for (ex, ey, ew, eh) in eyes:
        center = (int(x + ex + ew/2), int(y + ey + eh/2))

        if center[0] < x + w/2:
            if center[1] < y + h/2:
                if 'eye_left' in features:
                    continue

                left_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]

                features['eye_left'] = [center[0], center[1], left_eye_height]
                cv2.circle(img2, center, 5, (0, 0, 255), 2)
                print("Left eye detected at:", center)
            
        if center[0] > x + w/2:
            if center[1] < y + h/2:
                if 'eye_right' in features:
                    continue
                
                right_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]
                
                features['eye_right'] = [center[0], center[1], right_eye_height]
                cv2.circle(img2, center, 5, (0, 0, 255), 2)
                print("Right eye detected at:", center)

    # Validate eyes before proceeding
    if 'eye_left' not in features or 'eye_right' not in features:
        print("EYES not valid – insufficient eye detections\n")
        return img2

    if abs(features['eye_left'][1] - features["eye_right"][1]) >= (h/2 - h/3):
        print("EYES not valid – excessive vertical misalignment\n")
        return img2

    print("EYES valid\n")

    left_eye_vec = np.array(features["eye_left"])[:2]
    right_eye_vec = np.array(features["eye_right"])[:2]

    eye_axis_vec = right_eye_vec - left_eye_vec
    inter_eye_distance = np.linalg.norm(eye_axis_vec)
    if inter_eye_distance == 0:
        print("EYES invalid – zero inter-eye distance\n")
        return img2

    eye_axis_unit = eye_axis_vec / inter_eye_distance

    features["eye_midpoint"] = [
        int((left_eye_vec[0] + right_eye_vec[0]) / 2),
        int((left_eye_vec[1] + right_eye_vec[1]) / 2),
        height_m[int((left_eye_vec[0] + right_eye_vec[0]) / 2), int((left_eye_vec[1] + right_eye_vec[1]) / 2)]
    ]

    # Finding nose
    nose_found = False
    for (nx, ny, nw, nh) in nose:
        if (nw * nh) / (h * w) < 0.02:
            continue

        nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))
        if ('eye_left' in features and 'eye_right' in features and
            nose_center[1] > features['eye_left'][1] and
            nose_center[1] > features['eye_right'][1] and
            nose_center[0] > features['eye_left'][0] and
            nose_center[0] < features['eye_right'][0]):

            nose_height_map = height_m[y + ny:y + ny + nh, x + nx:x + nx + nw]
            if nose_height_map.size == 0:
                continue

            min_index = np.argmin(nose_height_map)
            row, col = np.unravel_index(min_index, nose_height_map.shape)
            nose_depth = np.min(nose_height_map)
            nose_center = (int(x + nx + col), int(y + ny + row))
            cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)
            print("Nose detected at:", nose_center)
            features['nose'] = [nose_center[0], nose_center[1], nose_depth]
            nose_found = True
            break

    if not nose_found:
        print("No valid nose found\n")
        return img2

    # Finding Mouth
    mouth_found = False
    exp_x = 0
    exp_y = 0
    for (mx, my, mw, mh) in mouth: 
        mouth_center = (int(x + mx + mw/2), int(y + my + mh/2))
        if 'nose' in features and mouth_center[1] < features['nose'][1] + 0.2 * h:
            continue
        
        exp_y = int(mh * 0.15)
        mouth_region = img1[y + my + exp_y : y + my + mh - exp_y, x + mx - exp_x: x + mx + mw + exp_x]
        if mouth_region.size == 0:
            continue

        cv2.circle(img2, mouth_center, 5, (0, 0, 255), 2)
        features['mouth'] = mouth_center
        print("Mouth detected at:", mouth_center)

        mouth_region_gray = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2GRAY)
        mouth_region_gray = cv2.equalizeHist(mouth_region_gray)

        mouth_hist = cv2.calcHist([mouth_region_gray], [0], None, [256], [0, 256])
        mean = np.mean(mouth_hist)
        stddev = np.std(mouth_hist)

        A = np.min(mouth_hist)
        B = mean + stddev
        C = A
        D = np.max(mouth_hist)

        if B > A:
            mouth_region_gray_adjusted = ((D - C) / (B - A)) * (mouth_region_gray - A) + C
            mouth_region_gray_adjusted = np.clip(mouth_region_gray_adjusted, 0, 255).astype('uint8')
        else:
            mouth_region_gray_adjusted = mouth_region_gray

        thresh_val = 240
        _, thresh_im1 = cv2.threshold(mouth_region_gray_adjusted, thresh_val, 255, cv2.THRESH_BINARY)
        mouth_region_binary_dilate = cv2.dilate(thresh_im1, None, iterations=1)

        black_mask = mouth_region_binary_dilate == 0
        rows, cols = np.where(black_mask)
        if len(rows) == 0:
            continue

        leftmost_col = np.min(cols)
        leftmost_row = rows[np.argmin(cols)]
        rightmost_col = np.max(cols)
        rightmost_row = rows[np.argmax(cols)]

        left_mouth = np.array([x + mx + leftmost_col + exp_x, y + my + leftmost_row + exp_y])
        right_mouth = np.array([x + mx + rightmost_col - exp_x, y + my + rightmost_row + exp_y])

        features["left_mouth"] = left_mouth
        features["right_mouth"] = right_mouth
        features["mid_mouth"] = np.array([
            int((right_mouth[0] + left_mouth[0]) // 2),
            int((right_mouth[1] + left_mouth[1]) // 2),
            height_m[int((right_mouth[0] + left_mouth[0]) // 2), int((right_mouth[1] + left_mouth[1]) // 2)]
        ])

        cv2.circle(img2, (left_mouth[0], left_mouth[1]), 2, (0, 255, 255), 1)
        cv2.circle(img2, (right_mouth[0], right_mouth[1]), 2, (0, 255, 255), 1)

        mouth_found = True
        break

    if not mouth_found:
        print("No valid mouth\n")
        return img2

    print(features["right_mouth"], "right mouth")
    print(features["left_mouth"], "left mouth\n")

    # Drawing lines
    cv2.line(img2, (int(features["eye_left"][0]), int(features["eye_left"][1])), (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["eye_left"][0]), int(features["eye_left"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["eye_left"][0]), int(features["eye_left"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (0, 255, 0), 1)

    # Improved Pose Estimation
    eye_mid = np.array(features['eye_midpoint'][:2])
    nose_vec = np.array(features['nose'][:2])
    mouth_mid = np.array(features['mid_mouth'][:2])

    # Yaw: signed horizontal offset from eye midline (±30° max)
    mid_to_nose_proj = np.dot(nose_vec - eye_mid, eye_axis_unit)
    yaw_deg = -mid_to_nose_proj / (inter_eye_distance * 0.5) * 30  # Negative for rightward yaw

    # Pitch: vertical deviation along eye-to-mouth axis (±25° max)
    eye_to_mouth = mouth_mid - eye_mid
    eye_to_mouth_norm = eye_to_mouth / np.linalg.norm(eye_to_mouth)
    eye_to_nose = nose_vec - eye_mid
    nose_proj_v = np.dot(eye_to_nose, eye_to_mouth_norm)
    expected_nose_v = np.linalg.norm(eye_to_mouth) * 0.45
    pitch_deg = (nose_proj_v - expected_nose_v) / expected_nose_v * 25

    # Roll: signed angle (deprecation-safe 3D cross)
    planar_vec = np.array([0.0, 1.0])
    orth_vec = np.array([-eye_axis_unit[1], eye_axis_unit[0]])
    planar_3d = np.append(planar_vec, 0.0)
    orth_3d = np.append(orth_vec, 0.0)
    cross_z = np.cross(planar_3d, orth_3d)[2]
    roll_rad = np.arctan2(cross_z, np.dot(planar_vec, orth_vec))
    roll_deg = np.degrees(roll_rad)

    print(f"Pitch: {pitch_deg:.2f}")
    print(f"Yaw: {yaw_deg:.2f}")
    print(f"Roll: {roll_deg:.2f}")

    # Compose rotation matrix (extrinsic Z-Y-X: yaw → pitch → roll)
    rvec = np.array([np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)])
    R, _ = cv2.Rodrigues(rvec)

    # Unit axes in local frame (X forward, Y down, Z right)
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    directions = R @ axes
    directions /= np.linalg.norm(directions, axis=0)
    directions *= 50  # Arrow length

    # Draw arrows from nose tip
    nose_pos = (int(features["nose"][0]), int(features["nose"][1]))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X-red, Y-green, Z-blue
    for i in range(3):
        dx, dy, _ = directions[:, i]
        end = (int(nose_pos[0] + dx), int(nose_pos[1] + dy))
        cv2.arrowedLine(img2, nose_pos, end, colors[i], 5, tipLength=0.3)

    # Text overlays
    cv2.putText(img2, f"Pitch: {pitch_deg:.2f}", (5, 30), font, 1, (255, 255, 255), 3)
    cv2.putText(img2, f"Yaw: {yaw_deg:.2f}", (5, 60), font, 1, (255, 255, 255), 3)
    cv2.putText(img2, f"Roll: {roll_deg:.2f}", (5, 90), font, 1, (255, 255, 255), 3)

    # ---------------------------------------------------------------------------------------------------------

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError("Image not found.")
    original = img.copy()
    h, w = img.shape[:2]

    # ------------------------------------------------------------------ #
    # 2. MediaPipe Face Mesh
    # ------------------------------------------------------------------ #
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected.")
        landmarks = results.multi_face_landmarks[0]

    # ------------------------------------------------------------------ #
    # 3. 3D model points – Y positive DOWN (matches image coordinates)
    # ------------------------------------------------------------------ #
    model_pts = np.array([
        [ 0.0,   0.0,   0.0],   # 0 – Nose tip
        [ 0.0, -63.6, -12.5],   # 1 – Chin
        [ 43.3,  32.7, -26.0],  # 2 – Left eye outer
        [-43.3,  32.7, -26.0],  # 3 – Right eye outer
        [ 28.9, -28.9, -24.1],  # 4 – Left mouth outer
        [-28.9, -28.9, -24.1],  # 5 – Right mouth outer
    ], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 4. 2D image points
    # ------------------------------------------------------------------ #
    img_pts = np.array([
        [landmarks.landmark[4].x   * w, landmarks.landmark[4].y   * h],  # Nose tip
        [landmarks.landmark[152].x * w, landmarks.landmark[152].y * h],  # Chin
        [landmarks.landmark[33].x  * w, landmarks.landmark[33].y  * h],  # Left eye outer
        [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],  # Right eye outer
        [landmarks.landmark[61].x  * w, landmarks.landmark[61].y  * h],  # Left mouth outer
        [landmarks.landmark[291].x * w, landmarks.landmark[291].y * h],  # Right mouth outer
    ], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 5. Camera intrinsics
    # ------------------------------------------------------------------ #
    focal = 1.5 * w
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 6. solvePnP
    # ------------------------------------------------------------------ #
    success, rvec, tvec = cv2.solvePnP(
        model_pts, img_pts, cam_mat, dist,
        flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        raise RuntimeError("solvePnP failed.")

    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_to_head = R_world_to_cam.T

    # ------------------------------------------------------------------ #
    # 7. Euler angles (Z-Y-X extrinsic)
    # ------------------------------------------------------------------ #
    sy = math.sqrt(R_cam_to_head[0,0]**2 + R_cam_to_head[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(-R_cam_to_head[2,1], R_cam_to_head[2,2])
        yaw   = math.atan2( R_cam_to_head[2,0], sy)
        roll  = math.atan2(-R_cam_to_head[1,0], R_cam_to_head[0,0])
    else:
        pitch = math.atan2( R_cam_to_head[1,2], R_cam_to_head[1,1])
        yaw   = math.atan2( R_cam_to_head[2,0], sy)
        roll  = 0.0

    pitch_deg_mp = math.degrees(pitch)
    yaw_deg_mp   = math.degrees(yaw)
    roll_deg_mp  = math.degrees(roll)

    # ------------------------------------------------------------------ #
    # 8. Arrows (MediaPipe PnP)
    # ------------------------------------------------------------------ #
    axes_local = np.array([
        [1, 0, 0],   # X – right
        [0, 1, 0],   # Y – down
        [0, 0, 1]    # Z – forward
    ], dtype=np.float32)

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

    # ------------------------------------------------------------------ #
    # 9. Text overlay (MediaPipe)
    # ------------------------------------------------------------------ #
    cv2.putText(original, f'Pitch: {pitch_deg_mp:+.2f} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Yaw:   {yaw_deg_mp:+.2f} deg', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Roll:  {roll_deg_mp:+.2f} deg', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ------------------------------------------------------------------ #
    # 10. Display
    # ------------------------------------------------------------------ #
    desired_width = 800
    desired_height = 600

    mine_window_name = "Mine"
    cv2.namedWindow(mine_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mine_window_name, desired_width, desired_height)
    cv2.imshow(mine_window_name, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    media_pipe_window_name = "MediaPipe"
    cv2.namedWindow(media_pipe_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(media_pipe_window_name, desired_width, desired_height)
    cv2.imshow(media_pipe_window_name, original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # 11. Return
    # ------------------------------------------------------------------ #
    return img2, {
        'angles': {'pitch': pitch_deg, 'yaw': yaw_deg, 'roll': roll_deg},
        'mp_angles': {'pitch': pitch_deg_mp, 'yaw': yaw_deg_mp, 'roll': roll_deg_mp},
        'features': features
    }


if __name__ == "__main__":
    heads = {}
    i = 0

    folder_path = os.getcwd()
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    jpg_files = png_files + jpg_files

    print(f"Found {len(jpg_files)} image files in {folder_path}")

    for image_path in jpg_files:
        image_name = os.path.basename(image_path)
        pickle_name = image_name[0:-7] + "height.pkl"
        
        try:
            with open(pickle_name, 'rb') as file:
                loaded_data = pickle.load(file)
        except FileNotFoundError:
            print(f"Error: The file {pickle_name} was not found.")
            continue
        except pickle.PicklingError:
            print("Error: Failed to load the pickle file.")
            continue

        image = cv2.imread(os.path.join(folder_path, image_name))
        
        sobel_x = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=3)
        
        heads[image_name] = {
            "im_rgb": cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            "im_bgr": image,
            "im_gray": cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            "im_height": loaded_data["height_m"],
            "im_gray_gaussian": cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0),
            "im_sobel_x": sobel_x,
            "im_sobel_y": sobel_y,
            "im_sobel_mag": cv2.normalize(np.sqrt(sobel_x**2 + sobel_y**2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        }
        
        print(f"\nProcessing {image_name}...")
        i += 1
        if i >= 70:
            break

    # Load classifiers
    face_cascade       = load_cascade('haarcascade_frontalface_default.xml')
    eye_cascade        = load_cascade('haarcascade_eye.xml')
    mouth_cascade      = load_cascade('haarcascade_mcs_mouth.xml')
    nose_cascade       = load_cascade('haarcascade_mcs_nose.xml')

    head = heads["jestinp_36_rgb.png"]
    img = head["im_rgb"].copy()
    height_m = head["im_height"].copy()

    annotated_img, results = GetHeadPose(face_cascade, eye_cascade, mouth_cascade, nose_cascade, img, height_m)
    print("Custom Head Pose Angles:", results['angles'])
    print("MediaPipe PnP Angles:", results['mp_angles'])