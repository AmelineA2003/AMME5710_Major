import sys, os  # Import system and operating system modules for path manipulation and environment settings
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))  # Determine the absolute path of the script's directory
sys.path.append(mainpath)  # Append the script's directory to the Python path for module imports
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')  # Set the Qt platform plugin to 'xcb' for Linux compatibility, avoiding display issues
import glob  # Import glob for filename pattern matching
import matplotlib.pyplot as plt  # Import Matplotlib for plotting (though not used in this script)
import matplotlib.image as mpimg  # Import Matplotlib image module (unused)
import cv2  # Import OpenCV for computer vision tasks
import numpy as np  # Import NumPy for numerical operations and array handling
import pickle  # Import pickle for serializing and deserializing Python objects
import math  # Import math for trigonometric and mathematical functions
# import mediapipe as mp  # Initially commented; later imported after warning suppression
import joblib  # Import joblib for efficient serialization (unused in this script)
import warnings  # Import warnings to suppress specific deprecation messages
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Arrays of 2-dimensional vectors.*")  # Suppress specific NumPy deprecation warnings
import mediapipe as mp  # Import MediaPipe for facial landmark detection
from PIL import Image  # Import PIL Image (unused in this script)
import time


# This function loads the cascade
def load_cascade(name):
    """
    Loads a Haar cascade classifier from the current working directory.
    
    Args:
        name (str): Filename of the cascade XML file.
    
    Returns:
        cv2.CascadeClassifier: Loaded classifier, or empty classifier if loading fails.
    """
    cascade_dir = os.getcwd()   # Get current working directory
    path = os.path.join(cascade_dir, name)  # Construct full path to cascade file
    if not os.path.isfile(path):
        print(f"ERROR: Cascade file not found → {path}")
        return cv2.CascadeClassifier()      # returns an empty classifier
    clf = cv2.CascadeClassifier(path)  # Load the cascade classifier
    if clf.empty():
        print(f"ERROR: Failed to load cascade → {path}")
    else:
        print(f"Loaded: {name}")
    return clf  # Return the classifier (empty if failed)


def GetHeadPose(face_cascade, eye_cascade, mouth_cascade, nose_cascade, img, height_map, name, folder_path):
    """
    Estimates head pose using Haar cascades for facial feature detection and depth information from a height map.
    Draws annotations, computes yaw, pitch, and roll angles, and displays results alongside MediaPipe comparison.
    
    Args:
        face_cascade, eye_cascade, mouth_cascade, nose_cascade: Pre-trained Haar cascade classifiers.
        img (np.ndarray): Input RGB image.
        height_map (np.ndarray): Corresponding depth/height map with NaN for invalid pixels.
        name (str): Image filename.
        folder_path (str): Directory path for saving output images.
    
    Returns:
        dict: Estimated pitch, yaw, and roll angles from MediaPipe (primary return); custom method results are visualized.
    """
    img1 = img.copy()  # Preserve original image for region extraction
    img2 = img.copy()  # Working copy for drawing annotations
    height_m = height_map.copy()  # Copy of height map for processing
    
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for cascade detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply blur to reduce noise

    """
    Detect Faces
    """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces with scale factor 1.3 and min neighbors 5

    if len(faces) == 0:
        print("No faces detected")
        return  # Exit if no face is found
    


    try:
        """
        Interpolate the NaN values in the height map
        """
        arr = height_m.copy().astype(float)  # Work with float copy to handle NaN
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
                        arr[i, j] = np.mean(neighbors)  # Fill NaN with average of valid 4-connected neighbors
        height_m = arr  # Update height map with interpolated values

        # Finding Face
        x, y, w, h = faces[0]  # We only expect to see one face within the frame  

        """
        Annotate the face
        """
        # Draw rectangle around face
        cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle around detected face

        # Annotate corners with coordinates
        tl = f"({x},{y})"
        tr = f"({x+w},{y})"
        bl = f"({x},{y+h})"
        br = f"({x+w},{y+h})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        # Draw corner labels with black outline and white fill for visibility
        cv2.putText(img2, tl, (x-5, y-5), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(img2, tr, (x+w+5, y-5), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(img2, bl, (x-5, y+h+15), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(img2, br, (x+w+5, y+h+15), font, font_scale, (0, 0, 0), thickness+1)
        cv2.putText(img2, tl, (x-5, y-5), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img2, tr, (x+w+5, y-5), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img2, bl, (x-5, y+h+15), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img2, br, (x+w+5, y+h+15), font, font_scale, (255, 255, 255), thickness)

        # Extract ROI for facial features
        roi_gray = gray[y:y+h, x:x+w]  # Grayscale region of interest (face)
        roi_height = height_m[y:y+h, x:x+w]  # Corresponding height map ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)  # Detect eyes in face ROI
        nose = nose_cascade.detectMultiScale(roi_gray)  # Detect nose
        mouth = mouth_cascade.detectMultiScale(roi_gray)  # Detect mouth

        features = {'face': (x, y, w, h)}  # Initialize features dictionary with face bounding box

        # Draw reference lines (horizontal divisions of face)
        cv2.line(img2, (x + int(w/2), y), (x + int(w/2), y + h), (255, 255, 255), 1)  # Vertical midline
        cv2.line(img2, (x, y + int(h/3)), (x + w, y + int(h/3)), (255, 255, 255), 1)  # Upper third
        cv2.line(img2, (x, y + int(h/2)), (x + w, y + int(h/2)), (255, 255, 255), 1)  # Midline
        cv2.line(img2, (x, y + int(3*h/5)), (x + w, y + int(3*h/5)), (255, 255, 255), 1)  # Lower division

        print("found a face")
        print(h, w, "head box dimensions\n")

        """
        Finding eyes
        """
        exp = 10  # Expansion pixels around detected eye region

        for (ex, ey, ew, eh) in eyes:
            center = (int(x + ex + ew/2), int(y + ey + eh/2))  # Eye center in full image coordinates

            # Check to see if the eye is on the left or right side of the face

            if center[0] < x + w/2:
                # A left eye is detected
                if center[1] < y + h/2:
                    # The left eye is in the upper 1/2 of the face
                    if 'eye_left' in features:
                        # There is already a left eye, ignore duplicate
                        continue

                    # Extract expanded region around left eye
                    left_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]  # Depth at eye center

                    features['eye_left'] = [center[0], center[1], left_eye_height]
                    cv2.circle(img2, center, 5, (0, 0, 255), 2)  # Red circle
                    print("Left eye detected at:", center)
                
            if center[0] > x + w/2:
                # A right eye is detected
                if center[1] < y + h/2:
                    # The right eye is in the upper 1/2 of the face
                    if 'eye_right' in features:
                        # Already have right eye, ignore
                        continue
                    
                    right_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]

                    features['eye_right'] = [center[0], center[1], right_eye_height]
                    cv2.circle(img2, center, 5, (0, 0, 255), 2)
                    print("Right eye detected at:", center)

        # Check to see if the two eyes are at approximately the same height
        if abs(features['eye_left'][1] - features["eye_right"][1]) < (h/2 - h/3):
            print("EYES valid\n")

            left_eye_vec = np.array(features["eye_left"])[:2]
            right_eye_vec = np.array(features["eye_right"])[:2]
            eye_axis_vec = right_eye_vec - left_eye_vec  # Vector from left to right eye
            perp_eye_axis_vec = np.array([-eye_axis_vec[1], eye_axis_vec[0]])  # Perpendicular vector

            # Midpoint between eyes with depth
            features["eye_midpoint"] = [
                (left_eye_vec[0] + right_eye_vec[0]) // 2,
                (left_eye_vec[1] + right_eye_vec[1]) // 2,
                height_m[int((left_eye_vec[0] + right_eye_vec[0]) // 2), int((left_eye_vec[1] + right_eye_vec[1]) // 2)]
            ]

        else:
            print("EYES not valid\n")

        """
        Finding Nose
        """
        for (nx, ny, nw, nh) in nose:
            if (nw*nh)/(h*w) < 0.02:  # Filter small detections
                continue

            nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))
            # Verify nose position relative to eyes
            if ('eye_left' in features and 'eye_right' in features and
                nose_center[1] > features['eye_left'][1] and
                nose_center[1] > features['eye_right'][1] and
                nose_center[0] > features['eye_left'][0] and
                nose_center[0] < features['eye_right'][0]):

                nose_height_map = height_m[y + ny:y + ny + nh, x + nx:x + nx + nw]
                expand = 35

                # Normalise Nose Region Height Map
                min_height = nose_height_map.min()    
                max_height = nose_height_map.max()
                normalized_nose = ((nose_height_map - min_height) / (max_height - min_height) * 255).astype(np.uint8)

                # Find mean, standard deviation, min and max
                normalized_nose_hist = cv2.calcHist([normalized_nose], [0], None, [256], [0, 256])
                mean = np.mean(normalized_nose_hist)
                stddev = np.std(normalized_nose_hist)
                min_val = np.min(normalized_nose_hist)
                max_val = np.max(normalized_nose_hist)

                # Contrast stretching parameters
                A = min_val
                B = mean + 1 * stddev
                C = min_val
                D = max_val

                # Apply contrast adjustment
                normalized_nose_adjusted = ((D-C)/(B-A))*(normalized_nose-A)+C
                normalized_nose_adjusted = ~np.clip(normalized_nose_adjusted, 0, 255).astype('uint8')  # Invert and clip

                # Apply binary threshold to isolate highest point on the nose
                thresh_val = 30
                set_val = 255
                ret, thresh_im1 = cv2.threshold(normalized_nose_adjusted, thresh_val, set_val, cv2.THRESH_BINARY)

                # Find the contour of the nose tip
                contours, hierarchy = cv2.findContours(thresh_im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                (y_nose, x_nose), radius = cv2.minEnclosingCircle(contours[0])  # Find nose tip via contour

                print("Tip of Nose detected at:", y_nose, x_nose)
                nose_depth = np.min(nose_height_map)  # Use minimum depth as nose tip depth

                # Draw Nose Tip and save nose tip information
                nose_center = (int(x + nx + x_nose), int(y + ny + y_nose))
                cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)  # Yellow circle
                print("Nose detected at:", nose_center)
                features['nose'] = [nose_center[0], nose_center[1], nose_depth]

            else:
                print("No nose position constraints not met.\n")

        try:
            print(features["nose"],"\n")
        except KeyError:
            print("No nose found\n")

        
        """
        Finding Mouth
        """
        exp_x = 0
        exp_y = 0
        i = 0
        for (mx, my, mw, mh) in mouth: 
            mouth_center = (int(x + mx + mw/2), int(y + my + mh/2))

            if mouth_center[1] < features['nose'][1] + 0.17*h:  # Mouth must be below nose
                print("Mouth y higher than nose y, skipping")
                continue
            
            exp_x = 0
            exp_y = int(mh * 0.15)  # Expand vertically by 15% of mouth height

            mouth_region = img1[y + my + exp_y : y + my + mh - exp_y, x + mx - exp_x: x + mx + mw + exp_x]
            cv2.circle(img2, mouth_center, 5, (0, 0, 255), 2)  # Red circle at mouth center
            
            features['mouth'] = mouth_center
            print("Mouth detected at:", mouth_center)

            # Enhance mouth region for corner detection
            mouth_region_gray = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2GRAY)
            mouth_region_gray = cv2.equalizeHist(mouth_region_gray)  # Histogram equalization

            # Finding mean, standard deviation, max and min of mouth region
            mouth_hist = cv2.calcHist([mouth_region_gray], [0], None, [256], [0, 256])
            mean = np.mean(mouth_hist)
            stddev = np.std(mouth_hist)
            min_val = np.min(mouth_hist)
            max_val = np.max(mouth_hist)


            # Contrast stretching parameters
            A = min_val
            B = mean + 1 * stddev
            C = min_val
            D = max_val

            # Apply contrast adjustment
            mouth_region_gray_adjusted = ((D-C)/(B-A))*(mouth_region_gray-A)+C
            mouth_region_gray_adjusted = np.clip(mouth_region_gray_adjusted, 0, 255).astype('uint8')


            # Apply binary threshold and dilate
            thresh_val = 240
            set_val = 255
            ret, thresh_im1 = cv2.threshold(mouth_region_gray_adjusted, thresh_val, set_val, cv2.THRESH_BINARY)
            mouth_region_binary_dilate = cv2.dilate(thresh_im1, None, iterations=1)  # Dilate to connect components

            # Find where there are black pixels - Expected to be mouth corners, underside of nose and goatee
            black_mask = mouth_region_binary_dilate == 0
            rows, cols = np.where(black_mask)

            # Find leftmost and rightmost dark pixels (mouth corners)
            min_col_idx = np.argmin(cols)
            leftmost_row = rows[min_col_idx]
            leftmost_col = cols[min_col_idx]
            left_mouth = np.array([x + mx + leftmost_col + exp_x, y + my + leftmost_row + exp_y])

            min_col_idx = np.argmax(cols)
            rightmost_row = rows[min_col_idx]
            rightmost_col = cols[min_col_idx]
            right_mouth = np.array([x + mx + rightmost_col - exp_x, y + my + rightmost_row + exp_y])

            # Annotate mouth cornes and save information
            features["right_mouth"] = right_mouth
            features["left_mouth"] = left_mouth
            features["mid_mouth"] = np.array([
                int((right_mouth[0] + left_mouth[0])//2),
                int((right_mouth[1] + left_mouth[1])//2),
                height_m[int((right_mouth[0] + left_mouth[0])//2), int((right_mouth[1] + left_mouth[1])//2)]
            ])
            
            cv2.circle(img2, (left_mouth[0], left_mouth[1]), 2, (0, 255, 255), 1)  # Yellow
            cv2.circle(img2, (right_mouth[0], right_mouth[1]), 2, (0, 255, 255), 1)
            break  # Use first valid mouth

        try:
            print(features["right_mouth"], "right mouth")
            print(features["left_mouth"], "left mouth\n")
        except KeyError:
            print("no valid mouth\n")

        # Draw connecting lines between facial features
        cv2.line(img2, (int(features["eye_left"][0]), int(features["eye_left"][1])), (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (0, 255, 0), 1)
        cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
        cv2.line(img2, (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
        cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["eye_left"][0]), int(features["eye_left"][1])), (0, 255, 0), 1)
        cv2.line(img2, (int(features["eye_left"][0]), int(features["eye_left"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
        cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (0, 255, 0), 1)

        # YAW calculation
        left_eye_vec = np.array(features["eye_left"])[:2]
        right_eye_vec = np.array(features["eye_right"])[:2]
        nose_vec = np.array([features["nose"]])[0][:2]

        eye_axis_vec = right_eye_vec - left_eye_vec
        inter_eye_distance = np.linalg.norm(eye_axis_vec)
        if inter_eye_distance == 0:
            raise ValueError("Left and right eye coordinates are identical.")

        eye_axis_unit = eye_axis_vec / inter_eye_distance
        left_to_nose = nose_vec - left_eye_vec
        nose_projection = np.dot(left_to_nose, eye_axis_unit)
        normalized_position = nose_projection / inter_eye_distance

        # Asymmetry ratio: deviation from center (0.0 = centered)
        denom_left = 0.5 - normalized_position
        denom_right = 0.5 + normalized_position
        asymmetry_ratio_yaw = (denom_right / denom_left) if denom_left != 0 else float('inf')
        print(asymmetry_ratio_yaw, " - asymmetry ratio (right/left)")
        yaw_deg = asymmetry_ratio_yaw * 180  # Scale to degrees

        # Roll calculation
        planar_vec = np.array([0, 1])
        orth_vec = np.array([-eye_axis_unit[1], eye_axis_unit[0]])
        planar_vec = planar_vec / np.linalg.norm(planar_vec)
        orth_vec = orth_vec / np.linalg.norm(orth_vec)

        roll_rad = np.arctan2(
            np.cross(planar_vec, orth_vec),
            np.dot(planar_vec, orth_vec)
        )
        roll_deg = np.degrees(roll_rad)

        # Pitch Calculation
        mid_to_mid = np.array([features["mid_mouth"][:2]]) - np.array([features["eye_midpoint"][:2]])
        eye_mid_to_nose = np.array([features["nose"][:2]]) - np.array([features["eye_midpoint"][:2]])

        A = eye_mid_to_nose.ravel()
        B = mid_to_mid.ravel()
        pitch_proj = np.dot(A, B) / np.linalg.norm(B)
        normalized_position = pitch_proj / np.linalg.norm(mid_to_mid) - 0.16

        denom_down = 0.5 - normalized_position
        denom_up = 0.5 + normalized_position
        asymmetry_ratio_pitch = (denom_down / denom_up) if denom_up != 0 else float('inf')
        pitch_deg = asymmetry_ratio_pitch * 180

        print(f"pitch: {pitch_deg:.2f} ")
        print(f"roll: {roll_deg:.2f} ")
        print(f"yaw: {yaw_deg:.2f}")

        # Compose rotation matrix (Z-Y-X extrinsic: yaw -> pitch -> roll)
        rvec = np.array([np.radians(pitch_deg), np.radians(roll_deg), np.radians(yaw_deg)])
        R, _ = cv2.Rodrigues(rvec)

        # Draw axis on face
        axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        directions = R @ axes
        directions /= np.linalg.norm(directions, axis=0)
        directions *= 50

        nose_pos = (int(features["nose"][0]), int(features["nose"][1]))
        axis_labels = ['X', 'Y', 'Z']
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        offset = 5

        for i in range(3):
            dx, dy, _ = directions[:, i]
            end = (int(nose_pos[0] + dx), int(nose_pos[1] + dy))
            color = colors[i]
            cv2.arrowedLine(img2, nose_pos, end, color, 2, tipLength=0.3)
            label_x = end[0] + offset
            label_y = end[1] - offset
            label = axis_labels[i]
            cv2.putText(img2, label, (label_x, label_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Display custom angles
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_roll = f"Roll: {roll_deg:.2f}"
        text_pitch = f"Pitch: {pitch_deg:.2f}"
        text_yaw = f"Yaw: {yaw_deg:.2f}"
        cv2.putText(img2, text_pitch, (5, 30), font, font_scale, (255, 255, 255), thickness+1)
        cv2.putText(img2, text_yaw, (5, 30+30), font, font_scale, (255, 255, 255), thickness+1)
        cv2.putText(img2, text_roll, (5, 30+60), font, font_scale, (255, 255, 255), thickness+1)

    except KeyError:
        print("a feature is not found")
        return  # Exit if any required feature is missing


    """
    Mediapipe Implementation for Comparison
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for MediaPipe and display

    if img is None:
        raise ValueError("Image not found.")
    original = img.copy()
    h, w = img.shape[:2]

    '''
    2. MediaPipe Face Mesh
    '''
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

    '''
    3. 3D model points – Y positive DOWN (matches image coordinates)
    '''
    model_pts = np.array([
        [ 0.0,   0.0,   0.0],   # 0 – Nose tip
        [ 0.0, -63.6, -12.5],   # 1 – Chin
        [ 43.3,  32.7, -26.0],  # 2 – Left eye outer
        [-43.3,  32.7, -26.0],  # 3 – Right eye outer
        [ 28.9, -28.9, -24.1],  # 4 – Left mouth outer
        [-28.9, -28.9, -24.1],  # 5 – Right mouth outer
    ], dtype=np.float64)

    '''
    4. 2D image points
    '''
    img_pts = np.array([
        [landmarks.landmark[4].x   * w, landmarks.landmark[4].y   * h],  # Nose tip
        [landmarks.landmark[152].x * w, landmarks.landmark[152].y * h],  # Chin
        [landmarks.landmark[33].x  * w, landmarks.landmark[33].y  * h],  # Left eye outer
        [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],  # Right eye outer
        [landmarks.landmark[61].x  * w, landmarks.landmark[61].y  * h],  # Left mouth outer
        [landmarks.landmark[291].x * w, landmarks.landmark[291].y * h],  # Right mouth outer
    ], dtype=np.float64)

    '''
    # 5. Camera intrinsics
    '''
    focal = 1.5 * w
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    '''
    # 6. solvePnP
    '''
    success, rvec, tvec = cv2.solvePnP(
        model_pts, img_pts, cam_mat, dist,
        flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        raise RuntimeError("solvePnP failed.")

    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_to_head = R_world_to_cam.T

    '''
    7. Euler angles (Z-Y-X)
    '''# ------------------------------------------------------------------ #'''
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

    pitch_deg = math.degrees(pitch)
    yaw_deg   = math.degrees(yaw)
    roll_deg  = math.degrees(roll)

    '''
    8. Local unit vectors
    '''
    axes_local = np.array([
        [1, 0, 0],   # X – right
        [0, 1, 0],   # Y – down
        [0, 0, 1]    # Z – forward
    ], dtype=np.float32)

    '''
    9. Transform to camera space
    '''
    arrow_len = 50.0
    dirs_cam = (R_cam_to_head @ axes_local.T).T
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1)[:, None]
    dirs_cam *= arrow_len
    offsets_2d = np.round(dirs_cam[:, :2]).astype(int)

    nose_2d = tuple(img_pts[0].astype(int))

    '''
    10. Draw arrows
    '''
    colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, (dx, dy) in enumerate(offsets_2d):
        end = (nose_2d[0] + dx, nose_2d[1] + dy)
        cv2.arrowedLine(original, nose_2d, end, colors_bgr[i], thickness=4, tipLength=0.25)

    # ------------------------------------------------------------------ #
    # 11. Text overlay
    # ------------------------------------------------------------------ #
    cv2.putText(original, f'Pitch: {pitch_deg:+.2f} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Yaw:   {yaw_deg:+.2f} deg', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Roll:  {roll_deg:+.2f} deg', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    '''
    # 12. Display and save both media pipe and develop algorithm outputs
    '''
    desired_width = 800
    desired_height = 600

    mine_window_name = "Mine"
    cv2.namedWindow(mine_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mine_window_name, desired_width, desired_height)
    cv2.imshow(mine_window_name, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    media_pipe_window_name = "media pipe"
    cv2.namedWindow(media_pipe_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(media_pipe_window_name, desired_width, desired_height)
    cv2.imshow(media_pipe_window_name, original)

    cv2.imwrite(folder_path + '/' + name[:-4] + "_30_" + "_media.png", original)
    cv2.imwrite(folder_path + '/' + name[:-4] + "_30_" + "_mine.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    13. Return MediaPipe angles
    '''
    return {
        'angles': {'pitch': pitch_deg, 'yaw': yaw_deg, 'roll': roll_deg},
    }


if __name__ == "__main__":


    """
    input the name of the RGB image and the code will search for the associated pickle file that contains a height map, ensure both files are
    within the working directory.

    Use Pickle_img to generate images and pickle files
    """
    
    i = 28
    image_name = f"jestinp_{i}_-30roll_rgb.png"
    pickle_name = image_name[0:-7] + "height.pkl"

    # Load Haar cascade classifiers
    face_cascade       = load_cascade('haarcascade_frontalface_default.xml')
    eye_cascade        = load_cascade('haarcascade_eye.xml')
    mouth_cascade      = load_cascade('haarcascade_mcs_mouth.xml')
    nose_cascade       = load_cascade('haarcascade_mcs_nose.xml')

    folder_path = os.getcwd()  # Use current directory
    start_time = time.time()
    
    

    # Code looks for the RGB image file and its associated .pkl file.
    try:
        image = cv2.imread(os.path.join(folder_path, image_name))
    except FileNotFoundError:
        print(f"Error: the file {image_name} not found")


    try:
        with open(pickle_name, 'rb') as file:
            loaded_data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: The file {pickle_name} was not found.")
    except pickle.PicklingError:
        print("Error: Failed to load the pickle file.")

    # Execute head pose estimation
    GetHeadPose(face_cascade, eye_cascade, mouth_cascade, nose_cascade, image, loaded_data['height_m'], image_name, folder_path)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")