import os
# Set Qt platform before importing cv2 to avoid plugin errors (use 'xcb' for X11)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import re
import time
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import cv2
import sys
import os


# ------------------------------------------------------------------
# Helper to load a cascade with a clear error message
# ------------------------------------------------------------------

def load_cascade(name):
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


cascade_dir = os.getcwd()              # e.g. /home/jestin/AMME5710_Major/src/stereo


# ------------------------------------------------------------------
# 3. Load every classifier
# ------------------------------------------------------------------
face_cascade       = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade        = load_cascade('haarcascade_eye.xml')
mouth_cascade      = load_cascade('haarcascade_mcs_mouth.xml')
nose_cascade       = load_cascade('haarcascade_mcs_nose.xml')
profileface_cascade= load_cascade('haarcascade_profileface.xml')
left_ear_cascade   = load_cascade('haarcascade_mcs_leftear.xml')
right_ear_cascade  = load_cascade('haarcascade_mcs_rightear.xml')

# ------------------------------------------------------------------
# 4. Final sanity check
# ------------------------------------------------------------------
classifiers = [
    ('Face',          face_cascade),
    ('Eye',           eye_cascade),
    ('Mouth',         mouth_cascade),
    ('Nose',          nose_cascade),
    ('Profile Face',  profileface_cascade),
    ('Left Ear',      left_ear_cascade),
    ('Right Ear',     right_ear_cascade),
]

failed = [name for name, clf in classifiers if clf.empty()]
if failed:
    raise RuntimeError(f"Cascade loading failed for: {', '.join(failed)}")
else:
    print("\nAll cascade classifiers loaded successfully.\n")


# Configure RealSense streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Check for device
ctx = rs.context()
if len(ctx.devices) == 0:
    print("No RealSense device found. Connect the camera and try again.")
    raise SystemExit(1)

pipeline.start(config)

# Obtain depth scale (meters per depth unit)
try:
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} m/unit")
except Exception:
    depth_scale = 1.0
    print("Warning: could not read depth scale, defaulting to 1.0")

# Align depth to color
align = rs.align(rs.stream.color)



try:
    timeout_count = 0
    max_timeouts = 3

    while True:
        try:
            frames = pipeline.wait_for_frames(5000)
        except RuntimeError as e:
            msg = str(e).lower()
            if "frame didn't arrive" in msg or "timeout" in msg:
                timeout_count += 1
                print(f"Warning: frame timeout ({timeout_count}/{max_timeouts})")
                if timeout_count >= max_timeouts:
                    print("Restarting pipeline to recover from repeated timeouts...")
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    pipeline.start(config)
                    timeout_count = 0
                continue
            else:
                raise

        timeout_count = 0

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        aligned = align.process(frames)
        aligned_depth = aligned.get_depth_frame()
        aligned_color = aligned.get_color_frame()

        depth_image = np.asanyarray(aligned_depth.get_data())
        color_image = np.asanyarray(aligned_color.get_data())

        img2 = color_image.copy()
        img1 = color_image.copy()
        height_m = depth_image.astype(np.float32) * float(depth_scale)


        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("No frontal face detected, trying profile face detector")
            faces = profileface_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("No faces detected")
            continue

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

        features = {'face': (x, y, w, h)}

        # Draw reference lines
        cv2.line(img2, (x + int(w/2), y), (x + int(w/2), y + h), (255, 255, 255), 1)
        cv2.line(img2, (x, y + int(h/3)), (x + w, y + int(h/3)), (255, 255, 255), 1)
        cv2.line(img2, (x, y + int(h/2)), (x + w, y + int(h/2)), (255, 255, 255), 1)


        exp = 10

        for (ex, ey, ew, eh) in eyes:
            center = (int(x + ex + ew/2), int(y + ey + eh/2))

            if center[0] < x + w/2:
                print("a left eye detected")
                if center[1] < y + 3*h/5:
                    print("Left eye is in upper 3/5 of face")
                    if 'eye_left' in features:
                        print("Multiple left eyes detected, skipping")
                        continue

                    left_eye_region = img1[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                    left_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]

                    cv2.rectangle(img2, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

                    left_eye_height_map = height_m[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                    features['eye_left'] = [center[0], center[1], left_eye_height]
                    cv2.circle(img2, center, 5, (0, 255, 0), 2)
                    print("Left eye detected at:", center)
                
            if center[0] > x + w/2:
                print("a right eye detected")

                if center[1] < y + 3*h/5:
                    if 'eye_right' in features:
                        print("Multiple right eyes detected, skipping")
                        continue
                    
                    right_eye_region = img1[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                    right_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]
                    
                    right_eye_height_map = height_m[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                    cv2.rectangle(img2, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                    features['eye_right'] = [center[0], center[1], right_eye_height]
                    cv2.circle(img2, center, 5, (0, 255, 0), 2)
                    print("Right eye detected at:", center)

        if features.get('eye_left') is None or features.get('eye_right') is None:
            print("Both eyes not detected, skipping face detection")
            continue

        for (nx, ny, nw, nh) in nose:
            nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))
            # Verify nose position relative to eyes
            if ('eye_left' in features and 'eye_right' in features and
                nose_center[1] > features['eye_left'][1] and
                nose_center[1] > features['eye_right'][1] and
                nose_center[0] > features['eye_left'][0] and
                nose_center[0] < features['eye_right'][0]):
                cv2.rectangle(img2, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (255, 0, 0), 2)
                # cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)
                # print("Nose detected at:", nose_center)
                # features['nose'] = nose_center

                
                # Extract nose height map (correct coordinates)
                nose_height_map = height_m[y + ny:y + ny + nh, x + nx:x + nx + nw]

                expand = 35
                nose_height_map_wide = height_m[y + ny - expand : y + ny + nh + expand, x + nx - expand : x + nx + nw + expand]

                

                # print(nose_height_map[0:10, :])
                min_index = np.argmin(nose_height_map)  # Result: 1 (position of 1)
                row, col = np.unravel_index(min_index, nose_height_map.shape)  # Result: (0, 1)

                print("Tip of Nose detected at:", row, col)

                nose_depth = np.min(nose_height_map)

                nose_center = (int(x + nx + col), int(y + ny + row))
                cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)
                print("Nose detected at:", nose_center)
                features['nose'] = [nose_center[0], nose_center[1], nose_depth]

            else:
                print("No nose detected or position constraints not met.")

        if features.get('nose') is None:
            print("Nose not detected, skipping face detection")
            continue
        
        # Yaw calculation
        left_eye_vec = np.array(features["eye_left"])[:2]
        right_ear_vec = np.array(features["eye_right"])[:2]
        nose_vec = np.array([features["nose"]])[0][:2]

        eye_axis_vec = right_ear_vec - left_eye_vec

        inter_eye_distance = np.linalg.norm(eye_axis_vec)
        if inter_eye_distance == 0:
            raise ValueError("Left and right eye coordinates are identical.")

        # Unit vector along the eye axis
        eye_axis_unit = eye_axis_vec / inter_eye_distance

        # Vector from left eye to nose
        left_to_nose = nose_vec - left_eye_vec

        # Scalar projection of nose position onto the eye axis
        nose_projection = np.dot(left_to_nose, eye_axis_unit)

        # print(nose_projection, " - distance from left eye to nose along the axis of the eyes")
        # Normalized position along the axis (-0.5 to +0.5 for centered nose)
        normalized_position = nose_projection / inter_eye_distance
        # print(normalized_position, " - inter eye distance to nose displacement ratio from left eye")

        # Vector from left eye to nose
        right_to_nose = nose_vec - right_ear_vec

        # Scalar projection of nose position onto the eye axis
        nose_projection = np.dot(right_to_nose, eye_axis_unit)

        # print(nose_projection, " - distance from the right eye to the nose along the axis of the eyes")
        # Normalized position along the axis (-0.5 to +0.5 for centered nose)
        normalized_position = nose_projection / inter_eye_distance
        # print(normalized_position, " - inter eye distance to nose displacement ratio from right eye")

        # Asymmetry ratio (optional alternative metric)
        # Maps normalized_position = 0 → 1.0 (symmetric)
        # Positive values indicate shift toward right eye
        denom_left = 0.5 - normalized_position
        denom_right = 0.5 + normalized_position
        asymmetry_ratio = (denom_right / denom_left) if denom_left != 0 else float('inf')

        # print(asymmetry_ratio, " - asymmetry ratio (right/left)")
        print(asymmetry_ratio * 180, " - yaw in degrees (approximate)")

        # roll calculation
        planar_vec = np.array([0, 1])  # reference up direction (y-axis)
        orth_vec = np.array([-eye_axis_unit[1], eye_axis_unit[0]])  # perpendicular to eye axis

        # Normalize both vectors (just to be safe)
        planar_vec = planar_vec / np.linalg.norm(planar_vec)
        orth_vec = orth_vec / np.linalg.norm(orth_vec)

        # Compute signed angle using atan2
        roll_rad = np.arctan2(
            np.cross(planar_vec, orth_vec),   # sine term (determines sign)
            np.dot(planar_vec, orth_vec)      # cosine term
        )

        roll_deg = np.degrees(roll_rad)
        print("Head roll (signed):", roll_deg)

        roll_text = f"Roll: {roll_deg:.2f} deg"
        cv2.putText(img2, roll_text, (10, 30), font, font_scale, (0, 0, 255), thickness)

        yaw_text = f"Yaw (approx): {asymmetry_ratio * 180:.2f} deg"
        cv2.putText(img2, yaw_text, (10, 60), font, font_scale, (0, 0, 255), thickness)

        # Annotate corners with coordinates
        # tl = f"({x},{y})"
        # tr = f"({x+w},{y})"
        # bl = f"({x},{y+h})"
        # br = f"({x+w},{y+h})"



        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.5
        # thickness = 1
        # cv2.putText(img2, tl, (x-5, y-5), font, font_scale, (0, 0, 0), thickness+1)

        cv2.imshow('Color Stream', img2)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow('Depth Stream', depth_colormap)


        key = cv2.waitKey(1) & 0xFF
        # if key == ord('k'):
        #     rgb_name = f"jestin_{save_index}_rgb.png"
        #     pickle_name = f"jestin_{save_index}_height.pkl"

        #     # Save RGB image (BGR)
        #     cv2.imwrite(rgb_name, color_image)

        #     # Create height map (meters). Mark invalid depth (0) as NaN.
        #     height_map = depth_image.astype(np.float32) * float(depth_scale)
        #     height_map[depth_image == 0] = np.nan

        #     # Save pickle with metadata and height map
        #     to_save = {
        #         'height_m': height_map,
        #         'depth_scale': float(depth_scale),
        #         'timestamp': time.time(),
        #         'shape': height_map.shape,
        #     }
        #     with open(pickle_name, 'wb') as f:
        #         pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        #     print(f"Saved {rgb_name} and {pickle_name}")
        #     save_index += 1

        if key == ord('q') or key == 27:
            break

finally:
    try:
        pipeline.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
