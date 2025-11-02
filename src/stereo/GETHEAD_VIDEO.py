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

        return img2
        # return

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
            # print("a left eye detected")
            # if center[1] < y + 3*h/5:
            if center[1] < y + h/2:
                # print("Left eye is in upper 1/2 of face")
                if 'eye_left' in features:
                    # print("Multiple left eyes detected, skipping")
                    continue

                left_eye_region = img1[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                left_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]

                # cv2.rectangle(img2, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

                left_eye_height_map = height_m[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                features['eye_left'] = [center[0], center[1], left_eye_height]
                cv2.circle(img2, center, 5, (0, 0, 255), 2)
                print("Left eye detected at:", center)
            
        if center[0] > x + w/2:
            # print("a right eye detected")

            # if center[1] < y + 3*h/5:
            if center[1] < y + h/2:
                # print("right eye is in upper 1/2 of face")
                if 'eye_right' in features:
                    # print("Multiple right eyes detected, skipping")
                    continue
                
                right_eye_region = img1[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                right_eye_height = height_m[y + ey + eh//2, x + ex + ew//2]
                
                right_eye_height_map = height_m[y + ey - exp : y + ey + eh + exp, x + ex - exp : x + ex + ew + exp]

                # cv2.rectangle(img2, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                features['eye_right'] = [center[0], center[1], right_eye_height]
                cv2.circle(img2, center, 5, (0, 0, 255), 2)
                print("Right eye detected at:", center)

    try:
        if abs(features['eye_left'][1] - features["eye_right"][1]) < (h/2 - h/3):
            print("EYES valid\n")

            left_eye_vec = np.array(features["eye_left"])[:2]
            right_eye_vec = np.array(features["eye_right"])[:2]

            eye_axis_vec = right_eye_vec - left_eye_vec

            perp_eye_axis_vec = np.array([-eye_axis_vec[1], eye_axis_vec[0]])
     
            print((left_eye_vec[0] + right_eye_vec[0]) // 2, (left_eye_vec[1] + right_eye_vec[1]) // 2)       
            features["eye_midpoint"] = [(left_eye_vec[0] + right_eye_vec[0]) // 2, (left_eye_vec[1] + right_eye_vec[1]) // 2, height_m[int((left_eye_vec[1] + right_eye_vec[1]) // 2), int((left_eye_vec[0] + right_eye_vec[0]) // 2)]]
   

            # features["eye_midpoint"] = np.array([int((left_eye_vec[0] + right_eye_vec[0]) // 2), int((left_eye_vec[1] + right_eye_vec[1]) // 2), height_m[int((left_eye_vec[0] + right_eye_vec[0]) // 2), int((left_eye_vec[1] + right_eye_vec[1]) // 2)]])


            planar_vec = np.array([0, 1])  # reference up direction (y-axis)


            theta = np.arctan2(
                np.cross(planar_vec, perp_eye_axis_vec),   # sine term (determines sign)
                np.dot(planar_vec, perp_eye_axis_vec)      # cosine term
            )

        else:
            print("EYES not valid\n")
            return img2
    except KeyError:
        print("didn't find two eyes")
        return img2
    # Finding face
    for (nx, ny, nw, nh) in nose:

        if (nw*nh)/(h*w) < 0.02:
            continue

        nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))
        # Verify nose position relative to eyes
        if ('eye_left' in features and 'eye_right' in features and
            nose_center[1] > features['eye_left'][1] and
            nose_center[1] > features['eye_right'][1] and
            nose_center[0] > features['eye_left'][0] and
            nose_center[0] < features['eye_right'][0]):
            # cv2.rectangle(img2, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (255, 0, 0), 2)
            # cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)
            # print("Nose detected at:", nose_center)
            # features['nose'] = nose_center
            # print(nw, nh)
            # print((nw*nh)/(h*w))
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

            # fig = plt.figure(figsize=(10, 30))
            # plt.subplot(311)
            # plt.imshow(nose_height_map, cmap='jet')
            # plt.subplot(312)
            # plt.imshow(img2)
            # plt.subplot(313)
            # plt.imshow(nose_height_map_wide, cmap='jet')

            # plt.imshow(img2)

        else:
            print("No nose position constraints not met.\n")

    try:
        print(features["nose"],"\n")
    except KeyError:
        print("No nose found\n")
        return img2


    # Finding Mouth
    exp_x = 0
    exp_y = 0

    # fig = plt.figure(figsize = (20,10))

    i = 0
    for (mx, my, mw, mh) in mouth: 

        # if i == 0:
        #     i = i + 1
        #     continue

        mouth_center = (int(x + mx + mw/2), int(y + my + mh/2))
        if mouth_center[1] < features['nose'][1] + 0.2*h:
            print("Mouth y higher than nose y, skipping")
            continue
        
        exp_x = 0
        exp_y = int(mh * 0.15)

        # mouth_region = img1[y + my - exp : y + my + mh + exp, x + mx - exp: x + mx + mw + exp]
        mouth_region = img1[y + my + exp_y : y + my + mh - exp_y, x + mx - exp_x: x + mx + mw + exp_x]
        # mouth_region = img1[y + my - exp : y + my + mh + exp, features["eye_left"][0] : features["eye_right"][0] ]


        cv2.circle(img2, mouth_center, 5, (0, 0, 255), 2)
        
        
        features['mouth'] = mouth_center
        print("Mouth detected at:", mouth_center)

        mouth_region_gray = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2GRAY)
        mouth_region_gray = cv2.equalizeHist(mouth_region_gray)

        mouth_hist = cv2.calcHist([mouth_region_gray], [0], None, [256], [0, 256])

        mean = np.mean(mouth_hist)
        stddev = np.std(mouth_hist)
        min = np.min(mouth_hist)
        max = np.max(mouth_hist)

        # contrast adjust the image
        A = min # some stretching parameters
        B = mean + 1 * stddev
        C = min
        D = max

        mouth_region_gray_adjusted = ((D-C)/(B-A))*(mouth_region_gray-A)+C # apply contrast adjustment
        mouth_region_gray_adjusted = np.clip(mouth_region_gray_adjusted, 0, 255).astype('uint8') # clip to [0,255] and convert to uint8

        thresh_val = 240
        set_val = 255
        ret, thresh_im1 = cv2.threshold(mouth_region_gray_adjusted, thresh_val, set_val, cv2.THRESH_BINARY)

        # plt.imshow(thresh_im1, cmap = "gray")

        mouth_region_binary_dilate= cv2.dilate(thresh_im1, None, iterations=1)

        # plt.imshow(mouth_region_binary_dilate, cmap = "gray")


        black_mask = mouth_region_binary_dilate == 0

        rows, cols = np.where(black_mask)

        min_col_idx = np.argmin(cols)
        leftmost_row = rows[min_col_idx]
        leftmost_col = cols[min_col_idx]

        left_mouth = np.array([x + mx + leftmost_col + exp_x, y + my + leftmost_row + exp_y])
        # left_mouth = np.array([x  + leftmost_col, y + leftmost_row])


        min_col_idx = np.argmax(cols)
        rightmost_row = rows[min_col_idx]
        rightmost_col = cols[min_col_idx]

        right_mouth = np.array([x + mx + rightmost_col - exp_x, y + my + rightmost_row + exp_y])
        # right_mouth = np.array([x + rightmost_col, y + rightmost_row])


        features["right_mouth"] = right_mouth
        features["left_mouth"] = left_mouth
        features["mid_mouth"] = np.array([int((right_mouth[0] + left_mouth[0])//2), int((right_mouth[1] + left_mouth[1])//2), height_m[int((right_mouth[1] + left_mouth[1])//2), int((right_mouth[0] + left_mouth[0])//2)]])
        # plt.imshow(mouth_region_binary_dilate, cmap="gray")
        # plt.imshow(mouth_region_gray_adjusted, cmap ="gray")
        # plt.imshow(thresh_im1, cmap="gray")
        # break

        cv2.circle(img2, (left_mouth[0], left_mouth[1]), 2, (0, 255, 255), 1)
        # cv2.circle(img2, (x + mx + leftmost_col, y + my + leftmost_row), 2, (0, 255, 255), 1)

        cv2.circle(img2, (right_mouth[0], right_mouth[1]), 2, (0, 255, 255), 1)

        # print(mh, mw)

        # plt.imshow(img2)
        break

    try:
        print(features["right_mouth"], "right mouth")
        print(features["left_mouth"], "left mouth\n")
        # plt.imshow(img2)
    except KeyError:
        print("no valid mouth\n")
        return img2

    
    # Drawing lines
    cv2.line(img2, (int(features["eye_left"][0]), int(features["eye_left"][1])), (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)

    cv2.line(img2, (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["eye_left"][0]), int(features["eye_left"][1])), (0, 255, 0), 1)

    cv2.line(img2, (int(features["eye_left"][0]), int(features["eye_left"][1])), (int(features["left_mouth"][0]), int(features["left_mouth"][1])), (0, 255, 0), 1)
    cv2.line(img2, (int(features["eye_right"][0]), int(features["eye_right"][1])), (int(features["right_mouth"][0]), int(features["right_mouth"][1])), (0, 255, 0), 1)

    # plt.imshow(img2)

    # YAW calculation

    left_eye_vec = np.array(features["eye_left"])[:2]
    right_eye_vec = np.array(features["eye_right"])[:2]
    nose_vec = np.array([features["nose"]])[0][:2]
    # print("huh")

    eye_axis_vec = right_eye_vec - left_eye_vec

    inter_eye_distance = np.linalg.norm(eye_axis_vec)
    if inter_eye_distance == 0:
        raise ValueError("Left and right eye coordinates are identical.")

    # print(inter_eye_distance)

    # print(inter_eye_distance, " - inter eye distance along the axis of the eyes")
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
    right_to_nose = nose_vec - right_eye_vec

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
    asymmetry_ratio_yaw = (denom_right / denom_left) if denom_left != 0 else float('inf')

    print(asymmetry_ratio_yaw, " - asymmetry ratio (right/left)")
    yaw_deg = asymmetry_ratio_yaw * 180
    # print(asymmetry_ratio * 180, " - yaw in degrees (approximate)")
    # print(np.degrees(np.arcsin(asymmetry_ratio)), " - yaw in degrees (approximate)")

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
    # print(roll_deg, "Head roll (signed)")

    # Pitch Calculation

    mid_to_mid = np.array([features["mid_mouth"][:2]]) - np.array([features["eye_midpoint"][:2]])
    # print(np.array([features["nose"][:2]]))
    # print(np.array([features["eye_midpoint"][:2]]))
    eye_mid_to_nose = np.array([features["nose"][:2]]) - np.array([features["eye_midpoint"][:2]])


    # Flatten to 1D vectors
    A = eye_mid_to_nose.ravel()   # or eye_mid_to_nose[0]
    B = mid_to_mid.ravel()        # or mid_to_mid[0]

    # Scalar projection of A onto B
    pitch_proj = np.dot(A, B) / np.linalg.norm(B)

    # print(mid_to_mid)
    # print(eye_mid_to_nose)

    # print(pitch_proj)

    # Normalized position along the axis (-0.5 to +0.5 for centered nose)
    normalized_position = pitch_proj /  np.linalg.norm(mid_to_mid)
    # print(normalized_position, " - inter eye distance to nose displacement ratio from right eye")

    # Asymmetry ratio (optional alternative metric)
    # Maps normalized_position = 0 → 1.0 (symmetric)
    # Positive values indicate shift toward right eye
    denom_down = 0.5 - normalized_position
    denom_up = 0.5 + normalized_position
    asymmetry_ratio_pitch = (denom_down / denom_up) if denom_up != 0 else float('inf')

    pitch_deg = asymmetry_ratio_pitch * 180

    # print(pitch_deg, "pitch")
    # print(asymmetry_ratio, " - asymmetry ratio (right/left)")
    # print(asymmetry_ratio_pitch * 180, " - pitch in degrees (approximate)")
    # print(np.degrees(np.arcsin(asymmetry_ratio)), " - yaw in degrees (approximate)")

    # Plotting midpoints
    # print(-features["eye_midpoint"][1] + features["mid_mouth"][1])

    eye_midpoint = features["eye_midpoint"]
    # t1 = f"{int(eye_midpoint[0])}, {int(eye_midpoint[1])}"
    # cv2.circle(img2, (int(features["eye_midpoint"][0]), int(features["eye_midpoint"][1])), 2, (255, 0, 0), 5)
    # cv2.putText(img2, t1, (int(eye_midpoint[0])-5, int(eye_midpoint[1])-5), font, font_scale, (0, 0, 0), thickness+1)

    mouth_midpoint = features["mid_mouth"]
    # t2 = f"{int(mouth_midpoint[0])}, {int(mouth_midpoint[1])}"
    # cv2.circle(img2, (int(features["mid_mouth"][0]), int(features["mid_mouth"][1])), 2, (255, 0, 0), 5)
    # cv2.putText(img2, t2, (int(mouth_midpoint[0])-5, int(mouth_midpoint[1])-5), font, font_scale, (0, 0, 0), thickness+1)

    # fig = plt.figure(figsize = (20,10))
    # plt.imshow(img2)

    print(f"pitch: {pitch_deg:.2f} ")
    print(f"roll: {roll_deg:.2f} ")
    print(f"yaw: {yaw_deg:.2f}")

    # Compose rotation matrix (extrinsic Z-Y-X: yaw -> pitch -> roll)
    rvec = np.array([np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)])
    R, _ = cv2.Rodrigues(rvec)  # Single 3x3 matrix from Rodriguez formula

    # Unit axes in local frame (X forward, Y down, Z right for head pose convention)
    axes = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    directions = R @ axes
    directions /= np.linalg.norm(directions, axis=0)  # Normalize
    directions *= 50  # Arrow length

    # Project to 2D and draw (BGR colors: X-red, Y-green, Z-blue)
    nose_pos = (int(features["nose"][0]), int(features["nose"][1]))
    for i in range(3):
        dx, dy, _ = directions[:, i]
        end = (int(nose_pos[0] + dx), int(nose_pos[1] + dy))
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i]
        cv2.arrowedLine(img2, nose_pos, end, color, 2, tipLength=0.3)

    # Rx = np.array([
    #     [1, 0, 0], 
    #     [0, math.cos(math.radians(roll_deg)), -math.sin(math.radians(roll_deg))], 
    #     [0, math.sin(math.radians(roll_deg)), math.cos(math.radians(roll_deg))]
    #     ])

    # Ry = np.array([
    #     [math.cos(math.radians(pitch_deg)), 0, math.sin(math.radians(pitch_deg))],
    #     [0, 1, 0],
    #     [-math.sin(math.radians(pitch_deg)), 0, math.cos(math.radians(pitch_deg))]
    #     ])

    # Rz = np.array([
    #     [math.cos(math.radians(yaw_deg)), -math.sin(math.radians(yaw_deg)), 0],
    #     [math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg)), 0],
    #     [0, 0, 1]
    #     ])

    # R = Rz @ Ry @ Rx

    # arrow_length = 50

    # # Define unit vectors in the local frame
    # axes = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]], 
    #     dtype=np.float32)  # X, Y, Z

    # # Transform and scale axes
    # directions = (R @ axes.T).T  # Shape: (3, 3)
    # directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize
    # directions *= arrow_length  # Scale to desired length

    # # Convert to pixel endpoints
    # endpoints = np.round(directions[:, :2]).astype(int)  # Only x, y components
    # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue

    # thickness = 5

    # for i, (dx, dy) in enumerate(endpoints):
    #     end_x = features["nose"][0] + dx
    #     end_y = features["nose"][1] + dy
    #     color = colors[i]
    #     cv2.arrowedLine(img2, (features["nose"][0], features["nose"][1]), (end_x, end_y), color, thickness, tipLength=0.3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_roll = f"Roll: {roll_deg:.2f}"
    text_pitch = f"Pitch: {pitch_deg:.2f}"
    text_yaw = f"Yaw: {yaw_deg:.2f}"
    cv2.putText(img2, text_pitch, (5, 30), font, font_scale, (255, 255, 255), thickness+1)
    cv2.putText(img2, text_yaw, (5, 30+30), font, font_scale, (255, 255, 255), thickness+1)
    cv2.putText(img2, text_roll, (5, 30+60), font, font_scale, (255, 255, 255), thickness+1)

    # # Display the annotated image and keep the window open until a key is pressed
    # cv2.imshow("head", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))  # 'img2' is your annotated image (ensure it is in BGR format for display)
    # cv2.waitKey(0)  # Blocks until any key is pressed; use 0 for indefinite wait
    # cv2.destroyAllWindows()  # Cleanly close the window after key press
    return img2

if __name__ == "__main__":

    # folder_path = "AFLW2000-3D/AFLW2000/"
    folder_path = os.getcwd()
    # png_files = glob.glob(os.path.join(folder_path, "*.png"))

    pickle_name = "jestin_video_1.pkl"
    try:
        with open(pickle_name, 'rb') as file:
            data = pickle.load(file)
        # print("Loaded data:", loaded_data)
        # print(loaded_data["height_m"].shape)
        # print(type(loaded_data["height_m"]))
    except FileNotFoundError:
        print(f"Error: The file {pickle_name} was not found.")
        # continue
    except pickle.PicklingError:
        print("Error: Failed to load the pickle file.")
        # continue
    # ------------------------------------------------------------------
    # Load every classifier
    # ------------------------------------------------------------------
    face_cascade       = load_cascade('haarcascade_frontalface_default.xml')
    eye_cascade        = load_cascade('haarcascade_eye.xml')
    mouth_cascade      = load_cascade('haarcascade_mcs_mouth.xml')
    nose_cascade       = load_cascade('haarcascade_mcs_nose.xml')


    rgb_frames = data['rgb_frames']       # List of BGR uint8 arrays (480x848x3)
    height_frames = data['height_frames'] # List of float32 arrays (480x640) in meters; NaN for invalid

    # Optional: Retrieve metadata if needed
    plane_point = data.get('plane_point')
    plane_normal = data.get('plane_normal')
    units = data.get('units', 'meters')
    height_map_shape = data.get('height_map_shape', (480, 640))

    print(f"Loaded {len(rgb_frames)} frames from '{pickle_name}'.")
    if len(rgb_frames) != len(height_frames):
        raise ValueError("Mismatch between RGB and height frame counts.")
    print(f"Height maps: {height_map_shape[0]}x{height_map_shape[1]} resolution, in {units} (signed distance; NaN for invalid).")

    # Playback parameters
    fps = 30  # Target frames per second (match capture rate)
    delay_ms = int(1000 / fps)  # Delay between frames in milliseconds

    # Playback loop
    print("Starting playback. Press 'q' to quit, 'p' to pause/resume.")
    paused = False
    frame_idx = 0

    while frame_idx < len(rgb_frames):
        if not paused:
            rgb = rgb_frames[frame_idx]
            height = height_frames[frame_idx]

            image = GetHeadPose(face_cascade, eye_cascade, mouth_cascade, nose_cascade, rgb, height)
            cv2.imshow("head pose", image)   
            
            # cv2.putText(rgb_display, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # # Display windows (side-by-side if desired; here stacked for single window)
            # combined = np.hstack((rgb_display, height_colormap)) if resize_factor == 1.0 else np.hstack((rgb_display, height_colormap))
            # cv2.imshow('RGB (Left) | Height Map (Right) - JET Colormap', combined)
            
            frame_idx += 1
            if frame_idx >= len(rgb_frames):
                print("Playback complete.")
                break
        
        # Handle key presses
        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF
        if key == ord('q'):
            print("Playback quit by user.")
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):  # Rewind to start
            frame_idx = 0
            print("Rewound to start.")
        
        # Manual frame navigation when paused
        if paused:
            if key == ord('n'):  # Next frame
                frame_idx = min(frame_idx + 1, len(rgb_frames) - 1)
            elif key == ord('b'):  # Previous frame
                frame_idx = max(frame_idx - 1, 0)

    cv2.destroyAllWindows()
    print("Playback window closed.")

    # GetHeadPose(face_cascade, eye_cascade, mouth_cascade, nose_cascade, head["im_rgb"], head["im_height"])