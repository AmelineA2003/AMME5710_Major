import math
import sys, os
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle


def show_nose(head):
    img = head["im_rgb"]
    height_m = head["im_height"]

    # Get original image dimensions
    original_height, original_width = img.shape[:2]
    resized_width, resized_height = 640, 360
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    img2 = cv2.resize(img, (resized_width, resized_height))
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("No frontal face detected, trying profile face detector")
        faces = profileface_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No profile face detected")
        return
    
    print(len(faces), "face(s) detected")
    
    for (x, y, w, h) in faces:
        # Map face ROI to original image
        x_orig = int(x * scale_x)
        y_orig = int(y * scale_y)
        w_orig = int(w * scale_x)
        h_orig = int(h * scale_y)
        
        # Head center in resized image
        center_x_resized = x + w/2
        center_y_resized = y + h/2
        
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
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img2[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        nose = nose_cascade.detectMultiScale(roi_gray)
        right_ear = right_ear_cascade.detectMultiScale(roi_gray)
        left_ear = left_ear_cascade.detectMultiScale(roi_gray)

        features = {}
        print("Face at:", (x, y, x+w, y+h))
        features['face'] = (x, y, w, h)

        # Draw reference lines relative to the face ROI
        cv2.line(img2, (x + int(w/2), y), (x + int(w/2), y + h), (255, 255, 255), 1)
        cv2.line(img2, (x, y + int(h/3)), (x + w, y + int(h/3)), (255, 255, 255), 1)
        cv2.line(img2, (x, y + int(h/2)), (x + w, y + int(h/2)), (255, 255, 255), 1)

        # Eye detection and depth calculation
        eye_depths = []
        for (ex, ey, ew, eh) in eyes:
            center = (int(x + ex + ew/2), int(y + ey + eh/2))
            # Map eye center to original image for depth
            eye_x_orig = int(center[0] * scale_x)
            eye_y_orig = int(center[1] * scale_y)
            eye_x_orig = max(0, min(eye_x_orig, original_width - 1))
            eye_y_orig = max(0, min(eye_y_orig, original_height - 1))
            eye_depth = height_m[eye_y_orig, eye_x_orig] if height_m[eye_y_orig, eye_x_orig] > 0 else 0
            
            if center[0] < x + w/2:
                if 'eye_left' in features:
                    print("Multiple left eyes detected, skipping")
                    continue
                features['eye_left'] = center
                eye_depths.append(eye_depth)
                print("Left eye detected at:", center, "Depth:", eye_depth)
                cv2.circle(img2, center, 5, (0, 255, 0), 2)
            else:
                if 'eye_right' in features:
                    print("Multiple right eyes detected, skipping")
                    continue
                features['eye_right'] = center
                eye_depths.append(eye_depth)
                print("Right eye detected at:", center, "Depth:", eye_depth)
                cv2.circle(img2, center, 5, (0, 255, 0), 2)

        # Calculate average eye depth
        avg_eye_depth = np.mean(eye_depths) if eye_depths else 0
        print("Average eye depth:", avg_eye_depth)

        # Initialize Euler angles
        yaw_deg, pitch_deg, roll_deg = 0, 0, 0
        nose_center = None
        nose_depth = 0
        center_depth = 0

        for (nx, ny, nw, nh) in nose:
            # Original nose center
            nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))
            # Map to original image for depth
            nose_x_orig = int(nose_center[0] * scale_x)
            nose_y_orig = int(nose_center[1] * scale_y)
            nose_x_orig = max(0, min(nose_x_orig, original_width - 1))
            nose_y_orig = max(0, min(nose_y_orig, original_height - 1))
            nose_depth = height_m[nose_y_orig, nose_x_orig] if height_m[nose_y_orig, nose_x_orig] > 0 else 0

            # Refine nose tip using height map
            nose_height_area = height_m[nose_y_orig:nose_y_orig+int(nh*scale_y), nose_x_orig:nose_x_orig+int(nw*scale_x)]
            nose_height_values = nose_height_area[nose_height_area > 0]
            if nose_height_values.size > 0:
                min_index_flat = np.argmin(nose_height_values)
                min_index_2d = np.unravel_index(min_index_flat, nose_height_area.shape)
                nose_tip_y, nose_tip_x = min_index_2d
                nose_tip_x_resized = int(x + nx + (nose_tip_x / scale_x))
                nose_tip_y_resized = int(y + ny + (nose_tip_y / scale_y))
                nose_center = (nose_tip_x_resized, nose_tip_y_resized)
                nose_x_orig = int(nose_center[0] * scale_x)
                nose_y_orig = int(nose_center[1] * scale_y)
                nose_x_orig = max(0, min(nose_x_orig, original_width - 1))
                nose_y_orig = max(0, min(nose_y_orig, original_height - 1))
                nose_depth = height_m[nose_y_orig, nose_x_orig] if height_m[nose_y_orig, nose_x_orig] > 0 else nose_depth
                print("Nose tip detected at (resized):", nose_center, "Depth:", nose_depth)
            else:
                print("No valid height values in nose region")

            cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)
            features['nose'] = nose_center

            # Calculate head center depth
            if avg_eye_depth > 0:
                depth_diff = nose_depth - avg_eye_depth
                center_depth = nose_depth + 3 * depth_diff
                print(f"Depth difference (nose - avg eyes): {depth_diff}, Estimated head center depth: {center_depth}")
            else:
                print("No valid eye depths, using nose depth as fallback")
                center_depth = nose_depth

            # Calculate Euler angles
            dx = nose_center[0] - center_x_resized
            dy = nose_center[1] - center_y_resized
            dz = nose_depth - center_depth
            yaw = math.atan2(dx, -dz)
            pitch = math.atan2(dy, -dz)
            yaw_deg = math.degrees(yaw)
            pitch_deg = math.degrees(pitch)

            # Roll estimation using eyes
            if 'eye_left' in features and 'eye_right' in features:
                eye_dx = features['eye_right'][0] - features['eye_left'][0]
                eye_dy = features['eye_right'][1] - features['eye_left'][1]
                roll = math.atan2(eye_dy, eye_dx)
                roll_deg = math.degrees(roll)

            print(f"Euler Angles: Yaw={yaw_deg:.2f}°, Pitch={pitch_deg:.2f}°, Roll={roll_deg:.2f}°")

        # Draw XYZ reference frame arrows
        if nose_center is not None:
            # Define rotation matrix for head orientation
            cy, sy = math.cos(yaw), math.sin(yaw)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cr, sr = math.cos(roll), math.sin(roll)

            # Rotation matrix (Z-Y-X convention: yaw, pitch, roll)
            R = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp,   cp*sr,            cp*cr]
            ])

            # Define unit vectors for head's X, Y, Z axes
            axis_length = 50  # Length of arrows in pixels
            axes = np.array([
                [axis_length, 0, 0],  # X-axis
                [0, axis_length, 0],  # Y-axis
                [0, 0, axis_length]   # Z-axis
            ])

            # Rotate axes
            rotated_axes = np.dot(R, axes.T).T

            # Project to 2D (ignoring depth for visualization, scaling z to image plane)
            focal_length = 500  # Arbitrary focal length for projection
            for i, axis in enumerate(rotated_axes):
                dx = axis[0] * focal_length / (focal_length + axis[2]) if axis[2] > -focal_length else 0
                dy = axis[1] * focal_length / (focal_length + axis[2]) if axis[2] > -focal_length else 0
                end_point = (int(center_x_resized + dx), int(center_y_resized + dy))
                
                # Choose color: X=red, Y=green, Z=blue
                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i]
                cv2.arrowedLine(img2, (int(center_x_resized), int(center_y_resized)), 
                              end_point, color, 2, tipLength=0.3)
                # Label axes
                cv2.putText(img2, ['X', 'Y', 'Z'][i], end_point, font, font_scale, color, thickness)

        # Display Euler angles on image
        euler_text = f"Yaw: {yaw_deg:.2f}°\nPitch: {pitch_deg:.2f}°\nRoll: {roll_deg:.2f}°"
        # Split text into lines for better rendering
        lines = euler_text.split('\n')
        for i, line in enumerate(lines):
            cv2.putText(img2, line, (x, y - 30 + i*15), font, font_scale, (0, 0, 0), thickness+1)
            cv2.putText(img2, line, (x, y - 30 + i*15), font, font_scale, (255, 255, 255), thickness)

        for (mx, my, mw, mh) in mouth:
            mouth_center = (int(x + mx + mw/2), int(y + my + mh/2))
            if mouth_center[1] < features['nose'][1] + 0.1*h:
                print("Mouth y higher than nose y, skipping")
                continue
            cv2.circle(img2, mouth_center, 5, (0, 0, 255), 2)
            features['mouth'] = mouth_center
            print("Mouth detected at:", mouth_center)

    cv2.imshow('Faces', img2)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()


heads = {}
i = 0

# folder_path = "AFLW2000-3D/AFLW2000/"
folder_path = os.getcwd()
jpg_files = glob.glob(os.path.join(folder_path, "*.png"))
print(f"Found {len(jpg_files)} .png files in {folder_path}")

for image_path in jpg_files:
    image_name = os.path.basename(image_path)

    pickle_name = image_name[0:-7] + "height.pkl"
    
    try:
        with open(pickle_name, 'rb') as file:
            loaded_data = pickle.load(file)
        # print("Loaded data:", loaded_data)
        # print(loaded_data["height_m"].shape)
        # print(type(loaded_data["height_m"]))
    except FileNotFoundError:
        print(f"Error: The file {pickle_name} was not found.")
    except pickle.PicklingError:
        print("Error: Failed to load the pickle file.")

    image = cv2.imread(os.path.join(folder_path, image_name))
    # print("Image:", image_name, "Shape:", image.shape)
    
    sobel_x = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0, ksize=3) # Horizontal
    sobel_y =  cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=3) # Vertical
    
    heads[image_name] = {
        "im_rgb": image,
        "im_bgr": cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        "im_gray": cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
        "im_height": loaded_data["height_m"],
        "im_gray_gaussian": cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0),
        "im_sobel_x": sobel_x,  # Horizontal
        "im_sobel_y": sobel_y,  # Vertical
        "im_sobel_mag": cv2.normalize(np.sqrt(sobel_x**2 + sobel_y**2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    }
    
    # output_path = f"{image_name[0:-4]}_head_pose.jpg"  # Save annotated image for each
    print(f"\nProcessing {image_name}...")
    # result = GetHeadPose(image_path, output_path)
    # print("Head Pose Euler Angles:", result['angles'])
    # print("Rotation Matrix (Camera to Head Frame):\n", result['camera_to_head_rotation'])
    i = i + 1
    if i >= 10:
        break
    #       # Wait for a key press indefinitely
   
    

face_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_eye.xml'))
mouth_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_mouth.xml'))
nose_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_nose.xml'))
profileface_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_profileface.xml'))
left_ear_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_leftear.xml'))
right_ear_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_rightear.xml'))


# Check if classifiers loaded successfully
classifiers = [face_cascade, eye_cascade, mouth_cascade, nose_cascade, 
              profileface_cascade, left_ear_cascade, right_ear_cascade]
if any(c.empty() for c in classifiers):
    print("Error: One or more cascade classifiers failed to load")


show_nose(heads["jestin_7_rgb.png"])
# show_webcam(heads["T2_seq-123456789_dist-1.09_rgb_4.jpg"]["im_rgb"])