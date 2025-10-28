import sys, os
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import math


def gethead(head):
    img2 = head["im_rgb"].copy()  # Create a copy to avoid modifying the original
    height_m = head["im_height"].copy()

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No frontal face detected, trying profile face detector")
        faces = profileface_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No faces detected")
        return

    for (x, y, w, h) in faces:
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


       
        # Detect eyes
        for (ex, ey, ew, eh) in eyes:
            center = (int(x + ex + ew/2), int(y + ey + eh/2))
            if center[0] < x + w/2:
                if 'eye_left' in features:
                    print("Multiple left eyes detected, skipping")
                    continue
                features['eye_left'] = center
                cv2.circle(img2, center, 5, (0, 255, 0), 2)
                print("Left eye detected at:", center)
            else:
                if 'eye_right' in features:
                    print("Multiple right eyes detected, skipping")
                    continue
                features['eye_right'] = center
                cv2.circle(img2, center, 5, (0, 255, 0), 2)
                print("Right eye detected at:", center)

        # Detect nose and generate heatmap
        for (nx, ny, nw, nh) in nose:
            nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))
            # Verify nose position relative to eyes
            if ('eye_left' in features and 'eye_right' in features and
                nose_center[1] > features['eye_left'][1] and
                nose_center[1] > features['eye_right'][1] and
                nose_center[0] > features['eye_left'][0] and
                nose_center[0] < features['eye_right'][0]):
                cv2.rectangle(img2, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (255, 0, 0), 2)
                cv2.circle(img2, nose_center, 5, (255, 255, 0), 2)
                print("Nose detected at:", nose_center)
                features['nose'] = nose_center

                # Extract nose height map (correct coordinates)
                nose_height_map = height_m[y + ny:y + ny + nh, x + nx:x + nx + nw]

                # Check if nose_height_map is valid
                if nose_height_map.size == 0:
                    print("Invalid nose height map region")
                    continue

                # Normalize height map
                min_height = np.min(nose_height_map)
                max_height = np.max(nose_height_map)
                if max_height <= min_height:
                    print("Invalid height range in nose region")
                    continue

                normalized_height = (nose_height_map - min_height) / (max_height - min_height) * 255
                normalized_height = normalized_height.astype(np.uint8)

                # Apply colormap
                heatmap = cv2.applyColorMap(normalized_height, cv2.COLORMAP_JET)

                # Resize heatmap for better visibility
                heatmap = cv2.resize(heatmap, (300, 300), interpolation=cv2.INTER_LINEAR)

                # Create color bar
                colorbar_width = 70
                colorbar_height = 300
                colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
                for i in range(colorbar_height):
                    value = int((i / colorbar_height) * 255)
                    colorbar[i, :] = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]

                # Add depth value labels to color bar
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                cv2.putText(colorbar, f"{max_height:.2f}", (5, 20), font, font_scale, (0, 0, 0), thickness+1)
                cv2.putText(colorbar, f"{max_height:.2f}", (5, 20), font, font_scale, (255, 255, 255), thickness)
                mid_height = min_height + (max_height - min_height) / 2
                cv2.putText(colorbar, f"{mid_height:.2f}", (5, colorbar_height//2 + 10), font, font_scale, (0, 0, 0), thickness+1)
                cv2.putText(colorbar, f"{mid_height:.2f}", (5, colorbar_height//2 + 10), font, font_scale, (255, 255, 255), thickness)
                cv2.putText(colorbar, f"{min_height:.2f}", (5, colorbar_height - 10), font, font_scale, (0, 0, 0), thickness+1)
                cv2.putText(colorbar, f"{min_height:.2f}", (5, colorbar_height - 10), font, font_scale, (255, 255, 255), thickness)

                # Combine heatmap and color bar
                combined_heatmap = np.hstack((heatmap, colorbar))
                combined_heatmap = cv2.copyMakeBorder(combined_heatmap, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                # Display heatmap
                cv2.imshow('Nose Height Map', combined_heatmap)

    # Display annotated image
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

key = "jestin_2_rgb.png"
gethead(heads[key])