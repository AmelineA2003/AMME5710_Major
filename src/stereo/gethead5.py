import sys, os
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import math



def show_nose(head):
    img = head["im_rgb"]
    height_m = head["im_height"]

    # Get original image dimensions
    original_height, original_width = img.shape[:2]
    resized_width, resized_height = 640, 360  # Resized image dimensions
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

        # Draw reference lines relative to the face ROI (for visualization)
        cv2.line(img2, (x + int(w/2), y), (x + int(w/2), y + h), (255, 255, 255), 1)
        cv2.line(img2, (x, y + int(h/3)), (x + w, y + int(h/3)), (255, 255, 255), 1)
        cv2.line(img2, (x, y + int(h/2)), (x + w, y + int(h/2)), (255, 255, 255), 1)

        for (ex, ey, ew, eh) in eyes:
            # Adjust eye coordinates to whole image
            center = (int(x + ex + ew/2), int(y + ey + eh/2))
            if center[0] < x + w/2:
                if 'eye_left' in features.keys():
                    print("Multiple left eyes detected, too much turn")
                    continue
                features['eye_left'] = center
                radius = 5
                cv2.circle(img2, center, radius, (0, 255, 0), 2)
                print("Left eye detected at:", center)
            if center[0] >= x + w/2:
                if 'eye_right' in features.keys():
                    print("Multiple right eyes detected, too much turn")
                    continue
                features['eye_right'] = center
                radius = 5
                cv2.circle(img2, center, radius, (0, 255, 0), 2)
                print("Right eye detected at:", center)

        for (nx, ny, nw, nh) in nose:
            # Original nose center (for reference, not used for final visualization)
            nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))
            if nose_center[1] > features['eye_left'][1] and nose_center[1] > features['eye_right'][1] and nose_center[0] > features['eye_left'][0] and nose_center[0] < features['eye_right'][0]:
                # Scale nose coordinates to original image dimensions for height map

                nose_height_map = height_m[x + nx : x + nx + nw, y + ny : y + ny + nh]

                # Draw circle at the nose center
                radius = 5
                cv2.circle(img2, nose_center, radius, (255, 255, 0), 2)
                print("Nose y lower than eye y")
                
                features['nose'] = nose_center
                # Draw line at nose level + offset, relative to whole image
                cv2.line(img2, (x, int(nose_center[1] + 0.1*h)), (x + w, int(nose_center[1] + 0.1*h)), (0, 0, 255), 1)


                # Create heatmap for height map
        if nose_height_map.size > 0:
            # Normalize height map to [0, 255] for visualization
            valid_heights = nose_height_map[nose_height_map > 0]
            if valid_heights.size > 0:
                min_height = np.min(valid_heights)
                max_height = np.max(valid_heights)
                # Avoid division by zero
                if max_height > min_height:
                    normalized_height = (nose_height_map - min_height) / (max_height - min_height) * 255
                else:
                    normalized_height = np.zeros_like(nose_height_map)
                normalized_height = normalized_height.astype(np.uint8)
                
                # Apply colormap
                heatmap = cv2.applyColorMap(normalized_height, cv2.COLORMAP_JET)
                
                # Resize heatmap to a fixed size for better visibility
                heatmap = cv2.resize(heatmap, (300, 300), interpolation=cv2.INTER_LINEAR)
                
                # Create color bar with increased width
                colorbar_width = 70
                colorbar_height = 300
                colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
                for i in range(colorbar_height):
                    # Map i from [0, colorbar_height] to [0, 255]
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
                
                # Add border to heatmap
                combined_heatmap = cv2.copyMakeBorder(combined_heatmap, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                
                # Display heatmap in a separate window
                cv2.imshow('Height Map', combined_heatmap)

        for (mx, my, mw, mh) in mouth:
            # Adjust mouth coordinates to whole image
            mouth_center = (int(x + mx + mw/2), int(y + my + mh/2))
            print("half box", h/2)
            print("Nose center", features['nose'][1] + 0.1*h)
            print("mouth center", mouth_center[1])
            
            if mouth_center[1] < features['nose'][1] + 0.1*h:
                print("Mouth y higher than nose y")
                continue
            
            print("still")
            radius = 5
            cv2.circle(img2, mouth_center, radius, (0, 0, 255), 2)
            features['mouth'] = mouth_center
            print("Mouth detected at:", mouth_center)

        # for (rex, rey, rew, reh) in right_ear:
        #     # Adjust right ear coordinates to whole image
        #     ear_center = (int(x + rex + rew/2), int(y + rey + reh/2))
        #     if ear_center[0] > x + w/2:
        #         radius = 5
        #         cv2.circle(img2, ear_center, radius, (255, 0, 255), 2)
        #         features['right_ear'] = ear_center
        #         print("Right ear detected at:", ear_center)
    
        # for (lex, ley, lew, leh) in left_ear:
        #     # Adjust left ear coordinates to whole image
        #     ear_center = (int(x + lex + lew/2), int(y + ley + leh/2))
        #     if ear_center[0] < x + w/2:
        #         radius = 5
        #         cv2.circle(img2, ear_center, radius, (0, 255, 255), 2)
        #         features['left_ear'] = ear_center
        #         print("Left ear detected at:", ear_center)

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