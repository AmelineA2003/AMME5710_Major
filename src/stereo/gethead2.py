import sys, os
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import math

#!/usr/bin/python

"""
run_face_track.py - OpenCV face/eye tracking using pre-trained Haar cascade classifiers
Packages required: OpenCV and numpy
"""


# def show_webcam(img):

#     img2 = cv2.resize(img, (640, 360))
#     gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0:
#         faces = profileface_cascade.detectMultiScale(gray, 1.3, 5)
    
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img2[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         mouth = mouth_cascade.detectMultiScale(roi_gray)
#         nose = nose_cascade.detectMultiScale(roi_gray)
#         right_ear_cascade.detectMultiScale(roi_gray)
#         left_ear_cascade.detectMultiScale(roi_gray)

#         features = {}

#         for (ex,ey,ew,eh) in eyes:
#             center = (int(ex + ew/2), int(ey + eh/2))
#             # radius = int(min(ew, eh)/2)
#             radius = 5
#             cv2.circle(roi_color, center, radius, (0,255,0), 2)

#         for (mx,my,mw,mh) in mouth:
#             if y + my > y + h/2:
#                 center = (int(mx + mw/2), int(my + mh/2))
#                 # radius = int(min(mw, mh)/2)
#                 radius = 5
#                 cv2.circle(roi_color, center, radius, (0,0,255), 2)

#         for (nx,ny,nw,nh) in nose:
#             center = (int(nx + nw/2), int(ny + nh/2))
#             # radius = int(min(nw, nh)/2)
#             radius = 5
#             cv2.circle(roi_color, center, radius, (255,255,0), 2)

#         for (rex,rey,rew,reh) in right_ear_cascade.detectMultiScale(roi_gray):
#             if x + rex > x + w/2:
#                 center = (int(rex + rew/2), int(rey + reh/2))
#                 # radius = int(min(rew, reh)/2)
#                 radius = 5
#                 cv2.circle(roi_color, center, radius, (255,0,255), 2)
    
#         for (lex,ley,lew,leh) in left_ear_cascade.detectMultiScale(roi_gray):
#             if x + lex < x + w/2:
#                 center = (int(lex + lew/2), int(ley + leh/2))
#                 # radius = int(min(lew, leh)/2)
#                 radius = 5
#                 cv2.circle(roi_color, center, radius, (0,255,255), 2)
        
#     cv2.imshow('Faces', img2)
#       # Wait for a key press indefinitely
#     while True:
#         key = cv2.waitKey(0) & 0xFF
#         if key == 27:  # ESC key
#             break
    
#     cv2.destroyAllWindows()

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
                nx_orig = int((x + nx) * scale_x)
                ny_orig = int((y + ny) * scale_y)
                nw_orig = int(nw * scale_x)
                nh_orig = int(nh * scale_y)

                # Ensure coordinates are within height map bounds
                ny_orig = max(0, min(ny_orig, original_height - 1))
                nx_orig = max(0, min(nx_orig, original_width - 1))
                nh_orig = min(nh_orig, original_height - ny_orig)
                nw_orig = min(nw_orig, original_width - nx_orig)

                # Extract nose height area from height map
                nose_height_area = height_m[ny_orig:ny_orig+nh_orig, nx_orig:nx_orig+nw_orig]
                nose_height_values = nose_height_area[nose_height_area > 0]  # Filter out zero values

                if nose_height_values.size > 0:  # Check if there are valid values
                    # Find the index of the minimum height value (nose tip)
                    min_index_flat = np.argmin(nose_height_values)
                    print("Index of minimum value (flattened):", min_index_flat)

                    # Convert flattened index to 2D coordinates in nose_height_area
                    min_index_2d = np.unravel_index(min_index_flat, nose_height_area.shape)
                    nose_tip_y, nose_tip_x = min_index_2d

                    # Map nose tip coordinates back to the resized image
                    nose_tip_x_resized = int(x + nx + (nose_tip_x / scale_x))
                    nose_tip_y_resized = int(y + ny + (nose_tip_y / scale_y))
                    nose_center = (nose_tip_x_resized, nose_tip_y_resized)

                    print("Nose tip detected at (resized image coordinates):", nose_center)
                else:
                    print("No valid height values in nose region")
                    # Fallback to original nose center
                    nose_center = (int(x + nx + nw/2), int(y + ny + nh/2))

                # Draw circle at the nose center
                radius = 5
                cv2.circle(img2, nose_center, radius, (255, 255, 0), 2)
                print("Nose y lower than eye y")
                
                features['nose'] = nose_center
                # Draw line at nose level + offset, relative to whole image
                cv2.line(img2, (x, int(nose_center[1] + 0.1*h)), (x + w, int(nose_center[1] + 0.1*h)), (0, 0, 255), 1)

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