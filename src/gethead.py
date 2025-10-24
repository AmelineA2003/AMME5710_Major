import sys, os
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np


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

def show_nose(img):

    img2 = cv2.resize(img, (640, 360))
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
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Annotate corners with coordinates
        tl = f"({x},{y})"  # top-left
        tr = f"({x+w},{y})"  # top-right
        bl = f"({x},{y+h})"  # bottom-left
        br = f"({x+w},{y+h})"  # bottom-right
        
        # Put text near each corner (white text with black outline for visibility)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Add black outline
        cv2.putText(img2, tl, (x-5, y-5), font, font_scale, (0,0,0), thickness+1)
        cv2.putText(img2, tr, (x+w+5, y-5), font, font_scale, (0,0,0), thickness+1)
        cv2.putText(img2, bl, (x-5, y+h+15), font, font_scale, (0,0,0), thickness+1)
        cv2.putText(img2, br, (x+w+5, y+h+15), font, font_scale, (0,0,0), thickness+1)
        
        # Add white text
        cv2.putText(img2, tl, (x-5, y-5), font, font_scale, (255,255,255), thickness)
        cv2.putText(img2, tr, (x+w+5, y-5), font, font_scale, (255,255,255), thickness)
        cv2.putText(img2, bl, (x-5, y+h+15), font, font_scale, (255,255,255), thickness)
        cv2.putText(img2, br, (x+w+5, y+h+15), font, font_scale, (255,255,255), thickness)
        
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img2[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        nose = nose_cascade.detectMultiScale(roi_gray)
        right_ear_cascade.detectMultiScale(roi_gray)
        left_ear_cascade.detectMultiScale(roi_gray)

        features = {}
        
        print("Face at:", (x,y,x+w,y+h))
        
        features['face'] = (x,y,w,h)

        cv2.line(roi_color, (int(w/2),0), (int(w/2),h), (255,255,255), 1)
        cv2.line(roi_color, (0,int(h/3)), (w,int(h/3)), (255,255,255), 1)
        # cv2.line(roi_color, (0,int(2*h/3)), (w,int(2*h/3)), (0,0,255), 1)
        cv2.line(roi_color, (0,int(h/2)), (w,int(h/2)), (255,255,255), 1)
        cv2.line(roi_color, (0,int(3*h/4)), (w,int(3*h/4)), (255,255,255), 1)

        for (ex,ey,ew,eh) in eyes:
            center = (int(ex + ew/2), int(ey + eh/2))
            # radius = int(min(ew, eh)/2)
            
            # if center[1] < h/3 + y:
            #     continue
            
            # if center[1] > 3*h/4 + y:
            #     continue
            
            if center[0] < w/2:
                if 'eye_left' in features.keys():
                    
                    print("Multiple left eyes detected, too much turn")    
                    continue
                    
                features['eye_left'] = center
                radius = 5
                cv2.circle(roi_color, center, radius, (0,255,0), 2)
                print("Left eye detected at:", center)
            
            if center[0] >= w/2:
                if 'eye_right' in features.keys():
                    
                    print("Multiple right eyes detected, too much turn")    
                    continue
                
                features['eye_right'] = center
                radius = 5
                cv2.circle(roi_color, center, radius, (0,255,0), 2)
                print("Right eye detected at:", center)
            # radius = 5
            # cv2.circle(roi_color, center, radius, (0,255,0), 2)

        for (nx,ny,nw,nh) in nose:
            nose_center = (int(nx + nw/2), int(ny + nh/2))

            
            if nose_center[1] > features['eye_left'][1] and nose_center[1] > features['eye_right'][1] and nose_center[0] > features['eye_left'][0] and nose_center[0] < features['eye_right'][0]:
                radius = 5
                cv2.circle(roi_color, nose_center, radius, (255,255,0), 2)
                print("Nose y lower than eye y")
                
                features['nose'] = nose_center
                cv2.line(roi_color, (0, int(features['nose'][1] + 0.1*features["face"][3])), (w, int(features['nose'][1] + 0.1*features["face"][3])), (0,0,255), 1)        # cv2.line(roi_color, (0,int(nose_center[1])), (w,int(nose_center[1])), (0,0,255), 1)

            

        for (mx,my,mw,mh) in mouth:
            
            mouth_center = (int(mx + mw/2), int(my + mh/2))
            # radius = int(min(mw, mh)/2)
            
            # if mouth_center[1] < 2*h/4:
            #     continue
            
            print("half box",h/2)
            print("Nose center", features['nose'][1] + 0.1*features["face"][3])
            print("mouth center", mouth_center[1])
            
            if  mouth_center[1] < features['nose'][1] + 0.1*features["face"][3]:
                print("Mouth y higher than nose y")
                continue
            
            print("still")
            radius = 5
            cv2.circle(roi_color, mouth_center, radius, (0,0,255), 2)

            # cv2.line(roi_color, (0, int(nose_center[1] + 0.05*h + y)), (w, int(nose_center[1] + 0.05*h + y)), (0,0,255), 1)        # cv2.line(roi_color, (0,int(nose_center[1])), (w,int(nose_center[1])), (0,0,255), 1)
        # cv2.line(roi_color, (0,int(0.05* (y+h - y))), (w,int(0.05* (y+h - y))), (0,0,255), 1)

        
    cv2.imshow('Faces', img2)
      # Wait for a key press indefinitely
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

heads = {}
i = 0

# folder_path = "AFLW2000-3D/AFLW2000/"
folder_path = os.getcwd()
jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
print(f"Found {len(jpg_files)} .jpg files in {folder_path}")

for image_path in jpg_files:
    image_name = os.path.basename(image_path)
    
    image = cv2.imread(os.path.join(folder_path, image_name))
    
    sobel_x = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0, ksize=3) # Horizontal
    sobel_y =  cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=3) # Vertical
    
    heads[image_name] = {
        "im_rgb": image,
        "im_bgr": cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        "im_gray": cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
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


show_nose(heads["emily8.jpg"]["im_rgb"])
# show_webcam(heads["T2_seq-123456789_dist-1.09_rgb_4.jpg"]["im_rgb"])