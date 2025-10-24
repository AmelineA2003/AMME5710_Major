#!/usr/bin/python

"""
run_face_track.py - OpenCV face/eye tracking using pre-trained Haar cascade classifiers
Packages required: OpenCV and numpy
"""

import sys, os
mainpath = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(mainpath)

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_eye.xml'))
mouth_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_mouth.xml'))
nose_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_nose.xml'))
profileface_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_profileface.xml'))
left_ear_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_leftear.xml'))
right_ear_cascade = cv2.CascadeClassifier(os.path.join(mainpath,'haarcascade_mcs_rightear.xml'))

# ...existing code...

# ...existing code...

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    count = 0
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        img2 = cv2.resize(img, (640, 360))
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
        if len(faces) == 0:
            faces = profileface_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img2[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            mouth = mouth_cascade.detectMultiScale(roi_gray)
            nose = nose_cascade.detectMultiScale(roi_gray)
            right_ear_cascade.detectMultiScale(roi_gray)
            left_ear_cascade.detectMultiScale(roi_gray)
   
            for (ex,ey,ew,eh) in eyes:
                center = (int(ex + ew/2), int(ey + eh/2))
                # radius = int(min(ew, eh)/2)
                radius = 5
                cv2.circle(roi_color, center, radius, (0,255,0), 2)

            for (mx,my,mw,mh) in mouth:
                if y + my > y + h/2:
                    center = (int(mx + mw/2), int(my + mh/2))
                    # radius = int(min(mw, mh)/2)
                    radius = 5
                    cv2.circle(roi_color, center, radius, (0,0,255), 2)

            for (nx,ny,nw,nh) in nose:
                center = (int(nx + nw/2), int(ny + nh/2))
                # radius = int(min(nw, nh)/2)
                radius = 5
                cv2.circle(roi_color, center, radius, (255,255,0), 2)
    
            for (rex,rey,rew,reh) in right_ear_cascade.detectMultiScale(roi_gray):
                if x + rex > x + w/2:
                    center = (int(rex + rew/2), int(rey + reh/2))
                    # radius = int(min(rew, reh)/2)
                    radius = 5
                    cv2.circle(roi_color, center, radius, (255,0,255), 2)
     
            for (lex,ley,lew,leh) in left_ear_cascade.detectMultiScale(roi_gray):
                if x + lex < x + w/2:
                    center = (int(lex + lew/2), int(ley + leh/2))
                    # radius = int(min(lew, leh)/2)
                    radius = 5
                    cv2.circle(roi_color, center, radius, (0,255,255), 2)
        
        cv2.imshow('Faces', img2)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        count += 1
    
    cv2.destroyAllWindows()

# ...existing code...


def main():
	show_webcam(mirror=True)


if __name__ == '__main__':
	main()

