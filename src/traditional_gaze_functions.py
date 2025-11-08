""" 
This file stores the functions for our traditional gaze tracking algorithms implemented. 
It includes: 
    - Method (1) of pupil detection 
    - Method (2) of pupil detection 
    - Method (1) of eye centre detection 
    - Method (2) of eye centre detection
    - Final integration function to integrate selected eye and pupil detection
"""

import cv2
import numpy as np


#################################### FUNCTIONS #################################
def detect_pupil_center_1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    kernel = np.ones((3,3), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)  # remove small bright noise
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1) # fill small dark holes

    _, mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,            # Inverse ratio of accumulator resolution
        minDist=gray.shape[0],  # Minimum distance between circle centers
        param1=100,      # Higher threshold for Canny edge detector
        param2=23,       # Accumulator threshold for circle detection (smaller -> more circles)
        minRadius=40,    # Minimum possible circle radius
        maxRadius=90     # Maximum possible circle radius
    )
    # If circles are found, draw them
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)  # outer circle
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)  # centroid
            cX = x 
            cY = y
    
        return (cX, cY) 
    else:
        return None



def detect_pupil_center_2(img):
    inverted = cv2.bitwise_not(img)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3,3), np.uint8)
    # eroded = cv2.erode(gray, kernel, iterations=1)
    eroded = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    _, thresh = cv2.threshold(eroded, 210, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def detect_eye_center(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = gray.copy()
    mean = img_copy.mean()
    stddev = img_copy.std()

    mask = img_copy < (mean - 1 * stddev)
    img_copy[mask] = 0

    ys, xs = np.where(img_copy == 0)

    if len(xs) > 0:
        leftmost_zero = xs.min()
        rightmost_zero = xs.max()

        y_left = ys[xs == leftmost_zero]
        y_right = ys[xs == rightmost_zero]

        top_y = min(y_left.min(), y_right.min())
        bottom_y = max(y_left.max(), y_right.max())
        vertical_mid = (top_y + bottom_y) // 2

        eye_center = ((leftmost_zero + rightmost_zero) // 2, bottom_y)
        return eye_center
    else: 
        return None
    

def detect_eye_center_2(img): 

    gray = cv2.medianBlur(img, 5)
    
    _, thresh = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )   

    # Morphological operations to merge regions 
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    eye_center = None
    if contours:
        # Take the largest contour
        biggest = max(contours, key=cv2.contourArea)
        
        if len(biggest) >= 5:  
            ellipse = cv2.fitEllipse(biggest)
            eye_center = (int(ellipse[0][0]), int(ellipse[0][1]))
    

    return eye_center
            



def traditional_gaze_tracker(cap, output_path="outputs/pupil_detection.mp4"):
    # Create outputs folder if not exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  
        fps = 20

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(150, 150))

        if len(faces) > 0:
            # Keep only the largest face
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            face_roi = frame[fy:fy+fh, fx:fx+fw]
            gray_face = gray[fy:fy+fh, fx:fx+fw]

            # Draw face rectangle
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10,10))
            if len(eyes) > 0:
                eyes = sorted(eyes, key=lambda e: e[1])[:2]  # top 2 eyes

                for (ex, ey, ew, eh) in eyes:
                    abs_x, abs_y = fx + ex, fy + ey
                    eye_crop = frame[abs_y:abs_y+eh, abs_x:abs_x+ew].copy()
                    gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

                    cv2.rectangle(frame, (abs_x, abs_y), (abs_x+ew, abs_y+eh), (0, 255, 0), 2)

                    pupil_center_rel = detect_pupil_center_2(eye_crop)
                    eye_center = detect_eye_center(eye_crop)

                    if pupil_center_rel is not None and eye_center is not None:
                        # Absolute coordinates
                        px, py = pupil_center_rel
                        pupil_center_abs = (abs_x + px, abs_y + py)
                        eye_center_abs = (abs_x + eye_center[0], abs_y + eye_center[1])

                        # Draw pupil and eye center
                        cv2.circle(frame, pupil_center_abs, 4, (0, 0, 255), -1)
                        cv2.circle(frame, eye_center_abs, 4, (255, 255, 0), -1)

        # Write frame to video
        out.write(frame)

        # Optional: display live
        cv2.imshow("Face + Eye + Pupil Detection", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")