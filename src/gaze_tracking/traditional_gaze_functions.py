import cv2
import numpy as np


#################################### FUNCTIONS #################################

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
    img_copy = img.copy()
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

    gray = cv2.GaussianBlur(img, (5,5), 0)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,  # neighborhood size
        C=5           # constant to subtract
    )
    
    # Morphological operations to merge regions 
    kernel = np.ones((3,3), np.uint8)
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
            


def traditional_gaze_tracker(frame, bbox, draw=True): 
    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    x, y, w, h = bbox
    info = {
        "faces": [],
        "eyes": [],
        "pupils": []
    }

    person_roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_roi, 1.1, 5, minSize=(50,50))

    if len(faces) > 0:
        # Sort faces by area (w*h) descending
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        faces = [faces[0]]  # Keep only the largest

        for (fx, fy, fw, fh) in faces: 
            abs_fx, abs_fy = x + fx, y + fy
            if draw:
                cv2.rectangle(frame, (abs_fx, abs_fy), (abs_fx+fw, abs_fy+fh), (255, 0, 0), 2)
            info["faces"].append((abs_fx, abs_fy, fw, fh))

            face_crop = person_roi[fy:fy+fh, fx:fx+fw]
            gray_face = gray_roi[fy:fy+fh, fx:fx+fw]

            eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10,10))
            eyes = sorted(eyes, key=lambda e: e[1])[:2]

            for (ex, ey, ew, eh) in eyes:
                abs_ex, abs_ey = abs_fx + ex, abs_fy + ey
                eye_crop = frame[abs_ey:abs_ey+eh, abs_ex:abs_ex+ew].copy()
                gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

                if draw:
                    cv2.rectangle(frame, (abs_ex, abs_ey), (abs_ex+ew, abs_ey+eh), (255, 0, 255), 2)

                pupil_center = detect_pupil_center_2(eye_crop)
                eye_center = detect_eye_center(gray_eye)

                if pupil_center is not None and eye_center is not None:
                    pupil_abs = (abs_ex + pupil_center[0], abs_ey + pupil_center[1])
                    eye_abs = (abs_ex + eye_center[0], abs_ey + eye_center[1])
                    info["eyes"].append(eye_abs)
                    info["pupils"].append(pupil_abs)
                    if draw:
                        cv2.circle(frame, pupil_abs, 4, (0, 0, 255), -1)
                        cv2.circle(frame, eye_abs, 4, (255, 255, 0), -1)

    return info