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
    
    # Morphological operations to merge regions (optional)
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
            


def traditional_gaze_tracker(cap): 
    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(150, 150))

        if len(faces) > 0:
            # Sort faces by area in descending order
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            faces = [faces[0]]  # Keep only the largest face

            for (fx, fy, fw, fh) in faces:
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                gray_face = gray[fy:fy+fh, fx:fx+fw]

                # Draw face rectangle
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

                # Detect eyes inside face
                eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10,10))


                if len(eyes) > 0:
                    # Sort eyes by vertical position (y) ascending → top eyes first
                    eyes = sorted(eyes, key=lambda e: e[1])
                    # Keep only the two highest eyes
                    eyes = eyes[:2]

                    for (ex, ey, ew, eh) in eyes:
                        abs_x, abs_y = fx + ex, fy + ey
                        eye_crop = frame[abs_y:abs_y+eh, abs_x:abs_x+ew].copy()
                        gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

                        cv2.rectangle(frame, (abs_x, abs_y), (abs_x+ew, abs_y+eh), (0, 255, 0), 2)

                        pupil_center_rel = detect_pupil_center_2(eye_crop)
                        eye_center = detect_eye_center(gray_eye)

                        if pupil_center_rel is not None and eye_center is not None:
                            # Absolute coordinates
                            px, py = pupil_center_rel
                            pupil_center_abs = (abs_x + px, abs_y + py)
                            eye_center_abs = (abs_x + eye_center[0], abs_y + eye_center[1])

                            # Draw eye rectangle, pupil, and eye center
                            cv2.circle(frame, pupil_center_abs, 4, (0, 0, 255), -1)
                            cv2.circle(frame, eye_center_abs, 4, (255, 255, 0), -1)

        cv2.imshow("Face + Eye + Pupil Detection", frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

