import cv2
import numpy as np

#################################### SETUP #####################################
# Video input
# cap = cv2.VideoCapture("test_AMELINE_30cm_eye_HD_rgb_1761970095.mp4")
# cap = cv2.VideoCapture("test_JESTIN_30cm_eye_HD_rgb_1761970129.mp4")
# cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")
cap = cv2.VideoCapture("ameline_with_light_rgb.mp4")

#################################### FUNCTIONS #################################

def detect_pupil_center_2(img):
    """
    Simple pupil detection using threshold + contour.
    Returns center (cx, cy) relative to the eye crop, or None.
    """
    inverted = cv2.bitwise_not(img)
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(gray, kernel, iterations=1)
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

def detect_pupil_center_1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess — blur helps reduce noise and improves circle detection
    # gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (5, 7), 2)
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
        x, y, r = circles[0, 0]  # take the first circle
        return (int(x), int(y))
    else:
        return None

#################################### MAIN LOOP #################################

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    for (fx, fy, fw, fh) in faces:
        face_roi = frame[fy:fy+fh, fx:fx+fw]
        gray_face = gray[fy:fy+fh, fx:fx+fw]

        # Draw face rectangle
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

        # Detect eyes **inside the face ROI**
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(20,20))

        for (ex, ey, ew, eh) in eyes:
            # Absolute coordinates in the original frame
            abs_x, abs_y = fx + ex, fy + ey
            eye_crop = frame[abs_y:abs_y+eh, abs_x:abs_x+ew].copy()
            pupil_center_rel = detect_pupil_center_2(eye_crop)
            if pupil_center_rel is None:
                continue

            px, py = pupil_center_rel
            pupil_center_abs = (abs_x + px, abs_y + py)
            eye_center = (abs_x + ew//2, abs_y + eh//2)

            # Draw eye rectangle, pupil, and eye center
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x+ew, abs_y+eh), (0, 255, 0), 2)
            cv2.circle(frame, pupil_center_abs, 4, (0, 0, 255), -1)
            cv2.circle(frame, eye_center, 4, (255, 255, 0), -1)

    cv2.imshow("Face + Eye + Pupil Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()