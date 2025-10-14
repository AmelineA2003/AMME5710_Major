import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # For each detected face
        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face ROI (for your eye mask processing)
            face_roi = color_image[y:y+h, x:x+w]

            # Show ROI window (optional)
            cv2.imshow("Face ROI", face_roi)

        # Show the main color image with bounding box
        cv2.imshow("RealSense Color Frame", color_image)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
