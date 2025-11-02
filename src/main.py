import cv2
import numpy as np
import mediapipe as mp


from traditional_gaze_functions import traditional_gaze_tracker
from ml_gaze_functions import mediapipe_gaze_tracker 

#################################### VIDEOS #####################################
# Video input
# cap = cv2.VideoCapture("test_AMELINE_30cm_eye_HD_rgb_1761970095.mp4")
# cap = cv2.VideoCapture("test_JESTIN_30cm_eye_HD_rgb_1761970129.mp4")
# cap = cv2.VideoCapture("ameline_test_d-_rgb_1761129562.mp4")
# cap = cv2.VideoCapture("ameline_with_light_rgb.mp4")
# cap = cv2.VideoCapture("IMG_0595.mp4")
# cap = cv2.VideoCapture("IMG_0588.mp4")
cap = cv2.VideoCapture("jestin_video_1_rgb.mp4")


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def main(): 
    traditional_gaze_tracker(cap)

    # mediapipe_gaze_tracker(cap, face_mesh)


    return 



if __name__ == '__main__': 
    main()