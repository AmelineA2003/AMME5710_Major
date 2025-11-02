import cv2
import numpy as np
import mediapipe as mp

import cv2
import mediapipe as mp
import numpy as np



def mediapipe_gaze_tracker(cap, face_mesh): 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            face_landmarks = results.multi_face_landmarks[0]

            # LEFT and RIGHT iris landmark indices 
            LEFT_IRIS = [474, 475, 476, 477]
            RIGHT_IRIS = [469, 470, 471, 472]

            # Eye corner landmarks
            LEFT_EYE = [33, 133]  # outer and inner corners
            RIGHT_EYE = [362, 263]

            def get_landmark_coords(index_list):
                """Convert landmark indices to (x, y) pixel coords."""
                return np.array([(int(face_landmarks.landmark[i].x * w),
                                int(face_landmarks.landmark[i].y * h)) for i in index_list])

            # Get iris centers
            left_iris = get_landmark_coords(LEFT_IRIS)
            right_iris = get_landmark_coords(RIGHT_IRIS)

            left_center = np.mean(left_iris, axis=0).astype(int)
            right_center = np.mean(right_iris, axis=0).astype(int)

            # Get eye corners for context
            left_eye_corners = get_landmark_coords(LEFT_EYE)
            right_eye_corners = get_landmark_coords(RIGHT_EYE)

            # Compute eye centers 
            left_eye_center = np.mean(left_eye_corners, axis=0).astype(int)
            right_eye_center = np.mean(right_eye_corners, axis=0).astype(int)

            # Draw pupil centers 
            cv2.circle(frame, tuple(left_center), 4, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_center), 4, (0, 0, 255), -1)

            # Draw eye centers 
            cv2.circle(frame, tuple(left_eye_center), 4, (255, 255, 0), -1)
            cv2.circle(frame, tuple(right_eye_center), 4, (255, 255, 0), -1)

        cv2.imshow("MediaPipe Face + Eye + Pupil Detection", frame)

        # Exit with ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
