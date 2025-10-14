#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# ----------------------------
# 1. RealSense Setup
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale:", depth_scale)

# ----------------------------
# 2. MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# 3. Helper for virtual board
# ----------------------------
board_w, board_h = 640, 480
board = np.ones((board_h, board_w, 3), dtype=np.uint8) * 255
gaze_dot = (board_w // 2, board_h // 2)

# Smoothing for gaze motion
alpha = 0.2  # smaller = smoother

# ----------------------------
# 4. Streaming Loop
# ----------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_for_mp = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_for_mp)

        h, w, _ = color_image.shape
        gaze_x, gaze_y = 0.5, 0.5  # normalized default

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Left and right iris centers
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]
            left_eye = np.array([left_iris.x, left_iris.y])
            right_eye = np.array([right_iris.x, right_iris.y])

            # Midpoint between eyes = gaze direction origin
            eye_center = (left_eye + right_eye) / 2.0

            # Map normalized coordinates to board
            gaze_x = 1 - np.clip(eye_center[0], 0, 1)  # invert x
            gaze_y = np.clip(eye_center[1], 0, 1)

            # Smooth motion
            new_dot = (
                int(alpha * gaze_x * board_w + (1 - alpha) * gaze_dot[0]),
                int(alpha * gaze_y * board_h + (1 - alpha) * gaze_dot[1])
            )
            gaze_dot = new_dot

            # Draw iris centers on color image
            for idx in [468, 473]:
                pt = face_landmarks.landmark[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(color_image, (cx, cy), 3, (0, 255, 0), -1)

        # ----------------------------
        # Draw virtual board
        # ----------------------------
        board = np.ones((board_h, board_w, 3), dtype=np.uint8) * 255
        cv2.circle(board, gaze_dot, 10, (0, 0, 255), -1)
        cv2.putText(board, "Approx. Gaze Position", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Combine RGB and board
        combined = np.hstack((color_image, board))

        cv2.imshow("RealSense RGB (left) + Virtual Gaze Board (right)", combined)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
