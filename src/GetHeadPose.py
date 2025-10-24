import cv2
import numpy as np
import mediapipe as mp
import os

def GetHeadPose(image_path, output_path=None):
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path.")
    
    original_image = image.copy()  # Keep a copy for annotation

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Detect face landmarks
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError(f"No face detected in image {image_path}].")

    # 3D model points of facial landmarks (nose tip, chin, left eye, right eye, left mouth, right mouth)
    # Y-axis is positive upward
    model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, 63.6, -12.5],       # Chin
        [-43.3, -32.7, -26.0],    # Left eye left corner
        [43.3, -32.7, -26.0],     # Right eye right corner
        [-28.9, 28.9, -24.1],     # Left mouth corner
        [28.9, 28.9, -24.1]       # Right mouth corner
    ], dtype='double')

    # Get corresponding 2D image points from detected landmarks
    face_landmarks = results.multi_face_landmarks[0]
    image_points = np.array([
        [face_landmarks.landmark[1].x * image.shape[1], face_landmarks.landmark[1].y * image.shape[0]],    # Nose tip
        [face_landmarks.landmark[152].x * image.shape[1], face_landmarks.landmark[152].y * image.shape[0]],# Chin
        [face_landmarks.landmark[263].x * image.shape[1], face_landmarks.landmark[263].y * image.shape[0]],# Left eye left corner
        [face_landmarks.landmark[33].x * image.shape[1], face_landmarks.landmark[33].y * image.shape[0]],  # Right eye right corner
        [face_landmarks.landmark[287].x * image.shape[1], face_landmarks.landmark[287].y * image.shape[0]],# Left mouth corner
        [face_landmarks.landmark[57].x * image.shape[1], face_landmarks.landmark[57].y * image.shape[0]]   # Right mouth corner
    ], dtype='double')

    # Camera internals
    focal_length = image.shape[1]
    center = (image.shape[1] / 2, image.shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # SolvePnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("Head pose estimation failed.")

    # Convert rotation vector to rotation matrix (head to camera)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Compute rotation matrix from camera to head frame (inverse = transpose)
    camera_to_head_rotation = rotation_matrix.T

    # Get Euler angles from rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    # Convert radians to degrees
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    roll_deg = np.degrees(roll)
    
    # y-axis is positive upward
    # x-axis is positive rightward
    # z-axis is positive forward
    angles = {'pitch': pitch_deg, 'yaw': yaw_deg, 'roll': roll_deg}

    # Mask non-head pixels to black using convex hull of detected face landmarks
    try:
        # Build array of landmark (x, y) points in image coordinates
        landmark_points = np.array([
            [lm.x * image.shape[1], lm.y * image.shape[0]] for lm in face_landmarks.landmark
        ], dtype=np.int32)

        # Compute convex hull over landmarks to get face region
        hull = cv2.convexHull(landmark_points)

        # Create mask and fill convex hull
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Apply mask: keep face pixels, set others to black
        masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        original_image = masked_image
    except Exception as _e:
        # If anything fails, fall back to original image (do not stop execution)
        print(f"Warning: failed to mask non-head pixels: {_e}")

    # Annotate the image with Euler angles text
    cv2.putText(original_image, f'Pitch: {pitch_deg:.2f} deg', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original_image, f'Yaw: {yaw_deg:.2f} deg', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original_image, f'Roll: {roll_deg:.2f} deg', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Visualize the pose by projecting 3D axes onto the image
    axis_length = 50  # Length of axes lines in model units
    axes_points = np.array([
        [0, 0, 0],           # Origin (nose tip)
        [axis_length, 0, 0], # X-axis end
        [0, axis_length, 0], # Y-axis end
        [0, 0, axis_length]  # Z-axis end
    ], dtype='double')

    # Project axes points to 2D image
    imgpts, _ = cv2.projectPoints(axes_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    nose_tip_2d = tuple(imgpts[0])  # Origin
    x_end = tuple(imgpts[1])
    y_end = tuple(imgpts[2])
    z_end = tuple(imgpts[3])

    # Draw axes: X (red), Y (green), Z (blue)
    cv2.line(original_image, nose_tip_2d, x_end, (0, 0, 255), 3)  # X-axis red
    cv2.line(original_image, nose_tip_2d, y_end, (0, 255, 0), 3)  # Y-axis green
    cv2.line(original_image, nose_tip_2d, z_end, (255, 0, 0), 3)  # Z-axis blue

    # Draw labels for axes
    cv2.putText(original_image, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(original_image, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(original_image, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display or save the annotated image
    if output_path:
        cv2.imwrite(output_path, original_image)
        print(f"Annotated image saved to {output_path}")
    else:
        cv2.imshow('Annotated Head Pose', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

    # Return both Euler angles and the rotation matrix from camera to head frame
    return {
        'angles': angles,
        'camera_to_head_rotation': camera_to_head_rotation,

    }

# Example usage:
if __name__ == "__main__":
    import glob
    # folder_path = "AFLW2000-3D/AFLW2000/"
    folder_path = os.getcwd()
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    print(f"Found {len(jpg_files)} .jpg files in {folder_path}")
    for image_path in jpg_files:
        image_name = os.path.basename(image_path)
        output_path = f"{image_name[0:-4]}_head_pose.jpg"  # Save annotated image for each
        print(f"\nProcessing {image_name}...")
        result = GetHeadPose(image_path, output_path)
        print("Head Pose Euler Angles:", result['angles'])
        print("Rotation Matrix (Camera to Head Frame):\n", result['camera_to_head_rotation'])