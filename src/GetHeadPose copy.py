import cv2
import numpy as np
import mediapipe as mp
import os
import glob
import math

def GetHeadPose(image_path, output_path=None):
    """
    Head pose estimation with:
      • pitch = 0° when face looks straight forward
      • positive pitch = looking up, negative = looking down
      • axes: X-right (red), Y-down (green), Z-forward (blue)
      • uses user-defined local unit vectors
    """
    # ------------------------------------------------------------------ #
    # 1. Load image
    # ------------------------------------------------------------------ #
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    original = img.copy()
    h, w = img.shape[:2]

    # ------------------------------------------------------------------ #
    # 2. MediaPipe Face Mesh
    # ------------------------------------------------------------------ #
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected.")
        landmarks = results.multi_face_landmarks[0]

    # ------------------------------------------------------------------ #
    # 3. 3D model points – Y positive DOWN (matches image coordinates)
    # ------------------------------------------------------------------ #
    model_pts = np.array([
        [ 0.0,   0.0,   0.0],   # 0 – Nose tip
        [ 0.0, -63.6, -12.5],   # 1 – Chin
        [ 43.3,  32.7, -26.0],  # 2 – Left eye outer
        [-43.3,  32.7, -26.0],  # 3 – Right eye outer
        [ 28.9, -28.9, -24.1],  # 4 – Left mouth outer
        [-28.9, -28.9, -24.1],  # 5 – Right mouth outer
    ], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 4. 2D image points
    # ------------------------------------------------------------------ #
    img_pts = np.array([
        [landmarks.landmark[4].x   * w, landmarks.landmark[4].y   * h],  # Nose tip (stable subnasale)
        [landmarks.landmark[152].x * w, landmarks.landmark[152].y * h],  # Chin
        [landmarks.landmark[33].x  * w, landmarks.landmark[33].y  * h],  # Left eye outer
        [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],  # Right eye outer
        [landmarks.landmark[61].x  * w, landmarks.landmark[61].y  * h],  # Left mouth outer
        [landmarks.landmark[291].x * w, landmarks.landmark[291].y * h],  # Right mouth outer
    ], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 5. Camera intrinsics (improved focal length for better convergence)
    # ------------------------------------------------------------------ #
    focal = 1.5 * w  # Adjusted for typical face capture; tunable if needed
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 6. solvePnP → world → camera (modern flag for stability)
    # ------------------------------------------------------------------ #
    success, rvec, tvec = cv2.solvePnP(
        model_pts, img_pts, cam_mat, dist,
        flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        raise RuntimeError("solvePnP failed.")

    R_world_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_to_head = R_world_to_cam.T  # camera → head

    # ------------------------------------------------------------------ #
    # 7. Euler angles from camera → head (Z-Y-X extrinsic)
    # ------------------------------------------------------------------ #
    sy = math.sqrt(R_cam_to_head[0,0]**2 + R_cam_to_head[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(-R_cam_to_head[2,1], R_cam_to_head[2,2])
        yaw   = math.atan2( R_cam_to_head[2,0], sy)
        roll  = math.atan2(-R_cam_to_head[1,0], R_cam_to_head[0,0])
    else:
        pitch = math.atan2( R_cam_to_head[1,2], R_cam_to_head[1,1])
        yaw   = math.atan2( R_cam_to_head[2,0], sy)
        roll  = 0.0

    pitch_deg = math.degrees(pitch)
    yaw_deg   = math.degrees(yaw)
    roll_deg  = math.degrees(roll)

    # ------------------------------------------------------------------ #
    # 8. User-defined local unit vectors (head frame)
    # ------------------------------------------------------------------ #
    axes_local = np.array([
        [1, 0, 0],   # X – right
        [0, 1, 0],   # Y – down (image +Y)
        [0, 0, 1]    # Z – forward
    ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 9. Transform to camera space
    # ------------------------------------------------------------------ #
    arrow_len = 50.0
    dirs_cam = (R_cam_to_head @ axes_local.T).T
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1)[:, None]
    dirs_cam *= arrow_len
    offsets_2d = np.round(dirs_cam[:, :2]).astype(int)

    # Nose tip in image
    nose_2d = tuple(img_pts[0].astype(int))

    # ------------------------------------------------------------------ #
    # 10. Draw arrows (Red=X, Green=Y, Blue=Z)
    # ------------------------------------------------------------------ #
    colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # R, G, B
    for i, (dx, dy) in enumerate(offsets_2d):
        end = (nose_2d[0] + dx, nose_2d[1] + dy)
        cv2.arrowedLine(original, nose_2d, end, colors_bgr[i], thickness=4, tipLength=0.25)

    # ------------------------------------------------------------------ #
    # 11. Text overlay
    # ------------------------------------------------------------------ #
    cv2.putText(original, f'Pitch: {pitch_deg:+.2f} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Yaw:   {yaw_deg:+.2f} deg', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(original, f'Roll:  {roll_deg:+.2f} deg', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ------------------------------------------------------------------ #
    # 12. Save / show
    # ------------------------------------------------------------------ #
    if output_path:
        cv2.imwrite(output_path, original)
        print(f"Annotated image saved → {output_path}")
    else:
        desired_width = 800   # In pixels
        desired_height = 600  # In pixels
        window_name = "head pose"
        # Create a resizable window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Set the window size
        cv2.resizeWindow(window_name, desired_width, desired_height)

        cv2.imshow(window_name, original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # 13. Return
    # ------------------------------------------------------------------ #
    return {
        'angles': {'pitch': pitch_deg, 'yaw': yaw_deg, 'roll': roll_deg},
        'camera_to_head_rotation': R_cam_to_head,
        'nose_2d': nose_2d,
        'axis_offsets_px': offsets_2d
    }

# ---------------------------------------------------------------------- #
# Example usage
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    folder = os.getcwd()
    png_files = glob.glob(os.path.join(folder, "*.png"))
    jpg_files = glob.glob(os.path.join(folder, "*.jpg"))
    image_files = png_files + jpg_files

    print(f"Found {len(image_files)} image(s) in {folder}")

    for img_path in image_files:
        name = os.path.basename(img_path)
        out_path = f"{os.path.splitext(name)[0]}_headpose.jpg"
        print(f"\nProcessing: {name}")
        try:
            result = GetHeadPose(img_path)
            print(f"   Angles → Pitch: {result['angles']['pitch']:+.2f}°, "
                  f"Yaw: {result['angles']['yaw']:+.2f}°, "
                  f"Roll: {result['angles']['roll']:+.2f}°")
        except Exception as e:
            print(f"   Failed: {e}")

        break