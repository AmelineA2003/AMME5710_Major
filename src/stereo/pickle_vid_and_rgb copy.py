import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
import time

# Configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start pipeline
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align RGB to depth (this projects RGB onto the depth frame's perspective: 640x480)
align = rs.align(rs.stream.depth)

# Optional: Spatial filter for depth smoothing
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)

# Pointcloud object for deprojection
pc = rs.pointcloud()

# Video writer for aligned RGB stream (MP4, 640x480 to match depth)
video_writer = None
fps = 30.0
video_filename = 'rgb_stream_aligned.mp4'  # Aligned to depth perspective

# Warm-up pipeline
print("Warming up pipeline...")
for _ in range(20):
    pipeline.wait_for_frames()

# Ground plane estimation with RANSAC (subsampled for efficiency)
def estimate_ground_plane(num_frames=10, subsample=1000):
    points_list = []
    for _ in range(num_frames):
        fs = pipeline.wait_for_frames()
        aligned_fs = align.process(fs)
        depth = aligned_fs.get_depth_frame()
        if not depth:
            continue
        depth = spatial.process(depth)
        pts = pc.calculate(depth)
        # Map aligned color (projected RGB) for potential use
        color = aligned_fs.get_color_frame()
        pc.map_to(color)
        vertices = np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)
        valid = np.all(vertices != 0, axis=1)
        if valid.sum() > subsample:
            indices = np.random.choice(np.where(valid)[0], subsample, replace=False)
            points_list.append(vertices[indices])
    if not points_list:
        raise RuntimeError("Failed to acquire points for plane estimation.")
    points = np.vstack(points_list)
    
    # Simplified RANSAC plane fit
    best_inliers = 0
    best_normal = np.array([0, 0, 1])
    best_point = points.mean(axis=0)
    for _ in range(50):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal) + 1e-6
        distances = np.abs(np.dot(points - p1, normal))
        inliers = np.sum(distances < 0.01)
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_point = points[distances < 0.01].mean(axis=0)
    return best_point, best_normal

plane_point, plane_normal = estimate_ground_plane()
print(f"Estimated plane: point {plane_point}, normal {plane_normal}")

# Frame storage for Pickle (aligned RGB for consistency with height maps)
rgb_frames = []  # Aligned 640x480 BGR uint8
height_frames = []  # Full 480x640 float32 in meters

# Capture parameters
max_duration = 10  # Seconds (adjust or remove for manual stop)
start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        if not frames:
            print("No frames received.")
            continue
        
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()  # This is now aligned (projected to 640x480)
        
        if not depth_frame or not color_frame:
            print("Missing depth or color frame.")
            continue

        # Apply spatial filter
        depth_frame = spatial.process(depth_frame)

        # Aligned RGB frame (640x480 BGR)
        rgb_data = np.asanyarray(color_frame.get_data())
        
        # Initialize MP4 writer on first frame (now 640x480)
        if video_writer is None and rgb_data.size > 0:
            height, width, _ = rgb_data.shape  # 480x640x3
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                print(f"Warning: Failed to open '{video_filename}' for writing.")
        
        # Write aligned RGB to MP4 (full image, no black spots from invalid depth)
        if video_writer is not None:
            # Ensure full RGB coverage: Holes from alignment are filled minimally if needed, but rs.align handles projection
            video_writer.write(rgb_data)
        
        rgb_frames.append(rgb_data.copy())

        # Full height map computation
        pts = pc.calculate(depth_frame)
        # Map aligned color (optional, for textured point cloud if needed later)
        pc.map_to(color_frame)
        points_3d = np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)
        
        vectors = points_3d - plane_point
        heights = np.dot(vectors, plane_normal)
        
        depths_m = np.asanyarray(depth_frame.get_data()).reshape(-1) * depth_scale
        valid = depths_m > 0
        heights[~valid] = np.nan

        height_map = heights.reshape(480, 640).astype(np.float32)
        height_frames.append(height_map.copy())

        # Diagnostics
        valid_heights = height_map[~np.isnan(height_map)]
        if valid_heights.size > 0:
            h_min, h_max, h_mean = valid_heights.min(), valid_heights.max(), valid_heights.mean()
            print(f"Captured frame {len(rgb_frames)} | Heights: Min {h_min:.3f}m, Max {h_max:.3f}m, Mean {h_mean:.3f}m", end='\r')
        
        if time.time() - start_time > max_duration:
            print("\nDuration reached.")
            break

except KeyboardInterrupt:
    print("\nCapture interrupted.")
except Exception as e:
    print(f"\nError: {e}")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Release MP4 writer
    if video_writer is not None:
        video_writer.release()
        print(f"Aligned RGB stream (640x480, full coverage) saved to '{video_filename}' ({len(rgb_frames)} frames).")

    # Save synchronized data to Pickle
    if rgb_frames and height_frames and len(rgb_frames) == len(height_frames):
        data = {
            'rgb_frames': rgb_frames,  # Aligned BGR uint8 (480x640x3), synchronized and hole-filled where possible
            'height_frames': height_frames,
            'plane_point': plane_point,
            'plane_normal': plane_normal,
            'units': 'meters',
            'height_map_shape': (480, 640),
            'rgb_shape': (480, 640, 3),
            'note': 'RGB frames are aligned to depth (projected to 640x480); minor holes filled by alignment process'
        }
        pickle_filename = 'realsense_rgb_height.pkl'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Aligned RGB and height streams saved to '{pickle_filename}' (full-precision heights in meters; NaN for invalid).")
    else:
        print("Capture incomplete; files not saved.")