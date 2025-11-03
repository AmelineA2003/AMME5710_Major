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

# Align RGB to depth
align = rs.align(rs.stream.depth)

# Optional: Spatial filter for depth smoothing
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)

# Pointcloud object for deprojection
pc = rs.pointcloud()

# Warm-up pipeline
print("Warming up pipeline...")
for _ in range(20):
    pipeline.wait_for_frames()

# Ground plane estimation with RANSAC (subsampled for speed; does not affect per-frame maps)
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
        pc.map_to(aligned_fs.get_color_frame())  # Optional mapping
        vertices = np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)
        valid = np.all(vertices != 0, axis=1)
        if valid.sum() > subsample:
            indices = np.random.choice(np.where(valid)[0], subsample, replace=False)
            points_list.append(vertices[indices])
    if not points_list:
        raise RuntimeError("Failed to acquire points for plane estimation.")
    points = np.vstack(points_list)
    
    # RANSAC plane fit (simplified iterative approach)
    best_inliers = 0
    best_normal = np.array([0, 0, 1])
    best_point = points.mean(axis=0)
    for _ in range(50):  # Iterations
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal) + 1e-6
        distances = np.abs(np.dot(points - p1, normal))
        inliers = np.sum(distances < 0.01)  # Threshold in meters
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_point = points[distances < 0.01].mean(axis=0)
    return best_point, best_normal

plane_point, plane_normal = estimate_ground_plane()
print(f"Estimated plane: point {plane_point}, normal {plane_normal}")

# Frame storage
rgb_frames = []
height_frames = []  # Full 480x640 float32 arrays in meters per frame

# Capture parameters
max_duration = 10  # Seconds (fallback)
start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        if not frames:
            print("No frames received.")
            continue
        
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Missing depth or color frame.")
            continue

        # Apply filter
        depth_frame = spatial.process(depth_frame)

        # RGB frame (full resolution)
        rgb_data = np.asanyarray(color_frame.get_data())
        rgb_frames.append(rgb_data.copy())

        # Point cloud for full 3D points (entire frame)
        pts = pc.calculate(depth_frame)
        points_3d = np.asanyarray(pts.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # Compute heights for all pixels (vectorized signed distance in meters)
        vectors = points_3d - plane_point
        heights = np.dot(vectors, plane_normal)
        
        # Mask invalid (zero-depth) points with NaN
        depths_m = np.asanyarray(depth_frame.get_data()).reshape(-1) * depth_scale
        valid = depths_m > 0
        heights[~valid] = np.nan

        # Reshape to full frame dimensions (480x640) with float32 precision
        height_map = heights.reshape(480, 640).astype(np.float32)
        height_frames.append(height_map.copy())

        # Diagnostic: Confirm full map and statistics
        nan_count = np.isnan(height_map).sum()
        valid_heights = height_map[~np.isnan(height_map)]
        if valid_heights.size > 0:
            h_min, h_max, h_mean = valid_heights.min(), valid_heights.max(), valid_heights.mean()
            print(f"Captured frame {len(rgb_frames)}: Height map (480x640) | Valid pixels: {valid_heights.size} | Min: {h_min:.3f}m | Max: {h_max:.3f}m | Mean: {h_mean:.3f}m | NaN: {nan_count}", end='\r')
        else:
            print(f"Captured frame {len(rgb_frames)}: Height map (480x640) | No valid pixels", end='\r')

        # Break on duration
        if time.time() - start_time > max_duration:
            print("\nDuration reached.")
            break

except KeyboardInterrupt:
    print("\nCapture interrupted by user.")
except Exception as e:
    print(f"\nError during capture: {e}")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    # Save to Pickle
    if rgb_frames and height_frames and len(rgb_frames) == len(height_frames):
        data = {
            'rgb_frames': rgb_frames,       # List of BGR uint8 arrays (480x848x3)
            'height_frames': height_frames, # List of full 480x640 float32 arrays in meters; NaN for invalid
            'plane_point': plane_point,     # Reference point on plane
            'plane_normal': plane_normal,   # Unit normal vector
            'units': 'meters',              # Metadata
            'height_map_shape': (480, 640)  # Explicit shape confirmation
        }
        with open('realsense_rgb_height.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f"\nSaved {len(rgb_frames)} synchronized frames to 'realsense_rgb_height.pkl'.")
        print("Each height map is the entire 480x640 array in meters (full resolution, signed distance to plane; NaN for invalid pixels).")
    else:
        print("Incomplete capture; Pickle file not created. Check camera connection and USB 3.0 port.")