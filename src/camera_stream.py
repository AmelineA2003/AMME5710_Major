""" 
This file is used to read in and save camera data from the Intel RealSense stereo camera. 
It includes features to save RGB and depth data, as well as pickle files containing data for each frame. 
"""

import os
import time
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2

# Set Qt platform before importing cv2 to avoid plugin errors
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# Filenames
timestamp = int(time.time())
name = 'ameline_with_light'
rgb_filename = f"{name}_rgb.mp4"
depth_filename = f"{name}_depth.mp4"
pkl_filename = f"{name}_heightmaps.pkl"

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Check for device
ctx = rs.context()
if len(ctx.devices) == 0:
    print("No RealSense device found. Connect the camera and try again.")
    raise SystemExit(1)

# Start pipeline
pipeline.start(config)

# Depth scale
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale} m/unit")

# Align depth to color
align = rs.align(rs.stream.color)

# Video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
frame_size = (1280, 720)
rgb_out = cv2.VideoWriter(rgb_filename, fourcc, fps, frame_size)
depth_out = cv2.VideoWriter(depth_filename, fourcc, fps, frame_size)

# Store all heightmaps and timestamps
heightmaps_list = []

print("Recording started. Press 'q' or ESC to stop.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Height map in meters (mark invalid depth as NaN)
        height_map = depth_image.astype(np.float32) * depth_scale
        height_map[depth_image == 0] = np.nan

        # Append to list with timestamp
        heightmaps_list.append({
            'height_m': height_map,
            'timestamp': time.time()
        })

        # Depth colormap for display/video
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Write videos
        rgb_out.write(color_image)
        depth_out.write(depth_colormap)

        # Show preview
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("RGB (left) + Depth (right)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Stopping recording...")
            break

finally:
    pipeline.stop()
    rgb_out.release()
    depth_out.release()
    cv2.destroyAllWindows()

    # Save all heightmaps as one pickle
    with open(pkl_filename, 'wb') as f:
        pickle.dump({
            'heightmaps': heightmaps_list,
            'depth_scale': depth_scale,
            'frame_shape': height_map.shape,
            'num_frames': len(heightmaps_list)
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"All recordings stopped.\nRGB video: {rgb_filename}\nDepth video: {depth_filename}\nHeightmaps pickle: {pkl_filename}")
