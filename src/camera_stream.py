import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Setup filenames with timestamp
timestamp = int(time.time())
name = 'ameline_test_d-'
num = '1'
bag_filename = f"{name}_{timestamp}.bag"
rgb_filename = f"{name}_rgb_{timestamp}.mp4"
depth_filename = f"{name}_depth_{timestamp}.mp4"

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Enable recording to bag file
config.enable_record_to_file(bag_filename)

# Start streaming
pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Setup video writers for RGB and Depth
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
frame_size = (640, 480)

rgb_out = cv2.VideoWriter(rgb_filename, fourcc, fps, frame_size)
depth_out = cv2.VideoWriter(depth_filename, fourcc, fps, frame_size)

print(f"Recording started.\nSaving RGB video as: {rgb_filename}\nSaving Depth video as: {depth_filename}\nRecording .bag file as: {bag_filename}")
print("Press Ctrl+C or ESC to stop recording.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Color map for depth for visualization and saving
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Write RGB and Depth videos
        rgb_out.write(color_image)
        depth_out.write(depth_colormap)

        # Show preview side by side
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("RGB (left) + Depth (right)", combined)

        key = cv2.waitKey(1)
        if key == 27:
            print("ESC pressed. Stopping recording...")
            break

except KeyboardInterrupt:
    print("\nCtrl+C detected. Stopping recording...")

finally:
    pipeline.stop()
    rgb_out.release()
    depth_out.release()
    cv2.destroyAllWindows()
    print("All recordings stopped and files saved.")
