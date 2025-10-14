import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Align depth to color frame
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply color map to depth image for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show the frames
        cv2.imshow('RealSense RGB + Depth', images)

        # Exit on ESC key
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
