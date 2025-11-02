import os
# Set Qt platform before importing cv2 to avoid plugin errors (use 'xcb' for X11)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import re
import time
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2

# Helper: find next numeric index for filenames like jestin_video_#.pkl
def get_next_index(prefix='jestin_video', directory='.', extension='.pkl'):
    files = os.listdir(directory)
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+){re.escape(extension)}$')
    idxs = []
    for f in files:
        m = pattern.match(f)
        if m:
            try:
                idxs.append(int(m.group(1)))
            except ValueError:
                pass
    return max(idxs) + 1 if idxs else 1

# Configure RealSense streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Check for device
ctx = rs.context()
if len(ctx.devices) == 0:
    print("No RealSense device found. Connect the camera and try again.")
    raise SystemExit(1)

pipeline.start(config)

# Obtain depth scale (meters per depth unit)
try:
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} m/unit")
except Exception:
    depth_scale = 1.0
    print("Warning: could not read depth scale, defaulting to 1.0")

# Align depth to color
align = rs.align(rs.stream.color)

# Next save index
save_index = get_next_index(prefix='jestin_video', directory='.', extension='.pkl')

# Lists to store the video sequence
rgb_frames = []    # List of aligned BGR uint8 images (1280x720)
height_frames = [] # List of height maps in meters (720x1280 float32, NaN for invalid)
timestamps = []    # List of float timestamps for each frame

# Video writer (will be initialized when recording starts)
video_writer = None

try:
    timeout_count = 0
    max_timeouts = 3
    recording = False
    fps = 30  # Target FPS for the output MP4 (matches stream configuration)

    while True:
        try:
            frames = pipeline.wait_for_frames(5000)
        except RuntimeError as e:
            msg = str(e).lower()
            if "frame didn't arrive" in msg or "timeout" in msg:
                timeout_count += 1
                print(f"Warning: frame timeout ({timeout_count}/{max_timeouts})")
                if timeout_count >= max_timeouts:
                    print("Restarting pipeline to recover from repeated timeouts...")
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    pipeline.start(config)
                    timeout_count = 0
                continue
            else:
                raise

        timeout_count = 0

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        aligned = align.process(frames)
        aligned_depth = aligned.get_depth_frame()
        aligned_color = aligned.get_color_frame()

        depth_image = np.asanyarray(aligned_depth.get_data())
        color_image = np.asanyarray(aligned_color.get_data())

        # Create height map (meters). Mark invalid depth (0) as NaN.
        height_map = depth_image.astype(np.float32) * float(depth_scale)
        height_map[depth_image == 0] = np.nan

        # Handle recording
        if recording:
            rgb_frames.append(color_image.copy())
            height_frames.append(height_map.copy())
            timestamps.append(time.time())
            
            # Write frame to MP4
            if video_writer is not None:
                video_writer.write(color_image)
            
            print(f"Recorded frame {len(rgb_frames)}")

        cv2.imshow('Color Stream', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Stream', depth_colormap)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            if not recording:
                # Start recording
                recording = True
                rgb_frames = []
                height_frames = []
                timestamps = []
                
                # Initialize video writer for MP4 (BGR frames)
                mp4_name = f"jestin_video_{save_index}_rgb.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
                video_writer = cv2.VideoWriter(mp4_name, fourcc, fps, (color_image.shape[1], color_image.shape[0]))
                
                if not video_writer.isOpened():
                    print("Error: Could not open VideoWriter for MP4. Recording RGB video disabled.")
                    video_writer = None
                else:
                    print(f"Recording started. RGB video will be saved to {mp4_name}. Press 'k' again to stop.")
            else:
                # Stop recording
                recording = False
                
                # Release video writer
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print(f"RGB video saved to {mp4_name} ({len(rgb_frames)} frames at {fps} FPS)")
                
                # Save the collected sequence to a single pickle file (including RGB frames)
                pickle_name = f"jestin_video_{save_index}.pkl"
                to_save = {
                    'rgb_frames': rgb_frames,          # List of BGR uint8 images
                    'height_frames': height_frames,    # List of float32 height maps in meters
                    'timestamps': timestamps,          # List of float timestamps
                    'depth_scale': float(depth_scale),
                    'frame_shape': color_image.shape,  # (height, width, channels) for RGB
                    'height_shape': height_map.shape,  # (height, width) for height maps
                    'num_frames': len(rgb_frames),
                    'start_time': timestamps[0] if timestamps else None,
                    'end_time': timestamps[-1] if timestamps else None,
                    'fps': fps,
                }
                with open(pickle_name, 'wb') as f:
                    pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Pickle saved to {pickle_name} ({len(rgb_frames)} frames)")
                save_index += 1

        if key == ord('q') or key == 27:
            break

finally:
    if recording and video_writer is not None:
        video_writer.release()
    try:
        pipeline.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()