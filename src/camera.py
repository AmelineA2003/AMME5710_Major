import os
# Set Qt platform before importing cv2 to avoid plugin errors (use 'xcb' for X11)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import re
import time
import pickle
import pyrealsense2 as rs
import numpy as np
import cv2

# Helper: find next numeric index for filenames like jestin_#_rgb.png
def get_next_index(prefix='jestin', directory='.'):
    files = os.listdir(directory)
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)_rgb(?:\..+)?$')
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
save_index = get_next_index(prefix='jestin', directory='.')

try:
    timeout_count = 0
    max_timeouts = 3

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

        cv2.imshow('Color Stream', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Stream', depth_colormap)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):
            rgb_name = f"jestinp_{save_index}_rgb.png"
            pickle_name = f"jestip_{save_index}_height.pkl"

            # Save RGB image (BGR)
            cv2.imwrite(rgb_name, color_image)

            # Create height map (meters). Mark invalid depth (0) as NaN.
            height_map = depth_image.astype(np.float32) * float(depth_scale)
            height_map[depth_image == 0] = np.nan

            # Save pickle with metadata and height map
            to_save = {
                'height_m': height_map,
                'depth_scale': float(depth_scale),
                'timestamp': time.time(),
                'shape': height_map.shape,
            }
            with open(pickle_name, 'wb') as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Saved {rgb_name} and {pickle_name}")
            save_index += 1

        if key == ord('q') or key == 27:
            break

finally:
    try:
        pipeline.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
