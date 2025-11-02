import pickle
import numpy as np
import cv2
import time

# Load the captured data from the Pickle file
pickle_file = 'realsense_rgb_height.pkl'  # Ensure this matches the saved filename

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

rgb_frames = data['rgb_frames']       # List of BGR uint8 arrays (480x848x3)
height_frames = data['height_frames'] # List of float32 arrays (480x640) in meters; NaN for invalid

# Optional: Retrieve metadata if needed
plane_point = data.get('plane_point')
plane_normal = data.get('plane_normal')
units = data.get('units', 'meters')
height_map_shape = data.get('height_map_shape', (480, 640))

print(f"Loaded {len(rgb_frames)} frames from '{pickle_file}'.")
if len(rgb_frames) != len(height_frames):
    raise ValueError("Mismatch between RGB and height frame counts.")
print(f"Height maps: {height_map_shape[0]}x{height_map_shape[1]} resolution, in {units} (signed distance; NaN for invalid).")

# Playback parameters
fps = 30  # Target frames per second (match capture rate)
delay_ms = int(1000 / fps)  # Delay between frames in milliseconds
height_display_range = (-1.0, 1.0)  # Adjust visualization clip range (min_height, max_height) in meters

# Optional: Resize RGB for display if too large (e.g., scale down by 50%)
resize_factor = 0.75  # 1.0 for original size; <1.0 to shrink
if resize_factor != 1.0:
    new_width = int(848 * resize_factor)
    new_height = int(480 * resize_factor)

# Playback loop
print("Starting playback. Press 'q' to quit, 'p' to pause/resume.")
paused = False
frame_idx = 0

while frame_idx < len(rgb_frames):
    if not paused:
        rgb = rgb_frames[frame_idx]
        height = height_frames[frame_idx]

        # Prepare RGB for display
        rgb_display = rgb.copy()
        if resize_factor != 1.0:
            rgb_display = cv2.resize(rgb_display, (new_width, new_height))

        # Prepare height map visualization (colormap for better contrast)
        height_viz = height.copy()
        # Clip to display range and handle NaN
        height_viz = np.clip(height_viz, height_display_range[0], height_display_range[1])
        height_viz[np.isnan(height)] = height_display_range[0]  # Treat NaN as minimum for visualization
        
        # Normalize to 0-1 for colormap
        norm = (height_viz - height_display_range[0]) / (height_display_range[1] - height_display_range[0])
        norm_8bit = (norm * 255).astype(np.uint8)
        
        # Apply JET colormap (blue: low, red: high)
        height_colormap = cv2.applyColorMap(norm_8bit, cv2.COLORMAP_JET)
        
        # Resize height map to match RGB display if resized
        if resize_factor != 1.0:
            height_colormap = cv2.resize(height_colormap, (new_width, new_height))
        
        # Overlay text with statistics
        valid_heights = height[~np.isnan(height)]
        if valid_heights.size > 0:
            h_min, h_max, h_mean = valid_heights.min(), valid_heights.max(), valid_heights.mean()
            stats_text = f"Frame {frame_idx + 1}/{len(rgb_frames)} | Min: {h_min:.3f}m | Max: {h_max:.3f}m | Mean: {h_mean:.3f}m"
        else:
            stats_text = f"Frame {frame_idx + 1}/{len(rgb_frames)} | No valid heights"
        
        cv2.putText(rgb_display, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display windows (side-by-side if desired; here stacked for single window)
        combined = np.hstack((rgb_display, height_colormap)) if resize_factor == 1.0 else np.hstack((rgb_display, height_colormap))
        cv2.imshow('RGB (Left) | Height Map (Right) - JET Colormap', combined)
        
        frame_idx += 1
        if frame_idx >= len(rgb_frames):
            print("Playback complete.")
            break
    
    # Handle key presses
    key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF
    if key == ord('q'):
        print("Playback quit by user.")
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif key == ord('r'):  # Rewind to start
        frame_idx = 0
        print("Rewound to start.")
    
    # Manual frame navigation when paused
    if paused:
        if key == ord('n'):  # Next frame
            frame_idx = min(frame_idx + 1, len(rgb_frames) - 1)
        elif key == ord('b'):  # Previous frame
            frame_idx = max(frame_idx - 1, 0)

cv2.destroyAllWindows()
print("Playback window closed.")

# Optional: Save playback as video (uncomment to enable)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_video = cv2.VideoWriter('playback_video.mp4', fourcc, fps, (combined.shape[1], combined.shape[0]))
# for i in range(len(rgb_frames)):
#     # Recompute combined for each frame (simplified; rerun visualization logic)
#     # ...
#     out_video.write(combined)
# out_video.release()
# print("Saved playback video to 'playback_video.mp4'.")