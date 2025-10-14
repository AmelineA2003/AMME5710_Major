import pyrealsense2 as rs
import numpy as np
import cv2

bag_file = "20251014_133748.bag"

pipeline = rs.pipeline()
config = rs.config()

# Set pipeline to read from the bag file (no explicit stream enabling)
config.enable_device_from_file(bag_file, repeat_playback=False)

pipeline.start(config)

# Prepare VideoWriter after starting pipeline (to get stream profile info)
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
width, height = intrinsics.width, intrinsics.height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            break

        color_image = np.asanyarray(color_frame.get_data())

        out.write(color_image)

        cv2.imshow('Color Frame', color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print(f"Playback finished or error: {e}")

finally:
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()

print("Done! Video saved as output_video.mp4")
