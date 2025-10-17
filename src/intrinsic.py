import pyrealsense2 as rs

# Start a pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get stream profiles
color_profile = profile.get_stream(rs.stream.color)
depth_profile = profile.get_stream(rs.stream.depth)

# Extract intrinsics
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

print("Color intrinsics:")
print(color_intrinsics)

print("\nDepth intrinsics:")
print(depth_intrinsics)

pipeline.stop()
