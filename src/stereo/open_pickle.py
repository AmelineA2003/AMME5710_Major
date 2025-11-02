import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open('realsense_rgb_height.pkl', 'rb') as f:
    data = pickle.load(f)

rgb_frames = data['rgb_frames']
height_frames = data['height_frames']

print(height_frames[0])

# for rgb, height in zip(rgb_frames, height_frames):
#     cv2.imshow('RGB', rgb)
#     cv2.imshow('Height Map', height)
#     if cv2.waitKey(30) & 0xFF == ord('q'):  # 30 ms for ~30 FPS
#         break
# cv2.destroyAllWindows()