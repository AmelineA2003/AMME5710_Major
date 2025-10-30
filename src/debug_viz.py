# debug_viz.py
import os
import cv2
from typing import List, Tuple

Color = Tuple[int,int,int]

def draw_boxes(img, dets: List[List[float]], color: Color, label: str = ""):
    """Draw boxes [x,y,w,h,score] in given color with an optional label."""
    out = img.copy()
    for d in dets or []:
        x,y,w,h = map(int, d[:4])
        cv2.rectangle(out, (x,y), (x+w,y+h), color, 2)
        if label:
            txt = f"{label}"
            if len(d) > 4: txt += f" {d[4]:.2f}"
            cv2.putText(out, txt, (x, max(0,y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out

class VideoSink:
    def __init__(self, path: str, fps: float, size: Tuple[int,int]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.w = cv2.VideoWriter(path, fourcc, fps, size)
        self.ok = self.w.isOpened()
        self.path = path
    def write(self, frame):
        if self.ok:
            self.w.write(frame)
    def release(self):
        if self.ok:
            self.w.release()
