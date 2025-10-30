# debug_viz.py
import cv2
import numpy as np
from typing import List, Tuple

Color = Tuple[int, int, int]

def draw_boxes(img, dets: List[List[float]], color: Color, label: str = "", with_score: bool = True):
    out = img.copy()
    for d in dets or []:
        x, y, w, h = map(int, d[:4])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if label:
            txt = label
            if with_score and len(d) > 4:
                try:
                    txt = f"{label} {float(d[4]):.2f}"
                except:
                    pass
            cv2.putText(out, txt, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out

def overlay_mask(img_bgr, mask_u8, alpha=0.6):
    """Overlay a binary/grayscale mask onto a BGR image using a heat colormap."""
    h, w = img_bgr.shape[:2]
    if mask_u8 is None:
        return img_bgr.copy()
    m = mask_u8
    if len(m.shape) == 2:
        # normalize to 0..255 if not already
        if m.max() <= 1:
            m = (m * 255).astype(np.uint8)
        else:
            m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)

def label(img_bgr, text: str):
    out = img_bgr.copy()
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return out

def tile_panels(panels: List[np.ndarray], cols=3, bg=(30,30,30)):
    """Tile images of equal size into a grid."""
    if not panels:
        return None
    h, w = panels[0].shape[:2]
    rows = int(np.ceil(len(panels) / cols))
    canvas = np.full((rows * h, cols * w, 3), bg, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = p
    return canvas

class VideoSink:
    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.w = cv2.VideoWriter(path, fourcc, fps, size)
        self.ok = self.w.isOpened()
        self.path = path
    def write(self, frame: np.ndarray):
        if self.ok:
            self.w.write(frame)
    def release(self):
        if self.ok:
            self.w.release()
