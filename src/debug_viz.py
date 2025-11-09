import cv2
import numpy as np
from typing import List, Tuple

Color = Tuple[int, int, int]


def draw_boxes(
    img: np.ndarray,
    dets: List[List[float]],
    color: Color,
    label: str = "",
    with_score: bool = True,
) -> np.ndarray:
    """Draw axis-aligned boxes (and optional scores) onto an image.

    Args:
        img: Input image (HxWx3, BGR, uint8).
        dets: Detections as [[x, y, w, h, score?], ...]. Score is optional.
        color: BGR color for rectangles and text, e.g., (0, 255, 0).
        label: Text prefix to render above each box (e.g., "RULES" or "HOG").
        with_score: If True and a score is present, append it to the label.

    Returns:
        A copy of `img` with rectangles (and labels) drawn.
    """
    out = img.copy()
    for d in dets or []:
        x, y, w, h = map(int, d[:4])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if label:
            txt = label
            if with_score and len(d) > 4:
                try:
                    txt = f"{label} {float(d[4]):.2f}"
                except Exception:
                    pass
            cv2.putText(
                out,
                txt,
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    return out


def overlay_mask(img_bgr: np.ndarray, mask_u8: np.ndarray | None, alpha: float = 0.6) -> np.ndarray:
    """Overlay a binary/gray mask as a heatmap on a BGR image.

    The mask is normalized to 0..255 if needed, resized to the image size,
    mapped with COLORMAP_JET, and alpha-blended over the input.

    Args:
        img_bgr: Base image (HxWx3, BGR, uint8).
        mask_u8: Mask image (HxW, uint8 or {0,1}) or None to return a copy of `img_bgr`.
        alpha: Heatmap opacity in [0, 1]; higher = stronger heatmap.

    Returns:
        A blended visualization image (same shape as `img_bgr`).
    """
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


def label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """Render a legible label at the top-left corner of an image.

    Draws white text with a thin black outline for contrast.

    Args:
        img_bgr: Input image (HxWx3, BGR, uint8).
        text: Text string to draw.

    Returns:
        A copy of `img_bgr` with the label rendered.
    """
    out = img_bgr.copy()
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return out


def tile_panels(panels: List[np.ndarray], cols: int = 3, bg: Tuple[int, int, int] = (30, 30, 30)) -> np.ndarray | None:
    """Tile equally sized images into a fixed-column grid.

    Grayscale inputs are auto-converted to BGR for consistent tiling.

    Args:
        panels: List of images (all HxW or HxWx3). Must all be the same size.
        cols: Number of columns in the grid (rows are computed automatically).
        bg: BGR background color for the canvas.

    Returns:
        A single tiled image (rows*h x cols*w x 3) or ``None`` if `panels` is empty.
    """
    if not panels:
        return None
    h, w = panels[0].shape[:2]
    rows = int(np.ceil(len(panels) / cols))
    canvas = np.full((rows * h, cols * w, 3), bg, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = p
    return canvas


class VideoSink:
    """Thin wrapper around OpenCV VideoWriter for MP4 output.

    Attributes:
        w: The underlying cv2.VideoWriter instance.
        ok: True if the writer opened successfully.
        path: Output file path.

    Notes:
        - The `size` argument must be (width, height) to match OpenCV's API.
        - Encodes using fourcc "mp4v".
    """

    def __init__(self, path: str, fps: float, size: Tuple[int, int]) -> None:
        """Create a video writer.

        Args:
            path: Output video path (e.g., "outputs/run.mp4").
            fps: Frames per second (e.g., 25.0).
            size: Frame size as (width, height).

        Side Effects:
            Opens an internal VideoWriter and sets `ok` to indicate success.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.w = cv2.VideoWriter(path, fourcc, fps, size)
        self.ok = self.w.isOpened()
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        """Append a frame to the video if the sink is open.

        Args:
            frame: Image frame (height, width) matching the initialized `size`.
        """
        if self.ok:
            self.w.write(frame)

    def release(self) -> None:
        """Finalize and close the video file if open.

        After calling this, the sink should not be used again.
        """
        if self.ok:
            self.w.release()
