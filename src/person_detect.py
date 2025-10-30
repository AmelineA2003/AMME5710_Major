# person_detect.py
# Notebook-faithful detector with optional HOG box merging (NMS or WBF-style),
# optional GrabCut refining, and a debug entrypoint to expose intermediate stages.
# - USE_YOLO flag controls backend
# - YOLOv8 if available (cls==0 persons only), else HOG
# - Returns list of [x, y, w, h, score]

from typing import List, Tuple
import cv2
import numpy as np

# -----------------------------
# Global config (notebook-like)
# -----------------------------
USE_YOLO = False          # Set True if you have `pip install ultralytics` + weights
detector_name = "HOG"
yolo_model = None

# ---- HOG merge options ----
HOG_MERGE = True          # enable/disable merging of overlapping HOG boxes
HOG_NMS_IOU = 0.50        # IoU for clustering (both NMS and WBF)
HOG_SCORE_THRESH = 0.20   # ignore very low HOG scores before merging
HOG_USE_WBF = False       # False = hard NMS; True = weighted-average (WBF-like)

# Grouping knobs (available on some OpenCV builds only; code falls back when unsupported)
HOG_USE_MEANSHIFT_GROUPING = True
HOG_FINAL_THRESHOLD = 2.0

# Full-body shaping after merge (nudges to human aspect and expands a bit)
FULLBODY_TARGET_AR = 0.45  # ~0.4–0.55 typical for standing person (w/h)
FULLBODY_EXPAND = 0.12     # fractional growth after shaping

# ---- GrabCut refinement (optional) ----
# Set to "grabcut" to enable or None to disable
HOG_REFINE = "grabcut"      # {"grabcut", None}
HOG_GRABCUT_ITERS = 3       # 2–5 typical
HOG_GRABCUT_INNER = 0.60    # inner seed size (fraction of box side)

# YOLO (optional)
if USE_YOLO:
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")   # change to your weights if needed
        detector_name = "YOLOv8"
    except Exception as e:
        print("YOLOv8 not available, falling back to HOG. Reason:", e)
        USE_YOLO = False
        detector_name = "HOG"

# ---- HOG setup ----
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# -----------------------------
# Utility helpers
# -----------------------------
def _normalize_nms_indices(idxs) -> List[int]:
    """Make OpenCV NMSBoxes outputs consistent across versions."""
    if idxs is None:
        return []
    try:
        arr = np.asarray(idxs).reshape(-1)
    except Exception:
        arr = np.array([idxs], dtype=np.int64)
    return [int(i) for i in arr.tolist() if i is not None]

def _clip_box_to_image(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, x); y = max(0, y)
    w = max(0, w); h = max(0, h)
    if w == 0 or h == 0:
        return x, y, w, h
    x2 = min(W, x + w); y2 = min(H, y + h)
    x = min(x, max(0, x2 - 1)); y = min(y, max(0, y2 - 1))
    w = max(0, x2 - x); h = max(0, y2 - y)
    return x, y, w, h


# -----------------------------
# Merging
# -----------------------------
def _hog_merge_nms(dets: List[List[float]],
                   score_thresh: float = 0.0,
                   iou: float = 0.5) -> List[List[float]]:
    """Hard NMS for HOG: choose one box per cluster."""
    if not dets:
        return []
    keep = [d for d in dets if d[4] >= score_thresh]
    if not keep:
        return []
    bboxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h, _ in keep]
    scores = [float(s) for *_, s in keep]
    idxs = cv2.dnn.NMSBoxes(bboxes=bboxes, scores=scores,
                            score_threshold=float(score_thresh),
                            nms_threshold=float(iou))
    idx_list = _normalize_nms_indices(idxs)
    if len(idx_list) == 0:
        return []
    return [keep[i] for i in idx_list]

def _hog_merge_wbf(dets: List[List[float]],
                   score_thresh: float = 0.0,
                   iou: float = 0.5) -> List[List[float]]:
    """
    Weighted-box-fusion-like merge for HOG:
    cluster by IoU, then weighted-average (x,y,w,h) by score.
    """
    if not dets:
        return []
    dets = sorted([d for d in dets if d[4] >= score_thresh], key=lambda d: d[4], reverse=True)
    if not dets:
        return []
    merged: List[List[float]] = []
    used = [False] * len(dets)

    def iou_xywh(a, b) -> float:
        ax, ay, aw, ah, _ = a
        bx, by, bw, bh, _ = b
        ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
        bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = aw * ah + bw * bh - inter
        return inter / ua if ua > 0 else 0.0

    for i, di in enumerate(dets):
        if used[i]:
            continue
        cluster = [di]
        used[i] = True
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            if iou_xywh(di, dets[j]) >= iou:
                used[j] = True
                cluster.append(dets[j])

        if len(cluster) == 1:
            merged.append(di)
            continue

        ws = np.array([c[4] for c in cluster], dtype=np.float32)
        ws /= (ws.sum() + 1e-8)
        xs = np.array([c[0] for c in cluster], dtype=np.float32)
        ys = np.array([c[1] for c in cluster], dtype=np.float32)
        ww = np.array([c[2] for c in cluster], dtype=np.float32)
        hh = np.array([c[3] for c in cluster], dtype=np.float32)
        x = float((xs * ws).sum())
        y = float((ys * ws).sum())
        w = float((ww * ws).sum())
        h = float((hh * ws).sum())
        score = float(max(c[4] for c in cluster))  # keep strongest
        merged.append([int(round(x)), int(round(y)), int(round(w)), int(round(h)), score])

    return merged


# -----------------------------
# Full-body shaper
# -----------------------------
def _shape_fullbody(dets, target_ar=0.45, expand=0.12):
    """
    Coerce boxes toward a full-body aspect (w/h ~ target_ar) and expand slightly.
    Keeps center fixed.
    """
    shaped = []
    for x, y, w, h, s in dets:
        if h <= 0:
            shaped.append([x, y, w, h, s]); continue
        ar = w / float(h)
        cx, cy = x + w * 0.5, y + h * 0.5
        if ar < target_ar:         # too tall/narrow → widen
            w2 = target_ar * h
            w2 = max(w2, w)
            w, h = w2, h
        else:                      # too wide/short → heighten
            h2 = w / max(target_ar, 1e-6)
            h2 = max(h2, h)
            w, h = w, h2
        w *= (1.0 + expand)
        h *= (1.0 + expand)
        x2 = int(round(cx - w * 0.5))
        y2 = int(round(cy - h * 0.5))
        w2 = int(round(w))
        h2 = int(round(h))
        shaped.append([x2, y2, w2, h2, s])
    return shaped


# -----------------------------
# GrabCut refinement
# -----------------------------
def _refine_box_with_grabcut(frame: np.ndarray, box: Tuple[int,int,int,int],
                             iters: int = 3, inner: float = 0.6) -> Tuple[int,int,int,int]:
    """
    Refine a box using GrabCut inside the ROI:
    - Seed an inner fraction as sure-foreground.
    - Return tight bounding box around GrabCut foreground.
    """
    H, W = frame.shape[:2]
    x, y, w, h = _clip_box_to_image(box[0], box[1], box[2], box[3], W, H)
    if w <= 2 or h <= 2:
        return box
    x1, y1, x2, y2 = x, y, x + w, y + h
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return box

    mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
    ih, iw = roi.shape[0], roi.shape[1]
    ix = int((1 - inner) / 2 * iw);  iy = int((1 - inner) / 2 * ih)
    iw2 = max(1, int(inner * iw));   ih2 = max(1, int(inner * ih))
    mask[iy:iy+ih2, ix:ix+iw2] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, None, bgdModel, fgdModel, max(1, iters), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return box  # be robust to OpenCV errors

    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return box

    xx1, xx2 = int(xs.min()), int(xs.max())
    yy1, yy2 = int(ys.min()), int(ys.max())
    rx, ry = x1 + xx1, y1 + yy1
    rw, rh = xx2 - xx1 + 1, yy2 - yy1 + 1
    return _clip_box_to_image(rx, ry, rw, rh, W, H)


# -----------------------------
# Backends
# -----------------------------
def detect_people_hog(frame) -> List[List[float]]:
    """
    Returns list of [x, y, w, h, score] using OpenCV HOG.
    'score' is the SVM distance (not a probability).
    """
    # Robust HOG call across OpenCV versions
    # person_detect.py (module knobs)
    HOG_HIT_THRESHOLD = 0.5   # try 0.2 → 0.8; higher = fewer detections

    # inside detect_people_hog(), call with positional args to avoid kw issues:
    try:
        rects, weights = hog.detectMultiScale(
            frame,
            HOG_HIT_THRESHOLD,      # <- hitThreshold
            (8, 8),                 # winStride
            (8, 8),                 # padding
            1.03                    # scale
            # omit finalThreshold/useMeanshiftGrouping for portability
        )
    except (TypeError, cv2.error):
        rects, weights = hog.detectMultiScale(
            frame,
            HOG_HIT_THRESHOLD,
            (8, 8),
            (8, 8),
            1.03
        )

    raw = [[int(x), int(y), int(w), int(h), float(s)] for (x, y, w, h), s in zip(rects, weights)]

    # Merge duplicates
    if HOG_MERGE:
        merged = _hog_merge_wbf(raw, HOG_SCORE_THRESH, HOG_NMS_IOU) if HOG_USE_WBF \
                 else _hog_merge_nms(raw, HOG_SCORE_THRESH, HOG_NMS_IOU)
    else:
        merged = raw

    # Shape to full body
    shaped = _shape_fullbody(merged, target_ar=FULLBODY_TARGET_AR, expand=FULLBODY_EXPAND)

    # Optional GrabCut refine per box
    if HOG_REFINE == "grabcut":
        refined = []
        for x, y, w, h, s in shaped:
            rx, ry, rw, rh = _refine_box_with_grabcut(frame, (x, y, w, h),
                                                      iters=HOG_GRABCUT_ITERS,
                                                      inner=HOG_GRABCUT_INNER)
            refined.append([rx, ry, rw, rh, s])
        return refined

    return shaped


def detect_people_yolo(frame) -> List[List[float]]:
    """
    Returns list of [x, y, w, h, score] using YOLOv8 (Ultralytics).
    Filters to class 0 (person).
    """
    results = yolo_model(frame, conf=0.60, iou=0.7, classes=[0], verbose=False)
    out: List[List[float]] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls.item()) if b.cls is not None else -1
            if cls != 0 and cls != -1:
                continue
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            w, h = x2 - x1, y2 - y1
            score = float(b.conf.item()) if b.conf is not None else 0.0
            out.append([x1, y1, w, h, score])
    return out


def detect_people(frame) -> List[List[float]]:
    """
    Dispatcher exactly as in the notebook:
    - If USE_YOLO and the model loaded, use YOLOv8.
    - Otherwise, use HOG (with optional merging and GrabCut).
    """
    if USE_YOLO and yolo_model is not None:
        return detect_people_yolo(frame)
    return detect_people_hog(frame)


# -----------------------------------------
# Debug entrypoint (for per-stage videos)
# -----------------------------------------
def detect_people_debug(frame):
    """
    Returns (final_dets, debug) where:
      final_dets: [[x,y,w,h,score], ...] (same as detect_people(frame))
      debug: dict with optional keys:
         'raw_hog', 'merged', 'shaped', 'grabcut' -> per-stage lists of boxes
    For YOLO backend, only 'final' is provided.
    """
    debug = {}

    # YOLO path
    if USE_YOLO and yolo_model is not None:
        final = detect_people_yolo(frame)
        debug["final"] = final
        return final, debug

    # person_detect.py (module knobs)
    HOG_HIT_THRESHOLD = 0.7   # try 0.2 → 0.8; higher = fewer detections

    # inside detect_people_hog(), call with positional args to avoid kw issues:
    try:
        rects, weights = hog.detectMultiScale(
            frame,
            HOG_HIT_THRESHOLD,      # <- hitThreshold
            (8, 8),                 # winStride
            (8, 8),                 # padding
            1.03                    # scale
            # omit finalThreshold/useMeanshiftGrouping for portability
        )
    except (TypeError, cv2.error):
        rects, weights = hog.detectMultiScale(
            frame,
            HOG_HIT_THRESHOLD,
            (8, 8),
            (8, 8),
            1.03
        )

    raw = [[int(x), int(y), int(w), int(h), float(s)] for (x, y, w, h), s in zip(rects, weights)]
    debug["raw_hog"] = raw

    # merge
    if HOG_MERGE:
        merged = _hog_merge_wbf(raw, HOG_SCORE_THRESH, HOG_NMS_IOU) if HOG_USE_WBF \
                 else _hog_merge_nms(raw, HOG_SCORE_THRESH, HOG_NMS_IOU)
    else:
        merged = raw
    debug["merged"] = merged

    # shape
    shaped = _shape_fullbody(merged, target_ar=FULLBODY_TARGET_AR, expand=FULLBODY_EXPAND)
    debug["shaped"] = shaped

    # grabcut
    if HOG_REFINE == "grabcut":
        refined = []
        for x, y, w, h, s in shaped:
            rx, ry, rw, rh = _refine_box_with_grabcut(frame, (x, y, w, h),
                                                      iters=HOG_GRABCUT_ITERS,
                                                      inner=HOG_GRABCUT_INNER)
            refined.append([rx, ry, rw, rh, s])
        debug["grabcut"] = refined
        final = refined
    else:
        final = shaped

    return final, debug
