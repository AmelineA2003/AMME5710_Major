# person_detect.py
# RULES (non-ML) detector with rich debug artefacts, resolution-invariant tuning,
# and selectable background models. ACCUM mode includes blur+EMA+tiny-CC pruning.
# HOG and YOLO back-ends are preserved.
#
# Public APIs:
#   - detect_people(frame, frame_idx=0) -> [[x,y,w,h,score], ...]
#   - detect_people_debug(frame, frame_idx=0) -> (final, debug_dict)
#
# Debug dictionary (RULES):
#   boxes: 'contours','filtered','edges_ok','nms','shaped','final'
#   images: 'img_bg','img_mask','img_diff','img_edge'

from typing import List, Tuple, Dict, Optional
from collections import deque
import cv2
import numpy as np

# -----------------------------
# Global config / modes
# -----------------------------
DETECTOR_MODE = "RULES"     # "RULES" | "HOG" | "YOLO"

USE_YOLO = False            # only if DETECTOR_MODE == "YOLO"
detector_name = "RULES"
yolo_model = None

# -----------------------------
# HOG config (if DETECTOR_MODE == "HOG")
# -----------------------------
HOG_HIT_THRESHOLD = 0.0
HOG_WINSTRIDE = (8, 8)
HOG_PADDING = (8, 8)
HOG_SCALE = 1.03

HOG_MERGE = True
HOG_NMS_IOU = 0.50
HOG_SCORE_THRESH = 0.20
HOG_USE_WBF = False  # simple WBF stub below; NMS is default/faster

# Full-body shaping (shared)
FULLBODY_TARGET_AR = 0.45
FULLBODY_EXPAND = 0.12

# -----------------------------
# RULES detector knobs (resolution-invariant)
# -----------------------------
# Background mode: "ACCUM" (running avg), "MEDIAN" (rolling median),
# "DIFF" (t-k frame differencing), or "MOG2" (OpenCV GMM)
RULES_BG_MODE = "MOG2"

# MOG2 params (used only if RULES_BG_MODE == "MOG2")
RULES_BG_HISTORY = 300
RULES_BG_VARTHRESH = 14
RULES_BG_SHADOWS = True
RULES_MOTION_BIN_THRESH = 165          # threshold on MOG2/other masks to binarize (0..255)

# ACCUM (running average) params — baseline for stop-and-go mall scene
RULES_ACCUM_ALPHA   = 0.010            # slowish learning; paused people don’t vanish
RULES_ACCUM_THRESH  = 32               # raise to cut flicker (26–32 typical)

# MEDIAN (rolling temporal median) params
RULES_MEDIAN_WINDOW = 21               # (unused in ACCUM)
RULES_MEDIAN_THRESH = 20

# DIFF (frame differencing) params
RULES_DIFF_STRIDE   = 5                # (unused in ACCUM)
RULES_DIFF_THRESH   = 24

# Motion overlap fraction (0..1) required inside a candidate box
RULES_MOTION_MIN_OVERLAP = 0.11        # allow brief pauses

# RESOLUTION-INVARIANT geometry/morph settings
RULES_MIN_AREA_FRAC = 0.009            # ~0.9% of frame; lower for tiny/far people
RULES_ASPECT_MIN = 0.35
RULES_ASPECT_MAX = 0.75

# Morphology kernel sizes as a fraction of the *short* side
RULES_MORPH_OPEN_FRAC  = 0.01         # denoise salt-and-pepper
RULES_MORPH_CLOSE_FRAC = 0.03         # fill holes in bodies

# Edge-density scoring
RULES_EDGE_MAG_THRESH = 28             # require stronger edges
RULES_EDGE_MIN_FRAC   = 0.06           # a bit more edge content

# NMS
RULES_NMS_IOU = 0.30

# Process throttling (1 = every frame)
RULES_PROCESS_EVERY_N = 1

# ---- Minimal stabilisers between diff -> mask ----
RULES_DIFF_BLUR_K = 12                 # Gaussian blur before threshold
RULES_MASK_TEMPORAL_ALPHA = 0.23       # EMA to dampen mask flicker (0.15–0.25 good)
RULES_MIN_CC_AREA_FRAC = 0.0007        # prune tiny components (~0.06% of frame)

# -----------------------------
# Init per-backend resources
# -----------------------------
# YOLO init (only if requested)
if DETECTOR_MODE == "YOLO" and USE_YOLO:
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        detector_name = "YOLOv8"
    except Exception as e:
        print("YOLOv8 not available. Falling back to RULES. Reason:", e)
        DETECTOR_MODE = "RULES"
        detector_name = "RULES"

# HOG init (only if requested)
hog = None
if DETECTOR_MODE == "HOG":
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    detector_name = "HOG"

# RULES init: MOG2 only if chosen
_bg = None
if DETECTOR_MODE == "RULES" and RULES_BG_MODE == "MOG2":
    _bg = cv2.createBackgroundSubtractorMOG2(
        history=RULES_BG_HISTORY, varThreshold=RULES_BG_VARTHRESH,
        detectShadows=RULES_BG_SHADOWS
    )
    detector_name = "RULES"

# ML-free background buffers
_bg_accum_f32: Optional[np.ndarray] = None      # for ACCUM
_bg_median_buf: deque = deque(maxlen=0)         # for MEDIAN
_prev_gray_buf: deque = deque(maxlen=0)         # for DIFF

# Temporal EMA for mask
_mask_ema: Optional[np.ndarray] = None

# -----------------------------
# Utils
# -----------------------------
def _nms_xywh(dets: List[List[float]], iou_thresh: float) -> List[List[float]]:
    if not dets:
        return []
    boxes = np.array([[d[0], d[1], d[0]+d[2], d[1]+d[3]] for d in dets], dtype=np.float32)
    scores = np.array([d[4] if len(d) > 4 else 1.0 for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    idxs = scores.argsort()[::-1]
    keep_idxs = []
    while idxs.size > 0:
        i = idxs[0]
        keep_idxs.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs[1:]] - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_thresh]
    return [dets[i] for i in keep_idxs]

def _clip_box_to_image(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    x2 = min(W, x + w); y2 = min(H, y + h)
    x = min(x, max(0, x2 - 1)); y = min(y, max(0, y2 - 1))
    return [x, y, max(0, x2 - x), max(0, y2 - y)]

def _get_mog2_background() -> Optional[np.ndarray]:
    """Return BGR background snapshot from MOG2 (None in early frames)."""
    try:
        return _bg.getBackgroundImage()
    except Exception:
        return None

def _edge_heatmap_u8(frame: np.ndarray) -> np.ndarray:
    """Return 8-bit gradient magnitude image for visualising edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

def _shape_fullbody(dets, target_ar=0.45, expand=0.12):
    shaped = []
    for x, y, w, h, s in dets:
        if h <= 0:
            shaped.append([x,y,w,h,s]); continue
        ar = w / max(h, 1e-6)
        cx, cy = x + w*0.5, y + h*0.5
        if ar < target_ar:           # too tall -> widen
            w2 = target_ar * h
            w2 = max(w2, w)
            w, h = w2, h
        else:                        # too wide -> heighten
            h2 = w / max(target_ar, 1e-6)
            h2 = max(h2, h)
            w, h = w, h2
        w *= (1.0 + expand)
        h *= (1.0 + expand)
        x2 = int(round(cx - w*0.5))
        y2 = int(round(cy - h*0.5))
        w2 = int(round(w))
        h2 = int(round(h))
        shaped.append([x2, y2, w2, h2, s])
    return shaped

# Resolution-aware derivation helpers
def _round_odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1

def _derive_rules_params(H: int, W: int):
    """Derive pixel thresholds from resolution-invariant fractions."""
    frame_area = float(H * W)
    short_side = float(min(H, W))

    min_area_px = int(round(RULES_MIN_AREA_FRAC * frame_area))
    k_open  = _round_odd(RULES_MORPH_OPEN_FRAC  * short_side)
    k_close = _round_odd(RULES_MORPH_CLOSE_FRAC * short_side)

    # guardrails
    min_area_px = max(1, min_area_px)
    k_open  = max(1, k_open)
    k_close = max(1, k_close)

    return {
        "MIN_AREA_PX": min_area_px,
        "K_OPEN":  k_open,
        "K_CLOSE": k_close,
    }

# -----------------------------
# ML-free background helpers
# -----------------------------
def _init_bg_buffers(H: int, W: int):
    """Ensure buffers have correct shapes/capacity for current resolution."""
    global _bg_accum_f32, _bg_median_buf, _prev_gray_buf
    if RULES_BG_MODE == "ACCUM":
        if _bg_accum_f32 is None or _bg_accum_f32.shape != (H, W):
            _bg_accum_f32 = None  # lazy init next call
    elif RULES_BG_MODE == "MEDIAN":
        if len(_bg_median_buf) == 0 or _bg_median_buf.maxlen != RULES_MEDIAN_WINDOW:
            _bg_median_buf = deque(maxlen=RULES_MEDIAN_WINDOW)
    elif RULES_BG_MODE == "DIFF":
        if len(_prev_gray_buf) == 0 or _prev_gray_buf.maxlen != (RULES_DIFF_STRIDE + 1):
            _prev_gray_buf = deque(maxlen=RULES_DIFF_STRIDE + 1)

def _apply_diff_blur_and_threshold(diff_u8: np.ndarray) -> np.ndarray:
    """Minimal stabilisation from abs-diff to binary mask (0/1)."""
    d = diff_u8
    # 1) Optional pre-threshold blur
    if RULES_DIFF_BLUR_K and RULES_DIFF_BLUR_K >= 3:
        k = int(RULES_DIFF_BLUR_K) | 1  # ensure odd
        d = cv2.GaussianBlur(d, (k, k), 0)

    # 2) Fixed threshold (ACCUM/MEDIAN/DIFF use their *_THRESH)
    if RULES_BG_MODE == "ACCUM":
        thr = int(RULES_ACCUM_THRESH)
    elif RULES_BG_MODE == "MEDIAN":
        thr = int(RULES_MEDIAN_THRESH)
    elif RULES_BG_MODE == "DIFF":
        thr = int(RULES_DIFF_THRESH)
    else:  # MOG2 shouldn't use this, but safe default
        thr = int(RULES_MOTION_BIN_THRESH)

    _, m = cv2.threshold(d, thr, 255, cv2.THRESH_BINARY)
    m = (m > 0).astype(np.uint8)  # {0,1}

    # 3) Optional temporal EMA smoothing on the binary mask
    global _mask_ema
    if RULES_MASK_TEMPORAL_ALPHA > 0.0:
        f = float(RULES_MASK_TEMPORAL_ALPHA)
        if _mask_ema is None or _mask_ema.shape != m.shape:
            _mask_ema = m.astype(np.float32) * 255.0
        else:
            _mask_ema = (1.0 - f) * _mask_ema + f * (m.astype(np.float32) * 255.0)
        m = (_mask_ema >= 127.0).astype(np.uint8)

    return m  # {0,1}

def _bg_and_mask(frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns (bg_bgr, mask_bin, absdiff_u8) for selected RULES_BG_MODE.
    bg_bgr may be None in very early frames.
    """
    H, W = frame.shape[:2]
    _init_bg_buffers(H, W)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if RULES_BG_MODE == "ACCUM":
        global _bg_accum_f32
        if _bg_accum_f32 is None:
            _bg_accum_f32 = gray.astype(np.float32)
            bg_u8 = gray.copy()
            diff  = np.zeros_like(gray)
            mask  = np.zeros_like(gray, dtype=np.uint8)
        else:
            cv2.accumulateWeighted(gray, _bg_accum_f32, RULES_ACCUM_ALPHA)
            bg_u8 = cv2.convertScaleAbs(_bg_accum_f32)
            diff  = cv2.absdiff(gray, bg_u8)
            mask  = _apply_diff_blur_and_threshold(diff)
        bg_bgr = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
        return bg_bgr, mask, diff

    elif RULES_BG_MODE == "MEDIAN":
        _bg_median_buf.append(gray.copy())
        if len(_bg_median_buf) < _bg_median_buf.maxlen:
            return None, np.zeros_like(gray, dtype=np.uint8), np.zeros_like(gray)
        stack = np.stack(list(_bg_median_buf), axis=0)  # T x H x W
        bg_u8 = np.median(stack, axis=0).astype(np.uint8)
        diff  = cv2.absdiff(gray, bg_u8)
        mask  = _apply_diff_blur_and_threshold(diff)
        bg_bgr = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
        return bg_bgr, mask, diff

    elif RULES_BG_MODE == "DIFF":
        _prev_gray_buf.append(gray.copy())
        if len(_prev_gray_buf) <= RULES_DIFF_STRIDE:
            return None, np.zeros_like(gray, dtype=np.uint8), np.zeros_like(gray)
        bg_u8 = _prev_gray_buf[0]  # t-k
        diff  = cv2.absdiff(gray, bg_u8)
        mask  = _apply_diff_blur_and_threshold(diff)
        bg_bgr = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
        return bg_bgr, mask, diff

    else:  # "MOG2"
        m_raw = _bg.apply(frame)  # 0..255 (shadows ~127)
        mask  = (m_raw > RULES_MOTION_BIN_THRESH).astype(np.uint8)  # {0,1}
        bg    = _get_mog2_background()
        if bg is None:
            diff = np.zeros((H, W), np.uint8)
        else:
            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(bg,   cv2.COLOR_BGR2GRAY))
        return bg, mask, diff

# OPTIONAL: WBF-like merge (HOG path)
def _wbf_merge(dets: List[List[float]], score_thresh: float, iou: float) -> List[List[float]]:
    dets = sorted([d for d in dets if d[4] >= score_thresh], key=lambda d: d[4], reverse=True)
    if not dets: return []
    used = [False]*len(dets)
    merged = []
    def iou_xywh(a,b):
        ax,ay,aw,ah,_=a; bx,by,bw,bh,_=b
        ax2,ay2=ax+aw, ay+ah; bx2,by2=bx+bw, by+bh
        ix1,iy1=max(ax,bx), max(ay,by); ix2,iy2=min(ax2,bx2), min(ay2,by2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1); ua=aw*ah + bw*bh - inter
        return inter/ua if ua>0 else 0.0
    for i, di in enumerate(dets):
        if used[i]: continue
        cluster=[di]; used[i]=True
        for j in range(i+1,len(dets)):
            if used[j]: continue
            if iou_xywh(di, dets[j]) >= iou:
                used[j]=True; cluster.append(dets[j])
        if len(cluster)==1:
            merged.append(di); continue
        ws = np.array([c[4] for c in cluster], dtype=np.float32); ws/= (ws.sum()+1e-8)
        xs = np.array([c[0] for c in cluster], dtype=np.float32)
        ys = np.array([c[1] for c in cluster], dtype=np.float32)
        ww = np.array([c[2] for c in cluster], dtype=np.float32)
        hh = np.array([c[3] for c in cluster], dtype=np.float32)
        x = float((xs*ws).sum()); y=float((ys*ws).sum()); w=float((ww*ws).sum()); h=float((hh*ws).sum())
        score = float(max(c[4] for c in cluster))
        merged.append([int(round(x)), int(round(y)), int(round(w)), int(round(h)), score])
    return merged

# -----------------------------
# RULES detector (rich debug, resolution-aware)
# -----------------------------
def detect_people_rules(frame: np.ndarray, frame_idx: int = 0) -> Tuple[List[List[float]], Dict[str, List[List[float]]]]:
    """RULES detector with debug artefacts. Returns (final_boxes, debug_dict)."""
    debug: Dict[str, List[List[float]]] = {}
    H, W = frame.shape[:2]

    # derive resolution-aware params
    d = _derive_rules_params(H, W)
    MIN_AREA_PX = d["MIN_AREA_PX"]
    K_OPEN  = d["K_OPEN"]
    K_CLOSE = d["K_CLOSE"]

    # 0–1) Background + mask + abs-diff (mode-dependent)
    bg_img, mask, diff_u8 = _bg_and_mask(frame)

    # 1a) Optional pre-contour tiny-component pruning (resolution-invariant)
    min_cc_area = max(1, int(round(RULES_MIN_CC_AREA_FRAC * H * W)))
    if min_cc_area > 1:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num):  # skip background 0
            if stats[i, cv2.CC_STAT_AREA] >= min_cc_area:
                keep[labels == i] = 1
        mask = keep

    # 1b) Morphology with resolution-aware kernels
    if K_OPEN > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_OPEN, K_OPEN))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    if K_CLOSE > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_CLOSE, K_CLOSE))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # 2) Contours -> candidate boxes (geometric gates)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < MIN_AREA_PX:
            continue
        ar = w / max(h, 1e-6)
        if not (RULES_ASPECT_MIN <= ar <= RULES_ASPECT_MAX):
            continue
        x, y, w, h = _clip_box_to_image(x, y, w, h, W, H)
        raw.append([x, y, w, h, 1.0])
    debug["contours"] = raw

    # 3) Motion-overlap gate (ensure boxes are foreground now)
    fg_bin = (mask > 0).astype(np.uint8)   # {0,1}
    filt = []
    for x, y, w, h, s in raw:
        roi = fg_bin[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        frac = float(roi.mean())           # mean on {0,1}
        if frac >= RULES_MOTION_MIN_OVERLAP:
            filt.append([x, y, w, h, s])
    debug["filtered"] = filt

    # 4) Edge-density score (visual heatmap too)
    edge_u8 = _edge_heatmap_u8(frame)
    scored = []
    for x, y, w, h, _ in filt:
        roi = edge_u8[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        frac = float((roi >= RULES_EDGE_MAG_THRESH).mean())
        if frac >= RULES_EDGE_MIN_FRAC:
            scored.append([x, y, w, h, float(frac)])
    debug["edges_ok"] = scored

    # 5) NMS
    nmsed = _nms_xywh(scored, RULES_NMS_IOU)
    debug["nms"] = nmsed

    # 6) Shape to full-body
    shaped = _shape_fullbody(nmsed, target_ar=FULLBODY_TARGET_AR, expand=FULLBODY_EXPAND)
    debug["shaped"] = shaped

    # Attach visual artefacts
    debug["img_mask"] = mask            # {0,1} uint8
    debug["img_bg"]   = bg_img          # BGR or None
    debug["img_diff"] = diff_u8         # uint8 0..255
    debug["img_edge"] = edge_u8         # uint8 0..255

    final = shaped
    debug["final"] = final
    return final, debug

# -----------------------------
# HOG backend (portable call)
# -----------------------------
def detect_people_hog(frame) -> List[List[float]]:
    try:
        rects, weights = hog.detectMultiScale(
            frame,
            HOG_HIT_THRESHOLD,
            HOG_WINSTRIDE,
            HOG_PADDING,
            HOG_SCALE
        )
    except Exception:
        rects, weights = hog.detectMultiScale(frame)

    raw = [[int(x), int(y), int(w), int(h), float(s)] for (x, y, w, h), s in zip(rects, weights)]
    if HOG_MERGE:
        if HOG_USE_WBF:
            merged = _wbf_merge(raw, HOG_SCORE_THRESH, HOG_NMS_IOU)
        else:
            merged = _nms_xywh([d for d in raw if d[4] >= HOG_SCORE_THRESH], HOG_NMS_IOU)
    else:
        merged = raw
    shaped = _shape_fullbody(merged, target_ar=FULLBODY_TARGET_AR, expand=FULLBODY_EXPAND)
    return shaped

# -----------------------------
# YOLO backend
# -----------------------------
def detect_people_yolo(frame) -> List[List[float]]:
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

# -----------------------------
# Public APIs
# -----------------------------
def detect_people(frame: np.ndarray, frame_idx: int = 0) -> List[List[float]]:
    """Return [[x,y,w,h,score], ...] from the selected backend."""
    if DETECTOR_MODE == "YOLO" and (yolo_model is not None):
        return detect_people_yolo(frame)
    if DETECTOR_MODE == "HOG" and (hog is not None):
        return detect_people_hog(frame)
    final, _ = detect_people_rules(frame, frame_idx)
    return final

def detect_people_debug(frame: np.ndarray, frame_idx: int = 0):
    """
    Returns (final_dets, debug) with per-stage outputs.
    For RULES: keys = 'contours','filtered','edges_ok','nms','shaped','final'
               and image artefacts 'img_bg','img_mask','img_diff','img_edge'
    For HOG:   keys = 'raw','nms','shaped','final'
    For YOLO:  keys = 'final'
    """
    debug: Dict[str, List[List[float]]] = {}
    if DETECTOR_MODE == "YOLO" and (yolo_model is not None):
        final = detect_people_yolo(frame)
        debug["final"] = final
        return final, debug
    if DETECTOR_MODE == "HOG" and (hog is not None):
        try:
            rects, weights = hog.detectMultiScale(
                frame, HOG_HIT_THRESHOLD, HOG_WINSTRIDE, HOG_PADDING, HOG_SCALE
            )
        except Exception:
            rects, weights = hog.detectMultiScale(frame)
        raw = [[int(x), int(y), int(w), int(h), float(s)] for (x,y,w,h), s in zip(rects, weights)]
        debug["raw"] = raw
        if HOG_MERGE:
            nmsed = _nms_xywh([d for d in raw if d[4] >= HOG_SCORE_THRESH], HOG_NMS_IOU) if not HOG_USE_WBF \
                else _wbf_merge(raw, HOG_SCORE_THRESH, HOG_NMS_IOU)
        else:
            nmsed = raw
        debug["nms"] = nmsed
        shaped = _shape_fullbody(nmsed, FULLBODY_TARGET_AR, FULLBODY_EXPAND)
        debug["shaped"] = shaped
        final = shaped
        debug["final"] = final
        return final, debug
    # RULES
    final, debug = detect_people_rules(frame, frame_idx)
    return final, debug
