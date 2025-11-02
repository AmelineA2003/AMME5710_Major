# person_detect.py
# Matches the original notebook functionality exactly:
# - USE_YOLO flag controls backend
# - YOLOv8 if available (cls==0 persons only), else HOG
# - Returns list of [x, y, w, h, score]
# - No extra post-filtering/NMS beyond what the backend does

import cv2
from typing import List

# ---- Config & globals (same shape as notebook) ----
USE_YOLO = True   # Set True if you have `pip install ultralytics` and weights available
detector_name = "HOG"
yolo_model = None

if USE_YOLO:
    try:
        from ultralytics import YOLO
        # You can change to a local/custom weights file as in the notebook (e.g., 'yolov8n.pt')
        yolo_model = YOLO("yolov8n.pt")
        detector_name = "YOLOv8"
    except Exception as e:
        print("YOLOv8 not available, falling back to HOG. Reason:", e)
        USE_YOLO = False
        detector_name = "HOG"

# ---- HOG setup (default pedestrian detector) ----
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_people_hog(frame) -> List[List[float]]:
    """
    Returns list of [x, y, w, h, score] using OpenCV HOG.
    'score' is the SVM distance (not a probability).
    """
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )
    out: List[List[float]] = []
    for (x, y, w, h), s in zip(rects, weights):
        out.append([int(x), int(y), int(w), int(h), float(s)])
    return out


def detect_people_yolo(frame) -> List[List[float]]:
    """
    Returns list of [x, y, w, h, score] using YOLOv8 (Ultralytics).
    Filters to class 0 (person). Uses model defaults (no extra NMS/filters here).
    """
    results = yolo_model(frame, conf=0.60, iou=0.3, classes=[0], verbose=False) 
    out: List[List[float]] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls.item()) if b.cls is not None else -1
            if cls != 0 and cls != -1:
                # keep persons only (COCO class 0); allow -1 just in case cls missing
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
    - Otherwise, use HOG.
    """
    if USE_YOLO and yolo_model is not None:
        return detect_people_yolo(frame)
    return detect_people_hog(frame)
