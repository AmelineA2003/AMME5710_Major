# main.py
import os
import cv2

from person_track import TrackerReID
from person_detect import detect_people
# we will try to import the debug entrypoint if present:
try:
    from person_detect import detect_people_debug
    HAVE_PD_DEBUG = True
except Exception:
    HAVE_PD_DEBUG = False

# optional face features
try:
    from face_features import FaceFeatureDetector, draw_face_boxes
    HAVE_FACE = True
except Exception:
    HAVE_FACE = False
    def draw_face_boxes(*args, **kwargs): pass

from debug_viz import draw_boxes, VideoSink

# ------------- config -------------
# VIDEO_PATH = "../data/MOTS/train/MOTS20-02/MOTS20-02.mp4"
VIDEO_PATH = "../data/MOTS/train/MOTS20-09/MOTS20-09.mp4"
OUTPUT_DIR = "outputs"
FINAL_OUT = os.path.join(OUTPUT_DIR, "labeled.mp4")

# turn on/off per-stage debug videos
DEBUG_VIDEOS = True
RAW_OUT      = os.path.join(OUTPUT_DIR, "raw_hog.mp4")
MERGED_OUT   = os.path.join(OUTPUT_DIR, "merged.mp4")
SHAPED_OUT   = os.path.join(OUTPUT_DIR, "shaped.mp4")
GRABCUT_OUT  = os.path.join(OUTPUT_DIR, "grabcut.mp4")  # written only if grabcut exists

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print("Could not open video at:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (W, H)

    # final writer
    final_writer = VideoSink(FINAL_OUT, fps, size)

    # debug writers
    raw_writer = merged_writer = shaped_writer = grabcut_writer = None
    if DEBUG_VIDEOS:
        raw_writer    = VideoSink(RAW_OUT, fps, size)
        merged_writer = VideoSink(MERGED_OUT, fps, size)
        shaped_writer = VideoSink(SHAPED_OUT, fps, size)
        grabcut_writer= VideoSink(GRABCUT_OUT, fps, size)

    tracker = TrackerReID(iou_thresh=0.35, appearance_thresh=0.65, max_age=45)

    if HAVE_FACE:
        tracker.face_detector = FaceFeatureDetector(
            prefer="mediapipe",
            roi_expand=0.12,
            head_region_ratio=0.25,
            eye_box_frac=0.24,
            mediapipe_model_selection=1,
            mediapipe_min_detection_confidence=0.80,
        )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- detection (with debug if available) ---
        if DEBUG_VIDEOS and HAVE_PD_DEBUG:
            final_dets, debug = detect_people_debug(frame)
        else:
            final_dets = detect_people(frame)
            debug = {}

        # --- write per-stage debug videos ---
        if DEBUG_VIDEOS:
            # stage: raw
            if "raw_hog" in debug and raw_writer:
                raw_img = draw_boxes(frame, debug["raw_hog"], (0,0,255), "raw")
                raw_writer.write(raw_img)
            # stage: merged
            if "merged" in debug and merged_writer:
                merged_img = draw_boxes(frame, debug["merged"], (0,165,255), "merged")
                merged_writer.write(merged_img)
            # stage: shaped
            if "shaped" in debug and shaped_writer:
                shaped_img = draw_boxes(frame, debug["shaped"], (0,255,255), "shaped")
                shaped_writer.write(shaped_img)
            # stage: grabcut (only if present)
            if "grabcut" in debug and grabcut_writer:
                gc_img = draw_boxes(frame, debug["grabcut"], (255,0,0), "grabcut")
                grabcut_writer.write(gc_img)

        # --- tracking + final labeled output ---
        states = tracker.update(frame, final_dets, frame_idx)
        vis = frame.copy()
        for s in states:
            x, y, w, h = s["box"]
            tid = s["id"]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(vis, f"ID {tid}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # optional faces
            if HAVE_FACE and getattr(tracker, "face_detector", None) is not None:
                try:
                    fobj = tracker.face_detector.detect(frame, (x,y,w,h))
                    if fobj is not None:
                        draw_face_boxes(vis, fobj)
                except Exception:
                    pass

        final_writer.write(vis)
        frame_idx += 1

    # cleanup
    cap.release()
    final_writer.release()
    if raw_writer: raw_writer.release()
    if merged_writer: merged_writer.release()
    if shaped_writer: shaped_writer.release()
    if grabcut_writer: grabcut_writer.release()

    print("Final:", FINAL_OUT)
    if DEBUG_VIDEOS:
        print("Raw HOG:", RAW_OUT)
        print("Merged:", MERGED_OUT)
        print("Shaped:", SHAPED_OUT)
        print("GrabCut:", GRABCUT_OUT)

if __name__ == "__main__":
    main()
