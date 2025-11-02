# main.py
import os
import cv2
import numpy as np  # needed for some debug guard paths

# --- your modules ---
import person_detect as pd
from person_detect import detect_people_debug
from person_track import TrackerReID

# Optional face features (safe import)
try:
    from face_features import FaceFeatureDetector, draw_face_boxes
    HAVE_FACE = True
except Exception:
    HAVE_FACE = False
    def draw_face_boxes(*args, **kwargs): pass

# Viz helpers
from debug_viz import VideoSink, draw_boxes, overlay_mask, label, tile_panels

# ---------------- config ----------------
# Switch detector backend here: "RULES" | "HOG" | "YOLO"
pd.DETECTOR_MODE = "RULES"

# Tuning for RULES (optional — adjust to taste)
pd.RULES_MIN_AREA = 35*35
pd.RULES_ASPECT_MIN = 0.35
pd.RULES_ASPECT_MAX = 0.85
pd.RULES_EDGE_MIN_FRAC = 0.06
pd.RULES_NMS_IOU = 0.50
pd.RULES_PROCESS_EVERY_N = 1

# Paths
# VIDEO_PATH = "../data/MOTS/train/MOTS20-02/MOTS20-02.mp4"
VIDEO_PATH = "../data/MOTS/train/MOTS20-09/MOTS20-09.mp4"
# VIDEO_PATH = "../data/collected/test_rgb_1760411494.mp4"
OUTPUT_DIR = "outputs"
FINAL_OUT   = os.path.join(OUTPUT_DIR, "labeled.mp4")

# Per-stage box videos (RULES)
DBG_CONTOURS = os.path.join(OUTPUT_DIR, "rules_contours.mp4")
DBG_FILTERED = os.path.join(OUTPUT_DIR, "rules_filtered.mp4")
DBG_EDGESOK  = os.path.join(OUTPUT_DIR, "rules_edges_ok.mp4")
DBG_NMS      = os.path.join(OUTPUT_DIR, "rules_nms.mp4")
DBG_SHAPED   = os.path.join(OUTPUT_DIR, "rules_shaped.mp4")
DBG_FINAL    = os.path.join(OUTPUT_DIR, "rules_final.mp4")

# Image artefact videos (RULES)
DBG_MASK  = os.path.join(OUTPUT_DIR, "rules_mask.mp4")
DBG_BG    = os.path.join(OUTPUT_DIR, "rules_bg.mp4")
DBG_DIFF  = os.path.join(OUTPUT_DIR, "rules_diff.mp4")
DBG_EDGE  = os.path.join(OUTPUT_DIR, "rules_edge.mp4")
DBG_PANEL = os.path.join(OUTPUT_DIR, "rules_panel.mp4")   # tiled overview

# HOG debug (if you flip to HOG)
DBG_HOG_RAW  = os.path.join(OUTPUT_DIR, "hog_raw.mp4")
DBG_HOG_NMS  = os.path.join(OUTPUT_DIR, "hog_nms.mp4")
DBG_HOG_SHAP = os.path.join(OUTPUT_DIR, "hog_shaped.mp4")
DBG_HOG_FIN  = os.path.join(OUTPUT_DIR, "hog_final.mp4")

DEBUG_VIDEOS = True  # turn off to disable all debug writers

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (W, H)

    # Final labeled output (with tracker IDs)
    final_writer = VideoSink(FINAL_OUT, fps, size)

    # Debug writers
    writers_boxes = {}
    writers_imgs  = {}
    panel_writer = None

    if DEBUG_VIDEOS:
        if pd.DETECTOR_MODE == "RULES":
            writers_boxes = {
                "contours": VideoSink(DBG_CONTOURS, fps, size),
                "filtered": VideoSink(DBG_FILTERED, fps, size),
                "edges_ok": VideoSink(DBG_EDGESOK,  fps, size),
                "nms":      VideoSink(DBG_NMS,      fps, size),
                "shaped":   VideoSink(DBG_SHAPED,   fps, size),
                "final":    VideoSink(DBG_FINAL,    fps, size),
            }
            writers_imgs = {
                "mask":  VideoSink(DBG_MASK,  fps, size),
                "bg":    VideoSink(DBG_BG,    fps, size),
                "diff":  VideoSink(DBG_DIFF,  fps, size),
                "edge":  VideoSink(DBG_EDGE,  fps, size),
            }
            panel_writer = VideoSink(DBG_PANEL, fps, size)
        elif pd.DETECTOR_MODE == "HOG":
            writers_boxes = {
                "raw":    VideoSink(DBG_HOG_RAW,  fps, size),
                "nms":    VideoSink(DBG_HOG_NMS,  fps, size),
                "shaped": VideoSink(DBG_HOG_SHAP, fps, size),
                "final":  VideoSink(DBG_HOG_FIN,  fps, size),
            }
        else:  # YOLO — only final
            writers_boxes = {"final": VideoSink(DBG_HOG_FIN, fps, size)}

    # Tracker
    tracker = TrackerReID(
        iou_thresh=0.35,
        appearance_thresh=0.65,
        max_age=45
    )

    # Optional face detector on the tracker
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
        if not ret:
            break

        # Use debug path so we can write stage videos
        final_dets, debug = detect_people_debug(frame, frame_idx)

        # ---- Per-stage BOX videos ----
        if DEBUG_VIDEOS:
            if pd.DETECTOR_MODE == "RULES":
                if "contours" in debug and "contours" in writers_boxes:
                    writers_boxes["contours"].write(draw_boxes(frame, debug["contours"], (0, 0, 255), "contours"))
                if "filtered" in debug and "filtered" in writers_boxes:
                    writers_boxes["filtered"].write(draw_boxes(frame, debug["filtered"], (0, 165, 255), "filtered"))
                if "edges_ok" in debug and "edges_ok" in writers_boxes:
                    writers_boxes["edges_ok"].write(draw_boxes(frame, debug["edges_ok"], (0, 255, 255), "edges_ok"))
                if "nms" in debug and "nms" in writers_boxes:
                    writers_boxes["nms"].write(draw_boxes(frame, debug["nms"], (0, 255, 0), "nms"))
                if "shaped" in debug and "shaped" in writers_boxes:
                    writers_boxes["shaped"].write(draw_boxes(frame, debug["shaped"], (255, 0, 0), "shaped"))
                if "final" in debug and "final" in writers_boxes:
                    writers_boxes["final"].write(draw_boxes(frame, debug["final"], (255, 0, 255), "final"))
            elif pd.DETECTOR_MODE == "HOG":
                if "raw" in debug and "raw" in writers_boxes:
                    writers_boxes["raw"].write(draw_boxes(frame, debug["raw"], (0, 0, 255), "raw"))
                if "nms" in debug and "nms" in writers_boxes:
                    writers_boxes["nms"].write(draw_boxes(frame, debug["nms"], (0, 165, 255), "nms"))
                if "shaped" in debug and "shaped" in writers_boxes:
                    writers_boxes["shaped"].write(draw_boxes(frame, debug["shaped"], (0, 255, 255), "shaped"))
                if "final" in debug and "final" in writers_boxes:
                    writers_boxes["final"].write(draw_boxes(frame, debug["final"], (255, 0, 255), "final"))
            else:  # YOLO
                if "final" in writers_boxes:
                    writers_boxes["final"].write(draw_boxes(frame, final_dets, (255, 0, 255), "final"))

        # ---- Image artefact videos (RULES) ----
        if DEBUG_VIDEOS and pd.DETECTOR_MODE == "RULES":
            bg_img   = debug.get("img_bg", None)
            mask_u8  = debug.get("img_mask", None)
            diff_u8  = debug.get("img_diff", None)
            edge_u8  = debug.get("img_edge", None)

            # Background stream (just label the bg image or blank if None)
            if "bg" in writers_imgs:
                if isinstance(bg_img, np.ndarray):
                    bg_vis = label(bg_img.copy(), "Background (MOG2)")
                else:
                    bg_vis = label(np.zeros_like(frame), "Background (MOG2)")
                writers_imgs["bg"].write(bg_vis)

            # Overlay streams
            if "mask" in writers_imgs:
                mask_vis = overlay_mask(frame, (mask_u8 * 255) if mask_u8 is not None and mask_u8.max() <= 1 else mask_u8, alpha=0.55)
                writers_imgs["mask"].write(label(mask_vis, "Motion mask (post-morph)"))
            if "diff" in writers_imgs:
                diff_vis = overlay_mask(frame, diff_u8, alpha=0.55)
                writers_imgs["diff"].write(label(diff_vis, "AbsDiff(gray, bg)"))
            if "edge" in writers_imgs:
                edge_vis = overlay_mask(frame, edge_u8, alpha=0.55)
                writers_imgs["edge"].write(label(edge_vis, "Gradient magnitude"))

            # 6-panel tile: Original, BG, Mask, Diff, Edge, Final
            if panel_writer:
                final_vis = draw_boxes(frame, debug.get("final", []), (255, 0, 255), "final")
                final_vis = label(final_vis, "Final boxes")

                if isinstance(bg_img, np.ndarray):
                    bg_panel = label(bg_img.copy(), "Background (MOG2)")
                else:
                    bg_panel = label(np.zeros_like(frame), "Background (MOG2)")

                mask_panel = label(overlay_mask(frame, (mask_u8 * 255) if mask_u8 is not None and mask_u8.max() <= 1 else mask_u8, 0.55), "Motion mask")
                diff_panel = label(overlay_mask(frame, diff_u8, 0.55), "AbsDiff")
                edge_panel = label(overlay_mask(frame, edge_u8, 0.55), "Grad magnitude")

                panel = tile_panels([label(frame, "Original"),
                                     bg_panel, mask_panel,
                                     diff_panel, edge_panel, final_vis], cols=3)
                panel_writer.write(panel)

        # ---- Tracking + labeled output ----
        states = tracker.update(frame, final_dets, frame_idx)
        vis = frame.copy()
        for s in states:
            x, y, w, h = s["box"]
            tid = s["id"]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {tid}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if HAVE_FACE and getattr(tracker, "face_detector", None) is not None:
                try:
                    fobj = tracker.face_detector.detect(frame, (x, y, w, h))
                    if fobj is not None:
                        draw_face_boxes(vis, fobj)
                except Exception:
                    pass

        final_writer.write(vis)
        frame_idx += 1

    # Cleanup
    cap.release()
    final_writer.release()
    for w in writers_boxes.values():
        w.release()
    for w in writers_imgs.values():
        w.release()
    if panel_writer:
        panel_writer.release()

    print("Final video:", FINAL_OUT)
    if DEBUG_VIDEOS:
        for name, w in {**writers_boxes, **writers_imgs}.items():
            print(f"{name}:", w.path)
        if panel_writer:
            print("panel:", panel_writer.path)

if __name__ == "__main__":
    main()
