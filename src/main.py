# main.py 
import os
import cv2
import time
import statistics as stats 


from person_track import TrackerReID
from person_detect import detect_people  

import person_detect as pd
pd.USE_YOLO = True   # or False to force HOG

try:
    from face_features import FaceFeatureDetector, draw_face_boxes
    HAVE_FACE = False
except Exception:
    HAVE_FACE = False
    FaceFeatureDetector = None
    def draw_face_boxes(*args, **kwargs): 
        pass

VIDEO_PATH = "data/MOTS/train/MOTS20-09/MOTS20-09.mp4"
OUTPUT_PATH = "outputs/labeled.mp4"
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

INCLUDE_IO = False

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video at:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

    # --- tracker ---
    tracker = TrackerReID(
        iou_thresh=0.35,          
        appearance_thresh=0.65,   
        max_age=45                
    )

    if HAVE_FACE:
        tracker.face_detector = FaceFeatureDetector(
            prefer="mediapipe",
            roi_expand=0.12,
            head_region_ratio=0.25,
            eye_box_frac=0.24,
            mediapipe_model_selection=1,
            mediapipe_min_detection_confidence=0.80,
        )
    else:
        tracker.face_detector = None

    frame_idx = 0
    frame_times = []  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timing
        t0 = time.perf_counter() 

        # Detect people: list of [x,y,w,h,score]
        detections = detect_people(frame)

        # Track + Re-ID
        states = tracker.update(frame, detections, frame_idx)

        # Draw
        for s in states:
            x, y, w, h = s["box"]
            tid = s["id"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Optional face features per track
            if HAVE_FACE and getattr(tracker, "face_detector", None) is not None:
                try:
                    face_obj = tracker.face_detector.detect(frame, (x, y, w, h))
                    if face_obj is not None:
                        draw_face_boxes(frame, face_obj)
                except Exception:
                    pass

        if INCLUDE_IO:
            writer.write(frame)
            t1 = time.perf_counter()
        else:
            t1 = time.perf_counter()
            writer.write(frame)

        # Save timing
        frame_times.append(t1 - t0)  
        frame_idx += 1

    cap.release()
    writer.release()

    # Output timing stats
    if frame_times:
        mean_s = stats.mean(frame_times)
        std_s = stats.stdev(frame_times) if len(frame_times) > 1 else 0.0
        mean_ms = mean_s * 1000.0
        std_ms = std_s * 1000.0
        eff_fps = 1.0 / mean_s if mean_s > 0 else float("inf")
        print(f"Frames processed: {len(frame_times)}")
        print(f"Per-frame time (mean ± std): {mean_ms:.2f} ± {std_ms:.2f} ms")
        print(f"Effective throughput: {eff_fps:.2f} FPS "
              f"({'incl' if INCLUDE_IO else 'excl'} write)")
    print("Done. Wrote:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
