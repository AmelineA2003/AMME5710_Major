# main.py (Option A import)
import os
import cv2

# --- person detection modules ---
from person_track import TrackerReID
from person_detect import detect_people  

# If you want to force HOG/YOLO at runtime, you can uncomment below:
import person_detect as pd 

# ---------------- configuration ----------------
# VIDEO_PATH = "../data/MOTS/train/MOTS20-02/MOTS20-02.mp4"
# VIDEO_PATH = "../data/MOTS/train/MOTS20-09/MOTS20-09.mp4"
VIDEO_PATH = "../data/collected/test_rgb_1760411494.mp4"
OUTPUT_PATH = "outputs/labeled.mp4"
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

DRAW = True

def main():

    # -------------- OPENING VIDEO STREAM ---------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video at:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if DRAW:
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

    # --------------- TRACKER CONFIG --------------------
    tracker = TrackerReID(
        iou_thresh=0.35,          # slightly higher
        appearance_thresh=0.65,   # higher = stricter re-ID reuse
        max_age=45                # time that boxes remain "active"
    )

    # ---------- ITERATE THROUGH EACH FRAME OF VIDEO -----------
    frame_idx = 0
    last_people_boxes = {}  # will hold the most recent frame's dict

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people: list of [x,y,w,h,score]
        detections = detect_people(frame)

        # Track + Re-ID
        states = tracker.update(frame, detections, frame_idx)

        # Build the requested dictionary: {ID: ((tl),(tr),(bl),(br))}
        people_boxes = {}
        for s in states:
            x, y, w, h = s["box"]
            tid = int(s["id"])

            tl = (int(x),       int(y))
            tr = (int(x + w),   int(y))
            bl = (int(x),       int(y + h))
            br = (int(x + w),   int(y + h))

            people_boxes[tid] = (tl, tr, bl, br)

            # Draw
            if DRAW:
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # START: AMELINE AND JESTINS PIPELINE HERE USING `PEOPLE_BOXES`


        # STOP: AMELINE AND JESTINS PIPELINE HERE

        # keep last frame's dict available after the loop
        last_people_boxes = people_boxes

        if DRAW:
            writer.write(frame)

        frame_idx += 1

    cap.release()

    if DRAW:
        writer.release()
        print("Done. Wrote:", OUTPUT_PATH)

    # DEBUG STATEMENT
    print("Last frame people_boxes dict:", last_people_boxes)

if __name__ == "__main__":
    main()
