
from dataclasses import dataclass
from typing import Optional, Tuple, List, Sequence

# All boxes are absolute XYWH in pixels
BBox = Tuple[int, int, int, int]


@dataclass
class Face:
    face_box: Optional[BBox]
    left_eye_box: Optional[BBox]
    right_eye_box: Optional[BBox]


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))


def _xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> BBox:
    return (int(x1), int(y1), max(0, int(x2 - x1)), max(0, int(y2 - y1)))


def _expand_xywh(x: int, y: int, w: int, h: int, frac: float, W: int, H: int) -> BBox:
    """Expand XYWH by a fraction of its size and clamp to image."""
    if frac <= 0.0 or w <= 0 or h <= 0:
        return (x, y, w, h)
    dx = int(w * frac / 2.0)
    dy = int(h * frac / 2.0)
    nx1 = _clip(x - dx, 0, W - 1)
    ny1 = _clip(y - dy, 0, H - 1)
    nx2 = _clip(x + w + dx, 0, W - 1)
    ny2 = _clip(y + h + dy, 0, H - 1)
    return _xyxy_to_xywh(nx1, ny1, nx2, ny2)


class FaceFeatureDetector:
    """Robust face/eye detector with tunable parameters.
    Backends (selected by `prefer` or auto-fallback):
      1) InsightFace (RetinaFace)  -> best; needs: pip install insightface onnxruntime
      2) MediaPipe FaceDetection   -> fast CPU; needs: pip install mediapipe
      3) OpenCV Haar (alt2)        -> fallback; no extra deps

    Returns Face with XYWH boxes; any missing feature is None.
    """

    def __init__(
        self,
        # Backend selection
        prefer: str = "auto",  # "auto" | "insightface" | "mediapipe" | "haar"

        # Global ROI controls
        roi_expand: float = 0.0,       # expand person box by this fraction (e.g., 0.1 = +10% overall)
        head_region_ratio: float = 1.0, # use top fraction of ROI (<=1.0). 1.0 = full person box
        eye_box_frac: float = 0.18,    # eye box size as fraction of face (for backends that return eye points)

        # InsightFace tuning
        insightface_det_size: Tuple[int, int] = (640, 640),
        insightface_providers: Optional[Sequence[str]] = None,  # e.g. ("CPUExecutionProvider",)
        insightface_score_thresh: Optional[float] = None,  # filter faces by det_score if provided

        # MediaPipe tuning
        mediapipe_model_selection: int = 1,          # 0=close, 1=far
        mediapipe_min_detection_confidence: float = 0.4,

        # Haar tuning
        haar_scale_factor: float = 1.05,
        haar_min_neighbors: int = 3,
        haar_min_size: Tuple[int, int] = (20, 20),
    ):
        self.backend = None
        self.impl = None

        self.roi_expand = float(roi_expand)
        self.head_region_ratio = float(head_region_ratio)
        self.eye_box_frac = float(eye_box_frac)

        # InsightFace options
        self.ins_det_size = insightface_det_size
        self.ins_providers = list(insightface_providers) if insightface_providers is not None else ["CPUExecutionProvider"]
        self.ins_score_thresh = insightface_score_thresh

        # MediaPipe options
        self.mp_model_selection = int(mediapipe_model_selection)
        self.mp_min_det_conf = float(mediapipe_min_detection_confidence)

        # Haar options
        self.haar_scale_factor = float(haar_scale_factor)
        self.haar_min_neighbors = int(haar_min_neighbors)
        self.haar_min_size = haar_min_size

        self._init_backend(prefer)

    # ------------------------ backend init ------------------------
    def _init_backend(self, prefer: str) -> None:
        def try_insightface():
            try:
                from insightface.app import FaceAnalysis  # type: ignore
                app = FaceAnalysis(name="buffalo_l", providers=list(self.ins_providers))
                app.prepare(ctx_id=0, det_size=tuple(self.ins_det_size))
                return "insightface", app
            except Exception:
                return None, None

        def try_mediapipe():
            try:
                import mediapipe as mp  # type: ignore
                return "mediapipe", mp
            except Exception:
                return None, None

        def try_haar():
            try:
                import cv2  # type: ignore
                fd = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
                ed = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                if fd.empty() or ed.empty():
                    return None, None
                return "haar", (fd, ed, cv2)
            except Exception:
                return None, None

        # Order by preference
        if prefer in ("insightface", "retinaface"):
            order = (try_insightface, try_mediapipe, try_haar)
        elif prefer in ("mediapipe", "mp"):
            order = (try_mediapipe, try_insightface, try_haar)
        elif prefer in ("haar",):
            order = (try_haar, try_mediapipe, try_insightface)
        else:
            order = (try_insightface, try_mediapipe, try_haar)

        for fn in order:
            name, impl = fn()
            if name:
                self.backend, self.impl = name, impl
                break

        if not self.backend:
            raise RuntimeError(
                "No face backend available. Install `insightface` or `mediapipe`, "
                "or ensure OpenCV Haar cascades are present."
            )

    # ------------------------ detection ------------------------
    def detect(self, frame_bgr, person_box_xywh: BBox) -> Face:
        x, y, w, h = map(int, person_box_xywh)
        H, W = frame_bgr.shape[:2]

        # Expand ROI if requested
        x, y, w, h = _expand_xywh(x, y, w, h, self.roi_expand, W, H)

        # Use only the head portion if requested
        if self.head_region_ratio < 1.0 and h > 0:
            h_head = max(1, int(h * self.head_region_ratio))
            person_roi = (x, y, w, h_head)
        else:
            person_roi = (x, y, w, h)

        x, y, w, h = person_roi
        x1, y1 = _clip(x, 0, W - 1), _clip(y, 0, H - 1)
        x2, y2 = _clip(x + w, 0, W - 1), _clip(y + h, 0, H - 1)
        if x2 <= x1 or y2 <= y1:
            return Face(None, None, None)

        roi = frame_bgr[y1:y2, x1:x2]

        if self.backend == "insightface":
            app = self.impl
            faces = app.get(roi)  # list of Face objects
            if not faces:
                return Face(None, None, None)

            # Optionally filter by det_score
            if self.ins_score_thresh is not None:
                faces = [f for f in faces if getattr(f, "det_score", 1.0) >= float(self.ins_score_thresh)]
                if not faces:
                    return Face(None, None, None)

            # choose top-most
            f = sorted(faces, key=lambda F: F.bbox[1])[0]
            bx1, by1, bx2, by2 = map(int, f.bbox)
            face_box = (x1 + bx1, y1 + by1, int(bx2 - bx1), int(by2 - by1))

            left_eye = right_eye = None
            if getattr(f, "kps", None) is not None:
                le = f.kps[0]  # (x,y)
                re = f.kps[1]
                fw, fh = face_box[2], face_box[3]
                ew = max(6, int(self.eye_box_frac * fw))
                eh = max(6, int(self.eye_box_frac * fh))
                left_eye = (int(x1 + le[0] - ew / 2), int(y1 + le[1] - eh / 2), ew, eh)
                right_eye = (int(x1 + re[0] - ew / 2), int(y1 + re[1] - eh / 2), ew, eh)

            return Face(face_box, left_eye, right_eye)

        if self.backend == "mediapipe":
            mp = self.impl
            with mp.solutions.face_detection.FaceDetection(
                model_selection=self.mp_model_selection,
                min_detection_confidence=self.mp_min_det_conf
            ) as fd:
                rgb = roi[:, :, ::-1]
                res = fd.process(rgb)
                if not res.detections:
                    return Face(None, None, None)
                det = max(res.detections, key=lambda d: d.score[0])
                bb = det.location_data.relative_bounding_box
                fx1 = int(bb.xmin * roi.shape[1])
                fy1 = int(bb.ymin * roi.shape[0])
                fw = int(bb.width * roi.shape[1])
                fh = int(bb.height * roi.shape[0])
                fx1 = _clip(x1 + fx1, 0, W - 1)
                fy1 = _clip(y1 + fy1, 0, H - 1)
                fw = max(0, min(fw, W - fx1))
                fh = max(0, min(fh, H - fy1))
                face_box = (fx1, fy1, fw, fh)

                left_eye = right_eye = None
                kps = getattr(det.location_data, "relative_keypoints", None)
                if kps and len(kps) >= 2:
                    re = kps[0]
                    le = kps[1]
                    ew = max(6, int(self.eye_box_frac * fw))
                    eh = max(6, int(self.eye_box_frac * fh))
                    right_eye = (int(x1 + re.x * roi.shape[1]) - ew // 2,
                                 int(y1 + re.y * roi.shape[0]) - eh // 2, ew, eh)
                    left_eye = (int(x1 + le.x * roi.shape[1]) - ew // 2,
                                int(y1 + le.y * roi.shape[0]) - eh // 2, ew, eh)

                return Face(face_box, left_eye, right_eye)

        # --- Haar fallback ---
        fd, ed, cv2 = self.impl
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = fd.detectMultiScale(
            gray,
            scaleFactor=self.haar_scale_factor,
            minNeighbors=self.haar_min_neighbors,
            minSize=self.haar_min_size
        )
        if len(faces) == 0:
            return Face(None, None, None)
        fx, fy, fw, fh = sorted(list(faces), key=lambda f: f[1])[0]
        face_box = (x1 + fx, y1 + fy, fw, fh)

        eyes = ed.detectMultiScale(
            gray[fy:fy + fh, fx:fx + fw],
            scaleFactor=self.haar_scale_factor,
            minNeighbors=max(2, self.haar_min_neighbors - 1),
            minSize=(max(12, self.haar_min_size[0] // 2), max(12, self.haar_min_size[1] // 2))
        )
        left_eye = right_eye = None
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[-1]
            left_eye = (x1 + fx + ex1, y1 + fy + ey1, ew1, eh1)
            right_eye = (x1 + fx + ex2, y1 + fy + ey2, ew2, eh2)
        elif len(eyes) == 1:
            ex, ey, ew, eh = eyes[0]
            if ex + ew / 2 <= fw / 2:
                left_eye = (x1 + fx + ex, y1 + fy + ey, ew, eh)
            else:
                right_eye = (x1 + fx + ex, y1 + fy + ey, ew, eh)

        return Face(face_box, left_eye, right_eye)


def draw_face_boxes(frame_bgr, face: Face) -> None:
    """Draws boxes in-place for quick debugging."""
    import cv2  # type: ignore

    def _rect(b, col):
        if b is None:
            return
        x, y, w, h = map(int, b)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), col, 2)

    if face is None:
        return
    _rect(face.face_box, (0, 255, 255))   # face
    _rect(face.left_eye_box, (255, 0, 0))
    _rect(face.right_eye_box, (255, 0, 0))
