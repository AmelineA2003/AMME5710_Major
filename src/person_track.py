# tracker_reid.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import deque

import numpy as np
import cv2

Box = Tuple[int, int, int, int]  # x, y, w, h


# -------------------------
# Utility functions
# -------------------------

def _xywh_to_xyxy(b: Box) -> Tuple[int, int, int, int]:
    x, y, w, h = b
    return x, y, x + w, y + h


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _crop_box(img: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    return img[y1:y2, x1:x2]


def _appearance_embedding(img: np.ndarray, box: Box, bins=(8, 8, 8)) -> np.ndarray:
    """
    Simple appearance descriptor: HSV color histogram (L2-normalized).
    Returns a 1D float32 vector of length bins[0]*bins[1]*bins[2].
    """
    patch = _crop_box(img, box)
    if patch.size == 0:
        return np.zeros((bins[0] * bins[1] * bins[2],), dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    return num / den


# -------------------------
# Track data
# -------------------------

@dataclass
class _Track:
    track_id: int
    box: Box
    last_seen: int
    hits: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    embeddings: deque = field(default_factory=lambda: deque(maxlen=10))

    def avg_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return np.mean(np.stack(self.embeddings), axis=0).astype(np.float32)


# -------------------------
# Public Tracker (Re-ID)
# -------------------------

class TrackerReID:
    """
    Usage:
        tracker = TrackerReID(iou_thresh=0.35, appearance_thresh=0.65, max_age=45)
        for frame_idx, frame in enumerate(frames):
            detections = detect_people(frame)   # [[x,y,w,h,score], ...]
            states = tracker.update(frame, detections, frame_idx)

    update() returns:
        List[{"id": int, "box": (x,y,w,h), "hits": int}]
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        appearance_thresh: float = 0.6,
        max_age: int = 30,
        iou_weight: float = 0.6,
        hist_bins: Tuple[int, int, int] = (8, 8, 8),
    ) -> None:
        self.iou_thresh = iou_thresh
        self.appearance_thresh = appearance_thresh
        self.max_age = max_age
        self.iou_weight = iou_weight
        self.hist_bins = hist_bins

        self.next_id: int = 1
        self.tracks: Dict[int, _Track] = {}
        self.gallery: Dict[int, np.ndarray] = {}  # track_id -> representative embedding

    def update(self, frame: np.ndarray, detections: List[List[float]], frame_idx: int):
        """
        detections: list of [x,y,w,h,score] (score not used in association)
        """
        # expire old
        self._expire_stale(frame_idx)

        # no detections → just surface current states
        if not detections:
            return self._states()

        # prep
        boxes: List[Box] = [tuple(map(int, d[:4])) for d in detections]
        embs: List[np.ndarray] = [_appearance_embedding(frame, b, self.hist_bins) for b in boxes]

        # match to alive tracks
        det_to_track = self._associate_greedy(boxes, embs)

        # try Re-ID for the unmatched (revive old IDs from gallery)
        unmatched = set(range(len(boxes))) - set(det_to_track.keys())
        det_to_track.update(self._reid_from_gallery(boxes, embs, unmatched, frame_idx))

        # update/create
        assigned = set(det_to_track.keys())
        for di, box in enumerate(boxes):
            if di in assigned:
                tid = det_to_track[di]
                t = self.tracks[tid]
                t.box = box
                t.last_seen = frame_idx
                t.hits += 1
                t.embeddings.append(embs[di])
                self._update_gallery_for(t)
            else:
                self._new_track(box, embs[di], frame_idx)

        return self._states()

    # ---- internals ----

    def _states(self):
        return [{"id": tid, "box": tr.box, "hits": tr.hits} for tid, tr in self.tracks.items()]

    def _new_track(self, box: Box, emb: np.ndarray, frame_idx: int) -> _Track:
        tid = self.next_id
        self.next_id += 1
        t = _Track(track_id=tid, box=box, last_seen=frame_idx)
        t.embeddings.append(emb)
        self.tracks[tid] = t
        self._update_gallery_for(t)
        return t

    def _expire_stale(self, frame_idx: int) -> None:
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if frame_idx - tr.last_seen > self.max_age:
                self._update_gallery_for(tr)  # stash best embedding for future Re-ID
                self.tracks.pop(tid, None)

    def _update_gallery_for(self, t: _Track) -> None:
        if not t.embeddings:
            return
        self.gallery[t.track_id] = t.avg_embedding() if len(t.embeddings) > 1 else t.embeddings[-1]

    def _associate_greedy(self, detections: List[Box], embs: List[np.ndarray]) -> Dict[int, int]:
        det_to_track: Dict[int, int] = {}
        used_tracks = set()
        for di, db in enumerate(detections):
            best_tid, best_score = None, -1.0
            for tid, tr in self.tracks.items():
                iou_score = _iou(db, tr.box)
                if iou_score < self.iou_thresh:
                    continue
                gemb = self.gallery.get(tid, tr.avg_embedding())
                sim = _cosine_sim(embs[di], gemb) if gemb is not None else 0.0
                score = self.iou_weight * iou_score + (1.0 - self.iou_weight) * sim
                if score > best_score and tid not in used_tracks:
                    best_score, best_tid = score, tid
            if best_tid is not None:
                used_tracks.add(best_tid)
                det_to_track[di] = best_tid
        return det_to_track

    def _reid_from_gallery(
        self, detections: List[Box], embs: List[np.ndarray], skip: set, frame_idx: int
    ) -> Dict[int, int]:
        det_to_track: Dict[int, int] = {}
        active = set(self.tracks.keys())
        for di, db in enumerate(detections):
            if di in skip:
                continue
            best_tid, best_sim = None, -1.0
            for tid, gemb in self.gallery.items():
                if tid in active:
                    continue
                sim = _cosine_sim(embs[di], gemb)
                if sim > best_sim:
                    best_sim, best_tid = sim, tid
            if best_tid is not None and best_sim >= self.appearance_thresh:
                # revive that ID
                t = _Track(track_id=best_tid, box=db, last_seen=frame_idx)
                t.embeddings.append(embs[di])
                self.tracks[best_tid] = t
                det_to_track[di] = best_tid
        return det_to_track
