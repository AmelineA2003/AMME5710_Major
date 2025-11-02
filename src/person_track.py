# tracker_reid.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
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
    """HSV color histogram (L2-normalized) as a light appearance descriptor."""
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
    hits: int = 1                  # total matches since birth
    hit_streak: int = 1            # consecutive matched frames
    age: int = 0                   # frames since birth
    confirmed: bool = False        # becomes True after MIN_HITS
    deleted: bool = False
    # appearance (EMA + short history)
    emb: Optional[np.ndarray] = None
    emb_hist: deque = field(default_factory=lambda: deque(maxlen=10))

    def update_embedding(self, new_emb: np.ndarray, alpha: float) -> None:
        if self.emb is None:
            self.emb = new_emb.astype(np.float32)
        else:
            self.emb = (1.0 - alpha) * self.emb + alpha * new_emb
        self.emb_hist.append(new_emb)
        n = np.linalg.norm(self.emb) + 1e-8
        self.emb = (self.emb / n).astype(np.float32)


# -------------------------
# Public Tracker (Re-ID)
# -------------------------

class TrackerReID:
    """
    tracker = TrackerReID(iou_thresh=0.35, appearance_thresh=0.65, max_age=45)
    states = tracker.update(frame, detections, frame_idx)
    -> List[{"id": int, "box": (x,y,w,h), "hits": int}]
    """

    def __init__(
        self,
        iou_thresh: float = 0.35,
        appearance_thresh: float = 0.65,   # cosine similarity gate (0..1)
        max_age: int = 45,
        iou_weight: float = 0.3,           # appearance-dominant cost (1-w)*sim + w*iou
        hist_bins: Tuple[int, int, int] = (8, 8, 8),
        # Hygiene knobs
        min_hits: int = 3,                 # confirmation before surfacing
        ema_alpha: float = 0.20,           # embedding smoothing
        dup_iou: float = 0.70,             # duplicate suppression IoU
        dup_app: float = 0.80,             # duplicate suppression similarity (cosine)
        # Reattach & spawn control
        reattach_iou: float = 0.60,        # overlap to reattach after brief miss
        reattach_grace: int = 3,           # frames since last_seen for reattach
        relaxed_app: float = 0.50,         # relaxed appearance for reattach
        no_spawn_iou: float = 0.60,        # block births on top of active tracks
    ) -> None:
        self.iou_thresh = iou_thresh
        self.appearance_thresh = appearance_thresh
        self.max_age = max_age
        self.iou_weight = iou_weight
        self.hist_bins = hist_bins

        self.min_hits = min_hits
        self.ema_alpha = ema_alpha
        self.dup_iou = dup_iou
        self.dup_app = dup_app
        self.reattach_iou = reattach_iou
        self.reattach_grace = reattach_grace
        self.relaxed_app = relaxed_app
        self.no_spawn_iou = no_spawn_iou

        self.next_id: int = 1
        self.tracks: Dict[int, _Track] = {}
        self.gallery: Dict[int, np.ndarray] = {}  # track_id -> representative embedding

    # --------------- public API ---------------

    def update(self, frame: np.ndarray, detections: List[List[float]], frame_idx: int):
        """detections: [[x,y,w,h,score], ...]  (score unused here)"""
        # expire old & age survivors
        self._expire_stale(frame_idx)
        for tr in self.tracks.values():
            tr.age += 1

        if not detections:
            for tr in self.tracks.values():
                tr.hit_streak = 0
            return self._states()

        boxes: List[Box] = [tuple(map(int, d[:4])) for d in detections]
        embs: List[np.ndarray] = [_appearance_embedding(frame, b, self.hist_bins) for b in boxes]

        # Global association (Hungarian) with hard gating
        det_to_track, unmatched_dets, unmatched_tracks = self._associate(boxes, embs)

        # Reattach unmatched detections to recently-unmatched tracks (grace)
        det_to_track, unmatched_dets, unmatched_tracks = \
            self._reattach_grace(boxes, embs, det_to_track, unmatched_dets, unmatched_tracks, frame_idx)

        # Update matched tracks / create new (with no-spawn-on-active rule)
        assigned = set(det_to_track.keys())
        matched_now: Set[int] = set()

        for di, box in enumerate(boxes):
            if di in assigned:
                tid = det_to_track[di]
                t = self.tracks[tid]
                t.box = box
                t.last_seen = frame_idx
                t.hits += 1
                t.hit_streak += 1
                t.confirmed = t.confirmed or (t.hits >= self.min_hits)
                t.update_embedding(embs[di], self.ema_alpha)
                self._update_gallery_for(t)
                matched_now.add(tid)
            else:
                # Block births that sit on top of any active track (prevents forking)
                if self._overlaps_active(box):
                    continue
                self._new_track(box, embs[di], frame_idx)

        # unmatched tracks lose streak this frame
        for tid, tr in self.tracks.items():
            if tid not in matched_now:
                tr.hit_streak = 0

        # Duplicate-track suppression (overlap + appearance)
        self._suppress_duplicates()

        return self._states()

    # --------------- internals ---------------

    def _states(self):
        """Only surface confirmed, non-deleted tracks."""
        return [
            {"id": tid, "box": tr.box, "hits": tr.hits}
            for tid, tr in self.tracks.items()
            if not tr.deleted and tr.confirmed
        ]

    def _new_track(self, box: Box, emb: np.ndarray, frame_idx: int) -> _Track:
        tid = self.next_id
        self.next_id += 1
        t = _Track(track_id=tid, box=box, last_seen=frame_idx)
        t.update_embedding(emb, alpha=1.0)  # init
        self.tracks[tid] = t
        self._update_gallery_for(t)
        return t

    def _expire_stale(self, frame_idx: int) -> None:
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if frame_idx - tr.last_seen > self.max_age:
                self._update_gallery_for(tr)  # keep last good emb for future re-ID
                self.tracks.pop(tid, None)

    def _update_gallery_for(self, t: _Track) -> None:
        if t.emb is not None:
            self.gallery[t.track_id] = t.emb

    # -------- Association (Hungarian) --------

    def _associate(self, detections: List[Box], embs: List[np.ndarray]):
        """Global assignment with IoU + appearance gating."""
        tracks = list(self.tracks.items())
        if not tracks or not detections:
            return {}, set(range(len(detections))), set(tid for tid, _ in tracks)

        T, D = len(tracks), len(detections)
        BIG = 1e3
        cost = np.full((T, D), BIG, dtype=np.float32)

        for ti, (tid, tr) in enumerate(tracks):
            gemb = self.gallery.get(tid, tr.emb)
            for di, db in enumerate(detections):
                iou_score = _iou(db, tr.box)
                if iou_score < self.iou_thresh:
                    continue
                sim = _cosine_sim(embs[di], gemb) if gemb is not None else 0.0
                if sim < self.appearance_thresh:
                    continue
                score = (1.0 - self.iou_weight) * sim + self.iou_weight * iou_score
                cost[ti, di] = 1.0 - score  # lower is better

        det_to_track: Dict[int, int] = {}
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(cost)
            for ti, di in zip(r, c):
                if cost[ti, di] >= 1.0:  # disallowed/awful pair
                    continue
                tid = tracks[ti][0]
                det_to_track[di] = tid
        except Exception:
            # Greedy fallback
            used_t, used_d = set(), set()
            flat = [(cost[ti, di], ti, di) for ti in range(T) for di in range(D)]
            for val, ti, di in sorted(flat, key=lambda x: x[0]):
                if val >= 1.0 or ti in used_t or di in used_d:
                    continue
                used_t.add(ti); used_d.add(di)
                det_to_track[di] = tracks[ti][0]

        matched_tids = set(det_to_track.values())
        unmatched_tracks = set(tid for tid, _ in tracks) - matched_tids
        unmatched_dets = set(range(D)) - set(det_to_track.keys())
        return det_to_track, unmatched_dets, unmatched_tracks

    # -------- Reattach & spawn control --------

    def _reattach_grace(
        self,
        boxes: List[Box],
        embs: List[np.ndarray],
        det_to_track: Dict[int, int],
        unmatched_dets: Set[int],
        unmatched_tracks: Set[int],
        frame_idx: int,
    ):
        """Allow a relaxed re-lock to a recently-unmatched track."""
        unmatched_dets = set(unmatched_dets)
        unmatched_tracks = set(unmatched_tracks)

        for di in list(unmatched_dets):
            db = boxes[di]
            best_tid, best_iou = None, 0.0
            for tid in list(unmatched_tracks):
                tr = self.tracks[tid]
                if (frame_idx - tr.last_seen) > self.reattach_grace:
                    continue
                iou_val = _iou(db, tr.box)
                if iou_val < self.reattach_iou:
                    continue
                gemb = self.gallery.get(tid, tr.emb)
                sim = _cosine_sim(embs[di], gemb) if gemb is not None else 0.0
                if sim < self.relaxed_app:
                    continue
                if iou_val > best_iou:
                    best_iou, best_tid = iou_val, tid

            if best_tid is not None:
                det_to_track[di] = best_tid
                unmatched_dets.remove(di)
                unmatched_tracks.discard(best_tid)

        return det_to_track, unmatched_dets, unmatched_tracks

    def _overlaps_active(self, db: Box) -> bool:
        for tid, tr in self.tracks.items():
            if _iou(db, tr.box) >= self.no_spawn_iou:
                return True
        return False

    # -------- Duplicate suppression --------

    def _suppress_duplicates(self) -> None:
        """
        If two active tracks latch onto the same person, collapse the younger/weaker one.
        Uses IoU + appearance similarity thresholds (dup_iou, dup_app).
        """
        items = sorted(
            self.tracks.items(),
            key=lambda kv: (kv[1].confirmed, kv[1].hits), reverse=True
        )
        keep: Dict[int, _Track] = {}
        for tid, t in items:
            if t.deleted:
                continue
            drop = False
            for uid, u in keep.items():
                if _iou(t.box, u.box) >= self.dup_iou:
                    a = t.emb if t.emb is not None else self.gallery.get(t.track_id)
                    b = u.emb if u.emb is not None else self.gallery.get(u.track_id)
                    sim = _cosine_sim(a, b) if (a is not None and b is not None) else 0.0
                    if sim >= self.dup_app:
                        # keep the stronger/older
                        if (t.confirmed and not u.confirmed) or (t.hits > u.hits):
                            u.deleted = True
                            keep.pop(uid, None)
                            keep[tid] = t
                        else:
                            t.deleted = True
                        drop = True
                        break
            if not drop and not t.deleted:
                keep[tid] = t
        self.tracks = {tid: tr for tid, tr in keep.items() if not tr.deleted}
