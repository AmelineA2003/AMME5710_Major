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
    """Convert a box from (x, y, w, h) to (x1, y1, x2, y2).

    Args:
        b: Bounding box as (x, y, w, h), where (x, y) is top-left.

    Returns:
        A tuple (x1, y1, x2, y2) with bottom-right coordinates computed as
        (x + w, y + h).
    """
    x, y, w, h = b
    return x, y, x + w, y + h


def _iou(a: Box, b: Box) -> float:
    """Compute Intersection-over-Union (IoU) between two boxes.

    Args:
        a: First box in (x, y, w, h) format.
        b: Second box in (x, y, w, h) format.

    Returns:
        IoU between boxes in the range [0.0, 1.0]. Returns 0.0 if union is zero.
    """
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
    """Crop a region of interest from an image using a (x, y, w, h) box.

    Args:
        img: Input image (HxWx3 BGR or HxW single channel).
        box: Bounding box (x, y, w, h).

    Returns:
        The cropped image patch as a numpy array. If the box lies partially
        outside the image, it is clipped to image bounds. If it is fully
        outside or empty, returns an empty array.
    """
    x, y, w, h = box
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    return img[y1:y2, x1:x2]


def _appearance_embedding(img: np.ndarray, box: Box, bins=(8, 8, 8)) -> np.ndarray:
    """Compute a simple HSV color-histogram appearance embedding.

    The descriptor is an L2-normalized 3D histogram over HSV channels.

    Args:
        img: Input BGR image (HxWx3).
        box: Bounding box (x, y, w, h) used to extract the patch.
        bins: Number of histogram bins per channel as (H, S, V).

    Returns:
        A 1D float32 vector of length bins[0] * bins[1] * bins[2]. If the
        cropped patch is empty, returns a zero vector of that length.
    """
    patch = _crop_box(img, box)
    if patch.size == 0:
        return np.zeros((bins[0] * bins[1] * bins[2],), dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (1D).
        b: Second vector (1D).
        eps: Small constant added to denominator for numerical stability.

    Returns:
        Cosine similarity in the range [-1.0, 1.0].
    """
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    return num / den


# -------------------------
# Track data
# -------------------------

@dataclass
class _Track:
    """Internal track state for a single identity.

    Attributes:
        track_id: Unique integer identifier for the track.
        box: Latest bounding box (x, y, w, h).
        last_seen: Frame index when the track was last updated.
        hits: Number of successful associations (detections linked to this track).
        history: Recent box history (unused externally; reserved for extensions).
        embeddings: Recent appearance embeddings for this track.

    Methods:
        avg_embedding: Returns the average of stored embeddings, or None.
    """
    track_id: int
    box: Box
    last_seen: int
    hits: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    embeddings: deque = field(default_factory=lambda: deque(maxlen=10))

    def avg_embedding(self) -> np.ndarray | None:
        """Compute the mean embedding over recent samples.

        Returns:
            The averaged embedding as a float32 vector, or ``None`` if no
            embeddings are stored.
        """
        if not self.embeddings:
            return None
        return np.mean(np.stack(self.embeddings), axis=0).astype(np.float32)


# -------------------------
# Public Tracker (Re-ID)
# -------------------------

class TrackerReID:
    """Lightweight person tracker with IoU + appearance-based association.

    The tracker maintains active tracks, associates new detections using a
    weighted combination of IoU and cosine similarity of HSV histogram
    embeddings, and can re-identify recently expired tracks using a gallery
    of representative embeddings.

    Example:
        >>> tracker = TrackerReID(iou_thresh=0.35, appearance_thresh=0.65, max_age=45)
        >>> for frame_idx, frame in enumerate(frames):
        ...     detections = detect_people(frame)   # [[x,y,w,h,score], ...]
        ...     states = tracker.update(frame, detections, frame_idx)

    The ``update`` method returns a list of dicts:
        ``[{"id": int, "box": (x,y,w,h), "hits": int}, ...]``
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        appearance_thresh: float = 0.6,
        max_age: int = 30,
        iou_weight: float = 0.6,
        hist_bins: Tuple[int, int, int] = (8, 8, 8),
    ) -> None:
        """Initialize the tracker.

        Args:
            iou_thresh: Minimum IoU to consider a detection-track pair viable.
            appearance_thresh: Minimum cosine similarity to revive a track
                from the gallery during Re-ID.
            max_age: Maximum number of frames a track can be unseen before
                expiring (moved to gallery).
            iou_weight: Weight in [0,1] for IoU in the association score.
                The appearance weight is (1 - iou_weight).
            hist_bins: HSV histogram binning used in the appearance embedding.

        Attributes:
            next_id: Next track ID to assign for new tracks.
            tracks: Active tracks, mapping track_id -> _Track.
            gallery: Representative embeddings for expired/active tracks,
                mapping track_id -> embedding vector.
        """
        self.iou_thresh = iou_thresh
        self.appearance_thresh = appearance_thresh
        self.max_age = max_age
        self.iou_weight = iou_weight
        self.hist_bins = hist_bins

        self.next_id: int = 1
        self.tracks: Dict[int, _Track] = {}
        self.gallery: Dict[int, np.ndarray] = {}  # track_id -> representative embedding

    def update(self, frame: np.ndarray, detections: List[List[float]], frame_idx: int):
        """Update tracker state with detections from the current frame.

        Steps:
          1) Expire stale tracks (older than ``max_age``).
          2) Extract boxes and appearance embeddings for detections.
          3) Greedy associate detections to active tracks using IoU + appearance.
          4) Attempt Re-ID from gallery for remaining unmatched detections.
          5) Update matched tracks and create new tracks for unmatched detections.

        Args:
            frame: Current BGR frame (HxWx3).
            detections: List of detections as [x, y, w, h, score]. (Score is
                not used in association.)
            frame_idx: Index of the current frame.

        Returns:
            List of dictionaries for current active tracks:
            ``[{"id": int, "box": (x,y,w,h), "hits": int}, ...]``
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
        """Return simple public state of all active tracks.

        Returns:
            A list of dicts with fields:
            - ``id``: Track ID (int)
            - ``box``: Latest box (x, y, w, h)
            - ``hits``: Number of successful associations
        """
        return [{"id": tid, "box": tr.box, "hits": tr.hits} for tid, tr in self.tracks.items()]

    def _new_track(self, box: Box, emb: np.ndarray, frame_idx: int) -> _Track:
        """Create a new track with the next available ID.

        Args:
            box: Initial bounding box (x, y, w, h).
            emb: Initial appearance embedding for the track.
            frame_idx: Current frame index.

        Returns:
            The newly created _Track instance.
        """
        tid = self.next_id
        self.next_id += 1
        t = _Track(track_id=tid, box=box, last_seen=frame_idx)
        t.embeddings.append(emb)
        self.tracks[tid] = t
        self._update_gallery_for(t)
        return t

    def _expire_stale(self, frame_idx: int) -> None:
        """Expire tracks that have not been seen within ``max_age`` frames.

        Moves their best-known embedding to the gallery for potential Re-ID.

        Args:
            frame_idx: Current frame index used to compute staleness.
        """
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if frame_idx - tr.last_seen > self.max_age:
                self._update_gallery_for(tr)  # stash best embedding for future Re-ID
                self.tracks.pop(tid, None)

    def _update_gallery_for(self, t: _Track) -> None:
        """Update the gallery entry for a given track.

        If multiple embeddings are present, uses the average embedding.

        Args:
            t: Track whose embedding should be (re)stored in the gallery.
        """
        if not t.embeddings:
            return
        self.gallery[t.track_id] = t.avg_embedding() if len(t.embeddings) > 1 else t.embeddings[-1]

    def _associate_greedy(self, detections: List[Box], embs: List[np.ndarray]) -> Dict[int, int]:
        """Greedily associate detections to active tracks.

        The association score is a weighted sum:
            ``score = iou_weight * IoU + (1 - iou_weight) * cosine_similarity``

        Args:
            detections: Detected boxes [(x, y, w, h), ...].
            embs: Appearance embeddings corresponding to ``detections``.

        Returns:
            A mapping from detection index to track_id for matched pairs.
        """
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
        """Attempt to re-identify unmatched detections from the gallery.

        Considers only expired (inactive) track IDs. If an unmatched detection's
        embedding is similar enough to a gallery embedding, revive that ID.

        Args:
            detections: Detected boxes [(x, y, w, h), ...].
            embs: Appearance embeddings corresponding to ``detections``.
            skip: Indices of detections that are already matched.
            frame_idx: Current frame index (for revived track bookkeeping).

        Returns:
            A mapping from detection index to track_id for revived matches.
        """
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
