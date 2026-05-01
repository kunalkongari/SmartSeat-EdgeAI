"""
utils.py
IoU, occupancy logic, FPS tracker, JPEG encoder.
"""

import cv2
import time
import collections
import numpy as np


# ── IoU ────────────────────────────────────────────────────────────────────────

def compute_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ── Occupancy logic ────────────────────────────────────────────────────────────

IOU_THRESHOLD    = 0.15   # lower = more sensitive (person near chair → occupied)
PROXIMITY_PIXELS = 60     # fallback: person within N px of chair centre → occupied


def _box_centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _centre_distance(box_a, box_b):
    cx1, cy1 = _box_centre(box_a)
    cx2, cy2 = _box_centre(box_b)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def determine_occupancy(detections: list) -> list:
    """
    Determine which detected chairs are occupied.

    A chair is 'occupied' if:
      - IoU with any person box >= IOU_THRESHOLD, OR
      - Centre of any person is within PROXIMITY_PIXELS of the chair centre

    Returns list of seat dicts:
        {"seat_id": int, "occupied": bool, "box": [x1,y1,x2,y2]}
    """
    persons = [d for d in detections if d["label"] == "person"]
    chairs  = [d for d in detections if d["label"] == "chair"]

    seats = []
    for idx, chair in enumerate(chairs, start=1):
        occupied = False
        for person in persons:
            iou  = compute_iou(chair["box"], person["box"])
            dist = _centre_distance(chair["box"], person["box"])
            if iou >= IOU_THRESHOLD or dist <= PROXIMITY_PIXELS:
                occupied = True
                break

        seats.append({
            "seat_id":  idx,
            "occupied": occupied,
            "box":      chair["box"],
        })

    return seats


# ── FPS tracker ───────────────────────────────────────────────────────────────

class FPSTracker:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 20):
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self):
        self._times.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return round((len(self._times) - 1) / elapsed, 1) if elapsed > 0 else 0.0


# ── JPEG encoder ──────────────────────────────────────────────────────────────

def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
