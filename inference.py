"""
inference.py
YOLOv8-based object detection for Smart Seat Occupancy.
Detects: person (class 0), chair (class 56) — standard COCO indices.
"""

import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

# COCO class IDs we care about
PERSON_CLASS = 0
CHAIR_CLASS  = 56

TARGET_CLASSES = {
    PERSON_CLASS: "person",
    CHAIR_CLASS:  "chair",
}


class YOLODetector:
    """Wraps YOLOv8 (ultralytics) for person + chair detection."""

    def __init__(self, model_path: str, conf_threshold: float = 0.40, device: str = "cpu"):
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.device         = device
        self._model         = None
        self._loaded        = False

    def load(self) -> bool:
        """Load the YOLO model. Returns True on success."""
        try:
            from ultralytics import YOLO
            self._model  = YOLO(self.model_path)
            # Warm-up pass so first real frame isn't slow
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self._model(dummy, verbose=False, device=self.device)
            self._loaded = True
            logger.info(f"YOLOv8 model loaded: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, frame: np.ndarray) -> list:
        """
        Run YOLOv8 inference on a BGR frame.

        Returns list of dicts:
            {
                "class_id":   int,
                "label":      str,
                "confidence": float,
                "box":        [x1, y1, x2, y2]  # absolute pixel coords
            }
        """
        if not self._loaded:
            return []

        try:
            results = self._model(
                frame,
                conf=self.conf_threshold,
                classes=list(TARGET_CLASSES.keys()),  # only person + chair
                verbose=False,
                device=self.device,
                imgsz=320,   # fast inference on Pi
            )
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if class_id not in TARGET_CLASSES:
                    continue

                detections.append({
                    "class_id":   class_id,
                    "label":      TARGET_CLASSES[class_id],
                    "confidence": round(confidence, 3),
                    "box":        [x1, y1, x2, y2],
                })

        return detections


def draw_detections(frame: np.ndarray, detections: list, seats: list) -> np.ndarray:
    """
    Draw bounding boxes and seat overlays on the frame.

    detections : raw YOLO detections
    seats      : list of seat dicts from determine_occupancy()
    """
    vis = frame.copy()

    # ── Raw detections (thin boxes) ───────────────────────────────────────────
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        if det["label"] == "person":
            color = (0, 255, 100)   # green
        else:
            color = (255, 165, 0)   # orange

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        label_text = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(
            vis, label_text, (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    # ── Seat overlays (thick boxes with status) ───────────────────────────────
    for seat in seats:
        x1, y1, x2, y2 = seat["box"]
        color  = (0, 60, 220)  if seat["occupied"] else (0, 200, 80)
        status = "OCC"         if seat["occupied"] else "VAC"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"Seat {seat['seat_id']} [{status}]",
            (x1 + 2, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
        )

    return vis
