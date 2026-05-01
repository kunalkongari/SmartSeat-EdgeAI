"""
app.py
Flask server: PiCamera2 → YOLOv8 inference → occupancy → web dashboard.

Thread model
────────────
  InferenceThread : captures frames, runs YOLO, updates shared state + JPEG
  Flask (main)    : serves HTTP / MJPEG stream (reads shared state only)
"""

import os
import sys
import time
import signal
import threading
import logging
import copy
import json

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from inference import YOLODetector, draw_detections
from utils import determine_occupancy, FPSTracker, encode_jpeg

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.environ.get("MODEL_PATH",    "model/yolov8n.pt")
# MODEL_PATH     = os.environ.get("MODEL_PATH",    "model/yolov8n_int8.tflite")
# MODEL_PATH     = os.environ.get("MODEL_PATH",    "model/yolov8n_float16.tflite")
# MODEL_PATH     = os.environ.get("MODEL_PATH",    "model/yolov8n_float32.tflite")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.45"))
TARGET_FPS     = int(os.environ.get("TARGET_FPS",       "10"))
FRAME_INTERVAL = 1.0 / TARGET_FPS
CAP_WIDTH      = 640
CAP_HEIGHT     = 480
JPEG_QUALITY   = 80

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "seats":          [],
    "total_seats":    0,
    "occupied_seats": 0,
    "vacant_seats":   0,
    "fps":            0.0,
    "error":          None,
}

_jpeg_lock    = threading.Lock()
_latest_jpeg: bytes = b""

# placeholder black frame sent before first real frame arrives
_PLACEHOLDER: bytes = b""

def _make_placeholder():
    global _PLACEHOLDER
    img = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, "Initialising camera...", (20, CAP_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    _PLACEHOLDER = buf.tobytes()

_make_placeholder()


# ── Camera helpers ────────────────────────────────────────────────────────────

def _open_picamera():
    """Try Picamera2 first (Raspberry Pi Camera Module)."""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cfg = cam.create_preview_configuration(
            main={"size": (CAP_WIDTH, CAP_HEIGHT), "format": "RGB888"}
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(1.0)   # warm-up
        logger.info("Picamera2 opened successfully.")
        return ("picamera2", cam)
    except Exception as e:
        logger.warning(f"Picamera2 unavailable ({e}); falling back to OpenCV.")
        return None


def _open_opencv_camera():
    """Fallback: USB cam or virtual device."""
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize lag
            logger.info(f"OpenCV VideoCapture opened on index {idx}.")
            return ("opencv", cap)
        cap.release()
    logger.error("No camera found via OpenCV.")
    return None


def _read_frame(cam_tuple):
    """Read one BGR frame. Returns np.ndarray or None."""
    kind, cam = cam_tuple
    if kind == "picamera2":
        try:
            frame = cam.capture_array()       # RGB888
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            logger.error(f"Picamera2 read error: {e}")
            return None
    else:
        ok, frame = cam.read()
        return frame if ok else None


def _close_camera(cam_tuple):
    if cam_tuple is None:
        return
    kind, cam = cam_tuple
    try:
        if kind == "picamera2":
            cam.stop()
        else:
            cam.release()
    except Exception:
        pass


# ── Inference loop ─────────────────────────────────────────────────────────────

def inference_loop():
    global _latest_jpeg, _state

    # ── Load model ────────────────────────────────────────────────────────────
    detector = YOLODetector(MODEL_PATH, conf_threshold=CONF_THRESHOLD)
    if not detector.load():
        with _lock:
            _state["error"] = f"Failed to load model: {MODEL_PATH}"
        logger.error(_state["error"])
        while True:
            _push_error_frame("Model load failed — check logs")
            time.sleep(2)

    # ── Open camera ───────────────────────────────────────────────────────────
    cam_tuple = _open_picamera() or _open_opencv_camera()

    if cam_tuple is None:
        with _lock:
            _state["error"] = "No camera found"
        while True:
            _push_error_frame("No camera available")
            time.sleep(2)

    fps_tracker = FPSTracker(window=20)
    logger.info("Inference loop running.")

    while True:
        t0 = time.monotonic()

        # ── Capture ───────────────────────────────────────────────────────────
        frame = _read_frame(cam_tuple)
        if frame is None:
            logger.warning("Empty frame; retrying.")
            time.sleep(0.05)
            continue

        # ── Detect ────────────────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Occupancy ─────────────────────────────────────────────────────────
        seats    = determine_occupancy(detections)
        occupied = sum(1 for s in seats if s["occupied"])
        vacant   = len(seats) - occupied

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = draw_detections(frame, detections, seats)

        fps_tracker.tick()
        fps = fps_tracker.fps

        # FPS overlay
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (6, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # ── Encode JPEG ───────────────────────────────────────────────────────
        try:
            jpeg = encode_jpeg(annotated, quality=JPEG_QUALITY)
        except Exception:
            jpeg = _PLACEHOLDER

        # ── Update shared state ───────────────────────────────────────────────
        with _lock:
            _state["seats"]          = copy.deepcopy(seats)
            _state["total_seats"]    = len(seats)
            _state["occupied_seats"] = occupied
            _state["vacant_seats"]   = vacant
            _state["fps"]            = fps
            _state["error"]          = None

        with _jpeg_lock:
            _latest_jpeg = jpeg

        # ── Rate-limit ────────────────────────────────────────────────────────
        elapsed = time.monotonic() - t0
        wait    = FRAME_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)


def _push_error_frame(message: str):
    global _latest_jpeg
    img = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, message, (20, CAP_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 220), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    with _jpeg_lock:
        _latest_jpeg = buf.tobytes()


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video_feed():
    """MJPEG stream."""
    def generate():
        while True:
            with _jpeg_lock:
                frame = _latest_jpeg or _PLACEHOLDER
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
            time.sleep(FRAME_INTERVAL)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    with _lock:
        payload = {
            "total_seats":    _state["total_seats"],
            "occupied_seats": _state["occupied_seats"],
            "vacant_seats":   _state["vacant_seats"],
            "fps":            _state["fps"],
            "error":          _state["error"],
            "seats": [
                {"seat_id": s["seat_id"], "occupied": s["occupied"]}
                for s in _state["seats"]
            ],
        }
    return jsonify(payload)


@app.route("/api/config")
def api_config():
    return jsonify({
        "model":          MODEL_PATH,
        "conf_threshold": CONF_THRESHOLD,
        "target_fps":     TARGET_FPS,
        "resolution":     f"{CAP_WIDTH}x{CAP_HEIGHT}",
    })


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _shutdown(sig, frame_ref):
    logger.info("Shutting down...")
    with _lock:
        _state["seats"]          = []
        _state["total_seats"]    = 0
        _state["occupied_seats"] = 0
        _state["vacant_seats"]   = 0
        _state["fps"]            = 0.0
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        logger.error(
            f"Model not found at '{MODEL_PATH}'.\n"
            f"Run:  python3 download_model.py"
        )
        sys.exit(1)

    t = threading.Thread(target=inference_loop, daemon=True, name="InferenceThread")
    t.start()
    logger.info("Inference thread started.")

    logger.info("Starting Flask on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
