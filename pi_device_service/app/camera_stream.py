"""
camera_stream.py — picamera2 MJPEG stream server for the Raspberry Pi.

Starts a background thread that continuously captures frames from the Pi Camera
Module (CSI) and serves them as an MJPEG stream on port 8080 at GET /stream.

Also exposes capture_jpeg() to grab a single JPEG frame for use by
FastAPI endpoints without going through the HTTP stream.

Falls back to a stub (black frame) if picamera2 is unavailable.
"""

import io
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

STREAM_PORT   = 8080
JPEG_QUALITY  = 80
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# Thread-safe: latest JPEG bytes
_frame_lock   = threading.Lock()
_latest_frame: bytes | None = None
_camera_ready = threading.Event()

# ── Camera capture thread ─────────────────────────────────────────────────────

def _capture_loop() -> None:
    global _latest_frame
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        _camera_ready.set()
        logger.info("Pi Camera started (%dx%d)", FRAME_WIDTH, FRAME_HEIGHT)

        import numpy as np
        import cv2

        while True:
            frame = cam.capture_array()
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with _frame_lock:
                _latest_frame = jpeg.tobytes()
            time.sleep(0.04)  # ~25 fps

    except ImportError:
        logger.warning("picamera2 not installed — using stub black frame")
        import numpy as np
        import cv2
        stub = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", stub)
        with _frame_lock:
            _latest_frame = jpeg.tobytes()
        _camera_ready.set()

    except Exception as exc:
        logger.exception("Camera capture loop failed: %s", exc)
        _camera_ready.set()


# ── MJPEG HTTP server ─────────────────────────────────────────────────────────

class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence request logs
        pass

    def do_GET(self):
        if self.path not in ("/stream", "/"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                with _frame_lock:
                    frame = _latest_frame

                if frame:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")

                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _mjpeg_server_thread() -> None:
    server = HTTPServer(("0.0.0.0", STREAM_PORT), _MJPEGHandler)
    logger.info("MJPEG stream server started on port %d", STREAM_PORT)
    server.serve_forever()


# ── Public API ─────────────────────────────────────────────────────────────────

def capture_jpeg() -> bytes | None:
    """Return the latest JPEG frame bytes, or None if camera not ready."""
    with _frame_lock:
        return _latest_frame


def start() -> None:
    """Start the camera capture loop and MJPEG HTTP server (call once at startup)."""
    capture_thread = threading.Thread(target=_capture_loop, daemon=True, name="cam-capture")
    capture_thread.start()

    # Wait up to 5 s for camera to initialise before starting HTTP server
    _camera_ready.wait(timeout=5.0)

    stream_thread = threading.Thread(target=_mjpeg_server_thread, daemon=True, name="cam-stream")
    stream_thread.start()
    logger.info("Camera stream threads started")
