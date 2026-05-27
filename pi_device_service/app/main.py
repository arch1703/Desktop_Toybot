"""
main.py — Raspberry Pi 3B actuation service.

Only exposes:
1. LED endpoints
2. Motor endpoints
3. Camera feed endpoints

No LLM logic.
No mode logic.
No display logic.
No buttons.
No audio.
"""

import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.motor_controller import (
    apply_motor_profile,
    execute_motor_action,
    cleanup_gpio,
)
from app.led_controller import set_led

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------

class LedColorRequest(BaseModel):
    color: str


class MotorActionRequest(BaseModel):
    action: str
    mode: str = ""
    motor_profile: str = ""


class MotorProfileRequest(BaseModel):
    profile: str


# ---------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------

app = FastAPI(
    title="Baymax Raspberry Pi Actuation Service",
    description="Raspberry Pi service for LED, motors, and camera feed only.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Camera Setup
# ---------------------------------------------------------------------

try:
    from picamera2 import Picamera2
    import cv2

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
    )

    CAMERA_AVAILABLE = True
    logger.info("Camera initialized successfully")

except Exception as e:
    logger.exception(f"Camera unavailable: {e}")
    picam2 = None
    CAMERA_AVAILABLE = False


def generate_frames():
    while True:
        if not CAMERA_AVAILABLE or picam2 is None:
            time.sleep(1)
            continue

        try:
            frame = picam2.capture_array()

            success, buffer = cv2.imencode(".jpg", frame)

            if not success:
                continue

            jpg_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg_bytes
                + b"\r\n"
            )

            time.sleep(0.03)

        except Exception as e:
            logger.exception(f"Camera frame error: {e}")
            time.sleep(1)


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    if CAMERA_AVAILABLE and picam2 is not None:
        try:
            picam2.start()
            logger.info("Pi camera started")
        except Exception as e:
            logger.exception(f"Failed to start camera: {e}")

    logger.info("Baymax Raspberry Pi actuation service started")


@app.on_event("shutdown")
async def shutdown():
    try:
        if CAMERA_AVAILABLE and picam2 is not None:
            picam2.stop()
            logger.info("Pi camera stopped")
    except Exception as e:
        logger.exception(f"Failed to stop camera cleanly: {e}")

    cleanup_gpio()
    logger.info("GPIO cleaned up")


# ---------------------------------------------------------------------
# Health / Root
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return HTMLResponse("""
    <html>
        <head>
            <title>Baymax Pi Actuation Service</title>
        </head>
        <body>
            <h1>Baymax Raspberry Pi Actuation Service</h1>

            <p>Status: running</p>

            <h2>Camera Stream</h2>
            <img src="/camera/stream" width="640" height="480" />

            <h2>Useful Endpoints</h2>
            <ul>
                <li><a href="/docs">FastAPI Docs</a></li>
                <li><a href="/health">Health</a></li>
                <li><a href="/camera/stream">Camera Stream</a></li>
                <li><a href="/motor/actions">Motor Actions</a></li>
            </ul>
        </body>
    </html>
    """)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Baymax Raspberry Pi Actuation Service",
        "camera_available": CAMERA_AVAILABLE,
        "features": {
            "camera": True,
            "motor": True,
            "led": True,
            "llm": False,
            "audio": False,
            "display": False,
            "buttons": False,
            "modes": False,
        },
    }


@app.get("/device/status")
def device_status():
    return {
        "status": "success",
        "camera_available": CAMERA_AVAILABLE,
        "available_systems": [
            "camera",
            "motor",
            "led",
        ],
    }


# ---------------------------------------------------------------------
# Camera Endpoints
# ---------------------------------------------------------------------

@app.get("/camera/stream")
def camera_stream():
    if not CAMERA_AVAILABLE or picam2 is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Camera not available"},
        )

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera/status")
def camera_status():
    return {
        "camera_available": CAMERA_AVAILABLE,
    }


# ---------------------------------------------------------------------
# LED Endpoints
# ---------------------------------------------------------------------

@app.post("/device/led")
def set_device_led(request: LedColorRequest):
    result = set_led(request.color)

    return {
        "status": "ok" if result.get("success") else "error",
        "color": request.color,
        "result": result,
    }


@app.post("/led")
def led_shortcut(request: LedColorRequest):
    """
    Shortcut endpoint for LED control.

    Example:
    curl -X POST http://<rpi-ip>:9000/led \\
      -H "Content-Type: application/json" \\
      -d '{"color": "red"}'
    """
    result = set_led(request.color)

    return {
        "status": "ok" if result.get("success") else "error",
        "color": request.color,
        "result": result,
    }


# ---------------------------------------------------------------------
# Motor Endpoints
# ---------------------------------------------------------------------

@app.get("/motor/actions")
def list_motor_actions():
    return {
        "allowed_actions": [
            "excited_wiggle",
            "nod_yes",
            "lean_left",
            "lean_right",
            "stop",
        ]
    }


@app.post("/motor/action")
def motor_action(request: MotorActionRequest):
    """
    Run one expressive motor action.

    Example action values:
    - excited_wiggle
    - nod_yes
    - lean_left
    - lean_right
    - stop
    """
    result = execute_motor_action(
        action=request.action,
        mode=request.mode,
        motor_profile=request.motor_profile,
    )

    return {
        "status": "executed",
        "action": request.action,
        "result": result,
    }


@app.post("/device/action")
def device_action(request: MotorActionRequest):
    """
    Compatibility endpoint.

    This keeps the old /device/action route working,
    but it only triggers motors now.
    """
    result = execute_motor_action(
        action=request.action,
        mode=request.mode,
        motor_profile=request.motor_profile,
    )

    return {
        "status": "executed",
        "action": request.action,
        "result": result,
    }


@app.post("/motor/profile")
def motor_profile(request: MotorProfileRequest):
    """
    Apply a motor profile.

    If ENA/ENB are jumpered, this may be a no-op depending on your
    motor_controller.py implementation.
    """
    result = apply_motor_profile(request.profile)

    return {
        "status": "applied",
        "profile": request.profile,
        "result": result,
    }


@app.post("/motor/stop")
def motor_stop():
    """
    Immediately stop both motors.
    """
    result = execute_motor_action("stop")

    return {
        "status": "stopped",
        "result": result,
    }


@app.get("/test/motor")
def test_motor_endpoint():
    return {
        "status": "ok",
        "message": "Motor endpoint reachable",
    }
