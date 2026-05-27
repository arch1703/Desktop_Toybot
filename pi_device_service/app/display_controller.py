"""
display_controller.py — Launch animated_eyes.py as a subprocess and forward
expression commands to it via eye_client (TCP socket on port 6000).
"""

import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

_eye_process: subprocess.Popen | None = None

# Path to animated_eyes.py — adjust if your project layout differs
_EYES_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "rpi_eyes", "animated_eyes.py"
)


def _start_eyes_process() -> None:
    global _eye_process
    script = os.path.abspath(_EYES_SCRIPT)
    if not os.path.exists(script):
        logger.warning("animated_eyes.py not found at %s — eye display unavailable", script)
        return
    try:
        _eye_process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)  # give pygame time to initialise before socket connects
        logger.info("animated_eyes.py started (PID %d)", _eye_process.pid)
    except Exception as exc:
        logger.warning("Could not start animated_eyes.py: %s", exc)
        _eye_process = None


def update_display(message: str) -> dict:
    """
    Map a mode display message to an eye expression and forward it.
    Also used as a general 'show something on screen' call.
    """
    from app.eye_client import send_expression

    # Simple mapping: mode messages → expressions
    msg_lower = message.lower()
    if any(k in msg_lower for k in ("play", "kids", "game", "story")):
        expression = "excited"
    elif any(k in msg_lower for k in ("meditat", "calm", "focus", "adult")):
        expression = "sleeping"
    else:
        expression = "normal"

    eye_result = send_expression(expression)
    logger.info("[DISPLAY] %s → eye expression: %s", message, expression)
    return {
        "display_updated": True,
        "message":         message,
        "eye_expression":  expression,
        "eye_result":      eye_result,
    }


def set_expression(expression: str) -> dict:
    """Directly set an eye expression by name."""
    from app.eye_client import send_expression
    return send_expression(expression)


# Start the eye process when this module is first imported (on Pi startup)
_start_eyes_process()
