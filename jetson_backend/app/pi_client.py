"""
pi_client.py — All HTTP calls from the Jetson to the Raspberry Pi body service.

Pi address is configured via PI_BASE_URL environment variable.
Default: http://192.168.5.1:9000  (static Ethernet IP assigned to the Pi)
"""

import os
import requests
from app.schemas import Mode

PI_BASE_URL = os.environ.get("PI_BASE_URL", "http://192.168.5.1:9000")


def _post(path: str, payload: dict, timeout: float = 2.0) -> dict:
    try:
        response = requests.post(f"{PI_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return {"success": True, "pi_response": response.json()}
    except requests.RequestException as error:
        return {"success": False, "error": str(error), "message": f"POST {path} failed"}


def _get(path: str, timeout: float = 2.0) -> dict:
    try:
        response = requests.get(f"{PI_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return {"success": True, "pi_response": response.json()}
    except requests.RequestException as error:
        return {"success": False, "error": str(error), "message": f"GET {path} failed"}


# ── Mode / Action ─────────────────────────────────────────────────────────────

def push_mode_to_pi(mode: Mode, timeout: float = 2.0) -> dict:
    return _post("/device/mode", {"mode": mode.value}, timeout)


def push_action_to_pi(action: str, timeout: float = 2.0) -> dict:
    return _post("/device/action", {"action": action}, timeout)


def get_pi_status(timeout: float = 2.0) -> dict:
    return _get("/device/status", timeout)


# ── Eyes ──────────────────────────────────────────────────────────────────────

def push_eye_expression(expression: str, timeout: float = 2.0) -> dict:
    """Send an eye expression command to the Pi eye service."""
    return _post("/device/eyes", {"expression": expression}, timeout)


# ── LED ───────────────────────────────────────────────────────────────────────

def push_led_color(color: str, timeout: float = 2.0) -> dict:
    """Set the tri-color LED preset on the Pi."""
    return _post("/device/led", {"color": color}, timeout)
