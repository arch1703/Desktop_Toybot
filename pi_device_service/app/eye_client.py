"""
eye_client.py — TCP socket client to send expression commands to animated_eyes.py.

animated_eyes.py listens on port 6000 (localhost).
Protocol: send a single JSON line, receive {"status":"ok"}.
"""

import json
import socket
import logging

logger = logging.getLogger(__name__)

EYE_HOST = "localhost"
EYE_PORT = 6000


def send_expression(expression: str) -> dict:
    """
    Send an expression command to the running animated_eyes process.
    expression: one of normal | excited | disappointed | blink | sleeping
    """
    try:
        with socket.create_connection((EYE_HOST, EYE_PORT), timeout=2.0) as sock:
            payload = json.dumps({"expression": expression}).encode("utf-8")
            sock.sendall(payload)
            response = sock.recv(64).decode("utf-8", errors="ignore")
        logger.info("Eye expression sent: %s  response: %s", expression, response)
        return {"success": True, "expression": expression}
    except (OSError, socket.timeout) as exc:
        logger.warning("Eye client failed: %s", exc)
        return {"success": False, "expression": expression, "error": str(exc)}
