"""
motor_controller.py — L298N motor driver via RPi.GPIO (BCM numbering).

ENA and ENB are JUMPERED to VCC — motors run at full voltage when enabled.
Speed control via PWM is not used. All motion is timed digital on/off pulses.

Wiring (BCM pin numbers):
  Motor A (left):  IN1=5,  IN2=6
  Motor B (right): IN3=13, IN4=19

Expressive actions (what Baymax does on a desk):
  excited_wiggle  — rapid left-right shake
  nod_yes         — forward-back bob
  lean_left       — hold left tilt briefly
  lean_right      — hold right tilt briefly
  stop            — all motors off (neutral)
"""

import logging
import time

logger = logging.getLogger(__name__)

# ── GPIO pin assignments ───────────────────────────────────────────────────────
IN1 = 5
IN2 = 6
IN3 = 13
IN4 = 19

_GPIO_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (IN1, IN2, IN3, IN4):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    _GPIO_AVAILABLE = True
    logger.info("L298N GPIO initialised (BCM): IN1=%d IN2=%d | IN3=%d IN4=%d", IN1, IN2, IN3, IN4)
except (ImportError, RuntimeError) as e:
    logger.warning("RPi.GPIO unavailable (%s) — motor stub active", e)


# ── Low-level primitives ──────────────────────────────────────────────────────

def _both_off():
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        for pin in (IN1, IN2, IN3, IN4):
            GPIO.output(pin, GPIO.LOW)
    else:
        logger.info("[MOTORS stub] both off")


def _left_forward():
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    else:
        logger.info("[MOTORS stub] left forward")


def _left_backward():
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        logger.info("[MOTORS stub] left backward")


def _right_forward():
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    else:
        logger.info("[MOTORS stub] right forward")


def _right_backward():
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        logger.info("[MOTORS stub] right backward")


def _pulse(left_fn, right_fn, duration_s: float):
    """Run a motor pattern for duration_s seconds then stop."""
    left_fn()
    right_fn()
    time.sleep(duration_s)
    _both_off()


# ── Expressive action sequences ───────────────────────────────────────────────

def _excited_wiggle():
    """Rapid alternating tilt: left-right-left-right."""
    for _ in range(3):
        _pulse(_left_forward, _right_backward, 0.15)
        time.sleep(0.05)
        _pulse(_left_backward, _right_forward, 0.15)
        time.sleep(0.05)
    _both_off()


def _nod_yes():
    """Forward-back bob twice."""
    for _ in range(2):
        _pulse(_left_forward, _right_forward, 0.18)
        time.sleep(0.05)
        _pulse(_left_backward, _right_backward, 0.18)
        time.sleep(0.05)
    _both_off()


def _lean_left():
    """Hold a left tilt for 400ms."""
    _pulse(_left_forward, _right_backward, 0.4)


def _lean_right():
    """Hold a right tilt for 400ms."""
    _pulse(_left_backward, _right_forward, 0.4)


# ── Public API ────────────────────────────────────────────────────────────────

_ACTION_MAP = {
    "excited_wiggle": _excited_wiggle,
    "nod_yes":        _nod_yes,
    "lean_left":      _lean_left,
    "lean_right":     _lean_right,
    "stop":           _both_off,
}


def execute_motor_action(action: str, mode: str = "", motor_profile: str = "") -> dict:
    fn = _ACTION_MAP.get(action)
    if fn is None:
        return {
            "motor_executed": False,
            "action": action,
            "message": f"Unknown action '{action}'. Allowed: {list(_ACTION_MAP.keys())}",
        }
    logger.info("[MOTOR] action=%s", action)
    fn()
    return {"motor_executed": True, "action": action}


def apply_motor_profile(profile: str) -> dict:
    """No-op: ENA/ENB are jumpered, speed profiles don't apply."""
    logger.info("[MOTOR PROFILE] jumpered — profile '%s' ignored", profile)
    return {"motor_profile_applied": False, "reason": "ENA/ENB jumpered, no PWM speed control"}


def cleanup_gpio() -> None:
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        _both_off()
        GPIO.cleanup()
        logger.info("GPIO cleaned up")


