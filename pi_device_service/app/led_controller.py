"""
led_controller.py — Active-low tri-color LED via RPi.GPIO (BCM numbering).

Wiring (BCM pin numbers, active LOW — tie cathode to GPIO, anode to 3.3 V via resistor):
  RED    = BCM 17
  GREEN  = BCM 27
  YELLOW = BCM 22

Active low: GPIO.LOW = LED ON, GPIO.HIGH = LED OFF

Color presets exposed via set_led(color):
  "red"     — red on, others off
  "green"   — green on, others off
  "yellow"  — yellow on, others off
  "all_on"  — all three on (produces white/mixed)
  "off"     — all off

Falls back to a stub logger if RPi.GPIO is unavailable.
"""

import logging

logger = logging.getLogger(__name__)

# ── Pin assignments ────────────────────────────────────────────────────────────
PIN_RED    = 17
PIN_GREEN  = 27
PIN_YELLOW = 22

# ── GPIO initialisation ────────────────────────────────────────────────────────
_GPIO_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (PIN_RED, PIN_GREEN, PIN_YELLOW):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)  # HIGH = OFF for active-low
    _GPIO_AVAILABLE = True
    logger.info("LED GPIO initialised (BCM): RED=%d GREEN=%d YELLOW=%d",
                PIN_RED, PIN_GREEN, PIN_YELLOW)
except (ImportError, RuntimeError) as e:
    logger.warning("RPi.GPIO unavailable (%s) — LED stub active", e)


# ── Core control ───────────────────────────────────────────────────────────────

def _write_pins(red: bool, green: bool, yellow: bool) -> None:
    """
    Set LED states. True = ON (GPIO LOW), False = OFF (GPIO HIGH).
    """
    if _GPIO_AVAILABLE:
        import RPi.GPIO as GPIO
        GPIO.output(PIN_RED,    GPIO.LOW if red    else GPIO.HIGH)
        GPIO.output(PIN_GREEN,  GPIO.LOW if green  else GPIO.HIGH)
        GPIO.output(PIN_YELLOW, GPIO.LOW if yellow else GPIO.HIGH)
    else:
        logger.info("[LED stub] red=%s green=%s yellow=%s", red, green, yellow)


# ── Named presets ──────────────────────────────────────────────────────────────

_PRESETS: dict[str, tuple[bool, bool, bool]] = {
    "red":    (True,  False, False),
    "green":  (False, True,  False),
    "yellow": (False, False, True),
    "all_on": (True,  True,  True),
    "off":    (False, False, False),
}


def set_led(color: str) -> dict:
    """
    Set the LED to a named color preset.
    color: red | green | yellow | all_on | off
    """
    preset = _PRESETS.get(color)
    if preset is None:
        return {
            "success": False,
            "error":   f"Unknown color preset '{color}'. Choose: {list(_PRESETS)}",
        }

    r, g, y = preset
    _write_pins(r, g, y)
    logger.info("[LED] color=%s  (R=%s G=%s Y=%s)", color, r, g, y)
    return {"success": True, "color": color, "red": r, "green": g, "yellow": y}


def red_only()    -> dict: return set_led("red")
def green_only()  -> dict: return set_led("green")
def yellow_only() -> dict: return set_led("yellow")
def all_on()      -> dict: return set_led("all_on")
def all_off()     -> dict: return set_led("off")
