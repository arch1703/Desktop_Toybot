"""
llm_engine.py — Gemini-backed LLM brain with tool-calling loop.

Uses the Google GenAI Python SDK to call Gemini via the API.
On each turn:
  1. Append the user message to conversation history.
  2. Send full history + registered tools to Gemini.
  3. If the model calls a tool, dispatch it, append result, loop back to step 2.
  4. Return the final text response.

Configure via environment variables:
  GEMINI_API_KEY  — Google AI Studio API key (required)
  GEMINI_MODEL    — model tag (default: gemini-2.5-flash)
  GEMINI_PRO_MODEL — model used for complex/story tasks (default: gemini-2.5-pro)
"""

from __future__ import annotations
import logging
import os
import time

from google import genai
from google.genai import types

from app.tools_registry import get_gemini_tools, dispatch
from app.state import current_state

logger = logging.getLogger(__name__)

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PRO_MODEL = os.environ.get("GEMINI_PRO_MODEL", "gemini-2.5-flash")

_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are Baymax, a friendly and helpful healthcare companion robot.
You control a physical robot body through a set of tools.

Available subsystems you can control:
- Expressive body motion (excited_wiggle, nod_yes, lean_left, lean_right, stop)
- Tri-color LED (red, green, yellow, or off — use to signal mood/status)
- Animated eyes (normal, excited, disappointed, blink, sleeping)
- Camera vision (check if player is raising the correct hand)
- Audio speech synthesis (say something out loud)
- Operating mode switching (kids, young_adult, adult)

Guidelines:
- Be warm, reassuring, and engaging. You are modelled after Disney's Baymax.
- You support full natural conversation: answer questions, tell stories, run quizzes,
  guide meditations, and carry out multi-turn interactions across button presses.
- ALWAYS call speak() to deliver your response audibly — every meaningful reply must
  be spoken, not just returned as text.
- When asked to do a physical action, call the relevant tool immediately.
- When asked to switch modes, call change_mode() immediately.
- For the hand-raising game, always call analyze_hand_raise() before declaring success/failure.
- For stories, meditations, or quizzes, speak the full content — break long passages into
  natural call sequences (one speak() per paragraph or instruction step).
- After completing a physical action, briefly acknowledge it in your spoken response.
- If a tool fails, report the issue simply and suggest a fix.
- Never invent tool results — only report what the tool actually returned.
"""


def _get_history() -> list[types.Content]:
    return current_state.setdefault("conversation_history", [])


def chat(user_message: str, max_tool_rounds: int = 16, use_pro: bool = False) -> dict:
    """
    Process one user message through the Gemini tool-calling loop.

    Returns:
        {
          "response": str,          # final assistant text
          "tool_calls": list[dict], # log of tools that were called
          "rounds": int,
        }
    """
    model = GEMINI_PRO_MODEL if use_pro else GEMINI_MODEL
    history = _get_history()
    history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    current_state["last_voice_command"] = user_message

    gemini_tools = get_gemini_tools()
    tool_config   = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
    )
    tool_call_log: list[dict] = []

    for round_num in range(max_tool_rounds):
        response = None
        _max_retries = 3
        for _attempt in range(_max_retries):
            try:
                response = _client.models.generate_content(
                    model=model,
                    contents=history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=gemini_tools if gemini_tools else None,
                        tool_config=tool_config if gemini_tools else None,
                    ),
                )
                break  # success
            except Exception as exc:
                _err = str(exc)
                _transient = any(k in _err for k in ("503", "overloaded", "unavailable", "Unavailable", "502", "529"))
                if _transient and _attempt < _max_retries - 1:
                    _wait = 2 ** _attempt  # 1s, 2s
                    logger.warning("Gemini transient error (attempt %d/%d), retrying in %ds: %s", _attempt + 1, _max_retries, _wait, _err)
                    time.sleep(_wait)
                    continue
                logger.exception("Gemini chat failed")
                error_text = f"I'm having trouble thinking right now: {exc}"
                history.append(types.Content(role="model", parts=[types.Part(text=error_text)]))
                return {"response": error_text, "tool_calls": tool_call_log, "rounds": round_num}
        if response is None:  # shouldn't happen, but guard anyway
            error_text = "I'm having trouble thinking right now. Please try again."
            history.append(types.Content(role="model", parts=[types.Part(text=error_text)]))
            return {"response": error_text, "tool_calls": tool_call_log, "rounds": round_num}

        if not response.candidates:
            error_text = "I received an empty response from my brain. Please try again."
            history.append(types.Content(role="model", parts=[types.Part(text=error_text)]))
            return {"response": error_text, "tool_calls": tool_call_log, "rounds": round_num + 1}

        candidate = response.candidates[0]

        # Guard against blocked or empty content (safety filter, finish_reason != STOP, etc.)
        if not candidate.content or not candidate.content.parts:
            finish_reason = str(getattr(candidate, "finish_reason", "UNKNOWN"))
            error_text = f"I wasn't able to respond properly (reason: {finish_reason}). Please try again."
            history.append(types.Content(role="model", parts=[types.Part(text=error_text)]))
            return {"response": error_text, "tool_calls": tool_call_log, "rounds": round_num + 1}

        parts = candidate.content.parts

        # Collect any function calls in this response
        fn_calls = [p for p in parts if p.function_call is not None]
        text_parts = [p.text for p in parts if p.text is not None]

        if not fn_calls:
            # Final text response — no more tool calls
            text = " ".join(text_parts).strip()
            history.append(types.Content(role="model", parts=parts))
            return {"response": text, "tool_calls": tool_call_log, "rounds": round_num + 1}

        # Append model's turn (which contains function_call parts)
        history.append(types.Content(role="model", parts=parts))

        # Dispatch all function calls, collect responses
        fn_response_parts: list[types.Part] = []
        for p in fn_calls:
            fn_name = p.function_call.name
            fn_args = dict(p.function_call.args) if p.function_call.args else {}
            logger.info("Tool call: %s(%s)", fn_name, fn_args)

            result = dispatch(fn_name, fn_args)
            tool_call_log.append({"tool": fn_name, "args": fn_args, "result": result})
            logger.info("Tool result: %s", result)

            fn_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response={"result": result},
                    )
                )
            )

        history.append(types.Content(role="user", parts=fn_response_parts))

    # Exceeded max rounds — force a plain text response
    logger.warning("Reached max_tool_rounds (%d), forcing text response", max_tool_rounds)
    try:
        final = _client.models.generate_content(
            model=model,
            contents=history,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        text = final.text or "I ran out of steps trying to complete your request."
    except Exception:
        text = "I ran out of steps trying to complete your request."
    history.append(types.Content(role="model", parts=[types.Part(text=text)]))
    return {"response": text, "tool_calls": tool_call_log, "rounds": max_tool_rounds}


def reset_conversation() -> None:
    """Clear conversation history."""
    current_state["conversation_history"] = []


def register_builtin_tools() -> None:
    """
    Register all built-in Baymax tools.
    Called once at application startup from main.py.
    Deferred here to avoid circular imports.
    """
    import app.pi_client as pi
    import app.vision_processor as vision
    from app.tools_registry import register_tool

    # ── Expressive motion ─────────────────────────────────────────────────
    def move_robot(action: str) -> str:
        allowed = ["excited_wiggle", "nod_yes", "lean_left", "lean_right", "stop"]
        if action not in allowed:
            return f"Unknown action '{action}'. Allowed: {allowed}"
        result = pi.push_action_to_pi(action)
        current_state["last_robot_action"] = action
        current_state["pi_connected"] = result["success"]
        return f"Motion '{action}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="move_robot",
        description=(
            "Trigger an expressive body motion on the robot. "
            "excited_wiggle: rapid left-right shake to show excitement. "
            "nod_yes: forward-back bob to agree or encourage. "
            "lean_left / lean_right: tilt in that direction. "
            "stop: return to neutral upright position."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["excited_wiggle", "nod_yes", "lean_left", "lean_right", "stop"],
                    "description": "The expressive motion to perform.",
                }
            },
            "required": ["action"],
        },
        handler=move_robot,
        tags=["motor", "actuator"],
    )

    # ── LED ───────────────────────────────────────────────────────────────
    from app.pi_client import push_led_color as _push_led

    def set_led(color: str) -> str:
        result = _push_led(color)
        current_state["led_state"] = color
        current_state["pi_connected"] = result["success"]
        return f"LED '{color}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="set_led",
        description=(
            "Set the robot's LED ring color to signal state or mood. "
            "red: wrong answer / alert. green: correct / success. "
            "yellow: thinking / waiting. all_on: celebration. off: idle."
        ),
        parameters={
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "enum": ["red", "green", "yellow", "all_on", "off"],
                    "description": "Color preset to set.",
                }
            },
            "required": ["color"],
        },
        handler=set_led,
        tags=["led", "actuator"],
    )

    # ── Eyes ──────────────────────────────────────────────────────────────
    def set_eye_expression(expression: str) -> str:
        allowed = ["normal", "excited", "disappointed", "blink", "sleeping"]
        if expression not in allowed:
            return f"Unknown expression '{expression}'. Allowed: {allowed}"
        result = pi.push_eye_expression(expression)
        current_state["eye_state"] = expression
        current_state["pi_connected"] = result["success"]
        return f"Eye expression '{expression}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="set_eye_expression",
        description=(
            "Change Baymax's eye display expression. "
            "excited: happy crescent eyes with glow flutter. "
            "disappointed: hooded heavy eyes. "
            "blink: single blink. "
            "sleeping: eyes nearly closed. "
            "normal: return to default open eyes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "enum": ["normal", "excited", "disappointed", "blink", "sleeping"],
                    "description": "The eye expression to display.",
                }
            },
            "required": ["expression"],
        },
        handler=set_eye_expression,
        tags=["eyes", "actuator"],
    )

    # ── Speech ────────────────────────────────────────────────────────────
    from app.audio_manager import speak as _local_speak

    def speak(text: str) -> str:
        success = _local_speak(text)
        return f"Spoke: '{text}' — {'ok' if success else 'playback failed'}"

    register_tool(
        name="speak",
        description=(
            "Make Baymax speak text out loud through the robot's speaker. "
            "Length should match the content — brief for acknowledgements, full paragraphs "
            "for stories, meditation instructions, or quiz questions. "
            "Always call this to deliver your response audibly."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to synthesize and play through the speaker.",
                }
            },
            "required": ["text"],
        },
        handler=speak,
        tags=["audio", "actuator"],
    )

    # ── Vision ────────────────────────────────────────────────────────────
    def analyze_hand_raise(expected: str) -> str:
        result = vision.analyze_hand_raise(expected)
        return result.get("details", str(result))

    register_tool(
        name="analyze_hand_raise",
        description=(
            "Capture a frame from the Pi Camera and use Gemini Vision to check "
            "whether the player is raising the expected hand. "
            "Pass expected='right' or expected='left'. "
            "Returns whether the correct hand is raised."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expected": {
                    "type": "string",
                    "enum": ["right", "left"],
                    "description": "Which hand should be raised.",
                }
            },
            "required": ["expected"],
        },
        handler=analyze_hand_raise,
        tags=["vision", "sensor"],
    )

    def describe_scene() -> str:
        result = vision.describe_scene()
        return result.get("description", str(result))

    register_tool(
        name="describe_scene",
        description=(
            "Capture a frame from the Pi Camera and describe what is visible using Gemini Vision. "
            "Useful for situational awareness or checking if a user looks engaged."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=describe_scene,
        tags=["vision", "sensor"],
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    def change_mode(mode: str) -> str:
        from app.schemas import Mode
        from app.mode_manager import set_current_mode
        try:
            m = Mode(mode)
        except ValueError:
            return f"Unknown mode '{mode}'. Choose from: kids, young_adult, adult"
        set_current_mode(m)
        result = pi.push_mode_to_pi(m)
        current_state["pi_connected"] = result["success"]
        return f"Mode changed to '{mode}': {'ok' if result['success'] else result.get('error', 'failed')}"

    register_tool(
        name="change_mode",
        description=(
            "Switch Baymax's operating mode. "
            "kids: safe, slow, playful — hand-raising game, storytelling. "
            "young_adult: medium speed, puzzles, tutoring. "
            "adult: wellness, guided meditation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["kids", "young_adult", "adult"],
                    "description": "The operating mode to activate.",
                }
            },
            "required": ["mode"],
        },
        handler=change_mode,
        tags=["mode"],
    )

    logger.info("Built-in tools registered: %s", list_tools())


def list_tools() -> list[str]:
    from app.tools_registry import list_tools as _lt
    return _lt()
