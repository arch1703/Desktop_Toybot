"""
voice_commands.py — Routes text commands through the LLM brain.

All intent recognition is handled by the LLM; there are no fast-path shortcuts.
"""

from app.state import current_state


def normalize_command(command: str) -> str:
    return command.strip().lower()


def handle_voice_command(command: str) -> dict:
    """
    Delegate every command to the LLM.  The LLM calls registered tools
    (change_mode, move_robot, speak, set_led, set_eye_expression, …) as needed.
    """
    from app.llm_engine import chat

    normalized = normalize_command(command)
    current_state["last_voice_command"] = normalized

    result = chat(normalized)

    return {
        "recognized": True,
        "command": normalized,
        "mode_changed": False,
        "message": result["response"],
        "tool_calls": result["tool_calls"],
        "llm_used": True,
    }


    # LLM path: send to Baymax brain
    try:
        from app.llm_engine import chat
        llm_result = chat(command)
        return {
            "recognized":   True,
            "command":      normalized,
            "mode_changed": False,
            "new_mode":     current_state["mode"],
            "message":      llm_result["response"],
            "tool_calls":   llm_result["tool_calls"],
            "llm_used":     True,
        }
    except Exception as exc:
        return {
            "recognized":   False,
            "command":      normalized,
            "mode_changed": False,
            "new_mode":     current_state["mode"],
            "message":      f"Command not processed: {exc}",
            "llm_used":     True,
        }