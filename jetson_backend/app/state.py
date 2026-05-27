from app.schemas import Mode


current_state = {
    "mode": Mode.kids,
    "last_voice_command": None,
    "last_robot_action": None,
    "pi_connected": False,
    # LLM
    "conversation_history": [],
    # Peripheral states (updated by LLM tool calls)
    "eye_state": "normal",
    "led_state": "off",
    "camera_active": False,
}