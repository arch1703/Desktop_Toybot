from app.schemas import Mode
from app.state import current_state


MOTOR_ACTIONS = ["excited_wiggle", "nod_yes", "lean_left", "lean_right", "stop"]

FEATURE_ACTIONS = {
    Mode.kids: [
        "start_trajectory_mapping_mat",
        "start_storytelling",
        "start_video_game_snake",
        "start_video_game_memory",
    ],
    Mode.young_adult: [
        "start_puzzles",
        "start_peer_tutoring",
        "start_trajectory_mapping_desktop",
    ],
    Mode.adult: [
        "start_guided_meditation",
        "start_facial_recognition",
    ],
}


def get_robot_behavior_for_mode(mode: Mode) -> dict:
    behaviors = {
        Mode.kids: {
            "lcd_message": "Kids Mode",
            "speaker_prompt": "Let's play!",
            "motor_profile": "slow_drive",
            "safety_level": "maximum",
        },
        Mode.young_adult: {
            "lcd_message": "Young Adult Mode",
            "speaker_prompt": "Choose a challenge.",
            "motor_profile": "medium_drive",
            "safety_level": "medium",
        },
        Mode.adult: {
            "lcd_message": "Adult Mode",
            "speaker_prompt": "How can I help you focus today?",
            "motor_profile": "slow_drive",
            "safety_level": "standard",
        },
    }

    return behaviors[mode]


def validate_robot_action(action: str) -> dict:
    mode = current_state["mode"]
    current_state["last_robot_action"] = action

    allowed = MOTOR_ACTIONS + FEATURE_ACTIONS[mode]

    if action not in allowed:
        return {
            "allowed": False,
            "mode": mode,
            "action": action,
            "message": f"Action '{action}' is not allowed in {mode.value} mode.",
        }

    return {
        "allowed": True,
        "mode": mode,
        "action": action,
        "message": f"Action '{action}' is allowed in {mode.value} mode.",
    }