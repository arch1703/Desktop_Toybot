from app.schemas import Mode
from app.state import current_state


MODE_FEATURES = {
    Mode.kids: [
        {"id": "trajectory_mapping_mat", "name": "Trajectory Mapping (Mat)", "action": "start_trajectory_mapping_mat"},
        {"id": "storytelling", "name": "Storytelling", "action": "start_storytelling"},
        {"id": "video_game_snake", "name": "Snake Game", "action": "start_video_game_snake"},
        {"id": "video_game_memory", "name": "Memory Game", "action": "start_video_game_memory"},
    ],
    Mode.young_adult: [
        {"id": "puzzles", "name": "Puzzles", "action": "start_puzzles"},
        {"id": "peer_tutoring", "name": "Peer Tutoring", "action": "start_peer_tutoring"},
        {"id": "trajectory_mapping_desktop", "name": "Trajectory Mapping (Desktop)", "action": "start_trajectory_mapping_desktop"},
    ],
    Mode.adult: [
        {"id": "guided_meditation", "name": "Guided Meditation", "action": "start_guided_meditation"},
        {"id": "facial_recognition", "name": "Facial Recognition", "action": "start_facial_recognition"},
    ],
}


def get_current_mode() -> Mode:
    return current_state["mode"]


def set_current_mode(mode: Mode) -> dict:
    current_state["mode"] = mode

    return {
        "mode": mode,
        "description": get_mode_description(mode),
        "config": get_mode_config(mode),
    }


def get_mode_description(mode: Mode) -> str:
    descriptions = {
        Mode.kids: (
            "Safe mode with trajectory mapping, storytelling, "
            "and simple video games. Restricted motor speed."
        ),
        Mode.young_adult: (
            "Interactive mode with puzzles, peer tutoring, "
            "and desktop trajectory mapping."
        ),
        Mode.adult: (
            "Wellness and productivity mode with guided meditation "
            "and facial recognition."
        ),
    }

    return descriptions[mode]


def get_mode_config(mode: Mode) -> dict:
    configs = {
        Mode.kids: {
            "features": MODE_FEATURES[Mode.kids],
            "voice_enabled": False,
            "motor_speed": "low",
            "tone": "playful",
        },
        Mode.young_adult: {
            "features": MODE_FEATURES[Mode.young_adult],
            "voice_enabled": True,
            "motor_speed": "medium",
            "tone": "casual",
        },
        Mode.adult: {
            "features": MODE_FEATURES[Mode.adult],
            "voice_enabled": True,
            "motor_speed": "low",
            "tone": "calm",
        },
    }

    return configs[mode]