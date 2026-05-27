import asyncio
import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from app.fer_onnx_manager import run_fer_once

# Load .env from Jetson backend folder
load_dotenv(dotenv_path="/home/vk-jn-or/Desktop/baymax/jetson_backend/.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Only needed if using Gemini TTS
AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE")
AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE")

# Raspberry Pi config
PI_BASE_URL = os.getenv("PI_BASE_URL", "http://192.168.10.2:9000")
PI_STREAM_URL = os.getenv("PI_STREAM_URL", f"{PI_BASE_URL}/camera/stream")

from app.schemas import (
    ModeRequest,
    VoiceCommandRequest,
    RobotActionRequest,
    LLMChatRequest,
    EyeRequest,
    LedRequest,
)
from app.state import current_state
from app.mode_manager import (
    get_current_mode,
    set_current_mode,
    get_mode_config,
    get_mode_description,
)
from app.voice_commands import handle_voice_command
from app.robot_controller import (
    get_robot_behavior_for_mode,
    validate_robot_action,
)
from app.pi_client import (
    push_mode_to_pi,
    push_action_to_pi,
    get_pi_status,
    push_eye_expression,
    push_led_color,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
CAPTURE_DIR = Path(__file__).resolve().parent.parent / "captures"
CAPTURE_DIR.mkdir(exist_ok=True)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Baymax Jetson Backend",
    description="LLM-driven robot brain running on Jetson Orin Nano.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── WebSocket connection manager ──────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        for ws in list(self._clients):
            try:
                await ws.send_text(msg)
            except Exception:
                if ws in self._clients:
                    self._clients.remove(ws)


ws_manager = _ConnectionManager()


async def _broadcast_state():
    """Push current state to all connected WebSocket clients."""
    mode = current_state["mode"]
    await ws_manager.broadcast({
        "type": "state_update",
        "state": {
            "mode": mode,
            "mode_description": get_mode_description(mode),
            "mode_config": get_mode_config(mode),
            "last_voice_command": current_state["last_voice_command"],
            "last_robot_action": current_state["last_robot_action"],
            "pi_connected": current_state["pi_connected"],
            "eye_state": current_state["eye_state"],
            "led_state": current_state["led_state"],
        },
    })




def _normalize_transcript(text: str) -> str:
    """Clean Whisper output — strip punctuation."""
    text = text.lower().strip()
    for ch in ".,!?":
        text = text.replace(ch, "")
    text = " ".join(text.split())
    return text


def _capture_camera_frames(duration_seconds: int = 5, fps: int = 2) -> dict:
    """
    Capture frames from the Raspberry Pi camera stream and save them locally.

    Saves into:
      jetson_backend/captures/capture_YYYYMMDD_HHMMSS/
    """
    try:
        import cv2
    except ImportError:
        return {
            "success": False,
            "message": "OpenCV is not installed. Run: python3 -m pip install opencv-python",
            "saved_files": [],
        }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = CAPTURE_DIR / f"capture_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(PI_STREAM_URL)

    if not cap.isOpened():
        return {
            "success": False,
            "message": f"Could not open Pi camera stream: {PI_STREAM_URL}",
            "saved_files": [],
        }

    saved_files = []
    interval = 1.0 / max(fps, 1)
    end_time = time.time() + duration_seconds
    next_save_time = 0
    frame_index = 0

    try:
        while time.time() < end_time:
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            now = time.time()

            if now >= next_save_time:
                frame_path = output_dir / f"frame_{frame_index:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_files.append(str(frame_path))
                frame_index += 1
                next_save_time = now + interval

    finally:
        cap.release()

    return {
        "success": len(saved_files) > 0,
        "message": f"Saved {len(saved_files)} frame(s).",
        "stream_url": PI_STREAM_URL,
        "capture_dir": str(output_dir),
        "saved_files": saved_files,
    }


MODE_ACTIONS = {
    "kids": {
        "led": "red",
        "motor": "excited_wiggle",
    },
    "young_adult": {
        "led": "green",
        "motor": "nod_yes",
    },
    "adult": {
        "led": "yellow",
        "motor": "lean_right",
    },
}

def _mode_led_and_motor(mode: str) -> tuple[str, str]:
    """Map Baymax mode to LED color and motor action."""

    mode_led_map = {
        "kids": "red",
        "young_adult": "green",
        "adult": "yellow",
    }

    mode_motor_action_map = {
        "kids": "excited_wiggle",
        "young_adult": "nod_yes",
        "adult": "lean_right",
    }

    return (
        mode_led_map.get(mode, "white"),
        mode_motor_action_map.get(mode, "stop"),
    )

async def _direct_mode_switch(mode: str) -> dict:
    """
    Direct voice mode switching:
    - updates backend state
    - pushes LED/motor to Pi
    - avoids enum mismatch crash
    """

    if mode not in MODE_ACTIONS:
        return {
            "success": False,
            "message": f"Unknown mode: {mode}",
        }

    led_color = MODE_ACTIONS[mode]["led"]
    motor_action = MODE_ACTIONS[mode]["motor"]

    # backend state only
    current_state["mode"] = mode

    # Pi actions
    pi_led_result = push_led_color(led_color)
    pi_motor_result = push_action_to_pi(motor_action)

    current_state["last_robot_action"] = motor_action
    current_state["led_state"] = led_color

    current_state["pi_connected"] = (
        pi_led_result.get("success", False)
        or pi_motor_result.get("success", False)
    )

    await _broadcast_state()

    return {
        "success": True,
        "mode": mode,
        "led_color": led_color,
        "motor_action": motor_action,
        "raspberry_pi": {
            "led": pi_led_result,
            "motor": pi_motor_result,
        },
    }
    

# async def _apply_mode_to_backend_and_pi(mode: str) -> dict:
#     """
#     Apply mode locally on Jetson and push mode, LED, and motor action to Pi.
#     """
#     led_color, motor_action = _mode_led_and_motor(mode)

#     backend_result = set_current_mode(mode)
#     pi_mode_result = push_mode_to_pi(mode)
#     pi_led_result = push_led_color(led_color)
#     pi_motor_result = push_action_to_pi(motor_action)

#     current_state["pi_connected"] = (
#         pi_mode_result.get("success", False)
#         or pi_led_result.get("success", False)
#         or pi_motor_result.get("success", False)
#     )

#     current_state["last_robot_action"] = motor_action

#     if pi_led_result.get("success", False):
#         current_state["led_state"] = led_color

#     await _broadcast_state()

#     return {
#         "mode": mode,
#         "led_color": led_color,
#         "motor_action": motor_action,
#         "backend": backend_result,
#         "raspberry_pi": {
#             "mode": pi_mode_result,
#             "led": pi_led_result,
#             "motor": pi_motor_result,
#         },
#     }


async def _handle_command_via_llm(command: str) -> dict:
    """
    Route any text command through the LLM.  The LLM calls registered tools
    (change_mode, move_robot, speak, set_led, …) as needed.
    """
    from app.llm_engine import chat

    current_state["last_voice_command"] = command
    result = await asyncio.get_event_loop().run_in_executor(None, chat, command)
    await _broadcast_state()

    return {
        "status": "success",
        "message": result["response"],
        "tool_calls": result["tool_calls"],
        "rounds": result["rounds"],
    }



def _generate_story_with_llama(user_request: str) -> dict:
    """
    Generate a short spoken story using local Ollama/LLaMA.

    Requires Ollama running locally:
      ollama serve

    Example model:
      ollama pull llama3.2:1b

    Environment variables:
      OLLAMA_BASE_URL=http://localhost:11434
      OLLAMA_MODEL=llama3.2:1b
    """
    try:
        import requests
    except ImportError:
        return {
            "success": False,
            "story": "",
            "error": "The requests package is not installed. Run: python3 -m pip install requests",
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        }

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    prompt = f"""
You are Baymax, a kind desk companion robot.

The user asked:
"{user_request}"

Write a warm, simple, spoken story based on the user's request.

Constraints:
- 120 to 180 words maximum
- Friendly and calm
- Suitable for children and general audiences
- No scary or violent content
- Speak naturally as Baymax
- Do not include stage directions
- Only return the story text
"""

    try:
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 240,
                },
            },
            timeout=120,
        )

        if response.status_code != 200:
            return {
                "success": False,
                "story": "",
                "error": f"Ollama returned HTTP {response.status_code}: {response.text}",
                "model": ollama_model,
            }

        data = response.json()
        story = (data.get("response") or "").strip()

        return {
            "success": bool(story),
            "story": story,
            "model": ollama_model,
        }

    except Exception as exc:
        return {
            "success": False,
            "story": "",
            "error": str(exc),
            "model": ollama_model,
        }


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    from app.llm_engine import register_builtin_tools
    from app.audio_manager import _warm_cache

    register_builtin_tools()

    # Pre-generate TTS for common phrases so first interactions are instant
    await asyncio.get_event_loop().run_in_executor(None, _warm_cache)

    logger.info("Baymax Jetson backend started — tools registered, TTS cache warmed")
    logger.info("PI_BASE_URL=%s", PI_BASE_URL)
    logger.info("PI_STREAM_URL=%s", PI_STREAM_URL)
    logger.info("AUDIO_INPUT_DEVICE=%s", AUDIO_INPUT_DEVICE or "default")
    logger.info("AUDIO_OUTPUT_DEVICE=%s", AUDIO_OUTPUT_DEVICE or "default")


# ── Static / root ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index = STATIC_DIR / "index.html"

    if index.exists():
        return FileResponse(index)

    return {
        "status": "running",
        "message": "Baymax Jetson backend is active.",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": os.environ.get("WHISPER_MODEL", "tiny"),
        "stt_engine": "local_whisper",
        "pi_base_url": PI_BASE_URL,
        "pi_stream_url": PI_STREAM_URL,
    }


# ── State / Mode ──────────────────────────────────────────────────────────────

@app.get("/state")
def get_state():
    mode = get_current_mode()

    return {
        "status": "success",
        "state": {
            "mode": mode,
            "mode_description": get_mode_description(mode),
            "mode_config": get_mode_config(mode),
            "last_voice_command": current_state["last_voice_command"],
            "last_robot_action": current_state["last_robot_action"],
            "pi_connected": current_state["pi_connected"],
            "eye_state": current_state["eye_state"],
            "led_state": current_state["led_state"],
        },
    }


@app.get("/mode")
def get_mode():
    mode = get_current_mode()

    return {
        "status": "success",
        "mode": mode,
        "description": get_mode_description(mode),
        "config": get_mode_config(mode),
    }


@app.post("/mode")
async def update_mode(request: ModeRequest):
    """
    Manual mode update endpoint.
    This also applies LED and motor behavior for consistency.
    """
    mode = request.mode
    result = await _apply_mode_to_backend_and_pi(mode)

    return {
        "status": "success",
        "message": (
            f"Mode changed to {mode}, LED set to {result['led_color']}, "
            f"motor action {result['motor_action']} triggered."
        ),
        **result,
    }

# ── Facial Expression Recognition (FER) ─────────────────────────────────────────────

@app.get("/fer/frame")
def fer_from_camera_frame():
	return run_fer_once()

@app.get("/camera/live")
def camera_live():
    """
    Returns Raspberry Pi live camera stream URL.
    """

    return {
        "stream_url": PI_STREAM_URL
    }

@app.post("/fer/speak")
async def fer_speak():
    """
    Run FER once and speak detected emotion aloud.
    """

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        run_fer_once,
    )

    if not result.get("success", False):
        return {
            "status": "error",
            "message": "Emotion detection failed.",
            "result": result,
        }

    emotion = result.get("emotion", "unknown")
    confidence = result.get("confidence", 0)

    emotion_phrases = {
        "happy": "You seem happy today.",
        "sad": "You seem a little sad.",
        "angry": "You seem frustrated.",
        "surprised": "You look surprised.",
        "neutral": "You seem calm.",
        "fear": "You seem worried.",
        "disgust": "You seem uncomfortable.",
    }

    spoken_text = emotion_phrases.get(
        emotion,
        f"I think you are feeling {emotion}",
    )

    try:
        from app.audio_manager import speak as _speak

        spoke = await asyncio.get_event_loop().run_in_executor(
            None,
            _speak,
            spoken_text,
        )

    except Exception as exc:
        logger.exception("FER speech failed")

        return {
            "status": "partial_success",
            "emotion": emotion,
            "confidence": confidence,
            "spoken_text": spoken_text,
            "speak_error": str(exc),
            "result": result,
        }

    return {
        "status": "success",
        "emotion": emotion,
        "confidence": confidence,
        "spoken_text": spoken_text,
        "spoke": spoke,
        "result": result,
    }

# ── Voice command ─────────────────────────────────────────────────────────────

@app.post("/voice-command")
async def voice_command(request: VoiceCommandRequest):
    """Text-based command endpoint — routes directly through the LLM."""
    return await _handle_command_via_llm(request.command)


@app.post("/audio/voice-command")
async def audio_voice_command(duration_seconds: int = 15):
    """
    Full Jetson voice command endpoint:
      1. Record once from Jetson mic
      2. Transcribe speech
      3. Save transcript
      4. Run command router
      5. Switch mode / capture camera / tell story

    Usage:
      curl -s -X POST "http://localhost:8000/audio/voice-command?duration_seconds=15"
    """
    from app.audio_manager import record, transcribe, save_transcript_log, save_wav_recording

    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None,
        record,
        duration_seconds,
    )

    if not wav_bytes:
        result = {
            "text": "",
            "language": "unknown",
            "success": False,
            "error": "Recording failed — check Jetson mic.",
            "bytes_recorded": 0,
        }
        saved_log = save_transcript_log(
            endpoint="audio_voice_command",
            duration_seconds=duration_seconds,
            result=result,
        )
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "Recording failed — check Jetson mic.",
            "duration_seconds": duration_seconds,
            "saved_log": saved_log,
        })

    wav_path = save_wav_recording(wav_bytes, endpoint="audio_voice_command")

    transcript_result = await asyncio.get_event_loop().run_in_executor(
        None,
        transcribe,
        wav_bytes,
    )

    transcript = transcript_result.get("text", "").strip()

    saved_log = save_transcript_log(
        endpoint="audio_voice_command",
        duration_seconds=duration_seconds,
        result={
            **transcript_result,
            "text": transcript,
            "bytes_recorded": len(wav_bytes),
            "wav_path": wav_path,
        },
    )

    if not transcript_result.get("success"):
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "Transcription failed.",
            "duration_seconds": duration_seconds,
            "saved_log": saved_log,
            "transcription": transcript_result,
        })

    if not transcript:
        return {
            "status": "error",
            "message": "No speech detected.",
            "duration_seconds": duration_seconds,
            "saved_log": saved_log,
            "transcription": transcript_result,
        }

    command_result = await _handle_command_via_llm(transcript)

    return {
        "status": command_result["status"],
        "duration_seconds": duration_seconds,
        "transcript": transcript,
        "wav_path": saved_log.get("wav_path"),
        "saved_log": saved_log,
        "transcription": transcript_result,
        "command_result": command_result,
    }


# ── Free voice chat (Talk to Baymax) ─────────────────────────────────────────

@app.post("/audio/chat")
async def audio_chat():
    """
    Free-form voice interaction endpoint.

    Flow:
      1. Record from Jetson mic using VAD — stops ~1.5 s after speech ends
         (hard cap: 30 s).
      2. Transcribe with local Whisper.
      3. Send transcript to LLM; LLM calls speak() and other tools autonomously.
      4. Broadcast updated state via WebSocket.
      5. Return {transcript, response, tool_calls, rounds}.
    """
    from app.audio_manager import (
        record_until_silence,
        transcribe,
        save_wav_recording,
        save_transcript_log,
    )
    from app.llm_engine import chat

    # 1. VAD recording
    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None,
        record_until_silence,
    )

    if not wav_bytes:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "Recording failed — check Jetson mic.",
        })

    wav_path = save_wav_recording(wav_bytes, endpoint="audio_chat")

    # 2. Transcribe
    transcript_result = await asyncio.get_event_loop().run_in_executor(
        None,
        transcribe,
        wav_bytes,
    )

    transcript = transcript_result.get("text", "").strip()

    saved_log = save_transcript_log(
        endpoint="audio_chat",
        duration_seconds=0,
        result={
            **transcript_result,
            "text": transcript,
            "bytes_recorded": len(wav_bytes),
            "wav_path": wav_path,
        },
    )

    if not transcript_result.get("success") or not transcript:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "No speech detected or transcription failed.",
            "saved_log": saved_log,
            "transcription": transcript_result,
        })

    current_state["last_voice_command"] = transcript

    # 3. LLM
    try:
        llm_result = await asyncio.get_event_loop().run_in_executor(None, chat, transcript)
    except Exception as exc:
        logger.exception("LLM chat failed in /audio/chat")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"LLM error: {exc}",
            "transcript": transcript,
        })

    await _broadcast_state()

    return {
        "status": "success",
        "transcript": transcript,
        "response": llm_result["response"],
        "tool_calls": llm_result["tool_calls"],
        "rounds": llm_result["rounds"],
        "wav_path": wav_path,
        "saved_log": saved_log,
    }


# ── LLM Chat ──────────────────────────────────────────────────────────────────

@app.post("/llm/chat")
async def llm_chat(request: LLMChatRequest):
    from app.llm_engine import chat

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        chat,
        request.message,
    )

    await _broadcast_state()

    return {
        "status": "success",
        "response": result["response"],
        "tool_calls": result["tool_calls"],
        "rounds": result["rounds"],
    }


@app.post("/llm/reset")
async def llm_reset():
    from app.llm_engine import reset_conversation

    reset_conversation()

    return {
        "status": "success",
        "message": "Conversation history cleared.",
    }


# ── Robot actions ─────────────────────────────────────────────────────────────

@app.get("/robot/behavior")
def get_robot_behavior():
    mode = get_current_mode()
    behavior = get_robot_behavior_for_mode(mode)

    return {
        "status": "success",
        "mode": mode,
        "behavior": behavior,
    }


@app.post("/robot/action")
async def robot_action(request: RobotActionRequest):
    validation = validate_robot_action(request.action)

    if not validation["allowed"]:
        return {
            "status": "blocked",
            "message": validation["message"],
            "data": validation,
        }

    pi_result = push_action_to_pi(request.action)

    current_state["pi_connected"] = pi_result["success"]
    current_state["last_robot_action"] = request.action

    await _broadcast_state()

    return {
        "status": "success" if pi_result["success"] else "partial_failure",
        "message": validation["message"],
        "validation": validation,
        "raspberry_pi": pi_result,
    }


# ── Eye control ───────────────────────────────────────────────────────────────

@app.post("/eyes")
async def set_eyes(request: EyeRequest):
    result = push_eye_expression(request.expression)

    if result["success"]:
        current_state["eye_state"] = request.expression

    current_state["pi_connected"] = result["success"]

    await _broadcast_state()

    return {
        "status": "success" if result["success"] else "error",
        **result,
    }


# ── LED control ───────────────────────────────────────────────────────────────

@app.post("/led")
async def set_led_endpoint(request: LedRequest):
    result = push_led_color(request.color)

    if result["success"]:
        current_state["led_state"] = request.color

    current_state["pi_connected"] = result["success"]

    await _broadcast_state()

    return {
        "status": "success" if result["success"] else "error",
        **result,
    }


# ── Vision ────────────────────────────────────────────────────────────────────

@app.get("/vision/analyze")
def vision_analyze(expected: str = "right"):
    from app.vision_processor import analyze_hand_raise

    result = analyze_hand_raise(expected)

    return {
        "status": "success",
        **result,
    }


@app.get("/vision/describe")
def vision_describe():
    from app.vision_processor import describe_scene

    result = describe_scene()

    return {
        "status": "success",
        **result,
    }


@app.post("/vision/capture-test")
async def vision_capture_test(duration_seconds: int = 5, fps: int = 2):
    """
    Direct camera capture test without voice.
    Saves frames from the Pi stream into jetson_backend/captures/.
    """
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        _capture_camera_frames,
        duration_seconds,
        fps,
    )

    return {
        "status": "success" if result["success"] else "error",
        **result,
    }


# ── Audio ─────────────────────────────────────────────────────────────────────

@app.get("/audio/status")
def audio_status():
    """Report Jetson USB audio device configuration."""
    from app.audio_manager import audio_status as _audio_status

    return {
        "status": "success",
        **_audio_status(),
        "message": "Jetson-local audio endpoints are active.",
    }


@app.get("/audio/devices")
def audio_devices():
    """
    List ALSA recording/playback devices from Jetson.
    """
    from app.audio_manager import list_audio_devices

    return {
        "status": "success",
        **list_audio_devices(),
    }


@app.get("/audio/transcripts")
def audio_transcripts(limit: int = 20):
    """
    Return recent saved transcription logs.
    """
    from app.audio_manager import read_transcript_logs

    return {
        "status": "success",
        "items": read_transcript_logs(limit=limit),
    }


@app.post("/audio/record")
async def audio_record(duration_seconds: int = 15):
    """
    Record once from the Jetson mic and return metadata.

    Usage:
      curl -s -X POST "http://localhost:8000/audio/record?duration_seconds=15"
    """
    from app.audio_manager import record, save_wav_recording

    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None,
        record,
        duration_seconds,
    )

    if not wav_bytes:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "success": False,
            "message": "Recording failed.",
            "duration_seconds": duration_seconds,
        })

    wav_path = save_wav_recording(wav_bytes, endpoint="audio_record")

    return {
        "status": "success",
        "success": True,
        "duration_seconds": duration_seconds,
        "bytes_recorded": len(wav_bytes),
        "wav_path": wav_path,
    }


@app.post("/audio/listen")
async def audio_listen(duration_seconds: int = 15):
    """
    Record once from the Jetson mic, transcribe, save transcript, and return text.
    """
    from app.audio_manager import listen_and_transcribe, save_transcript_log

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        listen_and_transcribe,
        duration_seconds,
        "audio_listen",
        True,
    )

    saved_log = save_transcript_log(
        endpoint="audio_listen",
        duration_seconds=duration_seconds,
        result=result,
    )

    return {
        "status": "success" if result.get("success") else "error",
        "saved_log": saved_log,
        **result,
    }


@app.post("/audio/play")
async def audio_play(file: UploadFile = File(...)):
    """Upload a WAV file and play it through the Jetson USB speaker."""
    from app.audio_manager import play

    wav_bytes = await file.read()

    ok = await asyncio.get_event_loop().run_in_executor(
        None,
        play,
        wav_bytes,
    )

    if not ok:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "success": False,
            "message": "Audio playback failed.",
            "filename": file.filename,
        })

    return {
        "status": "success",
        "success": True,
        "filename": file.filename,
        "bytes_played": len(wav_bytes),
    }


@app.post("/audio/speak")
async def audio_speak(request: LLMChatRequest):
    """Synthesize text and play through Jetson USB speaker."""
    from app.audio_manager import speak as _speak

    success = await asyncio.get_event_loop().run_in_executor(
        None,
        _speak,
        request.message,
    )

    return {
        "status": "success" if success else "error",
        "success": success,
        "text": request.message,
    }


@app.post("/audio/transcribe")
async def audio_transcribe(duration_seconds: int = 15):
    """
    Record once from Jetson USB mic, transcribe, save transcript, and return text.

    Usage:
      curl -s -X POST "http://localhost:8000/audio/transcribe?duration_seconds=15"
    """
    from app.audio_manager import listen_and_transcribe, save_transcript_log

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        listen_and_transcribe,
        duration_seconds,
        "audio_transcribe",
        True,
    )

    saved_log = save_transcript_log(
        endpoint="audio_transcribe",
        duration_seconds=duration_seconds,
        result=result,
    )

    return {
        "status": "success" if result.get("success") else "error",
        "saved_log": saved_log,
        **result,
    }


@app.post("/audio/story")
async def audio_story(duration_seconds: int = 15):
    """
    Storytelling endpoint.

    Flow:
      1. Record user's story request from Jetson mic
      2. Transcribe with local Whisper
      3. Save user request transcript + WAV path
      4. Generate story with local Ollama/LLaMA
      5. Speak story through Jetson speaker
      6. Save generated story log

    Usage:
      curl -s -X POST "http://localhost:8000/audio/story?duration_seconds=15" | python3 -m json.tool
    """
    from app.audio_manager import (
        listen_and_transcribe,
        save_transcript_log,
        speak as _speak,
    )

    # 1. Record + transcribe user story request
    transcription = await asyncio.get_event_loop().run_in_executor(
        None,
        listen_and_transcribe,
        duration_seconds,
        "audio_story",
        True,
    )

    request_log = save_transcript_log(
        endpoint="audio_story_request",
        duration_seconds=duration_seconds,
        result=transcription,
    )

    if not transcription.get("success"):
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "Could not transcribe story request.",
            "duration_seconds": duration_seconds,
            "transcription": transcription,
            "request_log": request_log,
        })

    user_request = transcription.get("text", "").strip()

    if not user_request:
        return {
            "status": "error",
            "message": "No story request detected.",
            "duration_seconds": duration_seconds,
            "transcription": transcription,
            "request_log": request_log,
        }

    # 2. Generate story with local LLaMA/Ollama
    story_result = await asyncio.get_event_loop().run_in_executor(
        None,
        _generate_story_with_llama,
        user_request,
    )

    if not story_result.get("success"):
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": "Story generation failed.",
            "duration_seconds": duration_seconds,
            "user_request": user_request,
            "transcription": transcription,
            "request_log": request_log,
            "story_result": story_result,
        })

    story = story_result["story"]

    # 3. Speak generated story using existing TTS path
    spoke = await asyncio.get_event_loop().run_in_executor(
        None,
        _speak,
        story,
    )

    # 4. Save generated story log
    story_log = save_transcript_log(
        endpoint="audio_story_generated",
        duration_seconds=duration_seconds,
        result={
            "success": True,
            "text": user_request,
            "language": transcription.get("language", "unknown"),
            "bytes_recorded": transcription.get("bytes_recorded"),
            "wav_path": transcription.get("wav_path"),
            "engine": "ollama",
            "model": story_result.get("model"),
        },
        extra={
            "generated_story": story,
            "spoke": spoke,
        },
    )

    await _broadcast_state()

    return {
        "status": "success",
        "duration_seconds": duration_seconds,
        "user_request": user_request,
        "story": story,
        "spoke": spoke,
        "story_model": story_result.get("model"),
        "wav_path": transcription.get("wav_path"),
        "transcription": transcription,
        "request_log": request_log,
        "story_log": story_log,
    }


# ── Pi status ─────────────────────────────────────────────────────────────────

@app.get("/pi/status")
def pi_status():
    result = get_pi_status()

    current_state["pi_connected"] = result["success"]

    return {
        "status": "success" if result["success"] else "error",
        **result,
    }


# ── WebSocket push ────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    await _broadcast_state()

    try:
        while True:
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "chat" and msg.get("message"):
                from app.llm_engine import chat

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    chat,
                    msg["message"],
                )

                await websocket.send_text(json.dumps({
                    "type": "chat_response",
                    "response": result["response"],
                    "tool_calls": result["tool_calls"],
                }))

                await _broadcast_state()

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
