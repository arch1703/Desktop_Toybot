"""
audio_manager.py — Jetson audio: local hardware I/O + local Whisper STT + optional Gemini TTS.

Hardware pipeline:
  arecord → WAV bytes → local Whisper STT → text
  text → Gemini TTS → WAV bytes → aplay

No API key is needed for transcription when USE_LOCAL_WHISPER=true.

Configure via environment variables:
  AUDIO_INPUT_DEVICE     — ALSA device for recording (default: plughw:2,0)
  AUDIO_OUTPUT_DEVICE    — ALSA device for playback (default: system default)
  AUDIO_RECORD_SECONDS   — default recording duration (default: 15)
  AUDIO_SAMPLE_RATE      — sample rate for arecord (default: 44100)
  AUDIO_CHANNELS         — channel count for arecord (default: 2)

  USE_LOCAL_WHISPER      — true/false, default true
  WHISPER_MODEL          — tiny, base, small, medium, large, default tiny
  WHISPER_LANGUAGE       — language code, default en
  WHISPER_FP16           — true/false, default false for Jetson CPU-safe mode

  GEMINI_API_KEY         — only needed for Gemini TTS, not for Whisper STT
  GEMINI_TTS_MODEL       — model for synthesis (default: gemini-2.5-flash-preview-tts)
  TTS_VOICE              — Gemini TTS voice name (default: Kore)

Transcript logging:
  transcripts/transcripts.jsonl

WAV recording storage:
  recordings/*.wav
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

TRANSCRIPT_DIR = BASE_DIR / "transcripts"
TRANSCRIPT_LOG_PATH = TRANSCRIPT_DIR / "transcripts.jsonl"
TRANSCRIPT_DIR.mkdir(exist_ok=True)

RECORDING_DIR = BASE_DIR / "recordings"
RECORDING_DIR.mkdir(exist_ok=True)

# ── Hardware config ───────────────────────────────────────────────────────────

AUDIO_INPUT_DEVICE  = os.environ.get("AUDIO_INPUT_DEVICE", "plughw:2,0")
AUDIO_OUTPUT_DEVICE = os.environ.get("AUDIO_OUTPUT_DEVICE", "")

DEFAULT_RECORD_SECONDS = int(os.environ.get("AUDIO_RECORD_SECONDS", "15"))
DEFAULT_SAMPLE_RATE    = int(os.environ.get("AUDIO_SAMPLE_RATE", "44100"))
DEFAULT_CHANNELS       = int(os.environ.get("AUDIO_CHANNELS", "2"))

# ── Local Whisper config ──────────────────────────────────────────────────────

USE_LOCAL_WHISPER = os.environ.get("USE_LOCAL_WHISPER", "true").lower() == "true"
WHISPER_MODEL     = os.environ.get("WHISPER_MODEL", "tiny")
WHISPER_LANGUAGE  = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_FP16      = os.environ.get("WHISPER_FP16", "false").lower() == "true"

_whisper_model = None

# ── Optional Gemini TTS config ─────────────────────────────────────────────────
# Gemini is no longer used for transcription in this file.
# It is only used by /audio/speak if you keep Gemini TTS enabled.

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TTS_MODEL      = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-pro-preview-tts")
TTS_VOICE      = os.environ.get("TTS_VOICE", "Kore")


def _get_gemini_client():
    """
    Lazy import Gemini only when TTS is used.
    This avoids crashing audio transcription if google-genai is unavailable.
    """
    if not GEMINI_API_KEY:
        return None

    try:
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        logger.exception("Failed to initialize Gemini client")
        return None


def _safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _run_cmd(cmd: list[str], timeout: int | None = None) -> dict:
    """
    Run a command safely and return stdout/stderr/returncode.
    Used for diagnostics such as arecord -l and aplay -l.
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "cmd": " ".join(cmd),
        }
    except Exception as exc:
        logger.exception("Command failed: %s", " ".join(cmd))
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "cmd": " ".join(cmd),
        }


def _pcm_to_wav(
    pcm: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _get_whisper_model():
    """
    Load local Whisper once and reuse it across requests.
    This avoids reloading the model every time /audio/transcribe is called.
    """
    global _whisper_model

    if _whisper_model is None:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Install it with: "
                "python3 -m pip install -U openai-whisper"
            ) from exc

        logger.info("Loading local Whisper model: %s", WHISPER_MODEL)
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        logger.info("Local Whisper model loaded: %s", WHISPER_MODEL)

    return _whisper_model


def audio_status() -> dict:
    """
    Return the current audio configuration used by this module.
    This does not prove hardware works; it reports config only.
    """
    return {
        "input_device": AUDIO_INPUT_DEVICE,
        "output_device": AUDIO_OUTPUT_DEVICE or "default",
        "record_seconds_default": DEFAULT_RECORD_SECONDS,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "channels": DEFAULT_CHANNELS,
        "stt_engine": "local_whisper" if USE_LOCAL_WHISPER else "disabled",
        "use_local_whisper": USE_LOCAL_WHISPER,
        "whisper_model": WHISPER_MODEL,
        "whisper_language": WHISPER_LANGUAGE,
        "whisper_fp16": WHISPER_FP16,
        "gemini_api_key_loaded_for_tts": bool(GEMINI_API_KEY),
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
        "transcript_log_path": str(TRANSCRIPT_LOG_PATH),
        "recording_dir": str(RECORDING_DIR),
    }


def list_audio_devices() -> dict:
    """
    List ALSA capture and playback devices visible to Jetson.
    """
    return {
        "recording_devices": _run_cmd(["arecord", "-l"], timeout=5),
        "playback_devices": _run_cmd(["aplay", "-l"], timeout=5),
        "current_config": audio_status(),
    }


# ── WAV recording storage ─────────────────────────────────────────────────────

def save_wav_recording(
    wav_bytes: bytes,
    endpoint: str = "audio",
) -> str | None:
    """
    Save recorded WAV bytes to recordings/.

    Returns:
      Absolute WAV file path as a string, or None on failure.
    """
    if not wav_bytes:
        return None

    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_endpoint = endpoint.replace("/", "_").replace(" ", "_")
        wav_path = RECORDING_DIR / f"{safe_endpoint}_{timestamp}.wav"

        with open(wav_path, "wb") as f:
            f.write(wav_bytes)

        logger.info("Saved WAV recording: %s", wav_path)
        return str(wav_path)

    except Exception:
        logger.exception("Failed to save WAV recording")
        return None


# ── Transcript logging ────────────────────────────────────────────────────────

def save_transcript_log(
    endpoint: str,
    duration_seconds: int,
    result: dict,
    extra: dict | None = None,
) -> dict:
    """
    Append one transcription result to transcripts/transcripts.jsonl.

    Returns the saved log entry.
    """
    entry = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "unix_time": time.time(),
        "endpoint": endpoint,
        "duration_seconds": duration_seconds,
        "success": bool(result.get("success")),
        "text": result.get("text", ""),
        "language": result.get("language", "unknown"),
        "error": result.get("error"),
        "bytes_recorded": result.get("bytes_recorded"),
        "wav_path": result.get("wav_path"),
        "engine": result.get("engine"),
        "model": result.get("model"),
    }

    if extra:
        entry.update(extra)

    with open(TRANSCRIPT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def read_transcript_logs(limit: int = 20) -> list[dict]:
    """
    Read the most recent transcript log entries.
    """
    if limit <= 0:
        limit = 20

    if not TRANSCRIPT_LOG_PATH.exists():
        return []

    try:
        lines = TRANSCRIPT_LOG_PATH.read_text(encoding="utf-8").splitlines()
        recent_lines = lines[-limit:]
        items = []

        for line in recent_lines:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return items

    except Exception:
        logger.exception("Failed to read transcript logs")
        return []


# ── Hardware recording ────────────────────────────────────────────────────────

def record_audio(
    duration_seconds: int = DEFAULT_RECORD_SECONDS,
    device: str = AUDIO_INPUT_DEVICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> bytes | None:
    """
    Record audio from the Jetson-connected mic using arecord.

    Returns WAV bytes, or None on failure.

    Physical behavior:
      The endpoint starts recording only when called.
      It records once for duration_seconds, then returns.
    """
    if duration_seconds <= 0:
        duration_seconds = DEFAULT_RECORD_SECONDS

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "arecord",
            "-D", device,
            "-f", "S16_LE",
            "-r", str(sample_rate),
            "-c", str(channels),
            "-t", "wav",
            "-d", str(duration_seconds),
            tmp_path,
        ]

        logger.info("Recording audio: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=duration_seconds + 5,
        )

        if result.returncode != 0:
            logger.error("arecord failed: %s", result.stderr)
            return None

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        logger.info("Recorded %d bytes of WAV audio", len(wav_bytes))
        return wav_bytes

    except subprocess.TimeoutExpired:
        logger.exception("arecord timed out")
        return None
    except Exception:
        logger.exception("Audio recording failed")
        return None
    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


# ── VAD recording ────────────────────────────────────────────────────────────

def record_until_silence(
    max_seconds: int = 30,
    silence_duration: float = 1.5,
    rms_threshold: int = 500,
    device: str = AUDIO_INPUT_DEVICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> bytes | None:
    """
    Record from the Jetson mic and stop automatically after ``silence_duration``
    seconds of continuous silence (RMS below ``rms_threshold``).

    Records in 0.5-second PCM chunks.  Stops when the silence cap is reached or
    ``max_seconds`` elapses, whichever comes first.  Returns assembled WAV bytes,
    or None on failure.
    """
    import math
    import struct

    chunk_seconds = 0.5
    bytes_per_sample = 2  # S16_LE
    chunk_frames = int(sample_rate * chunk_seconds)
    chunk_bytes  = chunk_frames * channels * bytes_per_sample

    silent_chunks_needed = math.ceil(silence_duration / chunk_seconds)  # = 3 for 1.5 s
    max_chunks = int(max_seconds / chunk_seconds)

    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", str(channels),
        "-t", "raw",           # raw PCM to stdout — no WAV header
    ]

    logger.info("VAD recording started (max=%ds silence=%.1fs rms_thresh=%d)", max_seconds, silence_duration, rms_threshold)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        logger.exception("Failed to launch arecord for VAD recording")
        return None

    pcm_chunks: list[bytes] = []
    silent_run = 0

    try:
        for _ in range(max_chunks):
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break

            pcm_chunks.append(chunk)

            # Compute RMS from 16-bit little-endian samples
            num_samples = len(chunk) // bytes_per_sample
            if num_samples == 0:
                silent_run += 1
                continue

            samples = struct.unpack_from(f"<{num_samples}h", chunk)
            rms = math.sqrt(sum(s * s for s in samples) / num_samples)

            if rms < rms_threshold:
                silent_run += 1
                logger.debug("Silent chunk #%d (rms=%.0f)", silent_run, rms)
                if silent_run >= silent_chunks_needed:
                    logger.info("VAD: silence detected after %.1f s — stopping", silence_duration)
                    break
            else:
                silent_run = 0

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not pcm_chunks:
        logger.warning("VAD recording produced no audio")
        return None

    pcm_all = b"".join(pcm_chunks)
    wav = _pcm_to_wav(pcm_all, sample_rate=sample_rate, channels=channels)
    logger.info("VAD recording complete: %d chunks, %d PCM bytes → %d WAV bytes", len(pcm_chunks), len(pcm_all), len(wav))
    return wav


# ── Hardware playback ─────────────────────────────────────────────────────────

def play_audio(wav_bytes: bytes, device: str = AUDIO_OUTPUT_DEVICE) -> bool:
    """
    Play WAV audio bytes through the Jetson speaker using aplay.
    Returns True if playback succeeded, False otherwise.
    """
    if not wav_bytes:
        logger.warning("No audio bytes provided to play_audio()")
        return False

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        cmd = ["aplay"]
        if device:
            cmd.extend(["-D", device])
        cmd.append(tmp_path)

        logger.info("Playing audio: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error("aplay failed: %s", result.stderr)
            return False

        logger.info("Audio playback completed")
        return True

    except Exception:
        logger.exception("Audio playback failed")
        return False
    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


def play_audio_file(path: str | Path, device: str = AUDIO_OUTPUT_DEVICE) -> bool:
    """Play an existing WAV file through the Jetson speaker."""
    path = Path(path)

    if not path.exists():
        logger.error("Audio file does not exist: %s", path)
        return False

    cmd = ["aplay"]

    if device:
        cmd.extend(["-D", device])

    cmd.append(str(path))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.error("aplay failed: %s", result.stderr)
            return False

        return True

    except Exception:
        logger.exception("Audio file playback failed")
        return False


# ── STT: Local Whisper ────────────────────────────────────────────────────────

def transcribe(wav_bytes: bytes) -> dict:
    """
    Transcribe WAV audio bytes locally using Whisper.

    No Gemini/OpenAI API key is needed.
    """
    if not wav_bytes:
        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "engine": "local_whisper",
            "model": WHISPER_MODEL,
            "error": "No WAV bytes provided",
        }

    if not USE_LOCAL_WHISPER:
        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "engine": "disabled",
            "model": None,
            "error": "USE_LOCAL_WHISPER is false",
        }

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        model = _get_whisper_model()

        logger.info(
            "Running local Whisper transcription: model=%s language=%s fp16=%s",
            WHISPER_MODEL,
            WHISPER_LANGUAGE,
            WHISPER_FP16,
        )

        result = model.transcribe(
            tmp_path,
            language=WHISPER_LANGUAGE,
            fp16=WHISPER_FP16,
        )

        text = (result.get("text") or "").strip()
        language = result.get("language") or WHISPER_LANGUAGE or "unknown"

        logger.info("Local Whisper transcribed: %s", text)

        return {
            "text": text,
            "language": language,
            "success": True,
            "engine": "local_whisper",
            "model": WHISPER_MODEL,
        }

    except Exception as exc:
        logger.exception("Local Whisper transcription failed")
        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "engine": "local_whisper",
            "model": WHISPER_MODEL,
            "error": str(exc),
        }

    finally:
        if tmp_path:
            _safe_unlink(tmp_path)


def listen_and_transcribe(
    duration_seconds: int = DEFAULT_RECORD_SECONDS,
    endpoint: str = "audio_transcribe",
    save_wav: bool = True,
) -> dict:
    """
    Record once from Jetson mic, optionally save the WAV, then transcribe locally.
    """
    wav_bytes = record_audio(duration_seconds=duration_seconds)

    if wav_bytes is None:
        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "duration_seconds": duration_seconds,
            "bytes_recorded": 0,
            "wav_path": None,
            "engine": "local_whisper",
            "model": WHISPER_MODEL,
            "error": "Recording failed",
        }

    wav_path = save_wav_recording(wav_bytes, endpoint=endpoint) if save_wav else None

    result = transcribe(wav_bytes)
    result["duration_seconds"] = duration_seconds
    result["bytes_recorded"] = len(wav_bytes)
    result["wav_path"] = wav_path

    return result


# ── TTS: Optional Gemini ──────────────────────────────────────────────────────

def synthesize(text: str) -> bytes | None:
    """
    Synthesize text to WAV audio bytes using Gemini TTS.

    This still requires GEMINI_API_KEY.
    Transcription does not require GEMINI_API_KEY.
    """
    if not text or not text.strip():
        logger.warning("No text provided to synthesize()")
        return None

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set. /audio/speak needs Gemini TTS or another local TTS.")
        return None

    try:
        from google.genai import types

        client = _get_gemini_client()
        if client is None:
            return None

        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                    )
                ),
            ),
        )

        pcm = response.candidates[0].content.parts[0].inline_data.data
        wav = _pcm_to_wav(pcm)

        logger.info("Synthesized %d chars → %d bytes WAV", len(text), len(wav))
        return wav

    except Exception:
        logger.exception("TTS synthesis failed")
        return None


def speak_text(text: str) -> bool:
    """text → Gemini TTS WAV → Jetson speaker."""
    wav_bytes = synthesize(text)

    if wav_bytes is None:
        return False

    return play_audio(wav_bytes)


# ── Startup hook ──────────────────────────────────────────────────────────────

def _warm_cache() -> None:
    """
    Warm local Whisper model once at startup if configured.

    This makes the first /audio/transcribe request faster, but startup may take longer.
    If startup becomes too slow, comment out _get_whisper_model().
    """
    logger.info("Audio manager ready (arecord/aplay + local Whisper STT + optional Gemini TTS)")
    logger.info("Audio config: %s", audio_status())

    if USE_LOCAL_WHISPER:
        try:
            _get_whisper_model()
        except Exception:
            logger.exception("Failed to warm local Whisper model")


# ── Compatibility aliases ─────────────────────────────────────────────────────

record = record_audio
play   = play_audio
speak  = speak_text
