"""
audio_controller.py — USB combo mic/speaker via PyAudio on the Raspberry Pi.

record_audio(duration)  → WAV bytes
play_audio(wav_bytes)   → plays through USB speaker
play_text(text)         → TTS via espeak (no heavy library needed on Pi)

Falls back to a stub if PyAudio or espeak is unavailable.
"""

import io
import logging
import os
import subprocess
import tempfile
import wave

logger = logging.getLogger(__name__)

# ── Device auto-detection ─────────────────────────────────────────────────────
# PyAudio device index for the USB combo mic/speaker.
# Set USB_AUDIO_INDEX env var to override auto-detection.
_AUDIO_INDEX: int | None = None
_PA_AVAILABLE = False


def _find_usb_audio_index() -> int | None:
    """Scan PyAudio devices and return the index of the first USB audio device."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if "usb" in name and info.get("maxInputChannels", 0) > 0:
                logger.info("Found USB audio device at index %d: %s", i, info["name"])
                pa.terminate()
                return i
        pa.terminate()
    except Exception as exc:
        logger.warning("PyAudio device scan failed: %s", exc)
    return None


def _init_audio() -> None:
    global _AUDIO_INDEX, _PA_AVAILABLE
    env_idx = os.environ.get("USB_AUDIO_INDEX")
    if env_idx is not None:
        _AUDIO_INDEX = int(env_idx)
    else:
        _AUDIO_INDEX = _find_usb_audio_index()

    try:
        import pyaudio
        _PA_AVAILABLE = True
    except ImportError:
        logger.warning("PyAudio not installed — audio stub active")


_init_audio()

SAMPLE_RATE   = 16000
CHANNELS      = 1
SAMPLE_WIDTH  = 2   # 16-bit
CHUNK_SIZE    = 1024


def record_audio(duration: int = 5) -> bytes:
    """
    Record audio from the USB mic, stopping early when silence is detected
    (Voice Activity Detection via webrtcvad).

    Falls back to fixed-duration recording if webrtcvad is unavailable.
    Returns raw WAV bytes.
    """
    if not _PA_AVAILABLE:
        logger.info("[AUDIO stub] record_audio duration=%d", duration)
        return b""

    # VAD config: 10ms frames at 16kHz = 160 samples per frame
    VAD_FRAME_MS   = 10
    VAD_FRAME_SAMP = SAMPLE_RATE * VAD_FRAME_MS // 1000  # 160
    SILENCE_LIMIT  = 0.5   # seconds of silence before stopping
    MAX_SECONDS    = duration

    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)  # aggressiveness 0-3; 2 is balanced
        use_vad = True
    except ImportError:
        logger.warning("webrtcvad not installed — using fixed-duration recording")
        use_vad = False

    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=_AUDIO_INDEX,
            frames_per_buffer=VAD_FRAME_SAMP if use_vad else CHUNK_SIZE,
        )

        frames: list[bytes] = []
        silent_frames = 0
        silence_trigger = int(SILENCE_LIMIT * 1000 / VAD_FRAME_MS)  # frames of silence to stop
        max_frames = int(MAX_SECONDS * 1000 / VAD_FRAME_MS)

        if use_vad:
            for _ in range(max_frames):
                chunk = stream.read(VAD_FRAME_SAMP, exception_on_overflow=False)
                frames.append(chunk)
                is_speech = vad.is_speech(chunk, SAMPLE_RATE)
                if not is_speech:
                    silent_frames += 1
                    if silent_frames >= silence_trigger and len(frames) > silence_trigger:
                        logger.info("VAD: silence detected — stopping early (%d frames)", len(frames))
                        break
                else:
                    silent_frames = 0
        else:
            num_chunks = int(SAMPLE_RATE / CHUNK_SIZE * MAX_SECONDS)
            for _ in range(num_chunks):
                frames.append(stream.read(CHUNK_SIZE, exception_on_overflow=False))

        stream.stop_stream()
        stream.close()
    finally:
        pa.terminate()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    logger.info("Recorded %d frames → %d bytes WAV", len(frames), buf.tell())
    return buf.getvalue()


def play_audio(wav_bytes: bytes) -> dict:
    """Play WAV bytes through the USB speaker using aplay."""
    if not wav_bytes:
        return {"audio_played": False, "message": "No audio data"}

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        cmd = ["aplay"]
        if _AUDIO_INDEX is not None:
            cmd += ["-D", f"hw:{_AUDIO_INDEX},0"]
        cmd.append(tmp_path)

        subprocess.run(cmd, check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(tmp_path)
        logger.info("Played %d bytes WAV", len(wav_bytes))
        return {"audio_played": True}

    except Exception as exc:
        logger.warning("play_audio failed: %s", exc)
        return {"audio_played": False, "message": str(exc)}


def play_text(text: str) -> dict:
    """
    TTS via espeak — lightweight, works offline on Pi.
    Falls back to logging if espeak is not installed.
    """
    if not text.strip():
        return {"audio_played": False, "message": "Empty text"}

    try:
        cmd = ["espeak", "-s", "150", "-a", "180", text]
        if _AUDIO_INDEX is not None:
            # Route espeak output to USB device via aplay pipe
            espeak_proc = subprocess.Popen(
                ["espeak", "-s", "150", "--stdout", text],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            aplay_proc = subprocess.Popen(
                ["aplay", "-D", f"hw:{_AUDIO_INDEX},0", "-"],
                stdin=espeak_proc.stdout, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            espeak_proc.stdout.close()
            aplay_proc.wait(timeout=30)
        else:
            subprocess.run(cmd, check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.info("[AUDIO TTS] %s", text)
        return {"audio_played": True, "prompt": text}

    except FileNotFoundError:
        logger.warning("espeak not found — text: %s", text)
        return {"audio_played": False, "message": "espeak not installed", "prompt": text}
    except Exception as exc:
        logger.warning("play_text failed: %s", exc)
        return {"audio_played": False, "message": str(exc), "prompt": text}


def play_prompt(prompt: str) -> dict:
    """Legacy shim — delegates to play_text."""
    return play_text(prompt)
