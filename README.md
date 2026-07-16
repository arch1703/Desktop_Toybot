# Desktop Toybot (Baymax)

A two-device robotics stack for a desktop companion robot:
- Jetson Orin Nano runs the main "brain" service and web UI
- Raspberry Pi runs hardware actuation and camera streaming
- Browser UI talks to Jetson, Jetson forwards commands to Pi

## System Architecture

Flow:

Web UI -> Jetson FastAPI -> Raspberry Pi FastAPI -> motors / LEDs / camera

Top-level services:
- jetson_backend: orchestration, mode logic, LLM/voice endpoints, FER inference, UI hosting
- pi_device_service: motor actions, LED control, camera stream, GPIO lifecycle

## Repository Layout

```text
jetson_backend/
	app/main.py                  # Jetson API + WebSocket + UI serving
	app/pi_client.py             # Jetson-to-Pi HTTP bridge
	app/voice_commands.py        # Voice command routing
	app/fer_onnx_manager.py      # FER ONNX inference helper
	static/                      # Browser frontend (index.html, app.js, style.css)
	fer_model/models/            # ONNX model assets

pi_device_service/
	app/main.py                  # Pi API: camera, LED, motor endpoints
	app/motor_controller.py
	app/led_controller.py

Facial-Emotion-Recognition-PyTorch-ONNX/
	# Reference FER training/conversion assets and experiments
```

## Quick Start

### 1. Start Raspberry Pi service (port 9000)

```bash
cd pi_device_service
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

### 2. Start Jetson backend (port 8000)

```bash
cd jetson_backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the UI

Visit:

http://<jetson-ip>:8000

## Required Configuration

The Jetson service reads environment variables for device routing and audio/LLM integration.

Key variables used in jetson_backend/app/main.py:
- PI_BASE_URL (default: http://192.168.10.2:9000)
- PI_STREAM_URL (default derived from PI_BASE_URL)
- GEMINI_API_KEY (needed only for Gemini-backed features)
- AUDIO_INPUT_DEVICE
- AUDIO_OUTPUT_DEVICE

Create a .env in jetson_backend with your deployment values.

## Health and Diagnostics

- Jetson health: GET /health
- Pi status: GET /device/status
- Pi camera stream: GET /camera/stream
- Jetson state: GET /state
- Jetson WebSocket: /ws

## Core API Highlights

Jetson endpoints include:
- mode/state control: /mode, /state
- voice + LLM: /voice-command, /audio/listen, /llm/chat
- robot actuation proxy: /robot/action, /led, /eyes
- vision helpers: /vision/analyze, /vision/describe, /vision/capture-test
- Pi connectivity: /pi/status

Pi endpoints include:
- LED control: /led and /device/led
- motor actions: /motor/action and /device/action
- camera stream: /camera/stream
- health/status: /health and /device/status

## Interaction Modes

| Mode | UI Color | Typical Focus |
| --- | --- | --- |
| kids | red | storytelling, games, guided interaction |
| young_adult | green | puzzles, tutoring, productivity support |
| adult | yellow | guided meditation, FER-assisted responses |

## Notes and Current Scope

- The browser frontend is served from jetson_backend/static.
- Some hardware-facing modules may still include placeholder behavior depending on your wiring and drivers.
- Replace GPIO/audio/motor internals with your board-specific production implementation when deploying to hardware.

## Team Credits

Built collaboratively with:
- Grishma Balgi
- Kamalam Sai Sivakumar

## Development Tips

- Run both services in separate terminals.
- Verify Pi service first before starting Jetson orchestration.
- Use /docs on each service for interactive OpenAPI testing.
	- Jetson docs: http://<jetson-ip>:8000/docs
	- Pi docs: http://<pi-ip>:9000/docs