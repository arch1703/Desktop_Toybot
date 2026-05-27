## Deskbot Baymax

Flow: Web app → Jetson FastAPI → Raspberry Pi FastAPI → motors/display/audio/buttons.

### Running

**1. Raspberry Pi** (port 9000):
```bash
cd pi_device_service
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

**2. Jetson Orin Nano** (port 8000):
```bash
cd jetson_backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**3. Open the web UI** at `http://<jetson-ip>:8000` in a browser.

### Health Checks

- Jetson: `GET /health`
- Pi: `GET /device/status`

### Modes (color-coded in UI)

| Mode | Color | Features |
|------|-------|----------|
| Kids | Red | Trajectory mapping (mat), storytelling, snake game, memory game |
| Young Adult | Green | Puzzles, peer tutoring, trajectory mapping (desktop) |
| Adult | Yellow | Guided meditation, facial recognition |

### Keyboard Shortcuts (Motor Controls)

- Arrow keys: drive forward/backward, turn left/right
- Spacebar: stop

### Architecture

- `jetson_backend/` — FastAPI brain + web UI (static files)
- `pi_device_service/` — FastAPI hardware interface (motors, display, audio, buttons)
- `jetson_backend/static/` — Frontend (HTML/CSS/JS, served by Jetson)

### Placeholders

Audio, display, and motor controllers use placeholder functions.
Replace with actual GPIO/PWM/audio drivers when hardware is connected.