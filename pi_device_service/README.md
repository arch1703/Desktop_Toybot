# DeskBot Raspberry Pi Device Service

This FastAPI service runs on the Raspberry Pi.

The Pi receives commands from the Jetson and controls:
- display
- audio prompts
- buttons
- two-motor drive base

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

API Docs: http://127.0.0.1:9000/docs