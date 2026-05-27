# DeskBot Jetson Backend

This FastAPI service runs on the Jetson Orin Nano.

The Jetson is the main brain. It handles:
- current mode
- voice command text
- robot action validation
- communication with the Raspberry Pi device service

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API Docs: http://127.0.0.1:8000/docs