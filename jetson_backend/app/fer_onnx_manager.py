from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests


# -----------------------------
# Config
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "fer_model" / "models" / "fer_model.onnx"

# Raspberry Pi camera stream
RPI_BASE_URL = "http://192.168.10.2:9000"
RPI_CAMERA_STREAM_URL = f"{RPI_BASE_URL}/camera/stream"

# OpenCV built-in face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


RAW_EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def map_to_baymax_emotion(raw_emotion: str) -> str:
    if raw_emotion == "happy":
        return "happy"

    if raw_emotion in ["neutral", "surprise"]:
        return "neutral"

    if raw_emotion in ["angry", "disgust", "fear", "sad"]:
        return "stressed"

    return "neutral"


# -----------------------------
# Load ONNX model once
# -----------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# -----------------------------
# Helpers
# -----------------------------

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def preprocess_face(face_gray):
    face = cv2.resize(face_gray, (48, 48))
    face = face.astype(np.float32) / 255.0

    # Shape: batch, channel, height, width
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=0)

    return face


def get_one_frame_from_rpi_stream():
    """
    Connects to the Raspberry Pi MJPEG stream and extracts one JPEG frame.
    """
    try:
        response = requests.get(
            RPI_CAMERA_STREAM_URL,
            stream=True,
            timeout=5,
        )

        if response.status_code != 200:
            return None, f"RPi stream returned status {response.status_code}"

        bytes_buffer = b""

        for chunk in response.iter_content(chunk_size=1024):
            bytes_buffer += chunk

            start = bytes_buffer.find(b"\xff\xd8")  # JPEG start
            end = bytes_buffer.find(b"\xff\xd9")    # JPEG end

            if start != -1 and end != -1:
                jpg_bytes = bytes_buffer[start:end + 2]

                image_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                response.close()

                if frame is None:
                    return None, "Could not decode JPEG frame"

                return frame, None

        return None, "Could not extract frame from stream"

    except requests.exceptions.RequestException as e:
        return None, f"Could not connect to RPi camera stream: {e}"


# -----------------------------
# Main FER function
# -----------------------------

def run_fer_once():
    if not MODEL_PATH.exists():
        return {
            "success": False,
            "error": f"ONNX model not found at {MODEL_PATH}",
        }

    if face_cascade.empty():
        return {
            "success": False,
            "error": "OpenCV Haar cascade could not be loaded",
        }

    frame, error = get_one_frame_from_rpi_stream()

    if error is not None:
        return {
            "success": False,
            "error": error,
        }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return {
            "success": False,
            "error": "No face detected",
        }

    # Use largest face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    face_gray = gray[y:y + h, x:x + w]
    input_tensor = preprocess_face(face_gray)

    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]

    probs = softmax(logits)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    raw_emotion = RAW_EMOTION_DICT[class_id]
    emotion = map_to_baymax_emotion(raw_emotion)

    return {
        "success": True,
        "emotion": emotion,
        "raw_emotion": raw_emotion,
        "confidence": confidence,
        "source": "raspberry_pi_camera_stream",
        "rpi_camera_url": RPI_CAMERA_STREAM_URL,
        "face_box": {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
        },
    }
