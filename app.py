import base64
import uuid
import os
import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from extract_features import extract_features_from_audio
import compat  # noqa: F401 — patches sklearn tree loading for old model files

API_KEY = os.getenv("API_KEY", "MYAPI")

app = FastAPI()

# CORS — allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


class VoiceRequest(BaseModel):
    language: str
    audioBase64: str


def classify_audio(audio_path: str, language: str = "auto"):
    """Shared classification logic for both endpoints."""
    features = extract_features_from_audio(audio_path)
    features_scaled = scaler.transform([features])

    probs = model.predict_proba(features_scaled)[0]
    prediction = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = "AI_GENERATED" if prediction == 1 else "HUMAN"

    return {
        "status": "success",
        "language": language,
        "classification": label,
        "confidenceScore": round(confidence, 2),
        "explanation": "Classification based on acoustic and spectral features",
    }


@app.post("/api/voice-detection")
def detect_voice(payload: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    mp3_path = None
    try:
        audio_bytes = base64.b64decode(payload.audioBase64, validate=True)
        mp3_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}.mp3")

        with open(mp3_path, "wb") as f:
            f.write(audio_bytes)

        return classify_audio(mp3_path, payload.language)

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Accept an audio file upload and classify it."""
    tmp_path = None
    try:
        ext = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
        tmp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}{ext}")

        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        return classify_audio(tmp_path)

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
