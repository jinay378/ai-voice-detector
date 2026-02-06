import base64
import uuid
import os
import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from extract_features import extract_features_from_audio

API_KEY = os.getenv("API_KEY", "MYAPI")

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


class VoiceRequest(BaseModel):
    language: str
    audioBase64: str


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

        features = extract_features_from_audio(mp3_path)
        features_scaled = scaler.transform([features])

        probs = model.predict_proba(features_scaled)[0]
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))

        label = "AI_GENERATED" if prediction == 1 else "HUMAN"

        return {
            "status": "success",
            "language": payload.language,
            "classification": label,
            "confidenceScore": round(confidence, 2),
            "explanation": "Classification based on acoustic and spectral features"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)
