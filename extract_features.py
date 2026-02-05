import numpy as np
import librosa
from pydub import AudioSegment
import os
import uuid

# 🔧 SET YOUR FFMPEG PATH HERE
AudioSegment.converter = r"C:\Users\Lenovo-PC\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"


def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")


def extract_features_from_audio(audio_path: str) -> np.ndarray:
    temp_wav = None

    # Load audio
    if audio_path.lower().endswith(".wav"):
        y, sr = librosa.load(audio_path, sr=16000)
    else:
        temp_wav = f"temp_{uuid.uuid4().hex}.wav"
        convert_mp3_to_wav(audio_path, temp_wav)
        y, sr = librosa.load(temp_wav, sr=16000)

    sr = int(sr)  # 🔥 FIX: ensure integer

    # Handle silent audio
    if np.all(y == 0):
        y = np.random.normal(0.0, 1e-6, size=sr)

    # Ensure minimum length (1 second)
    if len(y) < sr:
        pad_width = int(sr - len(y))  # 🔥 FIX: ensure integer
        y = np.pad(y, (0, pad_width), mode="constant")

    # ---------------- FEATURE EXTRACTION ----------------

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zcr
    ])

    # 🔥 CRITICAL: remove NaN / inf
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if temp_wav and os.path.exists(temp_wav):
        os.remove(temp_wav)

    return features