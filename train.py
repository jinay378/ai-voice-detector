import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features_from_audio

AI_DIR = "ai"
HUMAN_DIR = "human"

X = []
y = []

print("🔹 Loading AI WAV samples...")
ai_files = [f for f in os.listdir(AI_DIR) if f.lower().endswith(".wav")]
print(f"   Found {len(ai_files)} AI files")

for f in ai_files:
    features = extract_features_from_audio(os.path.join(AI_DIR, f))
    X.append(features)
    y.append(1)

print("🔹 Loading Human WAV samples...")
human_files = [f for f in os.listdir(HUMAN_DIR) if f.lower().endswith(".wav")]
print(f"   Found {len(human_files)} Human files")

for f in human_files:
    features = extract_features_from_audio(os.path.join(HUMAN_DIR, f))
    X.append(features)
    y.append(0)

if len(X) == 0:
    raise RuntimeError("❌ No WAV files found in ai/ or human/")

X = np.array(X)
y = np.array(y)

print(f"✅ Total samples: {len(X)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
model.fit(X_scaled, y)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🎉 Training complete. model.pkl & scaler.pkl saved.")