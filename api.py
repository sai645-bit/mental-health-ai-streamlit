from fastapi import FastAPI, UploadFile, File
import numpy as np
import joblib
import librosa
import tempfile
import os

app = FastAPI()

model = joblib.load("fusion_model.pkl")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.nanmean(pitch)

    energy = librosa.feature.rms(y=y)
    energy_mean = np.mean(energy)

    return np.hstack((mfcc_mean, pitch_mean, energy_mean))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    features = extract_features(temp_path)

    # dummy wearable
    wearable = np.array([80, 2.0, 2500, 6.5])

    final = np.hstack((features, wearable)).reshape(1, -1)

    pred = int(model.predict(final)[0])
    conf = float(model.predict_proba(final).max())

    os.remove(temp_path)

    return {
        "prediction": pred,
        "confidence": conf
    }