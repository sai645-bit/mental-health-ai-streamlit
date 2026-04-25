import plotly.graph_objects as go
import streamlit as st
import numpy as np
import joblib
import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

st.set_page_config(page_title="AI Mental Health Monitor", layout="centered")
st.title("🧠 AI Mental Health Monitoring System")

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fusion_model.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

EXPECTED_FEATURES = model.n_features_in_
st.write("Model expects:", EXPECTED_FEATURES)

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.nanmean(pitch)

    energy = librosa.feature.rms(y=y)
    energy_mean = np.mean(energy)

    features = np.hstack((mfcc_mean, pitch_mean, energy_mean))

    return features, y, sr, mfcc, pitch, energy

# -----------------------------
# Similarity
# -----------------------------
def compute_similarity(f1, f2):
    return np.linalg.norm(f1 - f2)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# INPUT (UPDATED WITH RECORDER)
# -----------------------------
with col1:
    st.subheader("🎙 Voice Input")

    input_mode = st.radio(
        "Choose Input",
        ["🎙 Record Voice", "📁 Upload Recording", "🎧 Sample"]
    )

    temp_path = None

    # 🎙 RECORD VOICE
    if input_mode == "🎙 Record Voice":
        audio_bytes = st.audio_input("Record your voice")

        if audio_bytes:
            temp_path = "temp.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes.read())

            st.success("Recording captured")

    # 📁 Upload
    elif input_mode == "📁 Upload Recording":
        audio_file = st.file_uploader(
            "Upload your voice recording",
            type=["wav", "mp3", "ogg"]
        )

        if audio_file:
            temp_path = "temp.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())

            st.success("Audio uploaded successfully")

    # 🎧 Sample
    elif input_mode == "🎧 Sample":
        if os.path.exists("sample.wav"):
            temp_path = "sample.wav"
            st.info("Using sample audio")
        else:
            st.warning("sample.wav not found")

    # Register doctor voice
    if st.button("Register Doctor Voice"):
        if temp_path:
            feat, *_ = extract_voice_features(temp_path)
            np.save("doctor_features.npy", feat)
            st.success("Doctor voice registered")
        else:
            st.warning("Please provide audio first")

# -----------------------------
# Wearable Data
# -----------------------------
with col2:
    st.subheader("⌚ Wearable Data")

    hr = st.slider("Heart Rate", 50, 120, 80)
    eda = st.slider("EDA", 0.5, 5.0, 2.0)
    act = st.slider("Activity", 500, 6000, 2500)
    sleep = st.slider("Sleep", 3.0, 9.0, 6.5)

wearable = np.array([hr, eda, act, sleep])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    if not temp_path:
        st.warning("Please provide audio input")
        st.stop()

    voice_feat, y, sr, mfcc, pitch, energy = extract_voice_features(temp_path)

    # Speaker filter
    if os.path.exists("doctor_features.npy"):
        doctor_feat = np.load("doctor_features.npy")
        sim = compute_similarity(voice_feat, doctor_feat)

        if sim < 50:
            st.error("Doctor voice detected")
            st.stop()

    # Combine features
    final = np.hstack((voice_feat, wearable))

    # Feature size fix
    if len(final) > EXPECTED_FEATURES:
        final = final[:EXPECTED_FEATURES]
    elif len(final) < EXPECTED_FEATURES:
        final = np.pad(final, (0, EXPECTED_FEATURES - len(final)))

    final = final.reshape(1, -1)

    st.write("Final input size:", final.shape)

    # Prediction
    pred = model.predict(final)[0]
    conf = model.predict_proba(final).max()

    # Dashboard
    st.subheader("📊 Dashboard")

    c1, c2 = st.columns(2)
    c1.metric("Risk", "HIGH" if pred else "LOW")
    c2.metric("Confidence", f"{conf:.2f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        title={'text': "Risk Score"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig)

    # Explainability
    st.subheader("🧠 Explainability")

    fig_mfcc, ax = plt.subplots()
    librosa.display.specshow(mfcc, ax=ax)
    st.pyplot(fig_mfcc)

    # Final insight
    if pred:
        st.warning("Stress detected")
    else:
        st.success("Normal")