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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fusion_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
EXPECTED_FEATURES = model.n_features_in_

st.write("Model expects:", EXPECTED_FEATURES)

def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch = np.nan_to_num(pitch)
    pitch_mean = np.mean(pitch)

    energy = librosa.feature.rms(y=y)
    energy_mean = np.mean(energy)

    features = np.hstack((mfcc_mean, pitch_mean, energy_mean))

    return features, y, sr, mfcc, pitch, energy

def compute_similarity(f1, f2):
    return np.linalg.norm(f1 - f2)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎙 Voice Input")

    input_mode = st.radio(
        "Choose Input",
        ["🎙 Record Voice", "📁 Upload Recording", "🎧 Sample"]
    )

    temp_path = None

    if input_mode == "🎙 Record Voice":
        audio_bytes = st.audio_input("Record your voice")

        if audio_bytes:
            temp_path = "temp.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes.read())
            st.success("Recording captured")

    elif input_mode == "📁 Upload Recording":
        audio_file = st.file_uploader(
            "Upload your voice recording",
            type=["wav", "mp3", "ogg"]
        )

        if audio_file:
            temp_path = "temp.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())
            st.success("Audio uploaded")

    elif input_mode == "🎧 Sample":
        if os.path.exists("sample.wav"):
            temp_path = "sample.wav"
            st.info("Using sample audio")
        else:
            st.warning("sample.wav not found")

    if st.button("Register Doctor Voice"):
        if temp_path:
            feat, *_ = extract_voice_features(temp_path)
            np.save("doctor_features.npy", feat)
            st.success("Doctor voice registered")
        else:
            st.warning("Please provide audio first")

with col2:
    st.subheader("⌚ Wearable Data")

    hr = st.slider("Heart Rate", 50, 120, 80)
    eda = st.slider("EDA", 0.5, 5.0, 2.0)
    act = st.slider("Activity", 500, 6000, 2500)
    sleep = st.slider("Sleep", 3.0, 9.0, 6.5)

wearable = np.array([hr, eda, act, sleep])

if st.button("🔍 Predict"):

    if not temp_path:
        st.warning("Please provide audio input")
        st.stop()

    voice_feat, y, sr, mfcc, pitch, energy = extract_voice_features(temp_path)

    if os.path.exists("doctor_features.npy"):
        doctor_feat = np.load("doctor_features.npy")
        sim = compute_similarity(voice_feat, doctor_feat)

        if sim < 50:
            st.error("Doctor voice detected")
            st.stop()

    final = np.hstack((voice_feat, wearable))

    if len(final) > EXPECTED_FEATURES:
        final = final[:EXPECTED_FEATURES]
    elif len(final) < EXPECTED_FEATURES:
        final = np.pad(final, (0, EXPECTED_FEATURES - len(final)))

    final = final.reshape(1, -1)

    st.write("Final input size:", final.shape)

    pred = model.predict(final)[0]
    conf = model.predict_proba(final).max()

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

    st.subheader("🧠 Explainability")

    fig_mfcc, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig_mfcc.colorbar(img, ax=ax)
    ax.set_title("MFCC Features")
    st.pyplot(fig_mfcc)

    st.subheader("📈 Pitch & Energy Analysis")

    fig2, ax2 = plt.subplots()
    ax2.plot(pitch, label="Pitch", color="blue")
    ax2.plot(energy[0] * 100, label="Energy", color="red")
    ax2.set_title("Pitch and Energy Over Time")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Value")
    ax2.legend()
    st.pyplot(fig2)

    if pred:
        st.warning("Stress detected")
    else:
        st.success("Normal")