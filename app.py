import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Mental Health Monitor", layout="centered")

st.markdown("""
<style>
.title {font-size:28px; font-weight:700;}
.section {font-size:20px; font-weight:600;}
.box {padding:15px; border-radius:10px; background-color:#1c1f26;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Mental Health Monitoring System</div>', unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fusion_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Audio Processor
# -----------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten()
        self.audio_frames.extend(audio)
        return frame

# -----------------------------
# Voice Feature Extraction
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
# Speaker Similarity
# -----------------------------
def compute_similarity(f1, f2):
    return np.linalg.norm(f1 - f2)

# -----------------------------
# UI Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section">🎙 Voice Input</div>', unsafe_allow_html=True)

    webrtc_ctx = webrtc_streamer(
        key="audio",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    st.info("Step 1: Register doctor voice → Step 2: Patient speaks")

    # -----------------------------
    # Register Doctor Voice
    # -----------------------------
    if st.button("🎙 Register Doctor Voice"):
        if webrtc_ctx.audio_processor and len(webrtc_ctx.audio_processor.audio_frames) > 0:
            audio_data = np.array(webrtc_ctx.audio_processor.audio_frames, dtype=np.float32)

            audio_data = audio_data / np.max(np.abs(audio_data))
            sf.write("doctor.wav", audio_data, 16000)

            doctor_feat, *_ = extract_voice_features("doctor.wav")
            np.save("doctor_features.npy", doctor_feat)

            st.success("Doctor voice registered successfully")
        else:
            st.warning("Record doctor voice first")

with col2:
    st.markdown('<div class="section">⌚ Wearable Data</div>', unsafe_allow_html=True)

    option = st.selectbox("Scenario", [
        "Normal",
        "Stress",
        "Anxiety",
        "Depression",
        "Custom"
    ])

    if option == "Normal":
        hr, eda, act, sleep = 72, 1.2, 3500, 7.5
    elif option == "Stress":
        hr, eda, act, sleep = 82, 2.0, 2600, 6.8
    elif option == "Anxiety":
        hr, eda, act, sleep = 95, 3.5, 2200, 5.8
    elif option == "Depression":
        hr, eda, act, sleep = 68, 1.8, 1200, 5.5
    else:
        hr = st.slider("Heart Rate", 50, 120, 80)
        eda = st.slider("EDA", 0.5, 5.0, 2.0)
        act = st.slider("Activity", 500, 6000, 2500)
        sleep = st.slider("Sleep", 3.0, 9.0, 6.5)

wearable = np.array([hr, eda, act, sleep])

# -----------------------------
# Predict
# -----------------------------
if st.button("🔍 Predict"):

    if webrtc_ctx.audio_processor is None or len(webrtc_ctx.audio_processor.audio_frames) == 0:
        st.warning("Please record voice first")
    else:
        try:
            audio_data = np.array(webrtc_ctx.audio_processor.audio_frames, dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))

            temp_path = "temp.wav"
            sf.write(temp_path, audio_data, 16000)

            voice_feat, y, sr, mfcc, pitch, energy = extract_voice_features(temp_path)

            # -----------------------------
            # Speaker Filter
            # -----------------------------
            if os.path.exists("doctor_features.npy"):
                doctor_feat = np.load("doctor_features.npy")

                similarity = compute_similarity(voice_feat, doctor_feat)

                if similarity < 50:
                    st.error("Doctor voice detected — please record patient voice")
                    st.stop()

            # -----------------------------
            # Prediction
            # -----------------------------
            final = np.hstack((voice_feat, wearable)).reshape(1, -1)

            pred = model.predict(final)[0]
            conf = model.predict_proba(final).max()

            # -----------------------------
            # Dashboard
            # -----------------------------
            st.markdown("## 📊 Dashboard")

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk", "HIGH" if pred else "LOW")
            c2.metric("Confidence", f"{conf:.2f}")
            c3.metric("HR", hr)

            # -----------------------------
            # Gauge
            # -----------------------------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf*100,
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                }
            ))
            st.plotly_chart(fig)

            # -----------------------------
            # Explainability
            # -----------------------------
            st.markdown("## 🧠 Voice Explainability")

            fig_mfcc, ax = plt.subplots()
            img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
            fig_mfcc.colorbar(img, ax=ax)
            st.pyplot(fig_mfcc)

            fig2, ax2 = plt.subplots()
            ax2.plot(pitch, label="Pitch")
            ax2.plot(energy[0]*100, label="Energy")
            ax2.legend()
            st.pyplot(fig2)

            # -----------------------------
            # Insight
            # -----------------------------
            st.markdown("## 🧠 AI Insight")

            if pred:
                st.warning("Signs of stress/anxiety detected")
            else:
                st.success("Voice appears stable")

        except Exception as e:
            st.error("Error")
            st.exception(e)